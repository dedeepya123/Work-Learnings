# RoPE: Math → Tensor Shapes → HuggingFace Code → Compile-Time Graph Requirements → Qualcomm Implementation

Design-review-style walkthrough, same rigor as the KV-cache and masking mentor documents.
No summarizing up front — build it step by step, the way it actually has to be explained
to another engineer who will go read the code right after this conversation.

Sources: `transformers/models/gemma4/modeling_gemma4.py` (`Gemma4TextRotaryEmbedding`,
`apply_rotary_pos_emb`, `rotate_half`), `transformers/modeling_rope_utils.py`
(`_compute_proportional_rope_parameters`), `pythonic_api/nano/models/gemma4_text/reauthoring.py`
(`QcApplyRopeSingle`, `create_position_embeddings`), `Nano/NanoV4/qlib/qadaptation.py`
(`ApplyRopeSingle`, `MulModule`), `Nano/NanoV4/air/nanov4/utils.py`
(`llm_create_position_embeddings`).

---

## Step 1 — The math, from first principles (not the code yet)

**Why RoPE exists at all.** Attention computes `Q · Kᵀ`. A transformer with no positional
signal treats `Q·K` identically regardless of where in the sequence `Q` and `K` came
from — attention is fundamentally a set operation. We need `Q·K` to depend on the
*relative* distance between the query token and the key token, not just their content.

**The core trick.** Take each pair of dimensions in a Q or K vector, treat it as one
complex number, and **rotate** that complex number by an angle proportional to the
token's absolute position:
```
z = x1 + i·x2                  (pair up two real dims as one complex number)
z' = z · e^{iθ·pos}             (rotate by angle θ·pos)
```
Do this for every token, at every pair of dimensions (each pair gets its **own** angle
`θ`, more on that below), for both Q and K.

**Why this makes the dot product relative-position-aware.** This is the one piece of
algebra worth actually doing, because everything downstream depends on trusting it:
```
Q at position m, rotated:  Q' = Q · e^{iθm}
K at position n, rotated:  K' = K · e^{iθn}

Q' · conj(K') = Q · e^{iθm} · conj(K · e^{iθn})
              = Q · conj(K) · e^{iθm} · e^{-iθn}          (conj of a product distributes; conj(e^{ix}) = e^{-ix})
              = Q · conj(K) · e^{iθ(m-n)}
```
The result depends on `θ(m − n)` — the **difference** between positions, not the absolute
positions themselves. That's the entire point of RoPE in one line of algebra: rotate Q and
K each by their own absolute angle, and the *relative* angle `θ(m-n)` is what survives
into the attention score.

**Why multiple angles, not just one.** A single frequency `θ` would make position
encoding periodic and ambiguous (position 0 and position `2π/θ` would look identical).
RoPE instead splits the head dimension into `head_dim/2` **pairs**, each pair rotating at
a *different* frequency — some pairs rotate slowly (capture long-range position), some
rotate fast (capture fine-grained local position). This is a direct analog of how
sinusoidal position encodings in the original Transformer paper used multiple
frequencies — RoPE's innovation is applying that idea as a *rotation of Q/K*, not as an
*additive* embedding.

**Checkpoint before touching code:** you should now be able to say, without looking
anything up: *"RoPE rotates Q and K by position-dependent angles, one angle per
dimension-pair, so that their dot product depends only on relative position — because
rotating two vectors by different angles and then taking their inner product leaves only
the angle difference."* If that sentence doesn't yet feel automatic, don't move on.

---

## Step 2 — Toy numeric example (tiny, hand-checkable)

`head_dim = 4` → 2 pairs. Say pair-0 uses frequency `θ0 = 1.0` rad/position, pair-1 uses
`θ1 = 0.1` rad/position (in reality these come from a formula — Step 4 — but for intuition
just pick two illustrative numbers).

Token at position `m=2`, query vector `Q = [1, 0, 1, 0]` (pair0 = `(1,0)`, pair1 = `(1,0)`,
i.e. both "pointing along the real axis" before rotation).

Rotate pair-0 by `θ0·m = 1.0·2 = 2.0` rad, pair-1 by `θ1·m = 0.1·2 = 0.2` rad:
```
pair0: (1,0) rotated by 2.0 rad → (cos2.0, sin2.0) ≈ (-0.416, 0.909)
pair1: (1,0) rotated by 0.2 rad → (cos0.2, sin0.2) ≈ (0.980, 0.199)
Q_rotated ≈ [-0.416, 0.909, 0.980, 0.199]
```
That's it — that's literally what RoPE computes, once per token, once per pair. Everything
in the code from here on is just "do this efficiently and for every position/pair at once,
using `cos`/`sin` tables instead of calling `rotate()` in a loop."

---

## Step 3 — Tensor shapes (before any formula, just the shape bookkeeping)

For a batch of `B` sequences, `S` tokens each, `head_dim = D`:
```
Q, K:            [B, num_heads, S, D]           (per-head, per-token vectors)
inv_freq:        [D/2]                           (one frequency per PAIR — computed once, config-derived)
position_ids:    [B, S]                           (which absolute position each token sits at)
freqs:           [B, S, D/2]                      (position × frequency, per pair)
cos, sin:        [B, S, D]                        (duplicated to full D — see Step 5 for why)
```
The shape that should stick in your head: **`inv_freq` has size `D/2`** — one frequency
per *pair* of dimensions, not per dimension. Everything about "why is this halved /
duplicated" in the code traces back to this one fact.

---

## Step 4 — Where the frequencies (`inv_freq`) actually come from

Formula (`modeling_gemma4.py:1610`, `compute_default_rope_parameters`):
```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
```
`torch.arange(0, dim, 2)` → `[0, 2, 4, ..., dim-2]` — `dim/2` values. Divide by `dim`,
raise `base` to that power, invert. Result: `dim/2` frequencies, **geometrically spaced**
between `1.0` (index 0) and roughly `1/base` (index `dim/2 - 1`).

**Why geometric spacing, intuitively:** low-index pairs get frequency ≈1 → rotate fast per
position → these pairs saturate (wrap around `2π`) quickly, useful for distinguishing
*nearby* positions. High-index pairs get frequency ≈`1/base` (base is typically 10,000 or
1,000,000 for Gemma4) → rotate extremely slowly → these pairs stay nearly unrotated even
across thousands of positions, useful for distinguishing *far-apart* positions without
wrapping ambiguity. This is the RoPE analog of the original Transformer's multi-frequency
sinusoidal encoding — same reasoning, different mechanism (rotation vs. addition).

**Gemma4-specific wrinkle — two separate `base` values per layer type**, confirmed in
`configuration_gemma4.py`:
```python
default_rope_params = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention":    {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
}
```
Sliding layers use plain default RoPE with `base=10,000` over the full `head_dim`. Full
layers use `"proportional"` RoPE (`modeling_rope_utils.py:179`) with `base=1,000,000` and
`partial_rotary_factor=0.25` — meaning **only 25% of `global_head_dim` actually gets
rotated**; the "proportional" variant is specifically designed so the *output* size is
still the full `head_dim` regardless of that factor (see its docstring: "proportional RoPE
will always return an encoding that is the size of `head_dim`"), by adjusting the
*effective base* internally rather than shrinking the returned tensor. This is why full
and sliding layers get **two entirely separate `inv_freq` buffers** — `Gemma4TextRotaryEmbedding.__init__`
loops over `self.layer_types` (a *set*, so just `{"full_attention","sliding_attention"}`,
two iterations regardless of `num_hidden_layers`) and registers
`full_attention_inv_freq` / `sliding_attention_inv_freq` as **two separate buffers**.

---

## Step 5 — `Gemma4TextRotaryEmbedding.forward` — the `cos`/`sin` table build

```python
# modeling_gemma4.py:1615-1633
def forward(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")              # [D/2], picked by layer_type
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

    inv_freq_expanded = inv_freq[None, :, None].expand(B, -1, 1)     # [B, D/2, 1]
    position_ids_expanded = position_ids[:, None, :].float()          # [B, 1, S]

    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)   # [B,D/2,1]@[B,1,S] -> [B,D/2,S] -> transpose -> [B,S,D/2]
    emb = torch.cat((freqs, freqs), dim=-1)                            # [B, S, D]   <- DUPLICATED, see below
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos, sin
```

**Purpose:** for every (token, pair) combination, compute the rotation angle
`θ_pair × position`, then the `cos`/`sin` of that angle — once, reused by every layer of
the matching type.

**Inputs:** `position_ids` `[B, S]`. **Outputs:** `cos`, `sin`, each `[B, S, D]`.

**The matmul, worked as a shape trace, not just symbols:**
`inv_freq_expanded` is `[B, D/2, 1]`, `position_ids_expanded` is `[B, 1, S]`. Batched
matmul → `[B, D/2, S]` — this is literally the **outer product** of frequencies and
positions, batched: entry `[b, i, s]` = `inv_freq[i] * position_ids[b,s]` = exactly
`θ_i × position_s`, for every pair `i` and every token `s` at once. `.transpose(1,2)` →
`[B, S, D/2]` just reorders axes so sequence comes before the frequency-pair axis (matching
how everything else in the model is laid out — `[B, S, ...]`).

**Why `torch.cat((freqs, freqs), dim=-1)` — the duplication, explained, not just noted.**
`freqs` is `[B,S,D/2]` — one angle per pair. But the *code representation* HF uses for
"rotate this pair by this angle" is **not** the direct complex-multiply from Step 1 — it's
the algebraically-equivalent **rotate-half trick**:
```python
def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, ...):
    return (x * cos) + (rotate_half(x) * sin)
```
This formula needs `cos`/`sin` to be **full `D`-length**, matching `x`'s full width — not
`D/2`. But there are only `D/2` *distinct* angles. So the fix is: **duplicate** the `D/2`
angles into both halves — `emb = cat(freqs, freqs)` → `cos`/`sin` become `[B,S,D]`, where
position `i` and position `i+D/2` hold the *same* angle. Verify this is mathematically
identical to Step 1's direct complex-multiply (do this derivation once, it's worth it):

Let `x1_i, x2_i` be the two halves of `x` at pair-index `i` (i.e. `x1_i = x[...,i]`,
`x2_i = x[...,i+D/2]`). Since `cos_i == cos_{i+D/2}` (duplicated), expand
`(x*cos + rotate_half(x)*sin)` at positions `i` and `i+D/2`:
```
output[i]     = x1_i·cos_i − x2_i·sin_i        (rotate_half puts -x2 at position i)
output[i+D/2] = x2_i·cos_i + x1_i·sin_i         (rotate_half puts x1 at position i+D/2)
```
Compare against Step 1's complex rotation `z' = z·e^{iθ} = (x1cosθ − x2sinθ) + i(x1sinθ + x2cosθ)`
— **exact match.** `rotate_half` + duplicated `cos`/`sin` is a vectorized, loop-free way to
express "rotate every `(x1_i, x2_i)` pair by `θ_i`," entirely through elementwise multiply
and one concat/negate — no explicit complex dtype, no per-pair loop.

**This duplication is the single most important fact for the rest of this document** —
everything Qualcomm changes about RoPE representation is, at its core, "we don't want this
duplication and this concat-negate; give us the two halves directly instead." Hold onto
that.

---

## Step 6 — `apply_rotary_pos_emb` — applying the table to Q and K

```python
# modeling_gemma4.py:1210-1232
def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1, rope_operator=None):
    cos = cos.unsqueeze(unsqueeze_dim)     # [B,S,D] -> [B,1,S,D], broadcasts over heads
    sin = sin.unsqueeze(unsqueeze_dim)
    if rope_operator is not None:
        return rope_operator(x, cos, sin)
    return (x * cos) + (rotate_half(x) * sin)
```
Called from `Gemma4TextAttention.forward`:
```python
query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)   # applied to Q
key_states   = apply_rotary_pos_emb(key_states,   cos, sin, unsqueeze_dim=2)   # applied to K
# value_states: NEVER passed through apply_rotary_pos_emb — see Step 1's algebra:
#               V is never dotted against anything, only weight-summed after softmax,
#               so rotating it would add cost for zero benefit and actively corrupt
#               its content direction.
```
Shapes at this call site: `query_states`/`key_states` are `[B, S, num_heads, D]` at the
point `apply_rotary_pos_emb` is called (before the `.transpose(1,2)` that happens right
after) — `unsqueeze_dim=2` inserts the broadcast axis at the *heads* position, since
`cos`/`sin` (`[B,S,D]`) don't have a heads dimension at all — the same rotation angle
applies identically to every attention head. (Contrast with the mask, which also has a
`1`-sized broadcast axis for heads — same underlying reason: this quantity doesn't vary
per head.)

---

## Step 7 — Why any of this is a *compile-time graph problem* at all

Stop and ask, explicitly, the same question asked for cache/mask: **what here is fixed at
compile time, and what varies at runtime?**

- `inv_freq` — **compile-time constant.** Purely a function of `config` (`head_dim`,
  `rope_theta`, `layer_type`) — never touches an input tensor. This is *why* it's
  registered as a `buffer`, not computed fresh: it's baked into the model at
  construction/load time.
- `position_ids` — **runtime value** (a scalar-derived tensor, per Lesson 1 of the earlier
  masking whiteboard — `cache_index + arange(ARN)`).
- `freqs = inv_freq_expanded @ position_ids_expanded` — this matmul **mixes a compile-time
  constant with a runtime value**. On an eager PyTorch graph, that's just... a matmul,
  every call, no issue. On a **compiled, fixed-shape HTP graph**, this raises the same
  question as everything else we've covered: is this matmul something the compiled graph
  can execute as-is, or does something about it need pre-baking?

**The actual constraint, stated precisely:** the matmul itself (`[B,D/2,1] @ [B,1,S]`) has
a perfectly static shape — `D/2` and `S(=ARN)` are both compile-time constants. So
*unlike* the cache/mask case, **this matmul is not the problem.** What *is* worth
re-examining is the **representation** of the output — `rotate_half` (a concat + negate,
re-run every call) and the `cos`/`sin` **duplication** (computing `D` values when only
`D/2` are distinct) — both are wasted, HTP-unfriendly work, not because they're dynamically
shaped, but because they're **inefficient/awkward ops for the target hardware and for
per-op quantization**, exactly the "op-representation adaptation" category from the
generic framework (recall: shape/staticness vs. op-representation vs.
interface/orchestration).

This is worth stating explicitly in a design review, because it's a genuinely different
*kind* of adaptation reason than the cache/mask ones: **RoPE's adaptation is not primarily
about fixed shapes — it's about avoiding wasteful/awkward ops** (concat+negate,
duplicated redundant values) **and giving the AIMET quantizer clean, individually-targetable
ops.** The compile-time-vs-runtime lens still applies (and explains why `inv_freq`/the
angle tables can be precomputed once, config-only), but it is not the dominant force here
the way it was for cache/mask.

---

## Step 8 — Qualcomm's representation: split real/imaginary, no duplication, no `rotate_half`

Two adapted implementations exist in the codebase (same two-repo situation as the
cache/mask docs) — same idea, same math, near-identical code:

**`reauthoring.py`'s `QcApplyRopeSingle`** and **`qadaptation.py`'s `ApplyRopeSingle`** —
both implement the literal complex-multiply from Step 1, directly, on **pre-halved**
real/imaginary inputs, instead of `rotate_half` on a full-width duplicated-`cos`/`sin`
input:

```python
# qadaptation.py — ApplyRopeSingle.forward
def forward(self, x_real, x_im, rope_vals):
    rope_real, rope_im = rope_vals                                    # each [.., D/2] — NOT duplicated
    x_prod_real = self.mul_x_real_rope_real(x_real, rope_real) - self.mul_x_im_rope_im(x_im, rope_im)
    x_prod_im   = self.mul_x_real_rope_im(x_real, rope_im)   + self.mul_x_im_rope_real(x_im, rope_real)
    x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x_real.shape[:-1], -1)
    return x
```
This is literally `z' = (x1+ix2)(cos+isin) = (x1·cos − x2·sin) + i(x1·sin + x2·cos)`,
written out as four real multiplies and two adds — **no `rotate_half`, no negate-via-concat,
no duplicated `cos`/`sin` table.** The caller passes in `x_real`/`x_im` as the two
**already-split halves** of Q or K, and `rope_real`/`rope_im` as the **half-length**
(`D/2`-sized) `cos`/`sin` — not the duplicated `D`-length version HF builds.

Concretely, from `reauthoring.py`'s `QcGemma4TextAttention.forward`:
```python
q_half = query_states.shape[-1] // 2
query_states = self.apply_rope_fn(query_states[..., :q_half], query_states[..., q_half:], position_embeddings)
```
`query_states[..., :q_half]` and `[..., q_half:]` **are** `x1`/`x2` from Step 1's algebra —
the split is done explicitly by the caller, once, rather than implicitly inside
`rotate_half` every time it's called.

And from `reauthoring.py`'s `create_position_embeddings` (the QC-side replacement for
`Gemma4TextRotaryEmbedding.forward`'s duplication step):
```python
def _rope_vals(layer_type):
    dim = config.head_dim if layer_type == "sliding_attention" else config.global_head_dim
    cos, sin = rotary_emb(x, position_ids, layer_type)         # still calls the SAME HF rotary_emb!
    cos, sin = cos.unsqueeze(dim=1), sin.unsqueeze(dim=1)
    return cos[..., :dim // 2], sin[..., :dim // 2]            # <-- SLICE OFF the duplicated half
```
This is the key move: **QC still calls the exact same `Gemma4TextRotaryEmbedding` HF
module** to get `inv_freq`-derived angles — no reimplementation of the frequency math at
all (Step 4's formula is untouched) — but immediately **slices off the redundant
duplicated half** (`cos[..., :dim//2]`) right after, since QC's `ApplyRopeSingle` only
ever needs `D/2` values, never `D`. Same confirmed independently in
`air/nanov4/utils.py:73-74`: `cos = cos[:,:,:,:dim//2]`.

---

## Step 9 — Why named `MulModule`s, specifically (the quantization-graph reason)

```python
class ApplyRopeSingle(nn.Module):
    def __init__(self):
        self.mul_x_real_rope_real = MulModule()
        self.mul_x_im_rope_im     = MulModule()
        self.mul_x_real_rope_im   = MulModule()
        self.mul_x_im_rope_real   = MulModule()
```
Four separate, individually-instantiated elementwise-multiply submodules, instead of four
inline `*` operators. **Why this matters, precisely:** when this model is traced into a
graph and handed to AIMET's `QuantizationSimModel`, each `nn.Module` instance becomes a
**distinct, individually-addressable node** the quantizer can assign its own calibrated
encoding (scale/offset) to. An inline `x_real * rope_real` Python operator, if it doesn't
get lifted into its own named module, risks being harder for the quantizer/graph tooling
to target individually — or at minimum, is less legible when inspecting the traced graph
during debugging (`onnx_visualizer.py`-style inspection, or reading a
`FloatActivationError` traceback). This is the same pattern already seen for `Add`, `Mul`,
`Matmul`, `Softmax` etc. wrapped as `CraftModule` subclasses in vanilla `modeling_gemma4.py`
itself (`Matmul`, `Div`, `Concat`, ... at the top of that file) — Gemma4's own authors
already anticipated this need generally; Qualcomm's `MulModule` wrapping inside RoPE is the
same idea applied specifically to the four multiplies RoPE's complex-multiply needs.

---

## Step 10 — Full side-by-side comparison table

| Aspect | HF (vanilla) | Qualcomm (adapted) |
|---|---|---|
| Frequency table (`inv_freq`) | `1/(base^(arange(0,D,2)/D))`, `[D/2]`, per layer_type — computed once, registered as buffer | **Identical** — same HF module (`Gemma4TextRotaryEmbedding`) reused as-is; no reimplementation |
| `cos`/`sin` table width | `D` (duplicated: `cat(freqs,freqs)`) | `D/2` — duplicated half **sliced off** right after calling the same HF module |
| Rotation formula | `x*cos + rotate_half(x)*sin` — `rotate_half` = slice + negate + concat | `x1·cos − x2·sin`, `x1·sin + x2·cos` computed directly as 4 multiplies + 2 adds — no concat/negate op |
| Input split | Implicit — `rotate_half` internally slices `x` into two halves every call | Explicit — caller slices Q/K into `x_real`/`x_im` **once**, passes both halves in |
| Elementwise multiplies | Inline `*` operators, not individually addressable in the traced graph | Four separate named `MulModule` instances — individually quantizable/inspectable graph nodes |
| Applied to V? | No (never called on `value_states`) | No — **identical reasoning**, unrelated to the adaptation; V is never dotted against anything |
| Per-layer-type angle split (full vs sliding, different `base`/`head_dim`) | Yes — two buffers, two dict entries, chosen by `layer_type` string | **Identical** — same two-buffer split, same selection logic; this part of RoPE was never touched by adaptation at all |
| What triggered the change | N/A | Op-representation adaptation (avoid concat+negate, avoid redundant duplicated values, give quantizer clean per-op targets) — **not** a fixed-shape/compile-time issue, unlike cache/mask |

---

## Step 11 — Worked numeric trace, HF vs QC, side by side

Reuse Step 2's toy numbers: `head_dim=4` (→ `D/2=2` pairs), position `m=2`,
`θ0=1.0, θ1=0.1`, `Q = [1,0,1,0]`.

**HF path:**
```
freqs = [θ0·m, θ1·m] = [2.0, 0.2]                                    shape [D/2] = [2]
emb = cat(freqs,freqs) = [2.0, 0.2, 2.0, 0.2]                         shape [D]   = [4]
cos = [cos2.0, cos0.2, cos2.0, cos0.2] ≈ [-0.416, 0.980, -0.416, 0.980]
sin = [sin2.0, sin0.2, sin2.0, sin0.2] ≈ [0.909, 0.199, 0.909, 0.199]

rotate_half(Q) = rotate_half([1,0,1,0]) = [-1,0,1,0]   (negate 2nd half, swap halves: x1=[1,0],x2=[1,0] -> [-1,0,1,0])
Q_rotated = Q*cos + rotate_half(Q)*sin
          = [1,0,1,0]*[-0.416,0.980,-0.416,0.980] + [-1,0,1,0]*[0.909,0.199,0.909,0.199]
          = [-0.416, 0, -0.416, 0] + [-0.909, 0, 0.909, 0]
          = [-1.325, 0, 0.493, 0]
```
Hmm — compare against Step 2's manually-computed answer `[-0.416, 0.909, 0.980, 0.199]`.
**These don't match — good, this is instructive, not an error to hide.** The mismatch is
because Step 2 used `Q=[1,0,1,0]` interpreted as **interleaved** pairs `(x[0],x[1])=(1,0)`
and `(x[2],x[3])=(1,0)`, but HF's `rotate_half` convention pairs `x[i]` with `x[i+D/2]` —
i.e. `(x[0],x[2])=(1,1)` and `(x[1],x[3])=(0,0)`. **This is exactly the "two different but
equivalent pairing conventions" point from the earlier RoPE math note** — recompute Step
2 with HF's actual convention to get a fair comparison:

**HF convention, correctly paired:** pair0 = `(x[0],x[2]) = (1,1)`, pair1 = `(x[1],x[3]) = (0,0)`.
```
pair0 (1,1) rotated by θ0·m=2.0: (1·cos2.0 − 1·sin2.0, 1·sin2.0 + 1·cos2.0) = (-0.416-0.909, 0.909-0.416) = (-1.325, 0.493)
pair1 (0,0) rotated by θ1·m=0.2: (0,0) rotated is still (0,0)
Result, de-interleaved back to HF's layout [x1_0,x1_1,x2_0,x2_1] = [-1.325, 0, 0.493, 0]
```
**Matches the direct HF computation above exactly.** Good — confirms both the formula and
the pairing convention are self-consistent; the earlier mismatch was purely from using the
wrong pairing convention for the comparison, not a bug in either derivation.

**QC path — same inputs, explicit half-split, no duplication:**
```
x_real = Q[..., :2] = [1, 0]          (first half)
x_im   = Q[..., 2:] = [1, 0]          (second half)
rope_real = cos[:2] = [-0.416, 0.980]  (HALF-width, no duplication)
rope_im   = sin[:2] = [0.909, 0.199]

x_prod_real = x_real*rope_real - x_im*rope_im = [1,0]*[-0.416,0.980] - [1,0]*[0.909,0.199]
            = [-0.416, 0] - [0.909, 0] = [-1.325, 0]
x_prod_im   = x_real*rope_im + x_im*rope_real = [1,0]*[0.909,0.199] + [1,0]*[-0.416,0.980]
            = [0.909, 0] + [-0.416, 0] = [0.493, 0]

result = cat(x_prod_real, x_prod_im) = [-1.325, 0, 0.493, 0]
```
**Bit-for-bit identical to HF's result** — `[-1.325, 0, 0.493, 0]` both ways. Confirms:
same math, same pairing convention (QC's `x_real`/`x_im` split is literally
`Q[...,:D/2]`/`Q[...,D/2:]` — the *same* halves HF's `rotate_half` implicitly operates on),
different op sequence (direct complex-multiply vs. concat-negate-duplicate). Exactly the
"adaptation = same math, different representation" principle, now verified with actual
numbers rather than asserted.

---

## Step 12 — Design-review summary

**One-sentence framing:** *"RoPE's math and frequency table are completely unchanged by
the Qualcomm adaptation — we still call the same HF `Gemma4TextRotaryEmbedding` module.
What changes is purely the application step: we slice off HF's redundant duplicated
`cos`/`sin` half, split Q/K into real/imaginary halves explicitly, and compute the
rotation as four named elementwise multiplies instead of `rotate_half`'s concat-negate —
verified bit-identical against HF's output for a hand-computed example."*

**Three points, in priority order for a review:**

1. **This is not a shape/compile-time adaptation — it's an op-representation adaptation.**
   Unlike the cache and mask, nothing about RoPE's tensor shapes changes at all between
   the fixed-ARN prefill graph and the ARN=1 decode graph — `D/2` and `S=ARN` are static
   in both worlds already. The motivation here is purely: fewer/cleaner ops for the
   quantizer and the HTP backend, not "this can't run on a static graph otherwise."
2. **The frequency math itself was never touched** — `inv_freq`, `base`, the two
   layer-type-specific tables, the `proportional` RoPE variant for full-attention layers —
   all reused verbatim from HF. Zero risk of numerical drift from re-deriving frequency
   math independently; the only new code is in how the *already-computed* angles get
   applied.
3. **Verified equivalence, not assumed.** The Step 11 trace is the artifact worth having
   ready if asked "how do we know this doesn't change the model's outputs" — a hand-worked
   example showing HF's `rotate_half`+duplicated-table path and QC's explicit-complex-multiply
   path land on the exact same numbers.
