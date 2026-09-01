# Gemma4 Text-Only Model — Adapted Architecture Reference (QAIRT/HTP, `qadaptation.py`)

Companion to `gemma4_text_architecture_reference.md` (vanilla HF). Same class-by-class
walk, restructured around: **what changes, why, what configs drive it, inputs, what it
does, outputs** — for each of `Gemma4ForCausalLM` → `Gemma4TextModel` →
`Gemma4TextDecoderLayer` → `Gemma4TextAttention` → `Gemma4TextMLP`. Scope: NanoV4's
`qlib/qadaptation.py`. (The other repo's `reauthoring.py` implements the same
requirements via a different mechanism in places — noted inline where it matters.)

---

## 0. The one constraint everything below exists to satisfy

**HF's graph is dynamic**: tensor shapes, which cache-layer class runs, how masks get
built — all resolved by live Python control flow, per call, using values only known at
that call's runtime (`past_key_values.get_seq_length()`, `torch.arange(...)`, `if
past_key_values is None`).

**HTP needs a static graph**: every tensor's shape, and every op that will ever execute,
must be fixed once, at compile time — before any real input ever arrives. There is no
Python `if` surviving into the compiled binary; whichever branch was traced is the only
branch that exists.

**So the adaptation problem, precisely stated:** take every place HF's code makes a
shape-affecting or control-flow decision using a *runtime* value, and rewrite it so that
decision is either (a) made once, from *config alone*, before compilation, or (b) reduced
to "fixed shape, computed from a runtime *scalar*" rather than "variable shape, computed
from a runtime *tensor whose size itself varies*."

Three things in this model need that treatment, and — to directly answer the "is it only
these three" framing — these are the three that affect **shape**. (Scalar constants,
per-layer structural dispatch, and quantization encodings are additional compile-time
concerns, but they don't change *shapes*, which is the specific problem this section is
about. See the earlier "what else is compile-time" discussion for the fuller inventory.)

| # | HF does this dynamically | Must become static because |
|---|---|---|
| 1 | KV cache grows via `torch.cat`, per-layer-type class split | cache tensor's shape can't change call to call on a compiled graph |
| 2 | Causal/sliding mask built fresh, shape tracking live cache length | mask shape must match the (now-fixed) cache shape |
| 3 | RoPE angles computed the same way regardless of hardware | not a shape problem at all (see the RoPE notes) — this one's motivation is op-representation/quantization cleanliness, included here only because it's the third "prepared once, reused" table alongside the other two |

---

## 1. Model inputs — what actually flows in, and why each one exists

Before the class-by-class walk, get the input list itself straight — this is where most
confusion happens if the concepts are still fuzzy (recall the earlier answer: only 3 of
these carry genuinely new information; everything else is a deterministic derivation of
those 3, precomputed outside the model).

| Input | What it is | Why it exists (vs. vanilla HF) |
|---|---|---|
| `input_ids` | token ids, `[B, ARN]` | same as HF — genuinely new info |
| `cache_index` | scalar — "where does the new data start, for full-attention layers" | HF derives this internally from `past_key_values.get_seq_length()`; here it must be supplied, because the model can no longer call a live method on a cache object |
| `swa_cache_index` | scalar — same idea, for sliding-attention layers | full and sliding layers can have **different** effective write offsets (window eviction) — HF's `get_seq_length(layer_idx=...)` handles this per-layer internally; adapted code needs it explicit and separate |
| `cache_position` / `swa_cache_position` | `[ARN]` — `cache_index + cache_tensor` (fixed ramp buffer) | replaces HF's live `torch.arange(...)`; same numeric idea, computed from a compile-time-fixed ramp plus one runtime scalar, so the op itself is a fixed-shape add, not a dynamic-length `arange` |
| `position_ids` / `swa_position_ids` | either raw position indices, or an already-computed `(cos,sin)` tuple, per layer type | HF computes `(cos,sin)` inside the model from a single `position_ids`; here, split by layer type (full vs sliding use different `base`/`head_dim`) and often precomputed entirely outside, passed in ready-to-use |
| `attention_mask` / `swa_attention_mask` | already-built 4D tensors, `[1,1,ARN,gcl]` / `[1,1,ARN,lcl]` | `ADAPTATION_6` — HF builds these internally via live cache calls; here they arrive as plain tensors, built by `qgenerator.py`, entirely outside the traced model |
| `per_layer_inputs` | PLE tensor, `[B,ARN,num_hidden_layers,ple_dim]` | deterministic function of `input_ids`; can be precomputed the same way as mask/position, though the model can also compute it internally if not supplied |
| `past_key_values` | the cache object/buffers | fixed-shape buffers now (see §3), not a growing `DynamicCache` |

---

## 2. `Gemma4ForCausalLM` (adapted)

**Why this class needs to change at all:** it's the entry point — it's where the
scalar-to-ramp conversion (`cache_index → cache_position`) has to happen, since that's the
one piece of "turn a runtime scalar into a fixed-shape tensor" logic that belongs above
the text-model call, not inside it.

**What changes vs. vanilla:**
- Registers `cache_tensor` — a **buffer**, not a computed value: `torch.arange(input_tokens_per_inference)`, fixed at model-construction time from `config.input_tokens_per_inference` (= ARN). This is the fixed ramp from §1.
- Converts `cache_index`/`swa_cache_index` (scalars) into `cache_position`/`swa_cache_position` (tensors) via `cache_position = cache_index + self.cache_tensor` — **before** calling the text model. This is the only new "compute something" logic at this level; everything else is just forwarding.
- Everything (mask, position embeddings, cache buffers) is assembled **before** this call reaches `self.model(...)` — this class doesn't build any of it itself, it just receives already-built tensors as `forward()` arguments and passes them straight through.

**Configs used:** `config.input_tokens_per_inference` (ARN — drives `cache_tensor`'s
length), `config.final_logit_softcapping` (unchanged from vanilla).

**Inputs:** `input_ids`, `attention_mask`, `swa_attention_mask`, `position_ids`,
`swa_position_ids`, `per_layer_inputs`, `cache_index`, `swa_cache_index`,
`cache_position`/`swa_cache_position` (optional — computed here if not supplied),
`past_key_values`, `logits_to_keep`.

**What it does:**
1. `cache_index + cache_tensor → cache_position` (if `cache_index` given; same for swa)
2. Calls `self.model(...)` (the adapted `Gemma4TextModel`) with everything, already-built, passed straight through
3. `lm_head(hidden_states[:, slice_indices, :])` → logits (slice_indices from `logits_to_keep`, same as vanilla)
4. Optional `final_logit_softcapping`: `tanh(logits/cap)*cap` — unchanged from vanilla

**Outputs:** `logits`, `past_key_values` (mutated), plus (adapted-only) raw `(swa_k,swa_v)`/`(global_k,global_v)` tuples when `return_dict=False`, for MTP drafter consumption.

**What stays exactly the same as vanilla:** the softcap formula, the tied `lm_head`
weight, the overall two-call structure (`self.model(...)` then `self.lm_head(...)`).

---

## 3. `Gemma4TextModel` (adapted)

**Why this class needs to change:** this is where HF's *internal* mask/cache/position
construction lived — `ADAPTATION_6` exists specifically to remove that construction from
here, turning this `forward()` into a pure consumer of already-concrete tensors.

**What changes vs. vanilla, mapped to the three compile-time problems from §0:**

**(1) KV cache — no longer auto-created, no longer type-split.**
```python
if use_cache and past_key_values is None:
    raise ValueError("use_cache is True but past_key_values are not provided")
```
Vanilla: `DynamicCache(config=self.config)` — auto-builds, and *splits cache-layer class*
by `layer_types` (`DynamicLayer` vs `DynamicSlidingWindowLayer`). Adapted: **never**
auto-builds; if a legacy tuple arrives, `from_legacy_cache()` wraps it in
`DynamicCache_adapted` — **one uniform class for every layer**, regardless of type. Why
uniform: qairt's HTP cache-patch (or here, the custom `.update()` itself) only needs to
target one class; a type-split would mean sliding layers silently running different code
than intended (this is the KV-cache trap covered earlier). The buffer itself is
conceptually `[B, kv_heads, CL_or_window, head_dim]` — fixed size, written via scatter,
not grown via concat (§ KV-cache mentor notes has the full mechanism).

**(2) Causal mask — no longer built here at all.**
```python
# ADAPATATION 6: Remove attn mask creation from inside model
```
Vanilla builds `causal_mask_mapping = {"full_attention": create_causal_mask(...),
"sliding_attention": create_sliding_window_causal_mask(...)}` internally, using live
`past_key_values.get_seq_length()`/`get_mask_sizes()` calls. Adapted: `attention_mask` and
`swa_attention_mask` arrive as plain **already-4D** tensors — `[1,1,arn,gcl]` /
`[1,1,arn,lcl]`, fixed shape, built by `qgenerator.py`'s `build_text()` (see the masking
mentor notes for the exact per-row construction). This model only *selects* per layer
type, exactly like vanilla:
```python
if self.config.layer_types[i] == "sliding_attention":
    attention_mask = swa_attention_mask
else:
    attention_mask = global_causal_mask
```

**(3) RoPE angles — same computation, split arrival paths.**
```python
if isinstance(position_ids, (tuple, list)) or position_ids is None:
    position_embeddings = position_ids                                    # already (cos,sin) — use directly
else:
    position_embeddings = self.rotary_emb(hidden_states, position_ids, "full_attention")   # raw indices — compute here
```
Same `rotary_emb` module as vanilla (frequency math untouched — see RoPE mentor notes) —
the only change is that this model accepts **either** raw position indices (computes
`(cos,sin)` itself, same as vanilla) **or** an already-computed `(cos,sin)` tuple (skips
computation entirely) — because the *typical* real path precomputes these outside
(`qgenerator.py`'s `swa_pos_emb = llm_create_position_embeddings(...)`), but the model
still supports computing them itself for tracing/flexibility.

**Configs used:** `config.layer_types` (per-layer mask/position-table selection, unchanged
from vanilla), `config.sliding_window_pattern` (fallback `swa_cache_position` derivation
when not supplied), `config.num_layers_to_run` (decoder-loop truncation), `config.return_dict`
(forced through — `ADAPTATION_4`).

**Inputs:** `input_ids`/`inputs_embeds`, `attention_mask`, `swa_attention_mask`,
`position_ids`, `swa_position_ids`, `per_layer_inputs`, `past_key_values`, `cache_position`,
`swa_cache_position`, `return_dict`.

**What it does:**
1. `input_ids → embed_tokens → inputs_embeds` (unchanged)
2. PLE: `get_per_layer_inputs` + `project_per_layer_inputs` if not precomputed (unchanged math, same as vanilla)
3. Legacy-cache conversion (`from_legacy_cache`) if `past_key_values` arrived as a tuple
4. `swa_cache_position` fallback derivation if not supplied (mirrors vanilla's own default logic, using `sliding_window_pattern` instead of a raw layer index)
5. `shared_kv_states = {}` reset (unchanged — KV-sharing logic itself untouched by adaptation)
6. Decoder loop: per layer, select `(cache_position, attention_mask, position_embeddings)` by `layer_types[i]`, call `Gemma4TextDecoderLayer`
7. Final `self.norm(...)` (unchanged)

**Outputs:** `last_hidden_state`, `past_key_values`, plus `(swa_k,swa_v)`/`(global_k,global_v)`
— the last-real-layer K/V per type, threaded out for MTP.

**What stays exactly the same as vanilla:** embedding lookup, PLE math, `shared_kv_states`
dict mechanics and KV-sharing selection logic, final norm, the per-layer-type
select-then-dispatch pattern itself (only *where the selected tensors came from* differs).

---

## 4. `Gemma4TextDecoderLayer` (adapted)

**Why this class barely changes:** none of the three compile-time problems (cache, mask,
RoPE) live at this layer — this class only *passes tensors through* to attention/MLP. Its
job is orchestration (norms + residuals), and that orchestration doesn't care whether the
mask/cache/position tensors it's handed were built statically or dynamically.

**What changes vs. vanilla:**
```python
hidden_states, _, key_states, value_states = self.self_attn(...)   # was: hidden_states, _ = self.self_attn(...)
...
return hidden_states, key_states, value_states                       # was: return hidden_states
```
That's the entire change. `cache_position` is passed as an explicit named arg (not buried
in `**kwargs`) so it's a real, traceable graph input rather than an implicit Python-side
value — worth naming explicitly since it's a small but easy-to-miss adaptation: **anything
that must survive into the traced graph has to be a real named tensor argument, not
something living only in `**kwargs`/closures.**

**Configs used:** none directly — `enable_moe_block`, `hidden_size_per_layer_input` checks
are unchanged from vanilla (this class doesn't introduce new config-gated behavior).

**Inputs:** `hidden_states`, `per_layer_input`, `shared_kv_states`, `position_embeddings`,
`attention_mask`, `past_key_values`, `cache_position`. (Identical to vanilla's inputs plus
the explicit `cache_position`.)

**What it does:** identical sandwich-norm structure to vanilla — `input_layernorm` → attn
→ `post_attention_layernorm` → residual; `pre_feedforward_layernorm` → mlp →
`post_feedforward_layernorm` → residual; optional PLE branch; final `layer_scalar` gate.
See `gemma4_text_architecture_reference.md` §3 for the full residual-structure diagram —
none of it changed here.

**Outputs:** `hidden_states`, `key_states`, `value_states` (the last two are new — threaded
up for MTP, see §3).

**What stays exactly the same as vanilla:** literally everything except the two lines
above. This is the "adaptation touches this file, but this class isn't where the
interesting decisions get made" case — worth saying plainly rather than implying every
class changes equally.

---

## 5. `Gemma4TextAttention` (adapted)

**Why this class changes the most:** this is where all three compile-time problems
(cache write, mask application, RoPE application) actually get *consumed* — so this is
where the representation changes are concentrated, even though (per the RoPE notes) not
all of them are "compile-time" problems in the shape sense.

**What changes, mapped to each of the three areas:**

**RoPE application** (op-representation adaptation, not a shape problem — see RoPE mentor
notes for full derivation):
```python
self.apply_rope_fn = ApplyRopeSingle()     # explicit real/imag complex-multiply, named MulModules
...
_q_half = query_states.shape[-1] // 2
query_states = self.apply_rope_fn(query_states[..., :_q_half], query_states[..., _q_half:], position_embeddings)
```
Same math as vanilla's `rotate_half`+duplicated-`cos`/`sin` approach (verified
bit-identical in the RoPE notes' worked trace) — different op sequence: explicit
half-split, four named multiplies, no concat-negate.

**KV cache write** (shape/staticness adaptation):
```python
if transposed_key_cache:
    key_states = key_states.transpose(2, 3)     # store K pre-transposed — avoids a live transpose in the QK matmul later
...
cache_kwargs = {"cache_position": cache_position, "transposed_key_cache": ..., "num_key_value_heads": ..., "return_new_key_value_only": ..., "head_dim": ...}
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```
Vanilla: `past_key_values.update(key_states, value_states, self.layer_idx)` — no extra
metadata needed, because vanilla's cache just concatenates. Adapted: the cache's `update()`
needs to know *where* to scatter (`cache_position`), *how* the tensor is laid out
(`transposed_key_cache`), and *what to keep* (`return_new_key_value_only`) — because
writing into a fixed pre-allocated buffer at a specific offset needs more information than
"just append."

**Mask application** (shape adaptation, consumed not built here):
```python
attention_interface(self, query_states, key_states, value_states, attention_mask, scaling=self.scaling, sliding_window=self.sliding_window, transposed_key_cache=transposed_key_cache, **kwargs)
```
The mask itself was built entirely outside (§3) — this class just passes it through to
`eager_attention_forward`, same call shape as vanilla, except the actual masking-application
logic inside that function now has two modes (additive vs. masked-softmax — see masking
mentor notes §5c).

**What does NOT change here — worth stating explicitly:** KV-sharing logic
(`is_kv_shared_layer`, `shared_kv_states[kv_shared_layer_index]` lookup,
`store_full_length_kv` publish) is **identical** to vanilla — config-driven,
hardware-agnostic, untouched by any adaptation. Same for `q_norm`/`k_norm`/`v_norm`
(QK-norm), `scaling=1.0`, GQA `repeat_kv`.

**Configs used:** `config.transposed_key_cache`, `config.return_new_key_value_only`
(cache-layout flags), `adaptations.enable_masked_softmax`, `adaptations.kv_clip_only`
(from `AdaptationFlags` — a single config object gating which HTP-friendly substitution is
active).

**Inputs:** `hidden_states`, `position_embeddings`, `attention_mask`, `shared_kv_states`,
`past_key_values`, `cache_position`.

**What it does:** project Q → norm → RoPE (via `ApplyRopeSingle`) → branch on
`is_kv_shared_layer` (read from `shared_kv_states` or compute+norm+RoPE+transpose+cache-update
K/V) → optional KV fake-quant → publish to `shared_kv_states` if `store_full_length_kv` →
`eager_attention_forward` → `o_proj`.

**Outputs:** `attn_output`, `attn_weights`, `key_states`, `value_states` (last two new, for
MTP threading — see §3/§4).

---

## 6. `Gemma4TextMLP` (adapted)

**Why this class changes at all — and why it's a *small*, isolated change:** unlike
attention, the MLP has no cache, no mask, no positional information flowing through it at
all — it's a pure per-token, per-position-independent transform. So none of the three
compile-time problems from §0 touch it. The one change here is a **different kind** of
adaptation entirely: activation-function *numerical* matching to the hardware backend's op
lowering, not a shape or caching concern.

**What changes:**
```python
class Gemma4TextMLP(Gemma4TextMLP_original):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.act_fn = Act2FN("gelu", craft_config=config.craft_config)     # erf-based GeLU
```
Vanilla uses `config.hidden_activation` (`"gelu_pytorch_tanh"` — the tanh-approximation of
GeLU, PyTorch's default fast approximation). Adapted overrides to plain `"gelu"` — the
**erf-based** (exact) GeLU formulation. **Why:** this mirrors the same rationale as
`QcGemma4RMSNorm`'s reformulation covered in the architecture-adaptation notes — "match
what MPP/HTP actually lowers the op to." The tanh-approximation and erf-based GeLU are
numerically close but not identical; if the HTP backend's fixed-function GeLU op is
erf-based internally, tracing/quantizing against the tanh-approximation would introduce a
small, avoidable train/inference mismatch. This is gated by `AdaptationFlags.use_erf_gelu`
(`qadaptation_flags.py:22`) — same "single flags object controls which substitution is
active" pattern as everywhere else.

**Configs used:** `config.craft_config` (unchanged — quantization-op wrapping, same as
vanilla), `AdaptationFlags.use_erf_gelu` (governs whether this override is even applied).

**Inputs:** `hidden_states` (post-`pre_feedforward_layernorm`) — identical to vanilla.

**What it does:** `gate = act_fn(gate_proj(x))` (erf-GeLU instead of tanh-GeLU) → `up =
up_proj(x)` → `fused = gate * up` → `down_proj(fused)` — same GeGLU structure as vanilla,
same call order, same shapes throughout.

**Outputs:** transformed `hidden_states`, same shape as input — identical to vanilla.

**What stays exactly the same as vanilla:** the gated-MLP structure itself, `gate_proj`/
`up_proj`/`down_proj` weight shapes, `use_double_wide_mlp` KV-shared-layer widening logic
(untouched — this adaptation doesn't interact with that at all).

---

## 7. Summary table — every class, one line each

| Class | Touched by adaptation? | Which of the 3 compile-time problems (§0) does it relate to | What actually changed |
|---|---|---|---|
| `Gemma4ForCausalLM` | Yes | cache (builds `cache_position` from scalar) | `cache_tensor` buffer + scalar→ramp conversion; everything else built upstream, passed through |
| `Gemma4TextModel` | Yes, heavily | all 3 (selects/consumes, doesn't build, mask/cache) | no auto-cache-creation, no internal mask-building (`ADAPTATION_6`), `(cos,sin)`-or-raw-indices dual input path, forced `return_dict` (`ADAPTATION_4`), K/V threaded out for MTP |
| `Gemma4TextDecoderLayer` | Minimally | none directly — pure pass-through | only returns extra `key_states`/`value_states`; residual/norm structure identical |
| `Gemma4TextAttention` | Yes, heavily | all 3 (RoPE application, cache write, mask consumption) | `ApplyRopeSingle` (explicit complex-multiply), `cache_kwargs`-driven scatter write, transposed-key storage, masked-softmax option |
| `Gemma4TextMLP` | Yes, minimally | none — orthogonal concern (op-numerical matching) | erf-GeLU instead of tanh-GeLU, gated by `use_erf_gelu` |

**The one-sentence takeaway for the whole document:** *the three compile-time problems
(cache, mask, RoPE-shape) are resolved almost entirely in `Gemma4ForCausalLM` (scalar→ramp)
and `Gemma4TextModel` (selection/consumption of already-built tensors) — by the time
execution reaches `Gemma4TextAttention`, it's mostly consuming already-static inputs, with
its own local adaptations (RoPE op sequence, cache scatter metadata) layered on top; and
`Gemma4TextDecoderLayer`/`Gemma4TextMLP` are largely untouched by the compile-time-shape
problem entirely, changing only for unrelated reasons (graph plumbing, op-numerical
matching).*
