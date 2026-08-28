# Attention Masking: Theory → HF Implementation → QAIRT Adaptation (`qgenerator.py` / `qadaptation.py`)

Companion to `gemma4_kv_cache_theory_to_adaptation_mentor_notes.md` — same template, same
"theory → HF function → adapted function, with shapes/values at every step" format,
applied to masking instead of caching. You already have the mask *theory* (see
`gemma4_causal_mask_notes.md` for the HF-only deep dive on offsets/shapes) — this file is
specifically the **HF → adaptation mapping**.

Sources: `transformers/masking_utils.py` (vanilla HF), `Nano/NanoV4/qlib/qgenerator.py`
(the actual mask-construction code for this pipeline — the real answer to "where do
`attention_mask`/`swa_attention_mask` come from" that `qadaptation.py`'s `ADAPTATION_6`
comment defers to), `Nano/NanoV4/qlib/qadaptation.py` (mask *consumption*, i.e.
`eager_attention_forward`), `Nano/NanoV4/test_framework_parity.py` (concrete numeric test
values).

---

## 1. Standard HF Masking Flow

```
Gemma4TextModel.forward(input_ids, attention_mask=None, past_key_values, ...)
      │
      ├─ if not isinstance(attention_mask, dict):
      │      mask_kwargs = {config, inputs_embeds, attention_mask, past_key_values, position_ids}
      │      causal_mask_mapping = {
      │          "full_attention":    create_causal_mask(**mask_kwargs),          ← BUILD, live call
      │          "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
      │      }
      │
      ├─ for each layer i:
      │      mask = causal_mask_mapping[config.layer_types[i]]     ← SELECT by layer type
      │      Gemma4TextAttention.forward(..., attention_mask=mask)
      │            └─ attn_weights = attn_weights + mask            ← APPLY, additive
      │
      └─ (mask is rebuilt from scratch, this call, every single forward pass)
```
Inside `create_causal_mask`/`create_sliding_window_causal_mask`: `_preprocess_mask_arguments`
pulls `q_offset`/`kv_length`/`kv_offset` **live** from `past_key_values.get_seq_length()`/
`get_mask_sizes()` (real Python method calls on the cache object, every call), then
`sdpa_mask`/`eager_mask` vectorize a boolean predicate (`causal_mask_function`,
`sliding_window_overlay`) over `torch.arange(...)` ranges built fresh each time.

---

## 2. Adapted (Qc) Masking Flow

```
ModelInputBuilder._derived()  [qgenerator.py:190-227]   ← ONE-TIME setup, memoized per (gcl,lcl,num_kv)
      │
      ├─ full_causal_mask = create_causal_mask(...).clamp_min(mask_neg)   ← [1,1,gcl,gcl] TEMPLATE
      ├─ full_swa_mask     = create_sliding_window_causal_mask(...).clamp_min(mask_neg)  ← [1,1,lcl,lcl] TEMPLATE
      │   (comment: "build() slices the relevant arn rows at runtime -- O(1) per call" —
      │    intent was cheap slicing; see §5 for why build_text() does NOT actually use these)
      │
ModelInputBuilder.build_text(arn, token_ids, unpadded_past_kv, ...)   ← PER-CALL, runs every prefill/decode step
      │
      ├─ kv_glb, kv_swa = real past-token counts (from unpadded_past_kv shapes)
      ├─ write_base_glb, write_base_swa = aligned write offsets (may differ from kv_glb/kv_swa!)
      │
      ├─ causal_mask  = torch.full((1,1,arn,gcl), mask_neg)   ← start FULLY MASKED
      ├─ sliding_mask = torch.full((1,1,arn,lcl), mask_neg)   ← start FULLY MASKED
      ├─ for r in range(num_real):                             ← per-row, flip valid cells to 0
      │      causal_mask[0,0,r, past_glb | new_glb]  = 0.0
      │      sliding_mask[0,0,r, past_swa | new_swa] = 0.0
      │
      └─ out["attention_mask"] = causal_mask; out["swa_attention_mask"] = sliding_mask
                    │
                    ▼ (passed as a plain input tensor, no cache-object calls inside the model)
Gemma4ForCausalLM.forward(attention_mask=..., swa_attention_mask=..., ...)   [qadaptation.py]
      │
      └─ self.model(attention_mask=attention_mask, swa_attention_mask=swa_attention_mask, ...)
             │
             └─ per-layer-type select (same branch as vanilla), then:
                    eager_attention_forward(..., causal_mask, mask_neg, enable_masked_softmax)
```

---

## 3. Side-by-Side Comparison

| Aspect | Standard HF | Adapted (Qc) |
|---|---|---|
| Who builds the mask | `Gemma4TextModel.forward`, **inside** the traced model | `ModelInputBuilder.build_text` (`qgenerator.py`), **entirely outside** the model — `ADAPTATION_6`: *"Remove attn mask creation from inside model"* |
| Mask shape | `[B,1,Q,KV]`, `KV` = however much history *actually exists* (`get_mask_sizes()` — grows with generation) | `[1,1,arn,gcl]` (causal) / `[1,1,arn,lcl]` (sliding) — **fixed** to compile-time constants (`gcl`=global context length, `lcl`=sliding buffer length), every call, forever |
| How validity is decided | boolean predicate (`kv_idx <= q_idx`, etc.) evaluated over `torch.arange` ranges, live | starts **fully masked** (`torch.full(..., mask_neg)`), then explicit index-set assignment flips valid `(row, col)` pairs to `0.0` |
| Masked value | `dtype_min` (≈ `-inf`) for additive mode | `mask_neg` (config-driven, **-200** by default) — a deliberately *mild* negative, not `-inf`, chosen for fixed-point/quantization safety |
| Masking application at attention | `attn_weights = attn_weights + mask` (always additive) | **two modes**, config-gated (`enable_masked_softmax`): additive (`+= mask`, same as HF) **or** masked-softmax: `where(mask==0, attn_weights, attn_weights_min + mask_neg)` |
| Source of "how much history" info | `past_key_values.get_seq_length()`/`get_mask_sizes()` — live method calls on a `Cache` object | `kv_glb`/`kv_swa` — plain Python ints computed from the *shapes* of `unpadded_past_kv` tensors passed in; no `Cache` object method calls at all |
| Precomputed template? | No — HF always builds fresh (though it's cheap: vectorized boolean ops) | **Yes, attempted** — `full_causal_mask`/`full_swa_mask` computed once via memoized `_derived()`, intended for O(1) per-call slicing — **but not actually used by `build_text()`** (see §5) |
| Row-by-row construction | No — whole mask built as one vectorized broadcast | **Yes** — explicit `for r in range(num_real): ...` loop, one row at a time, because write offset and real past count can diverge (padding/alignment) |

---

## 4. Tensor Shape Evolution

Real config: `gcl` (global context length) = `CONTEXT_LENGTH` = 15527, `lcl` (local/sliding
buffer length) = some multiple of `sliding_window` (512) with reserved slack, `arn` = 521.

| Stage | Standard HF | Adapted (Qc) |
|---|---|---|
| Mask template (one-time, `_derived()`) | *(none — HF never precomputes)* | `full_causal_mask`: `[1,1,15527,15527]`; `full_swa_mask`: `[1,1,lcl,lcl]` — built once, cached, then **not directly consumed** per-call (see §5) |
| Prefill (arn=521 tokens, kv_glb=0) | mask: `[1,1,521,521]` (KV grows to match query length — nothing prior exists) | `causal_mask`: `[1,1,521,15527]` — **already full width**, most columns still `mask_neg` (unwritten) |
| Decode step (arn=1, kv_glb=521 prior) | mask: `[1,1,1,522]` (KV length = 521+1) | `causal_mask`: `[1,1,1,15527]` — **same shape as prefill's**, just a different set of columns flipped to `0.0` |
| Shape stability across calls | **Changes every call** (grows) | **Never changes** — always `[1,1,arn,gcl]` / `[1,1,arn,lcl]`, `arn` fixed per compiled graph variant |

**One-sentence contrast, matching the cache notes' framing exactly:** HF's mask *shape*
tracks how much history exists (grows); the adapted mask's shape is *always* the fixed
`(arn, gcl)`/`(arn, lcl)`, and "how much history exists" lives entirely in *which columns
got flipped to 0* — same "shape fixed, content varies" principle as the cache.

---

## 5. Mask Lifecycle: Template (attempted) → Per-Call Build → Application

### 5a. The one-time template — `ModelInputBuilder._derived()` (`qgenerator.py:190-227`)

*Purpose:* precompute the full `gcl×gcl` causal mask and `lcl×lcl` sliding mask **once**,
using HF's own correct mask functions, so later calls could in principle just slice a
sub-block instead of rebuilding from scratch.
```python
full_causal_mask = create_causal_mask(
    config=llm, inputs_embeds=torch.zeros(1, gcl), attention_mask=torch.ones(1, gcl, dtype=torch.long),
    cache_position=torch.arange(gcl), past_key_values=None, position_ids=torch.arange(gcl),
).clamp_min(float(self._mask_neg))                                              # [1,1,gcl,gcl]

full_swa_mask = create_sliding_window_causal_mask(**lcl_mask_kwargs).clamp_min(float(self._mask_neg))  # [1,1,lcl,lcl]
```
*Inputs:* `llm` config, dummy full-length `inputs_embeds`/`attention_mask`/`position_ids`
covering the *entire* context length — this is a one-time, offline-style computation, not
tied to any real request. *Output:* two dense boolean-turned-float mask tensors, memoized
in `self._derived_cache[key]` keyed by `(gcl, lcl, num_kv)` — reused via `.to(device)`
across calls (`to()` method, line 235-242), never recomputed unless dims change.
`clamp_min(mask_neg)` replaces HF's default `-inf`/`dtype_min` with the milder `mask_neg`
(-200) right at template-build time.

**Important, and worth stating plainly in a design review:** this template is built with
the *intent* stated in the comment (*"build() slices the relevant arn rows at runtime —
O(1) per call"*), but **`build_text()` does not actually slice `full_causal_mask`/
`full_swa_mask` at all** — grep confirms these two dict keys are never read anywhere except
being copied device-to-device in `to()`. `build_text()` instead **rebuilds masks from
scratch every call** via the per-row loop (§5b). The comment at `qgenerator.py:342-344`
explains why the template approach was abandoned: *"the write base always differs from the
real past count, so the precomputed full_*_mask slices no longer apply."* This is a real,
worth-knowing gap between stated intent and actual behavior — the O(1)-slice optimization
was designed but isn't the live path.

### 5b. The actual per-call build — `ModelInputBuilder.build_text()` (`qgenerator.py:342-361`)

*Purpose:* build this call's real `attention_mask`/`swa_attention_mask`, accounting for the
fact that where new K/V gets *written* (`write_base_glb`/`write_base_swa` — possibly
alignment-padded) can differ from how many real past tokens exist (`kv_glb`/`kv_swa`).
```python
causal_mask  = torch.full((1, 1, arn, gcl), float(self._mask_neg), device=device)   # start ALL masked
sliding_mask = torch.full((1, 1, arn, lcl), float(self._mask_neg), device=device)   # start ALL masked
glb_cols = torch.arange(gcl, device=device)
swa_cols = torch.arange(lcl, device=device)

for r in range(num_real):                              # one real token = one row
    # Global (full-attention) row r:
    past_glb = glb_cols < kv_glb                                              # everything already real
    new_glb  = (glb_cols >= write_base_glb) & (glb_cols <= write_base_glb + r)  # this call's tokens up to r
    causal_mask[0, 0, r, past_glb | new_glb] = 0.0                            # flip valid cols to 0

    # Sliding (SWA) row r:
    window_lo = max(0, kv_swa + r - win + 1)                                  # left edge of the window
    past_swa  = (swa_cols >= window_lo) & (swa_cols < kv_swa)
    new_swa   = (swa_cols >= write_base_swa) & (swa_cols <= write_base_swa + r)
    sliding_mask[0, 0, r, past_swa | new_swa] = 0.0
```
*Inputs:* `arn` (fixed), `gcl`/`lcl` (fixed), `kv_glb`/`kv_swa` (real past-token counts,
from `unpadded_past_kv` tensor shapes), `write_base_glb`/`write_base_swa` (aligned write
offsets — see §7 for why these can diverge from `kv_glb`/`kv_swa`), `win` (sliding window
size). *Output:* `causal_mask` `[1,1,arn,gcl]`, `sliding_mask` `[1,1,arn,lcl]` — dense float
tensors, `0.0` where valid, `mask_neg` (-200) everywhere else. *Shape change:* none — same
shape every call, only which cells are `0.0` vs `mask_neg` changes.

**This is the direct generalization of the vanilla `causal_mask_function`/`sliding_window_overlay`
predicates** (`kv_idx <= q_idx`, `kv_idx > q_idx - window`) — same logical conditions,
expressed as boolean masks over `glb_cols`/`swa_cols` and applied via index-assignment
instead of a vectorized broadcast predicate. The *extra* complexity here (`past_X | new_X`,
separately tracking "already-real" vs "newly written this call") exists specifically
because `write_base` and `kv_glb`/`kv_swa` can differ — something vanilla HF's `q_offset`/
`kv_offset` scheme (single source of truth: `cache.get_seq_length()`) never has to handle,
because vanilla HF's cache position *is* the real past count, always (no separate
"alignment padding" concept).

### Mask-update worked format (matching the cache notes' template) — one real row

**Before:**
```
causal_mask[0,0,r,:] = mask_neg  (all gcl=15527 columns, freshly torch.full'd for this call)
kv_glb = 521          (521 real tokens already in the global cache)
write_base_glb = 521  (aligned; here it happens to equal kv_glb — no padding gap this step)
r = 0                  (first new token this call, decode step, arn=1)
```
**Current token:**
```
This row corresponds to query position kv_glb + r = 521 (the new decode-step token)
```
**Operation:**
```
past_glb = glb_cols < 521                       → columns 0..520 marked True
new_glb  = (glb_cols >= 521) & (glb_cols <= 521)  → column 521 marked True
causal_mask[0,0,0, past_glb | new_glb] = 0.0     → columns 0..521 set to 0.0
```
**After:**
```
causal_mask[0,0,0, 0:522]     = 0.0        (valid: all real history + this new token)
causal_mask[0,0,0, 522:15527] = mask_neg   (unwritten/future — still masked)
```
Directly comparable to the cache-notes' decode-step trace: cache wrote 1 new K/V slice at
offset 521 into a 15527-length buffer; the mask, independently, marks columns 0-521 valid
and 522+ invalid — **two separate mechanisms (cache write offset, mask column flip) that
must agree with each other for correctness**, but are computed by different code paths
(`DynamicLayer_adapted.update()`'s `cache_position` vs. `build_text()`'s `kv_glb`/`write_base_glb`).

### 5c. Application — `eager_attention_forward` (`qadaptation.py:201-252`)

*Purpose:* apply the precomputed mask to raw attention scores before softmax. *Inputs:*
`attn_weights` `[B,H,Q,KV]`, `causal_mask` `[B,1,Q,KV]` (from §5b), `module.enable_masked_softmax`
flag, `mask_neg`. *Output:* masked `attn_weights`, same shape, ready for softmax.
```python
if attention_mask is not None:
    causal_mask = attention_mask
    if attention_mask.shape[-1] != value_states.shape[-2]:      # defensive trace-time slice
        causal_mask = attention_mask[:, :, :, :value_states.shape[-2]]

    if module.enable_masked_softmax:
        attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
        minus_value = getattr(module.config, 'mask_neg', -200)
        attn_weights = torch.where(causal_mask == 0, attn_weights, attn_weights_min + minus_value)
    else:
        attn_weights = attn_weights + causal_mask                # vanilla-style additive
```
Two modes, both operating on the *same* precomputed mask tensor from §5b:
- **Additive** (`enable_masked_softmax=False`): identical in spirit to vanilla HF —
  `attn_weights + mask`, where `mask` is `0.0` (no-op) or `mask_neg` (pushes the score down).
- **Masked-softmax** (`enable_masked_softmax=True`, the pipeline's actual default per
  `qc_config.py`): instead of *adding* a negative offset, **replace** masked positions with
  "the smallest real score present, minus a further margin" — `attn_weights_min + mask_neg`.
  This produces the same *softmax outcome* (masked positions get near-zero probability)
  without ever computing `x + (-200)` on an already-extreme-valued tensor, which is friendlier
  for fixed-point/quantized arithmetic where very large additive swings are harder to
  represent precisely.

---

## 6. Numeric Confirmation — from the pipeline's own test file

`test_framework_parity.py:236` (and again at line 279) builds a parity-test mask directly,
confirming the additive-mode value in isolation:
```python
ATTN_MASK_VALUE = -3.4028234663852886e+38          # = torch.finfo(torch.float32).min, true "-inf" equivalent
attention_mask = (torch.triu(torch.ones(arn, arn), diagonal=1) * ATTN_MASK_VALUE).unsqueeze(0).unsqueeze(0)
```
This is a **plain upper-triangular causal mask** (`torch.triu(..., diagonal=1)` zeroes out
the lower triangle including diagonal, keeps upper triangle as `1`, times `-inf` gives
`0` on/below diagonal and `-inf` above it) — shape `[1,1,arn,arn]`. Two things worth
noting when comparing this against the real runtime path:
1. This test mask is **`[arn,arn]`**, not `[arn,gcl]` — a simplified same-size stand-in
   used specifically for isolated layer/attention-op parity checks (GG reference vs. QC
   model), not the real `[arn, gcl]`-shaped mask `build_text()` produces for actual
   inference.
2. It uses **true float32-min** (`-3.4e38`), not `mask_neg` (`-200`) — this is the
   *reference*-side value (matching what an unquantized/full-precision comparison target
   would use), while `-200` is specifically the *quantization-friendly* value chosen for
   the adapted model's own masked-softmax/additive path. The test file's `mask_neg=-200`
   usages (lines 77, 93 — config construction) confirm both values coexist deliberately: one
   for the GG/reference side, one for the QC/adapted side, and the parity tests check they
   still agree closely (`logits_error < 1e-4`) despite that difference.

---

## 7. Where This Connects Back to the Cache Notes

The mask and the cache are **not independent** — `build_text()`'s mask construction reads
`kv_glb`/`kv_swa`/`write_base_glb`/`write_base_swa`, all derived from the *same*
`unpadded_past_kv` tensors that `DynamicLayer_adapted.update()` scatters into. Two concrete
links:

1. **Why `write_base` can differ from `kv_glb`/`kv_swa` at all** (`qgenerator.py:325-326`,
   `_align_up(...)`): write offsets get rounded up to some alignment boundary (hardware/
   buffer-layout reasons), while the *real* past-token count doesn't need that rounding.
   So a row's "new" region (`write_base_glb .. write_base_glb+r`) and "past" region
   (`< kv_glb`) can have a small gap between them if `write_base_glb > kv_glb` — the
   `past_glb | new_glb` OR is specifically there to correctly mark both regions valid
   despite that gap, and anything strictly between `kv_glb` and `write_base_glb` (if any)
   stays masked, since it's neither real history nor a new write — just alignment padding.
2. **The mask and the cache-write offset must be computed consistently, but by separate
   code**: the cache's `cache_position` (used by `DynamicLayer_adapted.update()`'s scatter)
   and the mask's `write_base_glb`/`write_base_swa` (used by `build_text()`'s column-flip)
   both ultimately come from the same upstream bookkeeping (`cache_index` progression), but
   are threaded through two different functions/files. This is exactly the kind of place a
   real bug could hide if the two ever drifted out of sync — worth flagging as a design-review
   watch item, not just a "how it works" fact.

---

## Diagram — full lifecycle, both worlds side by side

```
                          STANDARD HF                          ADAPTED (Qc)
                    ┌──────────────────────┐          ┌──────────────────────────────┐
  TEMPLATE           (none — HF never          │          │ full_causal_mask [1,1,gcl,gcl] │
                     precomputes a mask         │          │ full_swa_mask    [1,1,lcl,lcl] │
                     template)                  │          │ built ONCE, memoized —         │
                    └──────────────────────┘          │ but NOT actually sliced/used   │
                                                        │ by build_text() (dead intent)  │
                                                        └──────────────────────────────┘
                               │                                       │
  PER-CALL BUILD    create_causal_mask() /              torch.full(mask_neg) then
                    create_sliding_window_causal_mask()  per-row loop flips valid
                    — vectorized predicate over           cells to 0.0, using
                    torch.arange, INSIDE forward()        kv_glb/write_base_glb —
                    reads cache.get_seq_length() LIVE      OUTSIDE the model entirely
                               │                                       │
  SHAPE              [B,1,Q,KV], KV grows every           [1,1,arn,gcl]/[1,1,arn,lcl],
                    call                                   FIXED forever, every call
                               │                                       │
  MASKED VALUE       -inf (dtype min)                     mask_neg = -200 (mild,
                                                             quantization-safe)
                               │                                       │
  APPLICATION        attn_weights + mask                  attn_weights + mask   (additive mode)
                    (always additive)                     OR
                                                            where(mask==0, x, min+mask_neg)
                                                            (masked-softmax mode, the default)
```

---

## Engineer-Level Summary (design-review pitch)

**One-sentence framing:** *"We moved mask construction entirely outside the compiled
model, fixed its shape to the compile-time context length instead of letting it track live
cache state, replaced `-inf` with a mild, quantization-safe constant, and added an
alternate masked-softmax application mode — all mirroring the same 'fixed shape, externally
supplied, hardware-friendly ops' pattern used for the KV cache."*

**Three concrete points, in priority order for a design review:**

1. **Shape fixed for the same reason as the cache.** The mask's `KV` axis is tied 1:1 to
   the cache buffer's allocated size (`gcl`/`lcl`), not to how much history actually
   exists — because both are compile-time decisions for the same static graph, not two
   independent choices.
2. **`mask_neg=-200` instead of `-inf` is a deliberate quantization-safety choice**, applied
   consistently at template-build time (`clamp_min`) and per-call build time (`torch.full`).
   `-inf` risks NaN/inf propagation and is unrepresentable cleanly in fixed-point; `-200` is
   large enough to still dominate the softmax (pushes masked positions to ~0 probability)
   while staying inside a safely representable numeric range.
3. **The precomputed-template optimization was designed but isn't live** — worth surfacing
   proactively in a review rather than waiting to be asked: `full_causal_mask`/`full_swa_mask`
   exist, are computed correctly, and are unused by the actual per-call path, because of the
   `write_base`-vs-`kv_glb` alignment mismatch. Either this is accepted tech debt (rebuild
   cost is apparently acceptable) or it's a real opportunity to wire up the intended O(1)
   slicing path if per-call mask-build cost ever becomes a bottleneck.
