# KV Cache: Theory → HF Implementation → QAIRT Adaptation (`qadaptation.py`)

Mentor-style walkthrough. You already have the theory. This maps theory → the exact HF
functions that implement it → the exact adapted functions in
`Nano/NanoV4/qlib/qadaptation.py` that replace them, with shapes and numbers at every step.

Reference config used throughout (from `misc/pipeline_config.json` and `qmodel.py`):
- `ARN = 521` (tokens processed per graph invocation — `input_tokens_per_inference`)
- `CONTEXT_LENGTH (CL) = 15527` (total cache capacity)
- `transposed_key_cache = True`, `return_new_key_value_only = True` (set in `qmodel.py:228-229`)
- Text config: `num_attention_heads=8`, `num_key_value_heads=4` (GQA), `head_dim=256`,
  `num_hidden_layers=35`, `num_kv_shared_layers=20`, `sliding_window=512`

For the worked numeric example (§8) we'll use small stand-in numbers (`ARN=4`, `CL=12`,
`sliding_window=4`) so the arithmetic is checkable by hand — the mechanism is identical at
real scale.

---

## 1. Standard HF KV Cache Flow

```
Gemma4TextModel.forward(input_ids, past_key_values=None, use_cache=True, ...)
      │
      ├─ if use_cache and past_key_values is None:
      │      past_key_values = DynamicCache(config=self.config)   ← CREATE, live Python object
      │
      ├─ for each layer i:
      │      Gemma4TextAttention.forward(..., past_key_values)
      │            │
      │            ├─ k_proj(hidden_states) → k_norm → rope(k)
      │            ├─ v_proj(hidden_states) → v_norm
      │            ├─ past_key_values.update(k, v, layer_idx)      ← UPDATE, grows tensor
      │            │        Cache.update() dispatches to:
      │            │        DynamicLayer.update()          (full-attention layers)
      │            │        DynamicSlidingWindowLayer.update()  (sliding layers, self-truncates)
      │            └─ returns full accumulated (k, v) for THIS call → used immediately
      │                    by attention_interface(q, k, v, mask)    ← RETRIEVE (same call)
      │
      └─ returns hidden_states, past_key_values (mutated cache object, carried to next call)
```

**Key property:** creation, update, and retrieval are three separate Python method calls on
a live object (`DynamicCache` instance), each doing real dynamic-shape work
(`torch.cat`, conditional truncation) at the time they're called. The cache is *not* a
model input — it's internal state threaded through the Python call stack across
`.generate()` steps.

---

## 2. Adapted (Qc) KV Cache Flow — `qadaptation.py`

```
Gemma4ForCausalLM.forward(input_ids, cache_index, swa_cache_index, past_key_values, ...)
      │
      ├─ cache_position     = cache_index     + self.cache_tensor   ← built OUTSIDE model logic,
      ├─ swa_cache_position = swa_cache_index + self.cache_tensor      from scalar + fixed buffer
      │
      ├─ self.model(..., past_key_values, cache_position, swa_cache_position)
      │      │
      │      ├─ past_key_values already exists (CREATED once, upstream, before the
      │      │   compiled graph ever runs — see §5). Model NEVER auto-creates it.
      │      │   If a legacy tuple arrives: from_legacy_cache() wraps it in
      │      │   DynamicCache_adapted (uniform class for ALL layers, no type-split)
      │      │
      │      ├─ for each layer i:
      │      │      Gemma4TextAttention.forward(..., past_key_values, cache_position)
      │      │            │
      │      │            ├─ k_proj → k_norm → apply_rope_fn(k)  → transpose(2,3) if transposed_key_cache
      │      │            ├─ v_proj → v_norm
      │      │            ├─ past_key_values.update(k, v, layer_idx, cache_kwargs)  ← UPDATE
      │      │            │        DynamicLayer_adapted.update() — SCATTER into fixed buffer
      │      │            │        at cache_position, or CONCAT only while still growing during trace
      │      │            └─ returns (k, v) per return_new_key_value_only flag  ← RETRIEVE
      │      │                    used immediately by eager_attention_forward(q, k, v, mask)
      │      │
      │      └─ returns hidden_states, past_key_values, (swa_k,swa_v), (global_k,global_v)
      │
      └─ lm_head(hidden_states) → logits
```

**Key property:** the cache buffer's **shape is fixed before compilation** and never
resized at runtime. Update is a **scatter-write to a known offset**, not a growing
concatenation. Creation happens once, outside the compiled graph's execution (the graph
only ever *consumes and mutates* a pre-existing buffer via named `past_key_i_in` /
`past_key_i_out` I/O tensors — confirmed real, e.g. `GENAI_BUILDER_STAGE.md`'s own graph
I/O listing: *"the 15 pairs of `past_key_i_in`/`past_value_i_in`"*).

---

## 3. Side-by-Side Comparison

| Aspect | Standard HF | Adapted (Qc) |
|---|---|---|
| Cache object | `DynamicCache` (or `DynamicSlidingWindowLayer` mix, per `layer_types`) | `DynamicCache_adapted` — **one uniform class for every layer**, regardless of type |
| Creation | Auto-created inside `forward()` if missing: `DynamicCache(config=self.config)` | **Never** auto-created inside `forward()`; either passed in already-built, or converted from a legacy tuple via `from_legacy_cache()` |
| Cache tensor growth | `torch.cat([self.keys, key_states], dim=-2)` — tensor **grows** every call | Pre-allocated to `CONTEXT_LENGTH` up front; `update()` **writes into** existing space via `scatter`, tensor size **never changes** post-allocation (real device path) |
| Where new K/V goes | Appended at the end (`cat`) | Written at the exact offset given by `cache_position`, via `tensor.scatter(dim=-2, index=cache_position, src=new_kv)` |
| Sliding-window truncation | `DynamicSlidingWindowLayer.update()` self-truncates to `sliding_window - 1` every call | **No truncation logic in the cache class at all** — every layer is a plain `DynamicLayer_adapted`; windowing is 100% delegated to the mask (`swa_attention_mask`, built externally) |
| Position tracking | `self.cumulative_length` (int, incremented in Python) | `cache_position` tensor (`cache_index + cache_tensor`), supplied from outside, no Python-side counter inside the cache object |
| K storage layout | `[B, num_kv_heads, seq_len, head_dim]` | `[B, num_kv_heads, head_dim, seq_len]` when `transposed_key_cache=True` — **last two axes swapped** |
| What `update()` returns | Always the **full accumulated** history (even for sliding layers, before their own truncation) | Governed by `return_new_key_value_only`: `True` → returns/stores **only this call's new slice**; `False` → full tensor, HF-like |
| Mask/position construction | Built **inside** `forward()`, from live `past_key_values.get_seq_length()`/`get_mask_sizes()` calls | Built **outside** the model entirely (by `generator.py`/runtime), passed in as plain 4D tensors |
| Dynamic ops used | `torch.arange`, `torch.cat`, Python `if past_key_values is None` | Fixed `cache_tensor` buffer + scalar add; `scatter`; branching kept to shape-comparison only |

---

## 4. Tensor Shape Evolution

Using `ARN=521`, `CL=15527`, `num_key_value_heads=4`, `head_dim=256`, `B=1`, for one
**real** (non-KV-shared) layer:

| Stage | Standard HF | Adapted (Qc, real device path) |
|---|---|---|
| Cache at t=0 (before any call) | `keys = torch.tensor([])` → shape `[0]` (lazy, unallocated) | `keys` shape `[1, 4, 15527, 256]` — **fully allocated**, e.g. zero-filled, before first inference call |
| Prefill (ARN=521 new tokens) K/V produced | `[1, 4, 521, 256]` | `[1, 4, 521, 256]` (untransposed) or `[1, 4, 256, 521]` (transposed) |
| Cache **after** prefill `update()` | `keys` shape becomes `[1, 4, 521, 256]` (cat of `[]` + 521 = 521) | `keys` shape **stays** `[1, 4, 15527, 256]` — the 521 new entries are scattered into positions `0..520`; buffer size unchanged |
| What `update()` **returns** at this step | `[1, 4, 521, 256]` — the full (only) history so far | Governed by `return_new_key_value_only=True` → returns `[1, 4, 521, 256]` (just the new slice, matching HF's per-call semantics for the *consumer*, but NOT because the buffer itself is that size) |
| Decode step 1 (ARN=1 new token) K/V produced | `[1, 4, 1, 256]` | `[1, 4, 1, 256]` |
| Cache **after** decode-step-1 `update()` | `keys` shape → `[1, 4, 522, 256]` (grew by 1) | `keys` shape **still** `[1, 4, 15527, 256]` — new entry scattered into position `521` |
| What attention actually reads | Full accumulated tensor, growing every step | Full pre-allocated buffer `[1, 4, 15527, 256]`, but **masked** so only valid (written) positions contribute — see §6 |

**The core shape difference in one sentence:** HF's cache tensor's *shape itself* is the
record of how much history exists; the adapted cache's shape is **always** the fixed
`CONTEXT_LENGTH`, and "how much history exists" lives entirely in `cache_position`/the mask
instead.

---

## 5. Cache Lifecycle: Creation → Update → Retrieval

### 5a. Creation

**Standard HF** — `Gemma4TextModel.forward`:
```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)
```
*Purpose:* lazily build a cache matching `config.layer_types` if the caller didn't supply
one. *Inputs:* `self.config`. *Output:* a `DynamicCache` with one `DynamicLayer` or
`DynamicSlidingWindowLayer` object per real (non-shared) decoder layer, each internally
`is_initialized=False`, no tensors yet (`keys=None`). *Shape at this point:* none — nothing
allocated until the first `update()` call per layer.

**Adapted** — `Gemma4TextModel.forward` (`qadaptation.py:762-764`):
```python
if use_cache and past_key_values is None:
    raise ValueError("use_cache is True but past_key_values are not provided")
    # past_key_values = DynamicCache(config=self.config)   ← commented out, deliberately dead code
```
*Purpose:* **forbid** auto-creation — same "ban `config=`, avoid the sliding-layer-class
trap" reasoning already established (KV-cache-trap notes) applies here too, since
`DynamicCache(config=...)` would still split layer classes by type. *Who actually creates
the cache instead:* upstream, outside this function — either the runtime allocates the real
fixed-size device buffer once before any inference call (real device path — the compiled
graph's `past_key_i_in` tensors *are* the allocation), or, for tracing/PyTorch-level work,
`from_legacy_cache()` builds a `DynamicCache_adapted` from a tuple:
```python
def from_legacy_cache(past_key_values):
    cache = DynamicCache_adapted()
    for layer_idx in range(len(past_key_values)):
        key_states, value_states = past_key_values[layer_idx]
        cache.update(key_states, value_states, layer_idx)
    return cache
```
*Inputs:* a tuple of `(key, value)` tuples, one per layer. *Output:* a
`DynamicCache_adapted` — every layer, `DynamicLayer_adapted` (uniform class, no
type-split). *Shape change:* whatever shape the incoming tuple's tensors already had —
this function doesn't allocate, it wraps.

### 5b. Update — the function to know cold

**Standard HF** — `DynamicLayer.update()` (`transformers/cache_utils.py:102`):
```python
def update(self, key_states, value_states, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
    self.keys = torch.cat([self.keys, key_states], dim=-2)
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values
```
*Purpose:* append this call's new K/V onto whatever's already stored. *Inputs:*
`key_states`/`value_states` shape `[B, num_kv_heads, new_len, head_dim]`. *Output:* the
**full** accumulated `[B, num_kv_heads, total_len, head_dim]` tensor. *Shape change:*
`total_len` grows by `new_len` every call, unbounded.

**Adapted** — `DynamicLayer_adapted.update()` (`qadaptation.py:2057-2096`):
```python
def update(self, key_states, value_states, cache_kwargs=None):
    if self.keys is None:
        self.keys, self.values = key_states, value_states
        return self.keys, self.values

    key_cat_dim = -1 if transposed_key_cache else -2
    if self.values.shape[-2] <= cache_position[-1]:
        # buffer not big enough yet (trace-time growth only) -> concat
        key_cache = torch.cat([self.keys, key_states], dim=key_cat_dim)
        value_cache = torch.cat([self.values, value_states], dim=-2)
    else:
        # cache_position lands INSIDE already-allocated space -> scatter (real device path)
        indices = cache_position.view(1,1,1,-1).expand(B, num_key_value_heads, head_dim, ARN)
        value_cache = self.values.scatter(dim=-2, index=indices.transpose(-1,-2), src=value_states)
        key_cache   = self.keys.scatter(dim=key_cat_dim, index=indices_correctly_oriented, src=key_states)

    if return_new_key_value_only:
        self.keys, self.values = key_states, value_states   # store only what's new
    else:
        self.keys, self.values = key_cache, value_cache      # store the full buffer
    return key_cache, value_cache
```
*Purpose:* write this call's new K/V **at the position `cache_position` specifies**, inside
a buffer whose size is otherwise fixed. *Inputs:* `key_states`/`value_states`
`[B, num_kv_heads, ARN, head_dim]` (or transposed), plus `cache_kwargs` (`cache_position`,
`transposed_key_cache`, `num_key_value_heads`, `head_dim`, `return_new_key_value_only`).
*Output:* `(key_cache, value_cache)` — shape depends on `return_new_key_value_only`:
`[B, num_kv_heads, ARN, head_dim]` if `True` (just the new slice, matching what
`eager_attention_forward` needs *this* step alongside the still-valid rest of the
pre-allocated buffer held elsewhere), or the full `[B, num_kv_heads, CL, head_dim]` buffer
if `False`. *Shape change:* **the stored buffer (`self.keys`) never resizes** after the
first allocation — this function only ever writes *into* fixed space (scatter) or, only
during trace-time construction, extends it once (concat), never both repeatedly like HF's
unconditional `cat`.

### Cache-update worked format (as requested) — one real step, prefill → decode boundary

**Before:**
```
cache.keys shape = [1, 4, 15527, 256]   (pre-allocated, positions 0..520 already written by prefill,
                                          positions 521..15526 still zero/unwritten)
cache_position for this call = [521]    (scalar cache_index=521 + cache_tensor[0]; ARN=1 decode step)
```
**Current token:**
```
K_new shape = [1, 4, 1, 256]   (or [1,4,256,1] if transposed_key_cache)
V_new shape = [1, 4, 1, 256]
```
**Operation:**
```
self.values.shape[-2] (=15527) > cache_position[-1] (=521)  →  scatter branch taken
indices = cache_position.view(1,1,1,-1).expand(1, 4, 256, 1)      # target offset = 521
value_cache = self.values.scatter(dim=-2, index=indices.transpose(-1,-2), src=V_new)
key_cache   = self.keys.scatter(dim=key_cat_dim, index=indices, src=K_new)
```
**After:**
```
cache.keys shape = [1, 4, 15527, 256]   (UNCHANGED shape — only position 521's slice overwritten)
return_new_key_value_only=True → self.keys, self.values <- K_new, V_new (only the new slice retained
                                    as this LAYER OBJECT's Python-side state; the real 15527-length
                                    buffer is the device-side persistent tensor, tracked outside this
                                    Python object on the real inference path)
returned to caller: (key_cache, value_cache) = the freshly-scattered full-shape tensors this call
```

### 5c. Retrieval

**Standard HF:** retrieval and update are the same call — `update()`'s return value *is*
the retrieval; there's no separate "read" step. Attention immediately consumes what
`update()` handed back.

**Adapted:** same pattern — `update()`'s return value feeds directly into
`eager_attention_forward(self, query_states, key_states, value_states, attention_mask, ...)`
in the same `Gemma4TextAttention.forward` call, no separate retrieval step either. The
distinction that matters operationally: what gets "read" for attention on the real device
path is effectively "the new slice just written" **plus** "whatever the mask says is still
valid from the rest of the persistent buffer" — retrieval semantics are carried jointly by
`(cache_position tracking, return_new_key_value_only flag, the mask)`, not by the shape of
what a single `update()` call returns.

---

## 6. Attention Computation — Before and After

**Before adaptation** (vanilla `eager_attention_forward`, `modeling_gemma4.py:1247`):
```python
key_states = repeat_kv(key, module.num_key_value_groups)          # GQA broadcast: 4 -> 8 heads
value_states = repeat_kv(value, module.num_key_value_groups)
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling   # explicit transpose here
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask                   # additive -inf mask
attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)
attn_output = torch.matmul(attn_weights, value_states)
```
Mask is `[B,1,Q,KV]` where `KV` = however much history the cache tensor *actually contains*
at this point (grows every step). Transpose happens live, inside the matmul call.

**After adaptation** (`qadaptation.py`'s `eager_attention_forward`, lines 201-252):
```python
key_states = repeat_kv(key, module.num_key_value_groups)
value_states = repeat_kv(value, module.num_key_value_groups)
if transposed_key_cache:
    attn_weights = torch.matmul(query, key_states) * scaling        # NO transpose — K already
                                                                       # stored pre-transposed
else:
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

if module.enable_masked_softmax:
    attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
    attn_weights = torch.where(causal_mask == 0, attn_weights, attn_weights_min + mask_neg)
else:
    attn_weights = attn_weights + causal_mask

attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)
attn_output = torch.matmul(attn_weights, value_states)
```
Two concrete differences that trace straight back to the cache adaptation:
1. **No live transpose** — because K is *stored* already transposed (`[B, kv_heads, head_dim, seq]`),
   the matmul that used to need `key_states.transpose(2,3)` doesn't anymore. This op was
   removed *because* of the cache layout change, not independently.
2. **Mask/`KV` axis size is fixed** (`CL` or `sliding_window`, not "however much has been
   written") — because the cache buffer's shape never changes, the mask built against it
   has a correspondingly fixed shape too (§3/§4). Masked-softmax (`where(mask==0, x,
   min+mask_neg)`) replaces additive `-inf` for fixed-point-safety reasons unrelated to the
   cache itself, but it's applied against that same fixed-shape mask.

---

## 7. Memory Layout — Before and After

**Before (HF, per real layer):**
```
keys:   [B, num_kv_heads, seq_len_so_far, head_dim]   — grows every call, reallocated by torch.cat
values: [B, num_kv_heads, seq_len_so_far, head_dim]   — same
Total memory in use at time t  ∝  seq_len_so_far  (grows linearly with generation length,
                                                     until you hit whatever HF/host memory allows)
Physical layout: standard row-major, last two axes = (seq, head_dim)
```

**After (Qc, per real layer, device path):**
```
keys:   [B, num_kv_heads, head_dim, CONTEXT_LENGTH]   — TRANSPOSED (head_dim, seq swapped),
                                                          allocated ONCE at CONTEXT_LENGTH, fixed forever
values: [B, num_kv_heads, CONTEXT_LENGTH, head_dim]    — values NOT transposed (only K is)
Total memory in use  =  fixed, CONTEXT_LENGTH-sized, for the ENTIRE session regardless of
                        how many tokens have actually been generated so far
Physical layout: K's last two axes swapped vs. HF — this is done specifically so the QK
                 matmul doesn't need a runtime transpose (§6); V is left in the natural
                 layout since it's never transposed for AV matmul either way.
```

**Key memory tradeoff, worth saying explicitly in a design review:** the adapted layout
trades **peak-allocation-at-session-start** (you commit `CONTEXT_LENGTH`-sized buffers
immediately, even for a 3-token prompt) for **zero reallocation ever** and
**hardware-predictable, constant-shape access patterns** — exactly what a fixed compiled
graph on an accelerator needs, and exactly what a flexible eager-mode Python session
doesn't need to pay for.

---

## 8. One Complete Worked Example, With Numbers

Small stand-in config for hand-checkable arithmetic: `ARN=4` (chunk size), `CL=12`
(context length), `sliding_window=4`, `B=1`, `num_kv_heads=1`, `head_dim=2` (tiny, just to
show real numbers), one **full-attention** layer, `transposed_key_cache=True`,
`return_new_key_value_only=True`.

### Step 0 — Allocation (before any inference call)
```
cache.keys   shape = [1, 1, 2, 12]     ← transposed: (head_dim=2, CL=12)
cache.values shape = [1, 1, 12, 2]     ← not transposed: (CL=12, head_dim=2)
All entries = 0 (unwritten placeholder)
```

### Step 1 — Prefill, ARN=4 new tokens (positions 0,1,2,3)
```
cache_index = 0  →  cache_position = 0 + cache_tensor([0,1,2,3]) = [0,1,2,3]

K_new (post k_proj/k_norm/rope/transpose) shape = [1,1,2,4]     (head_dim=2, new_seq=4)
V_new shape = [1,1,4,2]

Before:
  cache.keys[...,:,0:12]   = all zeros
  cache_position           = [0,1,2,3]

Operation:
  self.values.shape[-2] (=12) > cache_position[-1] (=3)  → SCATTER branch
  indices = [0,1,2,3] broadcast to shape [1,1,2,4] (for keys, transposed layout)
  key_cache   = cache.keys.scatter(dim=-1, index=indices, src=K_new)      # write into cols 0-3
  value_cache = cache.values.scatter(dim=-2, index=indices, src=V_new)    # write into rows 0-3

After:
  cache.keys[..., :, 0:4]   = K_new values   (cols 4..11 still zero)
  cache.values[..., 0:4, :] = V_new values   (rows 4..11 still zero)
  shape of cache.keys/values UNCHANGED: [1,1,2,12] / [1,1,12,2]
  returned (return_new_key_value_only=True): key_cache/value_cache = the SCATTERED full-width
    tensors this call ([1,1,2,12]/[1,1,12,2]) — used immediately for this step's attention;
    self.keys/self.values internally reset to just K_new/V_new ([1,1,2,4]/[1,1,4,2]) per the flag
```
Mask for this step (full attention, `q_idx,kv_idx ∈ [0..3]`, standard causal on the
written region — positions 4-11 are masked out entirely, matching "not written yet"):
```
mask[q=3] allows kv ∈ {0,1,2,3}, forbids {4..11}
```

### Step 2 — Decode, ARN=1 new token (position 4)
```
cache_index = 4  →  cache_position = 4 + cache_tensor([0]) = [4]

K_new shape = [1,1,2,1]
V_new shape = [1,1,1,2]

Before:
  cache.keys[..., :, 0:4] populated (step 1); rest zero
  cache_position = [4]

Operation:
  self.values.shape[-2] (=12) > cache_position[-1] (=4)  → SCATTER branch
  indices = [4] broadcast to [1,1,2,1]
  key_cache   = cache.keys.scatter(dim=-1, index=indices, src=K_new)      # write col 4 only
  value_cache = cache.values.scatter(dim=-2, index=indices, src=V_new)    # write row 4 only

After:
  cache.keys[..., :, 4]   = K_new   (cols 0-3 unchanged from step 1, cols 5-11 still zero)
  cache.values[..., 4, :] = V_new
  shape STILL [1,1,2,12] / [1,1,12,2] — identical to step 0's allocation, never resized
```
Mask for this step: `q_idx=4`, causal → allows `kv ∈ {0,1,2,3,4}`, forbids `{5..11}` — the
mask is what tells attention "only 5 of the 12 allocated slots are real so far," since the
buffer's shape gives no such signal on its own (unlike HF, where the buffer's *shape*
directly told you that).

### Sliding-window variant of step 2 (if this were a sliding layer, `sliding_window=4`)
Same scatter mechanics exactly — **the cache class does nothing different for sliding vs.
full layers** (recall: `DynamicLayer_adapted` is used uniformly, no
`DynamicSlidingWindowLayer_adapted` exists). The only difference is the **mask**:
```
mask[q=4] (sliding, window=4) allows kv ∈ {1,2,3,4}  (excludes kv=0 now — outside the window)
```
Even though `cache.keys[...,:,0]` (position 0's K) is still physically sitting in the
buffer, unmodified — the mask, not the cache, hides it.

---

## Diagram — full lifecycle, both worlds side by side

```
                          STANDARD HF                          ADAPTED (Qc)
                    ┌──────────────────────┐          ┌──────────────────────────────┐
  CREATE            │ DynamicCache(config)  │          │ Pre-allocated buffer          │
                    │ lazy, per-layer-type  │          │ [B,kv_heads,CL,head_dim]      │
                    │ split (Dynamic vs     │          │ SAME class every layer type   │
                    │ DynamicSlidingWindow) │          │ built ONCE, outside forward() │
                    └──────────┬───────────┘          └──────────────┬───────────────┘
                               │                                       │
  UPDATE            torch.cat([old, new], dim=-2)         tensor.scatter(index=cache_position,
                    tensor GROWS every call                            src=new)
                    sliding layers self-truncate           tensor shape NEVER changes
                    to sliding_window-1                    no type-specific truncation logic
                               │                                       │
  RETRIEVE          same call's return value =            same call's return value =
                    full accumulated history                new slice only (return_new_
                    (even for sliding, pre-truncate)         key_value_only) or full buffer
                               │                                       │
  MASK              built from live get_seq_length()/       built externally, fixed shape
                    get_mask_sizes() calls, INSIDE           (ARN x CL or ARN x window),
                    forward(), shape grows with history      content driven by cache_position
                               │                                       │
                    ATTENTION: Q @ K^T needs live            ATTENTION: Q @ K, no transpose
                    transpose (K stored [seq,head_dim])      (K stored PRE-transposed
                                                              [head_dim,seq])
```

---

## Engineer-Level Summary (design-review pitch)

**One-sentence framing:** *"We replaced a dynamically-growing, per-layer-type-specialized
KV cache with a single fixed-size buffer per layer that we write into at a known offset —
trading upfront memory commitment for zero reallocation, uniform per-layer behavior, and a
graph shape that a hardware compiler can actually target."*

**Three concrete engineering consequences, in priority order for a design review:**

1. **Compilability.** A static HTP graph cannot have a tensor whose shape changes call to
   call. `torch.cat`-based growth is fundamentally incompatible with that constraint;
   `scatter`-into-fixed-buffer is not. This is the *load-bearing* reason for the whole
   change — everything else follows from it.
2. **Uniform per-layer cache class.** Vanilla HF splits cache-layer *class* by
   `layer_types` (`DynamicLayer` vs `DynamicSlidingWindowLayer`), and that class owns its
   own truncation logic. We collapse both into one `DynamicLayer_adapted`, and move all
   windowing responsibility into the mask instead. This sidesteps a real correctness trap
   (documented separately: qairt's cache-patch only targets `DynamicLayer`, so
   `DynamicSlidingWindowLayer` would silently run unpatched vanilla code against our
   custom transposed-key convention) — one fewer class means the patch surface is total,
   not partial.
3. **Cost is paid upfront, not amortized.** We allocate `CONTEXT_LENGTH`-sized buffers
   regardless of actual prompt length — a 10-token prompt and a 10,000-token prompt commit
   the same cache memory at session start. This is the direct tradeoff for #1 and #2, and
   it's the number worth having ready if someone in the review asks "what does this cost
   us" — memory-flat, not memory-proportional-to-usage.
