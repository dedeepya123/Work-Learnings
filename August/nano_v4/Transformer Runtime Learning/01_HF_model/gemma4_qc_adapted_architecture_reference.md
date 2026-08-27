# Qc-Adapted Gemma4 Text Model — Architecture Reference (NanoV4 `qlib/qadaptation.py`)

Companion to `gemma4_text_architecture_reference.md` (vanilla HF). Same layer-by-layer
structure, but every section calls out **exactly what changed and why**, sourced from
`Nano/NanoV4/qlib/qadaptation.py` (the pipeline's own `ADAPTATION_N` comments are quoted
verbatim where present). This is NanoV4's adaptation layer — a different, more mature
codebase than the `pythonic_api/nano/models/gemma4_text/reauthoring.py` one covered earlier;
same goals (HTP-friendly graph), some mechanics differ (notably: cache scatter-writes here,
vs. plain concat there). Both are legitimate QAIRT adaptation strategies — worth knowing
which repo you're in when discussing specifics.

---

## 1. `Gemma4ForCausalLM` (adapted) — top level

```python
class Gemma4ForCausalLM(Gemma4ForCausalLM_original):
    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma4TextModel(config)          # the ADAPTED text model, not vanilla
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer("cache_tensor", torch.arange(config.input_tokens_per_inference),
                                  persistent=False)
```

### Why `cache_tensor` exists

**Problem it solves:** on-device / traced-graph inference can't take an arbitrary
Python-side `torch.arange(...)` call as a *dynamic* graph input every step — for a fixed,
compiled HTP graph, "how many new tokens are being processed this call" (`input_tokens_per_inference`,
aka **ARN** — the fixed prefill/decode chunk size the graph is compiled for) is a
**constant**, decided at compile time, not something recomputed at runtime.

`cache_tensor = torch.arange(input_tokens_per_inference)` is registered as a **buffer**
(so it's saved/loaded with the model like a weight, and moves with `.to(device)`, but isn't
trained) — a fixed `[0, 1, ..., ARN-1]` offset ramp. Then at call time:
```python
cache_position = cache_index + self.cache_tensor
```
`cache_index` is a **single scalar** — "where in the sequence does this chunk start" — supplied
by the runtime/caller each step. Adding the constant ramp gives the full per-token
`cache_position` vector *without* the graph needing a dynamic `arange` op at all. This is
the general pattern for building position/cache-index tensors on a fixed-shape compiled
graph: **precompute the ramp once as a buffer, feed only the scalar starting offset as a
real input, add them.**

This is the on-device equivalent of what vanilla HF does with a live Python
`torch.arange(past_seen_tokens, past_seen_tokens + q_len)` call every forward — vanilla can
afford to build that fresh each time since it's eager PyTorch; a compiled HTP graph cannot.

### Two independent ramps: `cache_index` / `cache_tensor` and `swa_cache_index`

```python
if cache_index is not None:
    cache_position = cache_index + self.cache_tensor            # global/full-attention layers
if swa_cache_index is not None:
    swa_cache_position = swa_cache_index + self.cache_tensor      # sliding-attention layers
```
Same buffer (`cache_tensor`), two different scalar starting offsets — because, as covered in
the KV-cache notes, full-attention layers' cache position tracks "how many tokens ever
seen" while sliding-attention layers' effective position can differ once the window starts
evicting. Vanilla HF derives both from `past_key_values.get_seq_length(layer_idx=...)`
internally, per layer type; here, **both scalar starting points are supplied from outside
the graph** (by the runtime driving inference), since a compiled graph can't call methods
on a Python cache object mid-execution.

### Everything gets built *before* calling `self.model(...)`

```python
outputs = self.model(
    input_ids=..., attention_mask=attention_mask, swa_attention_mask=swa_attention_mask,
    position_ids=position_ids, swa_position_ids=swa_position_ids,
    cache_position=cache_position, swa_cache_position=swa_cache_position,
    past_key_values=past_key_values, ...
)
```
Contrast with vanilla `Gemma4Model.forward`, which builds its own causal mask
(`create_causal_mask`/`create_sliding_window_causal_mask`) and its own cache
(`DynamicCache(config=...)`) **internally**, taking only `input_ids`/`attention_mask` (2D)
from the caller. Here, *all* of that — the 4D masks for both layer types, both cache-position
ramps, the cache object itself — must exist as concrete tensors **before** the model call,
because none of that mask/position logic (Python control flow, `config.layer_types` branching,
`torch.arange`) can live inside a traced/compiled graph. The model's `forward` becomes a
"pure consumer" of precomputed tensors, not a builder of them.

### Output — same as vanilla, still logits + optional softcap

```python
logits = self.lm_head(hidden_states[:, slice_indices, :])
if final_logit_softcapping is not None:
    logits = torch.tanh(logits / cap) * cap
```
`logits_to_keep`/`slice_indices` is unchanged from vanilla — only compute logits for the
positions actually needed (e.g. just the last token during decode), not the whole sequence.

---

## 2. `Gemma4TextModel` (adapted)

### Inputs — compare directly against vanilla

| | Vanilla | Adapted |
|---|---|---|
| Cache | builds `DynamicCache(config=...)` internally if missing | **raises** if `use_cache=True` and no cache passed — never auto-builds (see KV-cache-trap notes; same reasoning) |
| Masks | builds both dicts internally (`create_causal_mask`, `create_sliding_window_causal_mask`) | takes `attention_mask` (global) and `swa_attention_mask` (sliding) as **already-built 4D tensors** — comment: *"ADAPTATION 6: Remove attn mask creation from inside model"* |
| Position embeddings | computed once per `layer_type` from a single `position_ids`, internally | takes **separate** `position_ids` (full) and `swa_position_ids` (sliding) — each may already arrive as a precomputed `(cos,sin)`-like tuple, or as raw indices to run through `rotary_emb` here |
| Cache position | derived from `past_key_values.get_seq_length()` | `swa_cache_position` computed here **only as a fallback** if not supplied (mirrors HF's own default, using `sliding_window_pattern` instead of a raw layer index) — normally supplied precomputed from `ForCausalLM` |
| Output format | dataclass only | **`return_dict` is forced through explicitly** — comment: *"ADAPATATION 4: We need to return the output from the model in dict format otherwise jit trace cannot trace the outputs correctly and they are not seen in the onnx graph."* Also returns extra raw tuple outputs (`(swa_k, swa_v)`, `(global_k, global_v)`) when `return_dict=False`, for downstream MTP (multi-token-prediction/speculative-drafting) consumption. |

### Legacy-cache conversion — a different mechanism than vanilla's

```python
# ADAPTATION 5: We convert the past_key_values which flow into the model in the tuple format
# into the Cache object format. Even though we create a dynamic cache object here, the
# past_key_values tuple already contains the hybrid KV cache information i.e the sliding
# window layer have the KV cache of the size sliding_window.
if past_key_values is not None and not isinstance(past_key_values, Cache):
    past_key_values = from_legacy_cache(past_key_values)   # tuple-of-tuples -> DynamicCache_adapted
```
Vanilla does the same *kind* of thing (tuple ↔ `Cache` object, for ONNX-export tracer
compatibility — `Cache` objects can't be traced as graph I/O) but with a different target
class: vanilla builds a plain `DynamicCache()`; here it's `DynamicCache_adapted` (§5) — every
layer becomes the *same* adapted class regardless of `layer_type`, continuing the same "ban
`config=`, avoid the sliding-vs-full class split" strategy from the KV-cache-trap notes, just
implemented via a different route (custom `Cache` subclass) than the other repo's (raise +
never build one at all internally).

### Decoder loop returns K/V now, not just hidden states

```python
hidden_states, key_states, value_states = decoder_layer(...)
if self.config.layer_types[i] == "sliding_attention":
    swa_k, swa_v = key_states, value_states
else:
    global_k, global_v = key_states, value_states
```
Vanilla's decoder layer returns only `hidden_states`. This adapted version threads K/V back
out of *every* layer call, keeping the **last** sliding-type and **last** full-type K/V
around as `(swa_k, swa_v)` / `(global_k, global_v)` — exposed in the model's output tuple.
This is specifically for **MTP (multi-token prediction / speculative drafting)** — the small
drafter model (`Gemma4AssistantModel`, cross-attending via `Gemma4MtpCrossAttention`) needs
read access to the backbone's final-layer K/V without recomputing it.

Everything else — embedding, PLE injection, `shared_kv_states = {}` reset per call, the
per-layer-type branch selecting `(cache_position, attention_mask, position_embeddings)`,
final `self.norm(...)` — mirrors vanilla structurally; only the *source* of each precomputed
tensor differs (outside-supplied here vs. built-internally there).

---

## 3. `Gemma4TextDecoderLayer` (adapted)

Structurally **identical** sandwich-norm pattern to vanilla (input_layernorm → attn →
post_attention_layernorm → residual; pre_feedforward_layernorm → mlp → post_feedforward_layernorm
→ residual; optional PLE branch; final `layer_scalar` gate). The only change:

```python
hidden_states, _, key_states, value_states = self.self_attn(...)   # was: hidden_states, _ = self.self_attn(...)
...
return hidden_states, key_states, value_states                       # was: return hidden_states
```
Passes `cache_position` through explicitly as a named forward arg (so it's a real traceable
graph input, not buried in `**kwargs`) and threads K/V back up to the model level for the
MTP use described in §2. No change to the residual/norm math itself.

---

## 4. `Gemma4TextAttention` (adapted) — the real substance

### `__init__` — new fixed attributes, chosen via `AdaptationFlags`

```python
self.apply_rope_fn = ApplyRopeSingle()
adaptations = getattr(config, 'adaptations', AdaptationFlags())
self.enable_masked_softmax = adaptations.enable_masked_softmax
self._kv_fake_quant_fn = fake_quant / fake_quant_activation, depending on adaptations.kv_clip_only
```
`AdaptationFlags` (`qadaptation_flags.py`) is a small dataclass of toggles governing which
HTP-friendly substitutions are active (`rms_norm`, `linear_to_conv`, `use_erf_gelu`,
`kv_clip_only`, `enable_masked_softmax`, etc.) — a single place controlling adaptation
behavior across the whole model, rather than each class hardcoding a choice.

### RoPE application — `ApplyRopeSingle`, comment `ADAPTATION_1`

```python
## ADAPTATION_1: we apply the rope separately to query and key states for on-target efficiency
## Creating a separate class because we want to uniquely identify the EleMul operations in QuantSim
class ApplyRopeSingle(nn.Module):
    def forward(self, x_real, x_im, rope_vals):
        rope_real, rope_im = rope_vals
        x_prod_real = self.mul_x_real_rope_real(x_real, rope_real) - self.mul_x_im_rope_im(x_im, rope_im)
        x_prod_im   = self.mul_x_real_rope_im(x_real, rope_im)   + self.mul_x_im_rope_real(x_im, rope_real)
        x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x_real.shape[:-1], -1)
        return x
```
Same complex-multiply RoPE math derived in the attention notes (`z' = z * e^{iθ}`), but:
- Uses **four separate named `MulModule` submodules** instead of inline `*` operators —
  purely so each elementwise multiply is a distinct, individually-identifiable node in the
  traced graph for AIMET QuantSim to assign its own quantization encoding to. This is
  a recurring QAIRT pattern: wrap primitive ops (`Add`, `Mul`, `Matmul`, ...) as named
  `nn.Module`s so the quantizer's per-op calibration has a stable graph target — not
  something vanilla PyTorch code needs to care about at all.
- Called on **query and key separately** (`apply_rope_fn(q_real, q_im, ...)`,
  `apply_rope_fn(k_real, k_im, ...)`), each pre-split into real/imaginary halves by the
  caller — same "no `rotate_half` concat-negate trick, explicit half-split instead" pattern
  covered for the other repo's reauthoring code. Different repo, same underlying rationale
  (HTP-friendly graph shape).

### KV sharing logic — unchanged concept, same as vanilla

`is_kv_shared_layer` branch, `shared_kv_states[kv_shared_layer_index]` lookup,
`store_full_length_kv` publish — identical to what's covered in the `Gemma4TextAttention`
notes and the KV-cache-mechanics notes. No adaptation-specific change here; this logic is
config-driven and hardware-agnostic.

### Transposed key cache — `ADAPTATION_2` / `ADAPTATION_3`

```python
if transposed_key_cache:
    key_states = key_states.transpose(2, 3)     # store K with head_dim/seq axes swapped
...
# ADAPTATION_3: We require to redefine the attention class since we need to pass additional
# cache_kwargs (specifically the transposed key cache and the return_new_key_value_only)
cache_kwargs = {
    "cache_position": cache_position,
    "transposed_key_cache": transposed_key_cache,
    "num_key_value_heads": self.config.num_key_value_heads,
    "return_new_key_value_only": return_new_key_value_only,
    "head_dim": self.head_dim,
}
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```
Same transposed-key motivation covered before (avoids an expensive on-target transpose
inside the QK matmul — see `ADAPTATION_2` in `eager_attention_forward`, §5 below). What's
new here vs. the other repo: the cache's `update()` needs **extra metadata** (`transposed_key_cache`,
`num_key_value_heads`, `head_dim`, `return_new_key_value_only`) to know how to
scatter/concat correctly — hence a custom `cache_kwargs` dict threaded through, and a
custom `Cache`/`CacheLayer` pair to consume it (§5).

### V is normalized but never rope'd — same as vanilla, confirmed again here

`value_states = self.v_norm(value_states)` then straight to `transpose(1,2)` — no
`apply_rope_fn` call on V at all. Consistent with the "RoPE only matters for the Q·K dot
product" reasoning already established.

### Fake-quantizing the KV cache

```python
if self.k_cache_scale is not None and self.k_cache_num_bits is not None:
    key_states = self._kv_fake_quant_fn(key_states, int(self.k_cache_num_bits.item()), self.k_cache_scale)
```
Simulates the precision loss of an on-device quantized KV cache (clip-only or full
fake-quant, depending on `kv_clip_only`) *before* the tensor ever reaches `past_key_values.update()`
— so the numbers actually stored/returned already reflect what a real int8/int16 HTP cache
would produce, letting downstream AIMET calibration and accuracy tests see realistic
values instead of full-fp32 ones.

---

## 5. Cache mechanics — `DynamicCache_adapted` / `DynamicLayer_adapted`

This is the biggest structural difference from vanilla HF (and from the other repo's
"ban `config=`, force plain `DynamicLayer`" strategy) — here, **the cache's `update()`
itself is rewritten** to support a **scatter-write**, not just concat:

```python
class DynamicCache_adapted(Cache):
    def __init__(self):
        super().__init__(layer_class_to_replicate=DynamicLayer_adapted, ...)   # every layer, same adapted class

class DynamicLayer_adapted(DynamicLayer):
    def update(self, key_states, value_states, cache_kwargs=None):
        if self.keys is None:
            self.keys, self.values = key_states, value_states     # first call: just store as-is
            return self.keys, self.values

        key_cat_dim = -1 if transposed_key_cache else -2
        if self.values.shape[-2] <= cache_position[-1]:
            # cache_position runs past what's currently stored -> genuinely growing: concat
            key_cache = torch.cat([self.keys, key_states], dim=key_cat_dim)
            value_cache = torch.cat([self.values, value_states], dim=-2)
        else:
            # cache_position falls WITHIN already-allocated space -> overwrite in place: scatter
            indices = cache_position.view(1,1,1,-1).expand(...)
            value_cache = self.values.scatter(dim=-2, index=indices.transpose(-1,-2), src=value_states)
            key_cache = self.keys.scatter(dim=key_cat_dim, index=indices, src=key_states)

        if return_new_key_value_only:
            self.keys, self.values = key_states, value_states   # store only the new slice (not full history)
        else:
            self.keys, self.values = key_cache, value_cache     # store the full updated tensor
        return key_cache, value_cache
```

**Why scatter, not just concat:** on a fixed-shape compiled HTP graph, the KV cache tensor
is typically **pre-allocated to its full context length up front** (not grown dynamically
step by step, the way eager PyTorch's `torch.cat` does). Writing a new token's K/V means
placing it **at a specific known offset** inside that pre-allocated buffer —
`tensor.scatter(dim=..., index=cache_position, src=new_values)` — rather than reallocating
a bigger tensor every step (`torch.cat`), which a static graph can't do at all mid-execution.
The branch (`self.values.shape[-2] <= cache_position[-1]`) exists to still support the
"currently allocated space isn't big enough yet, actually extend it" case — needed
specifically when tracing/exporting the model itself (building intermediate MPP graphs),
per the inline comment — but on real fixed-shape device inference, the scatter branch is
what actually runs every step.

**`return_new_key_value_only`**: an extra flag controlling whether `self.keys`/`self.values`
get overwritten with just the newest slice (useful when a *separate*, externally-managed
buffer is what actually persists across calls — e.g. the runtime keeps the real KV$ buffer
device-side, and this Python-level cache object is only used for graph tracing / a single
call's bookkeeping) versus the full accumulated tensor (closer to vanilla `DynamicLayer`
behavior).

**Sliding-window truncation:** notice `DynamicLayer_adapted` extends `DynamicLayer`
directly, **not** `DynamicSlidingWindowLayer` — and `DynamicCache_adapted.__init__` uses
`layer_class_to_replicate=DynamicLayer_adapted` for **every** layer, matching the same
"ban the class-split, keep everything as one patchable class" strategy from the KV-cache
trap. Sliding-window behavior here is *entirely* delegated to the mask
(`swa_attention_mask`) and to how the *runtime* sizes/manages the pre-allocated buffer
outside this class — the cache-layer class itself carries no window-truncation logic at
all (contrast with vanilla `DynamicSlidingWindowLayer`, which self-truncates every
`update()` call).

---

## 6. Attention interface (adapted `eager_attention_forward`)

```python
# ADAPTATION_2: We send the transposed key cache to avoid the transpose inside matmul,
# it is expensive on target
if transposed_key_cache:
    attn_weights = torch.matmul(query, key_states) * scaling          # K already stored pre-transposed
else:
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling   # vanilla path

if module.enable_masked_softmax:
    attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
    attn_weights = torch.where(causal_mask == 0, attn_weights, attn_weights_min + mask_neg)
else:
    attn_weights = attn_weights + causal_mask     # vanilla-style additive mask
```

Two adaptation-specific choices beyond the transposed-key trick:

- **Masked softmax as an alternative to additive masking** — instead of adding `-inf`
  (unrepresentable/awkward for fixed-point HTP arithmetic) to masked positions, compute the
  **minimum** real attention score present, then `where(mask==0, attn_weights, min + mask_neg)`
  — i.e. replace masked positions with "the smallest real score minus a further margin
  (`mask_neg`, e.g. -200)." This achieves the same effect (masked positions get the lowest
  softmax probability) using only ops that are well-behaved in quantized/fixed-point math,
  no `-inf`/`NaN` risk.
- **Slicing the mask to match `value_states`' actual length** — a defensive slice
  (`causal_mask = attention_mask[:, :, :, :value_states.shape[-2]]`) needed specifically
  when tracing the adapted model for MPP-graph construction (intermediate KV-cache lengths
  during that process can legitimately differ from the mask's built length) — a
  tracing-pipeline concern, not a runtime behavior change.

`repeat_kv` (GQA broadcast) and the core `softmax`/`matmul` structure are otherwise
unchanged from vanilla.

---

## 7. Summary — adaptation-by-adaptation index

| Tag (as commented in code) | What it changes | Why |
|---|---|---|
| `ADAPTATION_1` | RoPE applied via `ApplyRopeSingle`, split real/imag, named `MulModule`s | HTP-friendly graph shape + stable per-op quantization targets for AIMET |
| `ADAPTATION_2` | Transposed key cache; skip transpose inside QK matmul | Avoids an expensive transpose op on-target |
| `ADAPTATION_3` | Custom `cache_kwargs` threaded into `past_key_values.update()` | Cache needs extra metadata (transposed layout, head_dim, scatter-vs-concat) to update correctly |
| `ADAPTATION_4` | `return_dict` forced through `Gemma4TextModel.forward` | JIT/ONNX tracer can't see outputs correctly from non-dict returns |
| `ADAPTATION_5` | Legacy tuple cache converted to `DynamicCache_adapted`, not vanilla `DynamicCache` | Keeps every layer on one patchable, scatter-capable class regardless of `layer_type` |
| `ADAPTATION_6` | Mask/cache/position tensors built **outside**, passed into the model | None of that Python-side construction logic can live inside a traced/compiled graph |
| (unlabeled) `cache_tensor` buffer | Fixed `arange(ARN)` ramp + scalar offset replaces live `torch.arange` | Fixed-shape compiled graph needs position tensors built from a constant + one scalar input, not a dynamic op |
| (unlabeled) masked softmax | `where(mask==0, x, min+mask_neg)` instead of additive `-inf` | Fixed-point/quantized-arithmetic friendly; avoids `-inf` |
| (unlabeled) scatter-write cache | `tensor.scatter(...)` instead of unconditional `torch.cat` | Pre-allocated fixed-size on-device buffer; writes go to a known offset, not a growing tensor |
