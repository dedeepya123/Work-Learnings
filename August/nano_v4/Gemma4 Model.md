# From NanoV4 to qairt's Pythonic API — a from-scratch walkthrough

Audience: someone who already knows the NanoV4 repo (`qlib/qmodel.py`, `qlib/qadaptation.py`) and
is now looking at `nano/` in this repo, which does the *same kind* of Gemma4 adaptation but through
qairt's own APIs (`QcAutoConfig`, `QcAutoModelForCausalLM`, `TransformersModuleMapping`,
`KVCacheMapping`). This doc explains, from zero, why these APIs exist, what each one does
internally, what changes vs. NanoV4 and what stays conceptually the same, and exactly what we
built for Gemma4 and where it lives.

---

## 1. Why this exists — the problem both repos are solving

A HuggingFace model (`Gemma4ForCausalLM`, `Qwen3ForCausalLM`, ...) is written to be correct and
general on a GPU/CPU. Running it well on Qualcomm's HTP (Hexagon Tensor Processor) backend needs a
handful of behavioral changes that have nothing to do with correctness — they exist purely because
of how the HTP backend and its KV cache work:

- **Transposed key cache** — store keys as `(..., head_dim, seq)` instead of `(..., seq,
  head_dim)`, because that's the layout the matmul on HTP wants.
- **Scatter-based KV cache updates** — instead of "concatenate the new token onto the cache"
  (`torch.cat`, which grows a tensor — bad for a static/traced graph), write the new token's K/V
  into a fixed-size pre-allocated cache at an explicit index (`cache_position`).
- **Masked softmax without `-inf`** — quantized/fixed-point arithmetic doesn't like `-inf`. Instead
  of the normal causal mask (additive `-inf` for masked positions), substitute a large-but-finite
  negative number (`mask_neg`, e.g. `-100`) relative to the row's own minimum score.
- **A fixed number of layers per forward call, static input length, etc.** — needed for exporting
  a static compute graph to run on-device.

NanoV4 solves this by writing its **own copies** of `Gemma4TextAttention`/`Gemma4TextModel` (in
`qadaptation.py`) with these behaviors built in, and monkey-patching them into the `transformers`
module namespace for the duration of model construction (`_patched_gemma4_classes`, a context
manager in `qmodel.py`).

qairt's pythonic API (`qairt.experimental.pipeline.torch.llm...`) solves the *same problem* for
several model families already (Llama, Qwen3, Phi3) with a **different, reusable mechanism**:
register replacement classes ahead of time, then swap them onto an already-built model's modules
in place. What we did for Gemma4 in this repo is write the *Gemma4-specific* replacement classes
qairt doesn't ship, but plug them into qairt's *generic* mechanism instead of inventing a new one.

So: **same target behavior, same "why", different "how you wire it in".**

---

## 2. Key concept: monkey-patch-before-build (NanoV4) vs. swap-class-after-build (qairt)

This is the one idea that explains almost every difference you'll see.

**NanoV4's way:**
```
with _patched_gemma4_classes():        # replaces ~25 names in the transformers.modeling_gemma4
    model = Gemma4ForCausalLM(config)  #   module namespace for the duration of this call
    load weights
# outside the `with`, the module namespace is back to vanilla transformers
```
The patched names are swapped *before* the model's `__init__` methods run, so when
`Gemma4TextAttention.__init__` executes, it's actually NanoV4's own subclass's `__init__` that
runs — the class was substituted at the point of construction.

**qairt's way:**
```
model = Gemma4ForCausalLM(text_config)   # 100% vanilla transformers, __init__ runs normally
load weights                              # weights land on vanilla-shaped modules
model = QcAutoModelForCausalLM._reauthor(model, qc_config=qc_config)
#        ^ walks the already-built model.modules() tree and does:
#              module.__class__ = ReplacementClass
#          for every module whose current class has a registered replacement.
#          __init__ is NOT called again — the instance just "becomes" a different class,
#          keeping every attribute (including weight tensors) it already had.
```

**Why this matters for how you write the replacement classes:**
- A class swap never re-runs `__init__`. So a NanoV4-style adaptation that adds a *new* attribute
  in `__init__` (e.g. a new buffer, a new sub-module) needs, in the qairt world, a separate small
  function (an "init_fn") that qairt calls *right after* the swap to add that attribute
  post-hoc — because the constructor of the replacement class never actually executes.
- Any adaptation that only changes `forward()` behavior — reads a new config flag, does the matmul
  differently — needs **no special handling** at all. The swapped-in class's `forward` is just
  what gets called on the next invocation. This is why most of what we wrote for Gemma4 is "just
  override `forward`", no `__init__` overrides, no init_fn.
- Config objects go through the same trick: `model.config.__class__` gets swapped too, so any new
  config attribute your adapted `forward()` wants to read must survive a swap-without-`__init__` —
  which is why every new config attribute is written as a **lazy property** (explained in §5).

---

## 3. Glossary — qairt terms you'll see, in plain English

| Term | What it actually is |
|---|---|
| **`TransformersModuleMapping`** | A registry + an `apply(model)` method. The registry is a plain dict `{VanillaClass: ReplacementClass}`. `apply()` walks `model.modules()` and does `module.__class__ = ReplacementClass` for every module whose class matches a registry key. This is the class-swap mechanism from §2. |
| **`register(from_cls, to_cls)`** | The function that adds one entry to that dict. Calling it does nothing to any model immediately — it's pure bookkeeping, read later when `apply()` runs. |
| **`QcConfigMixin`** | A mixin class that gives a config class a declarative way to add new fields: you list `{name: default}` pairs in `QC_BACKING_ATTRS`, and the mixin auto-generates a lazy `@property`/`@setter` for each. "Lazy" = the value is computed/defaulted on first *read*, not in `__init__` — required because of the no-`__init__`-on-swap rule from §2. |
| **`QcAttentionMixin`** | A mixin for attention classes. Its one real job: `_update_cache_kwargs_with_qc_config(self, cache_kwargs)` — takes a dict you're building and fills in the QC-specific keys (`transposed_key_cache`, `return_new_key_value_only`, `perform_scatter_kv_cache_update`, `num_key_value_heads`, `head_dim`) by reading `self.config`. It exists so every model's attention class builds the cache-update call in a consistent shape. |
| **`QcAutoConfig`** | A factory (`.from_config(hf_config, **kwargs)`) that: looks up the registered replacement config class for `hf_config`'s `model_type`, builds an instance of it from `hf_config`'s serialized dict, then applies your `**kwargs` onto it via `setattr` (skipping — with a warning — anything that isn't a real attribute). Produces a **detached config object** — it does not touch any model. |
| **`QcAutoModelForCausalLM`** | A wrapper around HF's `AutoModelForCausalLM`. `.from_pretrained(path, qc_config=...)` = (1) load the vanilla HF model from `path`, (2) call `._reauthor(model, qc_config)`. If you already loaded/built your model another way (as we do — see §6), you can skip straight to step 2 by calling `._reauthor(...)` yourself. |
| **`._reauthor(model, qc_config)`** | The actual adaptation step. Calls `TransformersModuleMapping.apply(model, qc_config)` (class-swap everything registered, swap `model.config`'s class too) then `KVCacheMapping.apply(model)` (see next row). This is genuinely the one function call where "the model becomes adapted." |
| **`KVCacheMapping`** | Unlike the above, this does **not** walk `model`'s modules. It globally monkey-patches two methods on `transformers.cache_utils.DynamicLayer` (`update` and `get_seq_length`) to QC-aware versions that understand `cache_kwargs` like `transposed_key_cache`/`cache_position`. Every `DynamicCache` instance created afterward, for any model, picks this up automatically because it's a patch on the shared library class, not a per-model change. |
| **`DynamicCache` / `DynamicLayer`** | HuggingFace's own generic KV-cache classes — one `DynamicLayer` per transformer layer, holding that layer's cached K/V tensors, with `update()` deciding how to fold in new K/V. |
| **`init_fn` / `revert_init_fn`** | Optional callables registered alongside a class mapping, run right after a class swap (`init_fn`) or right before un-swapping (`revert_init_fn`), to patch in/out attributes that a real `__init__` would have set. Not needed for Gemma4 (see §5) — needed by e.g. Qwen3 for its R3-Hadamard rotation buffers. |
| **`mappings.py` / `reauthoring.py`** | A convention, not a requirement: each model family gets its own subpackage under `.../llm/models/<model_type>/` with `reauthoring.py` (the actual replacement classes) and `mappings.py` (the `MAPPINGS` dict + the `register()` calls). We mirrored this for Gemma4 under `nano/models/gemma4_text/`. |
| **CRAFT** | qairt's own quantization-simulation wrapper modules baked into the *vanilla* `transformers` Gemma4 code (`attn_matmul_qk`, `attn_softmax`, rope "operator" modules, etc.) — used for calibration/export. Our adapted classes bypass these (call plain `torch.matmul`/`softmax` instead) since we're doing reauthoring-only, not calibration. This is a deliberate, documented behavior change, not an oversight. |

---

## 4. What stays the same vs. what changes, in general (not Gemma4-specific)

| | NanoV4 | qairt pythonic API |
|---|---|---|
| **The actual adapted math** (transposed-key matmul, scatter cache update, masked-softmax min-trick) | Hand-written in `qadaptation.py` | Same math, hand-written again — in our `reauthoring.py` — because qairt has no Gemma4 support out of the box. **This part had to be written either way; only where it lives changed.** |
| **When the swap happens** | Before `__init__` (monkey-patch module namespace, then construct) | After `__init__` (construct vanilla model, load weights, then swap `__class__` in place) |
| **How you enable it** | `with _patched_gemma4_classes(): model = Gemma4ForCausalLM(config)` | `model = Gemma4ForCausalLM(config); model = QcAutoModelForCausalLM._reauthor(model, qc_config=qc_config)` |
| **Runtime knobs** (`mask_neg`, `transposed_key_cache`, ...) | Plain `setattr(llm_config, "mask_neg", -100)` after building `llm_config` — nothing declares these as real fields | Declared once as `QC_BACKING_ATTRS` on a config mixin/subclass, exposed as real properties, settable via `QcAutoConfig.from_config(..., mask_neg=-100)` keyword args |
| **KV cache implementation** | A fully custom cache structure (flat pre-allocated tensors, own `DynamicCache_adapted`) | Reuses HF's stock `DynamicCache`/`DynamicLayer` — only `update()`/`get_seq_length()` are monkey-patched globally via `KVCacheMapping` |
| **Config discovery** | N/A — NanoV4 doesn't need to "discover" a config class, since it patches everything by hand for every model directly | `TransformersModuleMapping`'s registry + `QcAutoConfig.from_model_type` look up classes by `config.model_type` string — this is what makes the same generic call work for Llama, Qwen3, Phi3, and (after we registered it) Gemma4 |
| **Adding a brand-new model family (e.g. Gemma4) that qairt doesn't ship** | Not applicable — NanoV4 *is* the adaptation for its models, there's no separate "supported models list" | You write `reauthoring.py` + `mappings.py` for that family (exactly what we did) and call `register(...)` — no changes to any qairt-shipped file |

The high-level takeaway for someone coming from NanoV4: **the "what" (which tensor ops need to
change, and why) is identical — you already know that part.** What's different is purely
**mechanical: when/how the substitution is wired into the model**, and **how the new runtime knobs
are declared** so they survive the specific way qairt does substitution (class-swap-without-init).

---

## 5. What we actually built for Gemma4 — file by file

```
nano/
├── main.py                          # loads weights, builds qc_config, calls _reauthor, runs inference
└── models/
    └── gemma4_text/
        ├── mappings.py              # MAPPINGS dict + register() calls
        └── reauthoring.py           # QcGemma4TextConfig, QcGemma4TextAttention, QcGemma4TextModel
```

This mirrors qairt's own internal layout for Qwen3/Llama/Phi3 exactly
(`qairt/experimental/pipeline/torch/llm/models/<name>/{reauthoring,mappings}.py`) — we just keep it
in our own repo since qairt has no built-in Gemma4 package, and its auto-discovery only scans its
*own* installed package tree, never ours. That's why `main.py` has an explicit
`import models.gemma4_text.mappings` — that one import is what triggers the `register()` calls;
without it, `TransformersModuleMapping`'s registry simply has no Gemma4 entries.

### 5.1 `QcGemma4TextConfig` — the six NanoV4 runtime knobs, done as properties

NanoV4's `Gemma4Context.__init__` (`qmodel.py`) does, in effect:
```python
setattr(llm_config, "sliding_window_pattern", ...)
setattr(llm_config, "mask_neg", -100)
setattr(llm_config, "context_length", ...)
setattr(llm_config, "num_layers_to_run", ...)
setattr(llm_config, "pad_to_left", False)
setattr(llm_config, "modified_sliding_window", ...)
```
These are just attributes bolted onto a plain config object at runtime — nothing declares them as
real fields, so nothing would survive a class swap.

Our `QcGemma4TextConfig` (`reauthoring.py`) declares the same six names in a `QC_BACKING_ATTRS`
dict, and gives each one a real `@property`/`@setter` pair with **lazy defaulting**:
```python
@property
def mask_neg(self) -> int:
    if not hasattr(self, "_mask_neg"):
        self._mask_neg = -100
    return self._mask_neg
```
"Lazy" here specifically means: the default is only assigned the *first time the attribute is
read*, not in `__init__`. That's the detail that makes this survive `model.config.__class__ =
QcGemma4TextConfig` (a swap that never calls this class's `__init__`) — the first time
`model.config.mask_neg` is accessed after the swap, it just works, defaulting correctly if nobody
set it explicitly.

`sliding_window_pattern` additionally shows the "derive from something else if unset" pattern,
matching NanoV4's `_get_sliding_window_pattern(llm_config)` helper:
```python
if getattr(self, "_sliding_window_pattern", None) is None:
    self._sliding_window_pattern = self.layer_types.index("full_attention") + 1
```

### 5.2 `QcGemma4TextAttention` — the actual adapted math

This is the direct counterpart of NanoV4's adapted `Gemma4TextAttention` in `qadaptation.py`. Three
behaviors, each gated by a config flag read at call time (not baked in at construction):

1. **Rope, with the CRAFT bypass.** Vanilla HF's `apply_rotary_pos_emb(x, cos, sin,
   rope_operator=self.q_rope_operator)` takes an optional `rope_operator` — a CRAFT
   quantization-tracking module. We call the exact same function but **omit** that argument,
   which makes it fall through to HF's own plain-tensor branch (`(x * cos) + (rotate_half(x) *
   sin)`) — no separate rope class needed, just one omitted keyword.
   *(Why no rotary-embedding class swap at all, unlike some other models: Gemma4's `(cos, sin)` is
   computed once per layer-type at the model level and threaded down to every attention call —
   attention itself never calls `self.rotary_emb`, so there's nothing to intercept there.)*

2. **Transposed-key cache + scatter update.** Before caching:
   ```python
   if self.config.transposed_key_cache:
       key_states = key_states.transpose(2, 3)
   ```
   then, before calling `past_key_values.update(...)`, we build the `cache_kwargs` dict via
   `QcAttentionMixin._update_cache_kwargs_with_qc_config` — this is what lets the *globally
   monkey-patched* `DynamicLayer.update` (installed by `KVCacheMapping`, see 5.3 below) know
   whether to concatenate (prefill) or scatter-write at `cache_position` (decode step).
   *(NanoV4 builds an equivalent `cache_kwargs` dict by hand for its own custom cache function —
   same idea, different function on the receiving end.)*

3. **Masked softmax via `mask_neg`** (in `qc_gemma4_eager_attention_forward`, a plain function, not
   a CRAFT module):
   ```python
   if module.config.enable_masked_softmax:
       attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
       attn_weights = torch.where(attention_mask == 0, attn_weights, attn_weights_min + module.config.mask_neg)
   ```
   Nearly a verbatim port of NanoV4's own min-trick masking.

### 5.3 `QcGemma4TextModel` — only one change: layer truncation

```python
def forward(self, *args, **kwargs):
    num_layers_to_run = getattr(self.config, "num_layers_to_run", None) or self.config.num_hidden_layers
    original_layers = self.layers
    if num_layers_to_run != self.config.num_hidden_layers:
        self.layers = original_layers[:num_layers_to_run]
    try:
        return super().forward(*args, **kwargs)
    finally:
        self.layers = original_layers
```
Everything else — building the causal mask, computing rope once per layer-type, PLE (per-layer
embeddings), threading `shared_kv_states` between layers — is inherited unchanged from vanilla
`Gemma4TextModel.forward()`. NanoV4 also strips out internal mask creation because its own runtime
pre-builds masks for its custom flat-tensor cache; we kept HF's own mask-building utilities because
we're using HF's own `DynamicCache`, where they're already correct.

**No `QcGemma4ForCausalLM` and no rotary-embedding class exist**, and that's deliberate, not
missing work:
- `Gemma4ForCausalLM.forward` just calls `self.model(...)` and passes everything through — once
  `self.model`'s class is swapped, it dispatches correctly automatically.
- `Gemma4TextDecoderLayer.forward` does `hidden_states, _ = self.self_attn(...)` — a plain 2-tuple
  unpack. Our attention returns a 2-tuple too (`attn_output, attn_weights`) — NanoV4's decoder
  layer returns a 4-tuple to also feed drafter/MTP heads, which are out of scope here (text-only,
  no speculative decoding heads).

### 5.4 `mappings.py` — wiring it in

```python
MAPPINGS = {
    Gemma4TextConfig: QcGemma4TextConfig,
    Gemma4TextAttention: QcGemma4TextAttention,
    Gemma4TextModel: QcGemma4TextModel,
}
for _from_cls, _to_cls in MAPPINGS.items():
    TransformersModuleMapping.register(_from_cls, _to_cls)
```
`INIT_MAPPINGS`/`REVERT_INIT_MAPPINGS` are empty dicts for us — neither Qc class adds an attribute
the vanilla `__init__` didn't already create, so a plain class swap is sufficient (contrast: Qwen3
needs an `init_fn` to add its R3-Hadamard rotation buffers post-swap, because those genuinely don't
exist on a vanilla `Qwen3Attention` instance).

---

## 6. The end-to-end flow, in order, with exactly where things take effect

```python
# main.py

# 1) Build a 100% vanilla HF model and load weights onto it.
model = Gemma4ForCausalLM(text_config)
model.load_state_dict(remapped_state_dict, strict=False)
#    ^ nothing Qc-related has happened yet — this is a plain transformers model.

# 2) Trigger registration (pure bookkeeping — no model is touched).
import models.gemma4_text.mappings
#    ^ this import's top-level code calls TransformersModuleMapping.register(...) three times.
#      Nothing about `model` changes as a result of this line.

# 3) Build a detached Qc config object.
qc_config = QcAutoConfig.from_config(text_config, mask_neg=-100, transposed_key_cache=True, ...)
#    ^ looks up QcGemma4TextConfig via the registry from step 2 (keyed by text_config.model_type
#      == "gemma4_text"), builds an instance from text_config's data, applies your kwargs.
#      Still just a free-standing object — `model.config` is untouched.

# 4) THE ADAPTATION HAPPENS HERE.
model = QcAutoModelForCausalLM._reauthor(model, qc_config=qc_config)
#    ^ (a) model.config.__class__ = QcGemma4TextConfig, in place
#      (b) walks model.modules(): model.model.__class__ = QcGemma4TextModel
#                                  every layer's .self_attn.__class__ = QcGemma4TextAttention
#          (weights on these modules are untouched — only the class pointer changes)
#      (c) globally monkey-patches DynamicLayer.update/get_seq_length (KVCacheMapping)

# 5) Patch the one gap _reauthor's config-copy step leaves (our 6 extended attrs).
for key in ("sliding_window_pattern", "mask_neg", ...):
    setattr(model.config, key, getattr(qc_config, key))

# 6) Only NOW, on a forward call, does any adapted behavior actually execute:
outputs = model(input_ids=..., past_key_values=DynamicCache(), use_cache=True)
#    ^ QcGemma4TextModel.forward reads self.config.num_layers_to_run
#      QcGemma4TextAttention.forward reads self.config.transposed_key_cache, builds cache_kwargs,
#          calls past_key_values.update(...) -> dispatches into the step-4(c) patched function
#      qc_gemma4_eager_attention_forward reads self.config.enable_masked_softmax / mask_neg
```

**One-sentence version to say out loud in a demo:** *"We build the model and load its weights
completely normally, register our Gemma4-specific replacement classes once at import time, build a
small config object carrying the runtime knobs, then call qairt's `_reauthor` — which is the one
moment the live model's classes actually get swapped and the KV-cache function gets patched;
everything after that is just the adapted `forward()` methods reading `self.config` at call time,
same as NanoV4's adapted classes did, just wired in after construction instead of before."*

---

## 7. Known issue: Linear→Conv adaptation doesn't reach Gemma4 (documented, not yet fixed)

`_reauthor` is not the whole story. qairt has a *second*, separate adaptation step —
`Adapter.apply_adaptations(model, backend=BackendType.HTP, model_type="LLM")` — that's meant to run
**after** `_reauthor`, normally triggered automatically by the full pipeline's `ModelLoadingStage`
(gated by `config.apply_default_adaptations`, default `True`). Since our `main.py` calls `_reauthor`
directly and skips the pipeline/stage machinery entirely, we call `Adapter.apply_adaptations(...)`
ourselves to get the same default behavior. For Llama/Qwen3/Phi3, qairt's default adaptation list
for `(HTP, "LLM")` has exactly one entry: **`replace_linears_with_convs`** — replace every
`nn.Linear` with a `Conv2d`-based drop-in (a 1x1 convolution with the same weight/bias), because
the HTP backend runs convolutions more efficiently than generic linear layers.

**This is where Gemma4 runs into trouble that Llama/Qwen3/Phi3 never hit.**

### The three compounding gaps

**Gap 1 — qairt's matcher silently skips Gemma4's linear layers, no error at all.**
qairt's `replace_linears_with_convs` (`common/adaptations/linears_to_conv.py:99-102`) decides what
to convert with:
```python
if isinstance(module, (torch.nn.Linear, Conv1D)):
```
Gemma4's checkpoint doesn't use plain `nn.Linear` for `q_proj`/`k_proj`/`v_proj`/`o_proj`/
`gate_proj`/`up_proj`/`down_proj` — it uses `Gemma4QuantizableLinear`
(`transformers/models/gemma4/quantization_gemma4.py:96`), which subclasses **`nn.Module`
directly**, not `nn.Linear`. So the `isinstance` check is `False` for every one of these modules,
and the whole function silently walks past the entire model doing nothing — no exception, no
warning, no log line. We only noticed by explicitly dumping `model.named_modules()` afterward and
seeing `q_proj` was still `Gemma4QuantizableLinear`, not `Conv2d`.

**Gap 2 — even if matched, qairt's `ConvInplaceLinear` constructor has no branch for it.**
Say we widen the `isinstance` check to include `Gemma4QuantizableLinear`. The next step,
`ConvInplaceLinear.__init__` (`linears_to_conv.py:39-45`), does:
```python
if isinstance(mod, torch.nn.Linear):
    weight, bias = mod.weight, mod.bias
elif isinstance(mod, Conv1D):
    weight, bias = mod.weight.T, mod.bias
else:
    raise TypeError(f"ConvInplaceLinear expects a Linear or Conv1D module, got {type(mod).__name__}")
```
`Gemma4QuantizableLinear` still matches neither branch — it would just hit the `TypeError` instead
of being silently skipped. Widening the outer check without also widening this constructor just
trades a silent no-op for a hard crash.

**Gap 3 — `Gemma4QuantizableLinear`'s weight isn't a plain float tensor for this checkpoint.**
This is the one that makes the fix non-trivial rather than a one-line `isinstance` tweak.
`Gemma4QuantizableLinear` (`quantization_gemma4.py:112-202`) has three internal modes depending on
`config.use_quantized_model`/`use_clipped_linears`:
- **quantized mode**: `self.weight` is an **int8/int16** `nn.Parameter`, plus a separate
  `self.weight_scale` (`nn.Parameter`, shape `(out_features, 1)`) that dequantizes it, plus
  `input_scale`/`input_bits`/`output_scale`/`output_bits` buffers for activation fake-quant clipping.
- **clipped mode**: plain float weight, but `input_min`/`input_max`/`output_min`/`output_max`
  clamp bounds on activations.
- **plain mode**: just a normal float weight/bias, nothing extra.

We confirmed directly against the checkpoint (`model.safetensors`) that the language-model
projections **are** in quantized mode — the safetensors file has a `weight_scale` key alongside
`weight` for every `q_proj`/`k_proj`/etc. So this isn't a hypothetical edge case; it's the actual
storage format of the model we're adapting. qairt's `ConvInplaceLinear` has no concept of any of
this — it assumes `mod.weight` is already the real, usable float weight, and passes
`dtype=mod.weight.dtype` straight into `nn.Conv2d(...)`, which doesn't correctly support an int8
weight as a trainable `Parameter`.

### Why Llama/Qwen3/Phi3 never hit this

qairt's own model packages (`llm/models/llama/reauthoring.py`, `.../qwen3/`, `.../phi3/`) each
define their reauthored model using **plain `nn.Linear`** for every projection, and load checkpoint
weights into those plain layers as part of reauthoring itself — by the time
`Adapter.apply_adaptations` runs, there are no custom Linear subclasses left to miss. Gemma4 has no
such built-in qairt reauthoring package at all (confirmed: no `gemma` anywhere in the installed
`qairt` package), so `Gemma4QuantizableLinear` reaches the adaptation stage completely untouched —
this is a coverage gap specific to Gemma4 not (yet) being a qairt-native model, not a bug in
`linears_to_conv.py`'s logic given its assumptions.

### NanoV4 already solved this — for Gemma4 specifically

NanoV4's own `qlib/qlinear_to_conv.py` is the Gemma4-aware version of the exact same idea:
```python
if isinstance(module, (torch.nn.Linear, Gemma4QuantizableLinear)):
```
and its `ConvInplaceLinear.__init__` uses a duck-typed 3-way fallback instead of a closed
`isinstance` chain:
```python
if (hasattr(mod, 'get_weight') and callable(mod.get_weight)) and (hasattr(mod, 'get_bias') and callable(mod.get_bias)):
    weight, bias = mod.get_weight(), mod.get_bias()
elif isinstance(mod, torch.nn.Linear):
    weight, bias = mod.weight, mod.bias
elif isinstance(mod, Conv1D):
    weight, bias = mod.weight.T, mod.bias
```
It also: (a) handles non-floating-point weight storage by registering it as a **buffer** instead of
a `Parameter` when `weight_dtype.is_floating_point` is `False` (the exact case Gap 3 describes), and
(b) carries `input_bits`/`output_bits`/`input_scale`/`output_scale` through onto the new `Conv2d`
module and re-applies `fake_quant` to its input/output in `forward()`, so activation clipping isn't
silently dropped by the conversion.

**One important detail this hinges on:** `Gemma4QuantizableLinear.get_weight()`/`get_bias()` are
**not** methods on the vanilla `transformers` class NanoV4 (and we) import — NanoV4 only has them
because `qlib/qadaptation.py` defines its own monkey-patched subclass override:
```python
class Gemma4QuantizableLinear(Gemma4QuantizableLinear_original):
    def get_weight(self):
        return self.weight * self.weight_scale if self.weight_scale is not None else self.weight
    def get_bias(self):
        return None
```
Since we don't (yet) register an equivalent override, a direct copy-paste of NanoV4's
`ConvInplaceLinear` would still fail on Gemma4 in our repo — it would fall through to the
`isinstance(mod, torch.nn.Linear)` branch, which is also `False`, and hit the same wall. The actual
port needs either (a) a small `QcGemma4QuantizableLinear`-style override adding `get_weight`/
`get_bias`, registered the same way our other classes are, or (b) inlining the equivalent
`weight * weight_scale if weight_scale is not None else weight` duck-typing logic directly into a
Gemma4-specific `replace_linears_with_convs`.

**Resolved:** checked directly against `model.safetensors` — none of
`model.language_model.layers.0.self_attn.{q,k,v,o}_proj.*` has a `.bias` key (only `.weight`,
`.weight_scale`, `.input_bits`, `.input_scale`, `.output_bits`, `.output_scale`). So for *this*
checkpoint, `get_bias()` unconditionally returning `None` is correct, not a simplification that
happens to work — there is no bias tensor to drop. Still worth keeping the check in the ported
version rather than hardcoding `None`, in case a future checkpoint variant does set `bias=True`.

### Net takeaway

This isn't one missing `isinstance` check — it's three compounding gaps (silent skip → hard crash →
wrong dtype handling) that only surface because this specific checkpoint's `Gemma4QuantizableLinear`
layers are genuinely storing quantized (int8/16 + scale) weights, which is exactly the harder case
qairt's generic `ConvInplaceLinear` was never written to handle. NanoV4 already has the correct,
Gemma4-aware fix for all three; the work still ahead of us is porting that logic into our own
`nano/models/gemma4_text/` package (following the same `reauthoring.py`/`mappings.py` structure as
the rest of this repo) rather than trying to bend qairt's generic version to fit.

**Update — implemented.** `nano/models/gemma4_text/linear_to_conv.py` now provides
`QcGemma4ConvInplaceLinear`/`replace_linears_with_convs`, a Gemma4-aware port of the three fixes
above: matches `Gemma4QuantizableLinear` directly (inlining the `weight * weight_scale if
weight_scale is not None else weight` / `bias=None` dequantization logic — vanilla
`transformers.Gemma4QuantizableLinear` has no `get_weight`/`get_bias`, so no override class was
added), stores non-floating weight as a registered buffer instead of a `Conv2d` `Parameter`, and
carries `input_bits`/`output_bits`/`input_scale`/`output_scale` through to `forward()` via
`fake_quant_activation` (substituting for NanoV4's `air.nanov4.utils.fake_quant`, which isn't
importable in this environment). Wired into `main.py` via
`Adapter.apply_adaptations(..., apply_default_adaptations=False,
adaptations=[gemma4_replace_linears_with_convs])`, replacing (not supplementing) qairt's no-op
default. Verified via `named_modules()`: every `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/
`up_proj`/`down_proj`/`lm_head`/`per_layer_model_projection` now shows as
`QcGemma4ConvInplaceLinear`, and the existing reauthoring/inference/KV-cache-shape checks still pass.

## 8. `rms_norm` and `use_erf_gelu` — the other two NanoV4-only adaptations, now ported

Per the comparison in §6, qairt's entire default HTP+LLM adaptation list is
`replace_linears_with_convs` (§7) — everything else NanoV4's `Gemma4Context` does by default
(`qlib/qadaptation_flags.py`, all flags default `True`) is either N/A (vision/audio/LoRA/MTP, out
of this repo's text-only scope) or has no qairt-provided equivalent at all. Two flags were real,
unaddressed correctness gaps: `rms_norm` and `use_erf_gelu`. Both are now ported into
`nano/models/gemma4_text/reauthoring.py`.

**`rms_norm`** — NanoV4's override (`qadaptation.py:2029-2033`) replaces vanilla
`Gemma4RMSNorm._norm`'s `x * (mean(x^2) + eps)^-0.5` with `x / sqrt(mean(x^2) + eps)`.
Mathematically equivalent in exact arithmetic; the point is giving MPP/HTP a division instead of a
negative-power op, matching how MPP lowers RMSNorm. Ported as `QcGemma4RMSNorm(Gemma4RMSNorm)`,
overriding only `_norm`. No new attributes, so no `init_fn` needed — a plain `__class__` swap via
`TransformersModuleMapping.register(Gemma4RMSNorm, QcGemma4RMSNorm)` is sufficient, applied to
every `Gemma4RMSNorm` instance in the model (`input_layernorm`, `post_attention_layernorm`,
`pre_feedforward_layernorm`, `post_feedforward_layernorm`, `post_per_layer_input_norm`,
`per_layer_projection_norm`, ...).

**`use_erf_gelu`** — NanoV4's override (`qadaptation.py:1401-1409`) replaces
`Gemma4TextMLP.act_fn`/`Gemma4VisionMLP.act_fn` (built from `config.hidden_activation`, i.e.
`"gelu_pytorch_tanh"` for this checkpoint) with `Act2FN("gelu", ...)` — erf-based GeLU
(`nn.functional.gelu`) instead of the tanh approximation, again because MPP/HTP only supports the
erf form. Vision is out of scope (no `Gemma4VisionMLP` in a text-only model). Ported as
`QcGemma4TextMLP(Gemma4TextMLP)` — a body-less subclass, since the only change is which `act_fn`
instance gets built, and class-swap never re-runs `__init__`. The actual rebuild happens in an
`init_fn`, `_init_qc_gemma4_text_mlp_act_fn(module)`, registered via
`TransformersModuleMapping.register(Gemma4TextMLP, QcGemma4TextMLP,
init_fn=_init_qc_gemma4_text_mlp_act_fn)` — same pattern qairt's own `qwen3/mappings.py` uses for
`QcQwen3Attention`'s R3-Hadamard init and `QcQwen3ForCausalLM`'s cache-tensor buffer, both of which
also need to create/rebuild state after a class swap that skips `__init__`.

One scope note carried over directly from NanoV4: `Gemma4TextDecoderLayer` has its *own*, separate
`act_fn` (also built from `config.hidden_activation`) for the per-layer-input path
(`hidden_size_per_layer_input`, active for this checkpoint since it's 256, not 0) — NanoV4 does
not adapt this one (only `Gemma4TextMLP`/`Gemma4VisionMLP` are in `qmodel.py`'s `patches` dict for
`use_erf_gelu`), so this port doesn't either. Verified via `named_modules()`: `layers.0.mlp.act_fn`
shows `GELUActivation` (erf) while `layers.0.act_fn` (decoder-layer-level, per-layer-input) still
shows `GELUTanh` — matching NanoV4's scope exactly, not a broader "replace every GeLU" pass.

Both registrations live in `nano/models/gemma4_text/mappings.py` alongside the existing
config/attention/model entries, applied via the same `Adapter.apply_adaptations`-adjacent
`TransformersModuleMapping.register()` loop `main.py` already triggers by importing
`models.gemma4_text.mappings`. No `main.py` change was needed for these two (unlike linear→conv,
which required overriding `apply_default_adaptations`) — they ride the same module-class-swap
mechanism as the attention/model classes already handle.

## 9. Host-side RoPE/position-embedding precomputation — moving cos/sin out of the traced graph

**The gap.** Every adaptation up through §8 still left `Gemma4TextModel.forward`'s rope
computation untouched: `position_embeddings[layer_type] = self.rotary_emb(hidden_states,
position_ids, layer_type)` (`modeling_gemma4.py:2762`) runs *inside* the traced/exported graph on
every single call, recomputing the inv_freq outer product and `cos`/`sin` from scratch each time.
NanoV4 never does this — `qlib/qgenerator.py` calls `air/nanov4/utils.py`'s
`llm_create_position_embeddings(config, position_ids, layer_type)` on the **host**, once per
generation step, before the model is invoked, and passes the resulting `(cos, sin)` straight
through as the model's `position_ids` argument (`qlib/qmodel_classes.py`'s `TextModelQc.forward`
just forwards it, no rope math). This is a real, separate optimization from everything in §1-§8:
none of the earlier adaptations touch *when* or *where* cos/sin get computed, only how attention
consumes them once computed.

**qairt already has the identical pattern for Qwen3** — confirmed by directly reading
`llm/models/qwen3/reauthoring.py`. `QcQwen3RotaryEmbedding.forward` short-circuits: `if
isinstance(position_ids, tuple) and len(position_ids) == 2: return position_ids` — i.e. if
`position_ids` is already a precomputed `(cos, sin)` tuple, pass it straight through instead of
computing rope. `QcQwen3ForCausalLM.rotary_emb = QcQwen3RotaryEmbedding` (a **class** attribute)
is what lets qairt's generic host-side builder — `LLMGenerationMixin.create_position_embeddings`
in `llm/generation/generator.py:217-254` — find and instantiate it (`model.rotary_emb(model.config)`)
and call it once from the host (`rotary_emb(x, position_ids=position_ids)`) inside
`prepare_inputs` (`generator.py:695-742`), injecting the result into `prepared["position_ids"]`
before `LLMGenerator.forward` ever calls `self.model(**model_inputs)`.

**Why Gemma4 can't reuse that generic builder as-is, confirmed by direct file read.**
`create_position_embeddings` (both the base `LLMGenerationMixin` version and
`HybridLLMGenerator`'s override at `generator.py:1368-1377`) builds and returns exactly **one**
`(cos, sin)` pair — correct for Qwen3, which has one global rope table
(`Qwen3RotaryEmbedding.forward(self, x, position_ids)`, no `layer_type`). Gemma4 does not: its
`Gemma4TextRotaryEmbedding.forward(self, x, position_ids, layer_type=None)` reads
`getattr(self, f"{layer_type}_inv_freq")` and this checkpoint's `config.rope_parameters` has
genuinely different parameters per layer type (`full_attention`: `rope_theta=1e6`,
`partial_rotary_factor=0.25`, `rope_type="proportional"`; `sliding_attention`: `rope_theta=1e4`,
`rope_type="default"`) — confirmed via `Gemma4Config.from_pretrained(...).text_config`. So a
single `(cos, sin)` tuple can't represent what Gemma4 needs; it needs one pair *per layer_type*.

**The port**, in `nano/models/gemma4_text/reauthoring.py`:

- **`QcGemma4TextRotaryEmbedding(Gemma4TextRotaryEmbedding)`** — the Gemma4 analogue of
  `QcQwen3RotaryEmbedding`, but keyed by a **dict** instead of a bare tuple: `forward(self, x,
  position_ids, layer_type=None)` checks `isinstance(position_ids, dict)` and, if so, returns
  `position_ids[layer_type]` — short-circuiting per layer_type. This works with zero changes to
  `Gemma4TextModel.forward`'s existing per-layer-type loop (inherited unchanged into
  `QcGemma4TextModel`, §5): that loop already calls `self.rotary_emb(hidden_states, position_ids,
  layer_type)` once per `layer_type` in `unique_layer_types`, reusing the *same* `position_ids`
  argument for both calls — passing a `{layer_type: (cos, sin)}` dict as that argument means each
  of the two inherited calls resolves to the right precomputed tensor via the `layer_type` key,
  with no other line of `Gemma4TextModel.forward` needing to know precomputation happened at all.
- **`create_position_embeddings(rotary_emb, position_ids, dtype=torch.float32)`** — a
  module-level helper, not a class method, that builds the `{layer_type: (cos, sin)}` dict by
  calling the model's own (already class-swapped) `rotary_emb` submodule instance directly, once
  per `rotary_emb.layer_types`: `{lt: rotary_emb(x, position_ids, lt) for lt in
  rotary_emb.layer_types}`. This mirrors `LLMGenerationMixin.create_position_embeddings`'s "call
  the model's rotary_emb, capture the result before forward" mechanism — reusing qairt's existing
  approach rather than reimplementing rope math from `air/nanov4/utils.py`'s
  `llm_create_position_embeddings`/`_get_rotary_embedding` — just returning a dict instead of a
  single tuple, and taking the already-constructed submodule instance directly (`model.rotary_emb`,
  the instance) rather than a config class attribute (`model.rotary_emb(config)`, a fresh
  reconstruction) since we already have the live, class-swapped instance sitting on the model.

- **Registration**: `Gemma4TextRotaryEmbedding: QcGemma4TextRotaryEmbedding` added to `MAPPINGS`
  in `nano/models/gemma4_text/mappings.py`. No `init_fn` needed — no new attributes, plain
  `__class__` swap suffices (same reasoning as `QcGemma4RMSNorm`, §8).

- **`main.py`** now calls `create_position_embeddings(model.model.rotary_emb, position_ids)`
  explicitly on the host, once, before the single smoke-test forward call, and passes the
  resulting dict as `position_ids=` — the direct analogue of NanoV4's `qgenerator.py` call site.
  A real generation loop (not yet built here — `main.py` still only runs one forward pass, no
  `generate()`) would call this once per step, exactly as NanoV4's `qgenerator.py` does.

**Verified**: `python main.py` runs end-to-end; `model.model.rotary_emb` is
`QcGemma4TextRotaryEmbedding`; the forward pass succeeds and produces a sane next-token decode;
the existing transposed-key-cache check (§5/§6) still passes, confirming the reauthored attention
path — now fed precomputed rope tensors instead of computing its own — still runs correctly end to
end.

**Caveat, not yet addressed**: this is a single-call smoke test, not a generation loop.
`create_position_embeddings` would need to be called once per decode step (matching NanoV4's
`qgenerator.py`), and `position_ids` passed to each subsequent step would need to advance past
already-cached positions — neither is wired up yet since `main.py` doesn't have a `generate()`
loop at all (flagged already in §7/§8 as an explicit non-goal of this pass).

---

## 10. Auditing every NanoV4 adaptation flag against this port — `past_key_values` and beyond

Everything through §9 was built incrementally, one gap at a time, as each was noticed. This
section is the other direction: start from NanoV4's own **complete list** of adaptations
(`qlib/qadaptation_flags.py`'s `AdaptationFlags` dataclass — the single source of truth for "what
does NanoV4 change vs. vanilla Gemma4, and is it currently on") and go through every entry,
classifying it, instead of waiting to trip over the next gap by accident. Two real, previously
unaddressed gaps came out of this sweep — `enable_masked_softmax` and `kv_clip_only` — both fixed
below. Everything else in the list was confirmed either not applicable to this checkpoint, or
already correctly handled by qairt's generic machinery.

### 10.1 The full flag sweep

NanoV4's `AdaptationFlags` (`qlib/qadaptation_flags.py`, 8 fields, all default `True`):

| Flag | What it changes in NanoV4 | Status in this port |
|---|---|---|
| `vision_attention_forward` | Vision-tower attention (multi-head→single-head folding for `Gemma4VisionMLP`) | **N/A** — text-only scope, no vision tower is built at all (`main.py` only constructs `Gemma4ForCausalLM(text_config)`, never touches `Gemma4Model`'s vision/audio submodels) |
| `rms_norm` | `Gemma4RMSNorm._norm`: `x*(mean+eps)^-0.5` → `x/sqrt(mean+eps)` | **Ported**, §8 (`QcGemma4RMSNorm`) |
| `linear_to_conv` | `nn.Linear`/`Gemma4QuantizableLinear` → HTP-friendly `Conv2d` | **Ported**, §7 (`QcGemma4ConvInplaceLinear`) |
| `use_erf_gelu` | MLP `act_fn`: tanh-approx GeLU → erf GeLU | **Ported**, §8 (`QcGemma4TextMLP`) |
| `lora_clip_adaptation` | LoRA adapter weight clipping | **N/A for this checkpoint** — confirmed via direct `safetensors` key inspection of `model.safetensors`: zero keys containing `"lora"` (case-insensitive). If a future checkpoint ships LoRA weights, this would need porting; today there is nothing to adapt |
| `lora_matmul_to_conv` | LoRA matmuls → conv | **N/A**, same reason as above |
| `kv_clip_only` | K/V-cache "fake quant" dispatch: clamp-only vs. quantize-dequantize | **Real gap — fixed this section** (§10.3) |
| `enable_masked_softmax` | Additive-mask softmax: `+mask` vs. min-trick relative to `mask_neg` | **Real gap — fixed this section** (§10.2) |

Two more items came up from the user's explicit "`past_key_values` and all" framing, investigated
separately from the flag list above since they concern the cache *mechanism* itself, not an
`AdaptationFlags` entry:

| Item | Question | Answer |
|---|---|---|
| Is `KVCacheMapping.apply(model)` actually being called? | `main.py` never calls it explicitly — is that a missing step? | **No gap.** Read `QcAutoModelForCausalLM._reauthor()` directly (`auto_classes.py:290-322`): it calls `cls._transformers_module_mapping.apply(model, qc_config=qc_config)` **then** `cls._kv_cache_module_mapping.apply(model)` internally. `main.py`'s existing single `QcAutoModelForCausalLM._reauthor(model, qc_config=qc_config)` call already triggers both — nothing missing. |
| Does qairt's patched `DynamicLayer.update`/`get_seq_length` actually replicate NanoV4's own cache update math? | NanoV4 has its own `DynamicCache_adapted`/`DynamicLayer_adapted` (`qadaptation.py:2048-2117`) — is qairt's generic version equivalent, or just superficially similar? | **Confirmed equivalent**, field-by-field, against qairt's `_qc_dynamic_layer_update`/`_qc_dynamic_layer_get_seq_length` (`llm/models/utils.py:232-330`): same `key_cat_dim` handling, same scatter-index construction from `cache_position`, same `return_new_key_value_only` truncation logic. No port needed here — this is the one part of the whole KV-cache story where qairt's *generic*, model-family-agnostic code already does exactly what NanoV4's Gemma4-specific code does. |

### 10.2 Gap 1 — `enable_masked_softmax`: the masked-softmax branch was dead code

**HF baseline.** Vanilla `eager_attention_forward` (`modeling_gemma4.py`) does a plain additive
mask: `attn_weights = attn_weights + causal_mask`, where `causal_mask` is a float tensor with
`0` at valid positions and `torch.finfo(dtype).min` (i.e. `-inf`-ish) at masked ones, built by
`create_causal_mask`/`create_sliding_window_causal_mask` (`transformers.masking_utils`).

**Why NanoV4 changes it.** Adding a near-`-inf` float constant is fine in float32 on a GPU. It is
not fine once the graph is meant to run in fixed-point/quantized arithmetic on HTP — a huge
negative constant blows up the dynamic range the quantizer has to represent, wasting precision on
a value whose only job is "make softmax output ~0 here." NanoV4's `eager_attention_forward`
(`qadaptation.py:201-252`) replaces the additive mask with a **relative** one: instead of a fixed
huge negative number, subtract a modest constant (`mask_neg`, e.g. `-100`) from that *row's own
minimum* attention score:
```python
attn_weights_min, _ = attn_weights.min(dim=-1, keepdim=True)
attn_weights = torch.where(causal_mask == 0, attn_weights, attn_weights_min + minus_value)
```
This keeps every value in the tensor within a bounded range relative to the row's own data,
instead of introducing a constant many orders of magnitude larger than any real logit — much
friendlier to a quantizer. `Gemma4TextAttention.__init__` (`qadaptation.py:509-519`) wires this on
by reading `adaptations.enable_masked_softmax` (default `True` in `AdaptationFlags`) into
`self.enable_masked_softmax`.

**What we'd already built, and why it was inert.** `qc_gemma4_eager_attention_forward`
(`reauthoring.py`) already had this exact branch written, gated on
`module.config.enable_masked_softmax`:
```python
if module.config.enable_masked_softmax:
    attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
    attn_weights = torch.where(attention_mask == 0, attn_weights, attn_weights_min + module.config.mask_neg)
else:
    attn_weights = attn_weights + attention_mask
```
The bug wasn't in this function — it was one layer up. `enable_masked_softmax` is a base
`QcConfigMixin.QC_BACKING_ATTRS` field (`llm/models/utils.py:62`) that **defaults to `False`**,
and `main.py`'s `qc_config = QcAutoConfig.from_config(text_config, ...)` call never passed
`enable_masked_softmax=True`. So every forward call was silently taking the `else` branch — the
correct min-trick code existed but was unreachable at runtime. This is the qairt-world version of
a very ordinary bug: correct logic behind a flag nobody flipped.

**The fix** — one line added to `main.py`'s `QcAutoConfig.from_config(...)` call:
```python
enable_masked_softmax=True,
```
No new config property was needed — `enable_masked_softmax` already exists on the base
`QcConfigMixin` every `QcConfigMixin` subclass (including `QcGemma4TextConfig`) inherits; the gap
was purely "never set", not "doesn't exist".

### 10.3 Gap 2 — `kv_clip_only`: two different operations were being conflated

**HF baseline.** Vanilla `Gemma4TextAttention.forward` calls
`fake_quant_activation(key_states, self.k_cache_scale, self.k_cache_num_bits)` (and the value-cache
equivalent) whenever the checkpoint has per-layer `k_cache_scale`/`k_cache_num_bits` tensors —
confirmed present for all 35 layers of this checkpoint via direct `safetensors` key inspection
(140 matching `k_cache_scale`/`v_cache_scale`/`k_cache_num_bits`/`v_cache_num_bits` keys). This
function (`transformers/models/gemma4/quantization_gemma4.py:18-30`, delegating to
`static_fake_quant.fake_quant`, `static_fake_quant.py:151-178`) does a **full quantize-dequantize
round-trip**: divide by scale, round, clamp to the integer range, then multiply back by scale —
i.e. it actually simulates the rounding error a real int8/int16 cache would introduce.

**What NanoV4 does instead by default.** `Gemma4TextAttention.__init__` (`qadaptation.py:509-519`)
picks between two different KV-cache "fake quant" functions based on `adaptations.kv_clip_only`:
```python
if adaptations.kv_clip_only:
    self._kv_fake_quant_fn = lambda x, bits, scale: fake_quant(x, bits, scale)   # NanoV4's own fake_quant
else:
    self._kv_fake_quant_fn = lambda x, bits, scale: fake_quant_activation(x, scale, bits)  # HF's
```
`kv_clip_only` defaults `True`. NanoV4's own `fake_quant` (`air/nanov4/utils.py:82-90`) is **not**
a quantize-dequantize round-trip at all — it's a pure clamp, no rounding:
```python
def fake_quant(x, bw, scale):
    min_val = -(2 ** (bw - 1)) * scale
    max_val = ((2 ** (bw - 1)) - 1) * scale
    return torch.clamp(x, min_val, max_val)
```
So the *default* NanoV4 behavior for this checkpoint's KV cache is: clip values into the
representable int range, but don't actually simulate the rounding error of storing them at that
precision. Confirmed this is a deliberate default, not an oversight, by reading the dataclass
default (`kv_clip_only: bool = True`) directly.

**Why this is a real, live divergence, not a rounding-error footnote.** Our
`QcGemma4TextAttention.forward` (`reauthoring.py`) was unconditionally calling
`fake_quant_activation` — the full round-trip — on every key/value write, for every one of the 35
layers, on every forward call. That's the *opposite* of this checkpoint's actual default behavior
in NanoV4: we were introducing quantization rounding noise into the KV cache on every single
token, where NanoV4 (by default) does not.

**The port**, in `reauthoring.py`:
1. A new clamp-only function, deliberately kept a drop-in for `fake_quant_activation`'s call
   signature so the dispatch is a one-line ternary, not a rewritten call site:
   ```python
   def _kv_clip_only_fake_quant(x, scale, bw):
       scale = scale.squeeze()
       min_val = (-(2 ** (bw - 1))) * scale
       max_val = ((2 ** (bw - 1)) - 1) * scale
       return torch.clamp(x, min_val, max_val)
   ```
2. A new `kv_clip_only` property on `QcGemma4TextConfig`, following the exact same lazy-property
   pattern as `mask_neg`/`context_length`/etc. (§5.1) — added to `QC_BACKING_ATTRS` with default
   `True`, matching NanoV4's dataclass default:
   ```python
   QC_BACKING_ATTRS = {
       **QcConfigMixin.QC_BACKING_ATTRS,
       ...,
       "kv_clip_only": True,
   }
   ```
   This is a **Gemma4-specific** extension to `QC_BACKING_ATTRS`, same category as `mask_neg` —
   qairt's base `QcConfigMixin` has no concept of "which KV-cache quant function to use" because
   none of its shipped model families (Llama/Qwen3/Phi3) have this checkpoint-embedded
   `k_cache_scale`/`num_bits` quantization scheme at all; it's Gemma4-specific data, so the flag
   controlling how to consume it had to be added on our Gemma4 config subclass, not the shared
   mixin.
3. `QcGemma4TextAttention.forward`'s two `fake_quant_activation(...)` call sites became:
   ```python
   kv_fake_quant_fn = _kv_clip_only_fake_quant if self.config.kv_clip_only else fake_quant_activation
   key_states = kv_fake_quant_fn(key_states, self.k_cache_scale, int(self.k_cache_num_bits.item()))
   ```
   (and the same for `value_states`/`v_cache_scale`/`v_cache_num_bits`). Reading
   `self.config.kv_clip_only` at call time — not baking the choice in at construction — matches
   how every other config-driven branch in this file works (§2's "no `__init__` override" rule):
   the attribute has to be re-readable after a class-swap-without-`__init__`, so it's read fresh
   on every `forward()` call rather than cached on the instance.

### 10.4 A third, smaller finding — the silent `DynamicCache(config=...)` trap

While re-reading NanoV4's `Gemma4TextModel.forward` (`qadaptation.py:684-870`) to check the cache
construction path, one more divergence from vanilla stood out, unrelated to the two `AdaptationFlags`
above: NanoV4's override doesn't do vanilla's auto-construction of a cache when the caller forgets
to pass one. Vanilla `Gemma4TextModel.forward` (`modeling_gemma4.py:2734-2735`):
```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)
```
This matters because `DynamicCache(config=...)` (unlike the no-arg `DynamicCache()`) inspects
`config.layer_types` and picks `DynamicSlidingWindowLayer` for any `sliding_attention` layer —
and `KVCacheMapping`'s patch (`htp_mappings.py`'s `_apply_cache_patches`) only monkey-patches
`update`/`get_seq_length` on the **base** `DynamicLayer` class. `DynamicSlidingWindowLayer`
defines its own `update`/`get_seq_length` that don't delegate to the base class — so this
auto-construction path would silently produce a cache that never goes through our adapted
scatter/transposed-key logic at all, for exactly the layer type (`sliding_attention`) most of this
model's layers use. NanoV4 avoids the whole question by raising instead of ever taking this path
(that construction line is commented out in `qadaptation.py`, replaced with a `ValueError` if
`use_cache` and no cache was supplied).

`main.py` was already disciplined about always passing an explicit `past_key_values=DynamicCache()`
(no `config=`) into every forward call — so this was never actually triggered in practice. But
nothing stopped a *future* caller (e.g. inside a `generate()` loop added later) from omitting it
and silently falling into the broken auto-construction path. Ported the guard directly into
`QcGemma4TextModel.forward` (`reauthoring.py`), matching NanoV4's fail-loud behavior instead of
vanilla's fail-silent one:
```python
if kwargs.get("use_cache", True) and kwargs.get("past_key_values") is None:
    raise ValueError(
        "QcGemma4TextModel requires an explicit past_key_values=DynamicCache() "
        "(no config=) when use_cache is True; the vanilla auto-construction path "
        "is not compatible with KVCacheMapping's patched cache layers."
    )
```

### 10.5 Verification

```python
qc_config = QcAutoConfig.from_config(
    text_config,
    ...,
    enable_masked_softmax=True,   # new
    kv_clip_only=True,            # new (also QcGemma4TextConfig's own default, set explicitly for clarity)
)
```
`python main.py` end-to-end, exit code 0:
```
Missing keys: []
Unexpected keys: []
reauthoring check passed: model.model -> QcGemma4TextModel
reauthoring check passed: self_attn -> QcGemma4TextAttention
logits shape: torch.Size([1, 13, 262144])
next token: **
reauthoring check passed: rotary_emb -> QcGemma4TextRotaryEmbedding consumed a precomputed {layer_type: (cos, sin)} dict
reauthoring check passed: cached key shape -> (1, 1, 256, 13) (head_dim, seq) order confirms transposed_key_cache path ran)
```
No missing/unexpected weight keys, all reauthoring assertions pass, and the forward pass now
actually exercises the min-trick masked-softmax branch and the clamp-only KV-cache path on every
layer — previously both were either dead code or running the wrong operation. As before, the
decoded token itself is a plausible-shape sanity check, not a numerical-parity claim against
vanilla — the CRAFT bypass (§3's glossary entry) remains a documented, deliberate deviation from
bit-exact vanilla output.

### 10.6 What's still open after this pass

Flagged during the sweep, deliberately not acted on yet (none of these are `AdaptationFlags`
entries — they're separate observations that came up while reading the surrounding code):

- **`ApplyRopeSingle` vs. `apply_rotary_pos_emb`.** NanoV4's rope application
  (`qadaptation.py:156-176`) splits real/imaginary halves and does four elementwise
  multiplies/adds; our port uses vanilla HF's rotate-half `apply_rotary_pos_emb`. Both are
  standard ways to apply the same rotation and are very likely numerically equivalent, but this
  has not been verified tensor-for-tensor.
- **`num_logits_to_keep`: NanoV4 uses `0`, this port's `main.py` uses `1`.** Not yet investigated
  whether this matters beyond a single-forward-pass smoke test (it changes how many trailing
  positions' logits are computed/returned, not any KV-cache or attention behavior).
- **`modified_sliding_window`.** NanoV4's `Gemma4Context.__init__` asserts
  `args.sliding_window_length >= llm_config.sliding_window + self.arn` before setting this field;
  our config currently leaves it at `None` (i.e., "use the checkpoint's own `sliding_window`"),
  which trivially satisfies a not-yet-replicated invariant rather than actively upholding it.
- **`llm_create_causal_mask`.** NanoV4 precomputes masks on the host
  (`air/nanov4/utils.py:13-56`) the same way it precomputes rope (§9); this port still relies on
  vanilla `Gemma4TextModel.forward`'s internal `create_causal_mask` call. Deliberately deferred
  (design decision #2 from the original planning pass) since HF's own mask utility is already
  correct for the `DynamicCache` this port uses — revisit only if/when a static-graph export step
  actually needs masks precomputed outside the traced graph.
- **Generation-loop KV-cache handling.** Everything verified so far is a single forward call.
  qairt's `LLMGenerator`/`LLMGenerationMixin` (`llm/generation/generator.py`) cache-preparation
  path for a multi-step `generate()` loop has not yet been read/compared against NanoV4's
  `qgenerator.py` equivalent.


