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
