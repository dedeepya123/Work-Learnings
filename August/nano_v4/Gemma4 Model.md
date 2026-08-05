# NanoV4 → QAIRT: Gemma4 text-decoder adaptation, end to end

What this document is: a from-scratch, top-to-bottom explanation of how NanoV4's Gemma4
text-decoder adaptations were re-implemented on top of qairt's own reauthoring API, for anyone
new to either codebase. It follows the actual pipeline in `main.py`, in order: load checkpoint →
build Qc config → reauthor the model → verify → run one forward pass. Generation (`.generate()`,
multi-token loops) is **out of scope** — this pipeline stops at "one correct adapted forward pass."

---

## 0. The two systems in one sentence each

- **NanoV4** (`Nano/NanoV4/qlib/qadaptation.py`): hand-written subclasses of every vanilla HF
  Gemma4 module it needs to change, activated by monkey-patching (temporarily replacing) the
  original classes inside a context manager (`_patched_gemma4_classes`) while the model is built.
- **qairt** (`qairt/experimental/pipeline/torch/llm/...`): a generic "reauthoring" framework used
  for every model it supports (Llama, Qwen3, Phi3, ...). You register `{OriginalClass:
  AdaptedClass}` pairs once; a driver function then walks an already-built model and swaps
  `module.__class__` in place for every module whose class has a registered replacement. No context
  manager, no monkey-patching — the swap is permanent and happens after the model already exists.

Both approaches produce the same kind of result — a model whose modules run adapted
`forward()` code instead of vanilla HF code. This document is about how NanoV4's specific
adaptations were re-expressed as qairt registrations, and about the corners of qairt's design
(config construction, cache handling, `.generate()`) that shape how this had to be wired up.

See `QC_AUTOCONFIG.md` for a deeper dive into `QcAutoConfig`/`QcConfigMixin` internals (the
`from_model_type` registry lookup, lazy backing-attribute properties) — this document only
summarizes what's needed to follow the pipeline; that one explains the config machinery itself.

---

## 1. Overall flow diagram

```
 ┌─────────────────────┐
 │ load_checkpoint()    │  models/gemma4_text/checkpoint.py
 │  - Gemma4Config       │  Build vanilla Gemma4ForCausalLM(text_config), load only the
 │    .from_pretrained() │  language_model.* weights out of the multimodal checkpoint's
 │  - Gemma4ForCausalLM  │  single model.safetensors, load tokenizer.
 │  - safetensors remap  │
 └─────────┬────────────┘
           │ GemmaCheckpoint(model, text_config, tokenizer, ...)
           ▼
 ┌─────────────────────┐
 │ build_qc_config()    │  models/gemma4_text/qc_config.py
 │  QcAutoConfig         │  Wrap text_config with QC/HTP runtime flags
 │   .from_config(...)   │  (transposed_key_cache, mask_neg, kv_clip_only, ...).
 └─────────┬────────────┘  Model NOT touched yet — config object only.
           │ qc_config (a QcGemma4TextConfig)
           ▼
 ┌─────────────────────┐
 │ reauthor_model()      │  models/gemma4_text/adapt.py
 │  1. _reauthor()        │  Class-swap attention/model/config + patch KV-cache methods.
 │  2. copy extended      │  Copy the 6 Gemma4-specific config attrs qairt's copy loop misses.
 │     config attrs       │
 │  3. linear->conv        │  Gemma4-aware Linear/QuantizableLinear -> Conv2d (HTP-friendly).
 │     adaptation          │
 └─────────┬────────────┘
           │ model (now running Qc*-adapted classes)
           ▼
 ┌─────────────────────┐
 │ assert_reauthored()   │  models/gemma4_text/verify.py
 │  isinstance() checks   │  Fail loudly if the swap silently didn't happen.
 └─────────┬────────────┘
           │
           ▼
 ┌─────────────────────┐
 │ run_smoke_test()      │  models/gemma4_text/inference.py
 │  - DynamicCache()      │  ONE forward pass through the adapted model:
 │    (no config=)        │  host-precomputed rope, explicit empty cache, decode
 │  - host-side rope       │  one next-token, verify transposed KV-cache shape.
 │  - model(...)           │
 └──────────────────────┘
```

---

## 2. Loading the checkpoint — why not `AutoModelForCausalLM.from_pretrained(path)`

The checkpoint at `model_path` is a **multimodal** Gemma4 checkpoint: `config.json` describes a
`Gemma4Config` with vision/audio/assistant towers plus a `text_config` sub-config, and
`model.safetensors` stores every tower's weights together, each key prefixed by its tower name
(`model.language_model.*`, `model.vision_tower.*`, etc.).

We only want the text decoder. `Gemma4ForCausalLM.from_pretrained(model_path)` doesn't exist as a
"just the text half" loader — the standard HF path would load everything. So `checkpoint.py`
does it by hand:

1. `Gemma4Config.from_pretrained(model_path)` then `.text_config` — get just the text sub-config
   from the multimodal top-level config.
2. `Gemma4ForCausalLM(text_config)` — construct a *fresh, randomly-initialized* text-only model
   from that sub-config (no weights yet).
3. Walk `model.safetensors` keys, keep only ones starting with `model.language_model.` (renamed
   to `model.*` to match `Gemma4ForCausalLM`'s own naming) plus a few top-level lm-head/softcap
   ops kept under different prefixes in the checkpoint. Everything else (vision/audio/assistant
   weights) is never even materialized into a tensor.
4. `model.load_state_dict(state_dict, strict=False)` — `strict=False` because we're
   intentionally handing it a partial dict (only the text-tower keys); the `missing`/`unexpected`
   return values are then printed for a sanity check for the tensors we expect.

This is a one-off, checkpoint-specific detail — nothing about it is a qairt concept.

---

## 3. Building the Qc config — `from_config`, not `from_pretrained`

`QcAutoConfig` (qairt's config-wrapping API) has three entry points that all funnel to the same
place:

```
from_pretrained(path)      -> AutoConfig.from_pretrained(path) -> from_config(...)
from_config(base_config)   -> from_model_type(base_config.model_type, base_config, ...)
from_model_type(...)       -> the real implementation
```

**Why we call `from_config(text_config, ...)` directly, not `from_pretrained(model_path)`:**
`from_pretrained(model_path)` would call `AutoConfig.from_pretrained(model_path)`, which re-reads
`config.json` from scratch — landing back on the top-level **multimodal** `Gemma4Config`, not the
`text_config` we already extracted in step 2. We already have the right sub-config in hand
(`ckpt.text_config`); re-deriving it from the path would just re-introduce the multimodal
ambiguity `checkpoint.py` was written specifically to avoid. Calling `from_config` skips straight
to wrapping the object we already have.

**What `from_config` → `from_model_type` actually does:**

1. **Resolve the Qc config class for `text_config.model_type` (`"gemma4_text"`).** qairt has a
   built-in registry of `models/<type>/mappings.py` packages (llama, qwen3, phi3, eaglet) it
   auto-imports — gemma4 isn't one of them, so this lookup fails and falls back to
   `TransformersModuleMapping._module_mapping`, a plain dict anything can register into at
   runtime. This is the **plugin mechanism** we rely on: `nano/models/gemma4_text/mappings.py`
   calls `TransformersModuleMapping.register(Gemma4TextConfig, QcGemma4TextConfig)` (and the
   attention/model/rotary-embedding equivalents) at import time; `adapt.py` imports that module
   before ever calling into qairt, so the registration has already happened by the time
   `QcAutoConfig`/`_reauthor` look it up.
2. **Instantiate**: `QcGemma4TextConfig(**text_config.to_dict())` — every HF field
   (`vocab_size`, `layer_types`, `rope_parameters`, `craft_config`, ...) round-trips through as a
   kwarg.
3. **Apply our kwargs**: for every `key=value` passed to `from_config(...)`, if
   `hasattr(qc_config, key)`, `setattr` it; otherwise log a warning and skip (typos don't error,
   they silently no-op — worth double-checking kwarg names against `QC_BACKING_ATTRS` when adding
   new ones).

The result, `qc_config`, is a config **object only** — no model has been touched yet. It's the
single source of every QC/HTP runtime flag (`transposed_key_cache`, `mask_neg`, `kv_clip_only`,
...) that the reauthored modules will read off `self.config` at forward-time.

---

## 4. Reauthoring the model — qairt's class-swap mechanism

### 4.1 What "reauthoring" means mechanically

Given a model already built and weight-loaded (step 2) and a Qc config (step 3), qairt's
`_reauthor()` walks every submodule and, wherever `type(submodule)` has a registered replacement
(from the same `TransformersModuleMapping.register(...)` calls used for the config class), does:

```python
submodule.__class__ = RegisteredReplacementClass
```

This is an **in-place class swap on an already-constructed object** — not a re-instantiation.
`__init__` is never called again. Whatever attributes the original `__init__` already set stay;
only the method-resolution behavior (which `forward()` runs) changes. This is exactly why every
adapted class in `reauthoring.py` avoids adding brand-new attributes in `__init__` — there's no
`__init__` call at swap time to set them.

### 4.2 What actually gets swapped for Gemma4 text-only

| Vanilla class | Swapped to | What changes |
|---|---|---|
| `Gemma4TextConfig` | `QcGemma4TextConfig` | Adds QC + Gemma4Context-parity fields as lazy properties |
| `Gemma4TextAttention` | `QcGemma4TextAttention` | Transposed-key cache, scatter-cache hooks, CRAFT bypass, masked-softmax |
| `Gemma4TextModel` | `QcGemma4TextModel` | `num_layers_to_run` truncation + explicit-cache guard |
| `Gemma4TextRotaryEmbedding` | `QcGemma4TextRotaryEmbedding` | Dict-passthrough short-circuit for host-precomputed rope |
| `Gemma4RMSNorm` | `QcGemma4RMSNorm` | `x/sqrt(...)` instead of `x*(...)^-0.5` (HTP-friendly) |
| `Gemma4TextMLP` | `QcGemma4TextMLP` | erf-based GeLU instead of tanh-approx (matches MPP/HTP) |

Not swapped, deliberately: `Gemma4ForCausalLM`, `Gemma4TextDecoderLayer` — both just call into
already-swapped submodules and need no changes of their own (vanilla `forward()` dispatches
correctly once `self.model`/`self.self_attn` are swapped underneath them).

### 4.3 Why `reauthor_model()` calls `._reauthor()` directly, not `.from_pretrained()`

`QcAutoModelForCausalLM.from_pretrained(path, qc_config=...)` is the normal, all-in-one entry
point: it loads the base HF model itself, then calls `._reauthor()`, then (if a
`quantization_config` was given) quantizes. We can't use it as-is because loading is our own
custom, multimodal-aware safetensors remap (step 2) — not something `AutoModelForCausalLM`
knows how to do. So we build+load the model ourselves, then call the exact internal method
`from_pretrained` would have called next: `QcAutoModelForCausalLM._reauthor(model,
qc_config=qc_config)`. No `quantization_config` is passed, so the quantization branch never runs
— this pass is reauthoring-only.

### 4.4 The extended-config-attrs copy step

`_reauthor()` internally copies `QcConfigMixin.QC_BACKING_ATTRS` (9 generic fields:
`transposed_key_cache`, `input_tokens_per_inference`, ...) from `qc_config` onto `model.config`.
Our 6 Gemma4-specific fields (`mask_neg`, `context_length`, `sliding_window_pattern`, ...) are
stored as underscore-prefixed backing attributes and get skipped by that copy loop's `_`-prefix
guard. `adapt.py` copies these 6 explicitly right after `_reauthor()` returns. Because
`model.config` is the *same object* every submodule already holds a reference to, setting them
once here reaches every layer.

### 4.5 Linear→Conv adaptation

qairt's default HTP adaptation set (normally applied automatically inside its own pipeline stage,
which we bypass by calling `._reauthor()` directly) includes `replace_linears_with_convs` — but
the generic version only matches `nn.Linear`/`Conv1D`. Gemma4's projections are
`Gemma4QuantizableLinear`, a different base class entirely, so the generic pass silently no-ops on
every projection in the model. `adapt.py` substitutes a Gemma4-aware version
(`gemma4_replace_linears_with_convs`, in `linear_to_conv.py`) that also matches
`Gemma4QuantizableLinear`, dequantizing (`weight * weight_scale`) before building the `Conv2d`
replacement.

---

## 5. Two real adaptations worth calling out explicitly

### 5.1 CRAFT ops vs. plain ops — and why this matches NanoV4, not a shortcut

Vanilla Gemma4 code doesn't call `x * y` directly for its internal arithmetic — it wraps every
basic op (multiply, add, matmul, concat, ...) in a small class:

```python
class Mul(CraftModule):
    def forward(self, x, v):
        result = x * v
        return self.apply_output_sfq(result)   # <- extra step: simulated quantization
```

`apply_output_sfq` pipes the result through `StaticFakeQuant`, which can simulate what happens if
that value gets rounded to low-precision (int8/int16) — this is CRAFT: same math, plus a
quantization-calibration tap on the output. Vanilla rope (`RotaryEmbeddingOperator`) is built
entirely from these CRAFT ops.

Our adapted attention (`qc_gemma4_eager_attention_forward`) uses plain `torch.matmul`/`softmax`,
and calls `apply_rotary_pos_emb(...)` with no `rope_operator` argument — which falls through to
HF's plain-Python fallback, `(x * cos) + (rotate_half(x) * sin)`, no CRAFT tap. **This is not a
deviation from NanoV4** — NanoV4's own adapted text attention (`qadaptation.py`) does the exact
same thing: plain `torch.matmul` for QK/AV, and its own `ApplyRopeSingle` module uses a bare
`MulModule` (`return a * b`, no CRAFT). Both NanoV4 and this port skip CRAFT here because CRAFT's
calibration hooks are a vanilla-HF-only mechanism; anyone doing quantization/calibration on this
model would plug in through a different, separate mechanism (NanoV4 hints at this via
`MulModule`'s docstring: *"we want to uniquely identify the EleMul operations in QuantSim"* — a
different tool from CRAFT).

### 5.2 Host-side RoPE precomputation

Vanilla `Gemma4TextModel.forward` computes `cos`/`sin` itself, inside the traced graph, on every
call. NanoV4 never lets this happen — `qgenerator.py` calls
`llm_create_position_embeddings(...)` on the **host**, once per generation step, and passes the
result straight through as the model's `position_ids` argument.

qairt already has this exact pattern for Qwen3: `QcQwen3RotaryEmbedding.forward` short-circuits
if `position_ids` is already a precomputed `(cos, sin)` tuple. Gemma4 needs a **dict**, not a
tuple — it has two distinct rope tables (`full_attention` vs `sliding_attention`, different
`rope_theta`/`partial_rotary_factor`), so one tuple can't represent both. `QcGemma4TextRotaryEmbedding`
does `if isinstance(position_ids, dict): return position_ids[layer_type]`, and
`create_position_embeddings(rotary_emb, position_ids)` (called from `run_smoke_test`) builds that
`{layer_type: (cos, sin)} `dict from the host, once, before the forward call — the same place in
the pipeline NanoV4's `qgenerator.py` call sits.

---

## 6. Verification — why `assert_reauthored()` exists

Because the class-swap in §4.1 is a silent, non-erroring mutation (`module.__class__ = X` never
raises just because `X` is the wrong class), there's no built-in signal that reauthoring actually
took effect versus, say, a registration silently failing to match. `verify.py` checks this
explicitly with `isinstance()` on `model.model` and every layer's `self_attn`, plus two config
flags — so a broken registration fails loudly here instead of quietly running vanilla code
further down.

---

## 7. Running one forward pass — `DynamicCache()`, and why `.generate()` doesn't work here

### 7.1 The cache-layer trap

`transformers.cache_utils.DynamicCache` has two ways to be constructed:

```python
DynamicCache()                 # every layer gets the plain DynamicLayer class
DynamicCache(config=some_cfg)  # layers are chosen per-index from some_cfg.layer_types:
                                #   "sliding_attention" -> DynamicSlidingWindowLayer
                                #   "full_attention"    -> DynamicLayer
```

qairt's KV-cache patch (`KVCacheMapping`, applied inside `_reauthor()`) works by monkey-patching
methods (`update`, `get_seq_length`) **only onto the base `DynamicLayer` class**. It does this
once, generically, for every model qairt supports — it has no Gemma4-specific knowledge of
`DynamicSlidingWindowLayer` at all. `DynamicSlidingWindowLayer` defines its own `update`/
`get_seq_length` that don't call back into `DynamicLayer`'s versions — so the patch never reaches
it.

Consequence: if a `DynamicCache(config=self.config)` gets built anywhere (this is exactly what
vanilla `Gemma4TextModel.forward` does internally when `use_cache=True` and no cache was passed
in), every **sliding-attention** layer would silently run unpatched vanilla cache behavior (no
transposed-key layout, no scatter update, no `kv_clip_only` clamp) while every **full-attention**
layer runs the adapted path — same model, silently inconsistent behavior per layer, no error
anywhere.

**The fix**: `QcGemma4TextModel.forward` (our swapped-in model class) raises `ValueError` if
`use_cache=True` and no `past_key_values` was explicitly passed — this forecloses vanilla's
auto-construction path entirely. Every caller, including `run_smoke_test`, must construct
`DynamicCache()` **with no `config=`** and pass it in explicitly. With no `config`, HF falls back
to plain `DynamicLayer` for every layer uniformly — sliding or full — so the patch reaches all of
them identically.

### 7.2 Why `.generate()` doesn't just work

`.generate()` (HF's `GenerationMixin`) manages its own cache unless you hand it one. Left to its
own devices, it builds a cache the same way vanilla `forward()` would — i.e. exactly the
`DynamicCache(config=...)` trap above. Calling `model.generate(**inputs)` without an explicit
`past_key_values=DynamicCache()` would silently hit the broken sliding-layer path described in
§7.1.

Passing `past_key_values=DynamicCache()` into `generate()` yourself would dodge that specific
trap — but a second, smaller mismatch remains: `.generate()`'s internal
`prepare_inputs_for_generation` builds `position_ids` as a plain tensor each decode step, not our
`{layer_type: (cos, sin)}` dict — so `QcGemma4TextRotaryEmbedding`'s dict short-circuit (§5.2)
never engages, and rope silently falls back to being computed on-model per step. Not incorrect,
just skips that specific NanoV4-parity optimization.

**Bottom line**: the *model* itself is fully adapted either way — `.generate()` would still run
`QcGemma4TextModel`/`QcGemma4TextAttention`'s forward code. What breaks is `.generate()`'s own
cache-construction and rope-injection conveniences, which is why this pass drives one manual
`model(...)` call with a hand-built cache instead. A real multi-step generation loop (matching
NanoV4's `qgenerator.py` — advancing `cache_position`, rebuilding the rope dict per step) is a
**separate, later stage**, intentionally not built here.

### 7.3 Scatter vs. concat — the patched cache update, and why we never see scatter fire

`KVCacheMapping` patches `DynamicLayer.update` with `_qc_dynamic_layer_update` (qairt's
`llm/models/utils.py`). That function picks between two write strategies:

- **Concat/init** — the default: new key/value tensors are appended to whatever's already in the
  cache (`torch.cat`), growing it. This is what vanilla `DynamicCache` always does.
- **Scatter** — writes the new key/value into a **fixed index** of an already-sized buffer
  (`cache.index_copy_`-style), instead of growing anything. This is the real on-device target
  behavior: on HTP, the KV history lives in a preallocated external buffer at a fixed address, so
  each step must write to a known offset, not "append."

The dispatch rule: scatter only fires when the caller passes a `cache_position` that indexes
*inside* an already-preallocated buffer. `run_smoke_test`'s single forward call never passes
`cache_position`, so it always takes the concat/init branch — meaning `perform_scatter_kv_cache_update=True`
in `qc_config` is set, but its code path has never actually executed yet. Exercising it requires a
real multi-step loop with preallocated cache tensors and explicit `cache_position` advancement —
the same "separate, later stage" as §7.2, not built here.

---

## 8. Where each adaptation actually lives (code map)

In-code comments were kept short (2-3 lines) on purpose — the "why" for each of these lives here,
not in the source. Use this table to jump from a class/function name to the section that explains it.

| Symbol | File | Explained in |
|---|---|---|
| `load_checkpoint` | `checkpoint.py` | §2 |
| `build_qc_config` | `qc_config.py` | §3 |
| `reauthor_model` | `adapt.py` | §4.3, §4.4 |
| `gemma4_replace_linears_with_convs` | `linear_to_conv.py` | §4.5 |
| `TransformersModuleMapping.register(...)` calls | `mappings.py` | §4.1, §4.2, §3 (plugin lookup) |
| `QcGemma4TextConfig` | `reauthoring.py` | §4.2, `QC_AUTOCONFIG.md` |
| `qc_gemma4_eager_attention_forward` | `reauthoring.py` | §5.1 (CRAFT bypass) |
| `QcGemma4TextAttention` | `reauthoring.py` | §4.2, §7.3 (cache update call) |
| `QcGemma4TextModel` | `reauthoring.py` | §4.2, §7.1 (cache guard) |
| `QcGemma4TextRotaryEmbedding` / `create_position_embeddings` | `reauthoring.py` | §5.2 |
| `assert_reauthored` | `verify.py` | §6 |
| `run_smoke_test` | `inference.py` | §7.1, §5.2 |

---

## 9. What's deliberately out of scope for this stage

- Multi-token `.generate()` loop (§7.2) — needs manual `cache_position`/`position_ids` advancement
  per step; `return_new_key_value_only` would also need to flip to `False` for a host-Python loop
  to keep the cache's full history (currently `True`, matching the on-device target where the real
  KV history lives in an external buffer, not in this Python object).
- Scatter-based fixed-index KV writes — `perform_scatter_kv_cache_update=True` is set, but with a
  single forward call and no `cache_position` ever passed, the scatter branch of the patched cache
  update never actually fires; everything so far exercises only the initial-write path.
- CRAFT/QuantSim-based calibration — see §5.1; not wired up, and not the same mechanism NanoV4 uses
  for text-attention quantization anyway.
- Numerical parity check of HF's rotate-half rope formula vs. NanoV4's `ApplyRopeSingle`
  (split-real/imaginary) — very likely equivalent, not yet verified tensor-for-tensor.
- Vision/audio/assistant(drafter)/MTP towers — text-only scope throughout.
