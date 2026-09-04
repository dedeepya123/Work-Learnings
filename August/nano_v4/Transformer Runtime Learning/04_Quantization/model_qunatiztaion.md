# Quantization Stage — Call-Flow Trace, Generalized Mechanism, and NanoV4 Mapping

This doc does for the **quantization stage** what `MODEL_PREPARATION_STAGE.md`
does for model preparation: a precise, file:line-cited trace of what
actually happens internally, generalized across *any* qairt quantization
technique (not just our LPBQ usage), with LPBQ as the concrete worked
example and NanoV4's manual flow mapped onto the same skeleton at the end.

The core idea to hold onto throughout: **quantization is not one monolithic
step.** It's a fixed skeleton of shared sub-steps (prepare → build quantsim →
apply mixed precision → calibrate → export), and every technique — LPBQ,
plain calibration, SeqMSE, AdaScale, SpinQuant, PrefixQuant, and NanoV4's own
hand-rolled flow — just slots into that same skeleton, inserting its own
one or two technique-specific steps at a fixed point in the sequence.

All line numbers are exact citations against:
- qairt: `qairt_env_nano/lib/python3.10/site-packages/qairt/experimental/pipeline/torch/llm/`
- AIMET: `qairt_env_nano/lib/python3.10/site-packages/aimet_torch/`
- Our code: `nano/models/gemma4_text/`, `nano_pipeline/`
- NanoV4: `Nano/NanoV4/qlib/`

---

## 1. The Stage contract layer — what a "normal" quantization stage does

### 1a. Input / Config / Output

`QuantizationStage`'s Pydantic contract (`torch/llm/stages/quantization_stage.py`):

```python
class QuantizationInput(StageInput):        # :46-51
    model: torch.nn.Module
    tokenizer: Any | None = None
    config: Optional[Any] = None

class QuantizationConfig(StageConfig):       # :54-96
    recipe_name: str                          # RecipeRegistry key, e.g. "lpbq_seqmse"
    recipe_config: Optional[Union[str, Dict]] # YAML path/dict override, or None = recipe's default
    technique_kwargs: Dict                    # per-technique nested overrides
    config_overrides: Dict                    # generic top-level overrides
    filename_prefix: str
    export_format: str = "v2"                 # "v1" | "v2" | "all"
    model_preparation_path: ...

class QuantizationOutput(StageOutput):        # :99-131
    model: ...
    quantsim: Optional[Any]
    quantizer: Optional[AIMETQuantizer]        # kept so export() can delegate to it
    tokenizer, config, generator: ...
    model_path, encodings_path, tokenizer_path, config_path: ...
```

### 1b. Default `_pre_hook` — a pure existence check

```python
# quantization_stage.py:213-230
try:
    RecipeRegistry.get(config.recipe_name)
except ValueError as e:
    raise ValueError(f"[QuantizationStage] {e}") from e
```
Plus `assert input.model is not None`, `assert input.tokenizer is not None`.
It discards the resolved recipe class — this is validation only.

### 1c. Default `_execute` — the recipe dispatch path

For a standard model, `_execute` (`:280-385`) does, in order:

1. `recipe_class = RecipeRegistry.get(config.recipe_name)` (`:286`).
2. `resolved_config = self._resolve_recipe_config(config)` (`:287`) — merges,
   in order: the recipe's registered default YAML → explicit `recipe_config`
   → `config_overrides` → `technique_kwargs` → generator-derived shape
   overrides (`:232-278`).
3. Strips stage-only keys the recipe doesn't understand (`filename_prefix`,
   `export_format`, `model_preparation_path`, etc., `:290-294`).
4. Builds `apply_kwargs` by injecting `model=input.model`,
   `tokenizer=input.tokenizer`, and a few passthrough flags into the merged
   dict (`:297-306`).
5. `result = recipe_class().apply(**apply_kwargs)` (`:316`) — **this is
   where control passes into the actual quantization technique.**
6. Wraps `result` into a `QuantizationOutput` (`:330-336`), re-attaches the
   outer wrapper if `input.model` had one (`:338-344`), and **auto-exports**
   to a temp dir if a `quantizer` object exists (`:374-383`) — this is why
   `model_path`/`encodings_path` get populated even without an explicit
   downstream `export()` call, for a standard model.

### 1d. `RecipeRegistry` — how `recipe_name` resolves to a class

`RecipeRegistry` (`quantization/recipes/registry.py:20-60`) is a plain
`Dict[str, Type[AIMETQuantizationRecipe]]`. Registration happens via a
decorator, applied at module import time in `quantization/recipes/defaults.py`:

```python
@register_recipe("lpbq_seqmse")          # defaults.py:417
@register_recipe("calibrator")           # defaults.py:270
@register_recipe("spinquant_adascale")   # defaults.py:732
@register_recipe("seq_mse_opt")          # defaults.py:915
@register_recipe("gptaq_hf")             # defaults.py:975
@register_recipe("prefix_quant")         # defaults.py:1001
# ... etc
```

`config.recipe_name = "lpbq_seqmse"` (a plain string in the YAML) resolves
via `RecipeRegistry.get(name)` — a dict lookup, raising `ValueError` on miss.
The recipe class itself (e.g. `LPBQ_SeqMSE_Recipe`) subclasses
`AIMETQuantizationRecipe`, whose `apply()` is the real per-recipe entry
point — this is what actually builds the technique's `Params` object and
calls `quantizer.quantize(params)` underneath.

**Why we bypass all of this** (`nano_pipeline/stages/quantization_stage.py:83-99`):
our custom `Gemma4QuantizationStage._execute` skips steps 1-5 above entirely
and calls `quantize_model()` directly, because step 2's merge only accepts
YAML-serializable values — there's no way to hand it a live Python
`DataLoader` object (our actual calibration set). Everything from Section 2
onward, though, still applies exactly the same way — we just reach it by a
shorter, more direct path.

---

## 2. The shared skeleton every technique inherits — `AIMETQuantizer`

This is the part that generalizes across every technique, and the part
worth understanding deeply, since it's the same regardless of which recipe
name you pick.

### 2a. The canonical sequence

Confirmed by reading both `Calibrator.quantize()` and `LPBQQuantizer.quantize()`
— both follow this exact order, with no separate "setup" hook on the base
class (validation is inlined at the top of each concrete `quantize()`):

```
[inline: isinstance(params, ...) / dataloader is None checks]
        |
        v
_prepare_model(generator, path)          <-- runs Model Preparer Pro (see
        |                                     MODEL_PREPARATION_STAGE.md)
        v
_prepare_inputs(dummy_inputs, generator)  <-- builds dummy trace inputs for
        |                                     quantsim construction
        v
_create_quantsim(prepared_model, dummy_inputs, ...)  <-- builds the base
        |                                                 AIMET QuantizationSimModel
        v
[technique-specific step]                <-- THE part that actually varies
        |                                     (LPBQ: setup_blockwise; SeqMSE:
        |                                     apply_seq_mse; AdaScale: apply_adascale;
        |                                     Calibrator: nothing extra)
        v
_compute_encodings(quantsim, dataloader, forward_pass_callback, generator)
        |                                     <-- calibration: this is where AIMET's
        |                                         observers actually run (Section 4)
        v
return QuantizationResult(quantsim=quantsim, quantizer=self)
```

### 2b. `_create_quantsim` — what actually builds the AIMET quantsim

Concretely implemented in `Calibrator._create_quantsim`
(`quantization/techniques/calibration/calibrator.py:87-177`); `LPBQQuantizer`
and others reuse it directly rather than reimplementing it
(`lpbq_quantizer.py:136-143` is a thin delegating wrapper).

```python
quant_scheme = kwargs.pop("quant_scheme", QuantScheme.post_training_tf)  # == min_max
default_output_bw = kwargs.pop("default_output_bw", 16)
default_param_bw = kwargs.pop("default_param_bw", 4)
in_place = kwargs.pop("in_place", True)
config_file = _get_htp_config_path(dsp_arch)

quantsim = QuantizationSimModel(**quantsim_args)          # global default_param_bw applied to EVERY module
if tie_quantizers:
    propagate_output_encodings(quantsim, aimet_ops.Concat)
cls._apply_module_precisions(quantsim, module_precisions)  # per-module OVERRIDES applied AFTER
return quantsim
```

**Key ordering fact, worth calling out explicitly**: the base
`QuantizationSimModel(...)` constructor applies one **uniform, global**
`default_param_bw`/`default_output_bw` to every module first. Mixed
precision (`ModulePrecisions`/`custom` regex map, see Section 3) is a
**post-construction patch** applied on top, not baked in from the start.
Same ordering, confirmed identically, in NanoV4's manual flow (Section 6).

**The config file** — `config_file = _get_htp_config_path(dsp_arch)`:

```python
def _get_htp_config_path(dsp_arch: DspArchitecture) -> str:
    config_filename = f"htp_quantsim_config_{dsp_arch.value}.json"
    return str(importlib.resources.files("aimet_torch.common.quantsim_config").joinpath(config_filename))
```

This is a **fixed JSON file bundled inside the `aimet_torch` package itself**
(`aimet_torch/common/quantsim_config/htp_quantsim_config_v81.json`), selected
purely by which `dsp_arch` enum value you pass (v66/v68/v69/v73/v75/v79/v81).
It is not a path you configure directly — only indirectly, by choosing the
target architecture. This JSON is what actually encodes:

- **Activations are per-tensor by global default** (`per_channel_quantization: "False"` at the `defaults` level) — this is the CLAUDE.md rule "activations are always per-tensor quantized."
- **Weights are per-channel only for specific op types** — `Conv`/`ConvTranspose`/`Gemm`/`MatMul`/`PRelu` explicitly override to `per_channel_quantization: "True"` — this is the CLAUDE.md rule "parameters are mostly per-channel quantized."
- Biases unquantized by default (`params.bias.is_quantized: "False"`).
- Norm-weight quantization defaults to *asymmetric* (`LayerNormalization`/`RMSNormalization` weight → `is_symmetric: "False"`) — which is exactly why our flow explicitly overrides RMSNorm to symmetric via `ModulePrecisions`/NanoV4 overrides it via `_modify_weight_quantization` when a symmetric norm scheme is wanted.
- Op-fusion supergroups (`Conv+Relu`, `MatmulAdd`, etc.).

### 2c. `quant_scheme` — what "calibration scheme" actually means here

```python
class QuantScheme(Enum):        # aimet_torch/common/defs.py:18-30
    min_max = 1
    post_training_tf = min_max              # literal alias, same enum value
    post_training_tf_enhanced = 2
    post_training_percentile = 6
    # + deprecated range-learning variants
```

Our flow (and `Calibrator`'s own default) uses `QuantScheme.post_training_tf`
— which is just an alias for `min_max` (value 1). So concretely: **calibration
here means min/max observation**, not a percentile or range-learning scheme,
unless explicitly overridden via `quantsim_kwargs["quant_scheme"]`.

### 2d. `export()` — the full trace, and what "v2" actually means

`AIMETQuantizer.export()` (`aimet_quantizer.py:653-796`):

1. Validates `export_format in ("v1", "v2", "all")`.
2. If `onnx_export_args` not supplied, **auto-derives** input/output names
   from `self._export_generator.config.num_hidden_layers` (`:689-726`) —
   **this is the exact auto-derivation our custom `_build_onnx_export_args()`
   bypasses**, because it assumes every layer owns a KV cache, which is wrong
   for Gemma4's 15-of-35 KV-owning-layer architecture.
3. Temporarily flips AIMET globals (`EXPORT_TO_ONNX_DIRECT = True`,
   `RESTORE_ONNX_MODEL_INITIALIZERS = True`) and swaps
   `quantsim.encoding_version`.
4. **v1 branch**: sets `encoding_version = "1.0.0"`, calls AIMET's own
   (deprecated) `QuantizationSimModel.export(path, filename_prefix,
   dummy_input, onnx_export_args, ...)` — this single deprecated call
   internally handles **both** ONNX export and encodings-JSON write in one
   pass; no separate `torch.onnx.export` call is needed on our side either
   way.
5. **v2 branch** (our default): sets `encoding_version = "2.0.0"`, calls
   `quantsim.onnx.export(dummy_inputs, {export_path}_v2_encodings/{filename}.onnx,
   input_names=..., output_names=..., export_int32_bias=True)` — a newer,
   dedicated exporter class (`QuantizationSimModelOnnxExporter`, accessed via
   the `sim.onnx` property) that writes to the `_v2_encodings/` suffixed
   directory. **This confirms and generalizes the earlier finding**: the
   `_v2_encodings` suffix is a hardcoded consequence of `export_format="v2"`,
   not tied to any specific path string — it applies to any technique's
   export call.
6. Restores the AIMET globals in a `finally` block.

**"v2" = `encoding_version: "2.0.0"`** — a newer AIMET encodings-JSON schema,
paired with the newer `sim.onnx.export()` code path. "v1" = the older
`encoding_version` ("0.6.1"/"1.0.0") paired with the deprecated
`QuantizationSimModel.export()` method.

---

## 3. Mixed precision — where `ModulePrecisions` fits in the sequence

Directly answering "what happens internally, generally, for mixed
precision": `_apply_module_precisions` (`aimet_quantizer.py:323-385`)
resolves `ModulePrecisions`'s semantic fields (`lm_head`, `embedding`,
`kv_cache`, `rms_norm`) plus the `custom` name/regex map into one flat
`dict[name_or_regex, Precisions]` via `_resolve_precisions`
(`:387-442`), then `_apply_custom_precisions` (`:444-498`) walks
`quantsim.model.named_modules()` and, for each key, tries an exact
attribute match, then an exact `named_modules()` key match, then
`re.fullmatch(key, module_name)` — and on a match, directly mutates that
module's quantizer object: `module.param_quantizers["weight"].bitwidth = bw`,
`q.symmetric = ...`.

This runs, as established in Section 2b, **strictly after** the base
quantsim is constructed with one global default bitwidth — it's a
post-construction patch, applied identically whether the caller used
`Calibrator`, `LPBQQuantizer`, or NanoV4's own manual walk.

---

## 4. Calibration internals — the exact observer chain, arrow by arrow

This is the deepest layer — what actually happens, mechanically, when
"calibration" runs.

```
AIMETQuantizer._compute_encodings(quantsim, dataloader, callback, generator)   [aimet_quantizer.py:549-651]
        |  either forwards a user callback straight through, or builds
        |  the built-in _default_callback (Section 4b)
        v
QuantizationSimModel.compute_encodings(forward_pass_callback)                 [aimet_torch/quantsim/quantsim.py:394-445]
        |  with torch.no_grad(), utils.in_eval_mode(model):
        |      with aimet_nn.compute_encodings(model):
        |          forward_pass_callback(model)          <-- your data loop runs HERE
        v
aimet_nn.compute_encodings(model)  (context manager)                          [aimet_torch/nn/__init__.py:22-84]
        |  walks model.modules(), collects every quantizer, enters each
        |  quantizer's own compute_encodings() context, patches its forward()
        v
AffineQuantizerBase._compute_encodings / forward_wrapper                      [aimet_torch/quantization/affine/quantizer.py:600-670]
        |  every time this quantizer's tensor is touched during a forward call:
        v
encoding_analyzer.update_stats(input)                                          [aimet_torch/quantization/encoding_analyzer.py:341-351]
        |  self.observer.collect_stats(input_tensor)   <-- reads real fp32 values
        |  self.observer.merge_stats(new_stats)         <-- folds into running min/max
        v
[... repeats for every forward call across every calibration batch ...]
        v
(on context exit, after ALL calibration forward passes are done)
encoding_analyzer.compute_encodings(num_steps, symmetric)                      [encoding_analyzer.py:359-373, 417+]
        |  turns the ACCUMULATED min/max into the final scale/offset
        v
Each quantizer now has a frozen "encoding" (scale + offset)
```

**This confirms exactly what "calibration" means, mechanically**: every
quantized tensor's observer watches the real floating-point values flowing
through it, across every forward call the callback makes, accumulating a
running min/max; only after every calibration batch finishes does it
convert that accumulated range into the tensor's final scale/offset. It is
not per-batch — it's a single running accumulation across the whole
calibration set.

### 4a. The default callback, when none is provided (our case)

`AIMETQuantizer._compute_encodings`'s inner `_default_callback`
(`aimet_quantizer.py:598-642`):

```python
for batch in itertools.islice(dataloader, num_batches) if num_batches else dataloader:
    inputs = {k: v for k, v in batch.items()
              if k in {"input_ids","attention_mask","inputs_embeds","position_ids",
                        "hidden_states","past_key_values","cache_index"}}
    if generator is not None:
        _set_generator_runtime_model(model)     # swap generator.model -> the quantsim being calibrated
        generator(**inputs)
    else:
        model(inputs["input_ids"])
```

Two details worth internalizing:

1. **Only a fixed whitelist of dict keys survives the filter** — if your
   batch dict has extra keys, they're silently dropped unless you supply your
   own callback.
2. **The generator's `.model` gets swapped to the quantsim being calibrated**
   before each call, and restored afterward in a `finally` — this is *why*
   a custom `forward_pass_callback` that fully replaces this default must
   replicate that swap manually (established earlier in this session, when
   discussing multi-step decode calibration).

### 4b. Our end-to-end input → output for the Gemma4 stage

**Input**: `input.model` = `TextModelQc`-wrapped reauthored decoder (from
`Gemma4ModelLoadingStage`), `input.config` = `qc_config`, `input.tokenizer`
= HF tokenizer. `nano_pipeline/stages/quantization_stage.py:83-99`:

```python
qc_config = input.config
quantizer = quantize_model(
    input.model.qc_model, qc_config, input.tokenizer,     # unwrapped -- quantize_model() re-wraps its own
    prepare_path=config.prepare_path, export_path=config.export_path,
)
```

**Output** — `Gemma4QuantizationOutput.model_construct(...)`:

```python
model=quantizer.quantsim.model,        # the quantized model
quantsim=quantizer.quantsim,           # the AIMET QuantizationSimModel object
quantizer=quantizer,                   # the LPBQQuantizer instance
generator=None,                        # unlike the base stage's auto-export path, we don't build one here
model_path=f"{config.export_path}_v2_encodings/model.onnx",
encodings_path=f"{config.export_path}_v2_encodings/model.encodings",
tokenizer_path=config._pipeline_context.model_id_or_path,
```

Note `model_path`/`encodings_path` are **hand-constructed literal paths**,
not derived from the base class's globbing `export()` logic — because our
`_execute` bypasses the base class's `_execute` entirely (per Section 1d),
and `quantize_model()` internally already called `quantizer.export(...)`
with our custom `onnx_export_args` before returning.

---

## 5. Generalizing beyond LPBQ — what's shared vs. technique-specific

### 5a. `Calibrator` — the simplest possible technique

Confirms the skeleton has no hidden extra step: `Calibrator.quantize()`
(`calibration/calibrator.py:179-234`) goes straight from
`_create_quantsim` to `_compute_encodings` with **nothing** in between —
plain min/max calibration, no technique-specific insertion at all.

### 5b. `LPBQQuantizer` — one extra step

```python
quantsim = params.quantsim or self._create_quantsim(...)   # delegates to Calibrator's version, verbatim
self.setup_blockwise(quantsim, params)                       # <-- the ONLY thing LPBQ adds
self._compute_encodings(quantsim, params.dataloader, ...)
```
`setup_blockwise` calls AIMET's `set_grouped_blockwise_quantization_for_weights`
on modules tagged `_apply_technique=True` (set via `ModulePrecisions`'
`apply: true` flag) — LPBQ's blockwise weight-quantization scheme layered on
top of the identical base quantsim.

### 5c. Every other technique, one line of evidence each

| Technique | Where the shared steps show up | Its own inserted step |
|---|---|---|
| **SeqMSE** | `self._prepare_model()` → `self._prepare_inputs()` → `Calibrator._create_quantsim(...)` (directly reused) | `apply_seq_mse(...)` — replaces `compute_encodings` in SeqMSE's own flow; it's an *optimizer*, calibration happens later when a downstream quantizer runs on its output |
| **AdaScale** | `self._create_quantsim(...)` internally delegates to `Calibrator._create_quantsim` | `apply_adascale(qsim=quantsim, data_loader=..., forward_fn=...)` |
| **SpinQuant** | `self._prepare_model()` only — **never builds a quantsim at all** | `apply_spinquant(model=self.prepared_model)` — pure weight-transform, meant to run *before* a real quantizer step |
| **PrefixQuant** | None of the above — not an `AIMETQuantizer` subclass at all; an HF-adapter-style optimizer | `generate_prefix_kvcache(model, dummy_input, ...)` — precomputes and caches a KV state; produces no quantsim/encodings whatsoever |

**Takeaway for generalizing to "any technique"**: if it subclasses
`AIMETQuantizer` (Calibrator, LPBQ) or `AIMETOptimizer`
(SeqMSE, AdaScale, SpinQuant), it reuses `_prepare_model`/`_create_quantsim`
verbatim and only adds its own step in the middle. PrefixQuant is the one
genuine outlier — it sits entirely outside this skeleton, orchestrated
through a separate `HFQuantizationRecipe` step-list mechanism instead.

### 5d. Params classes — shared vs. technique-specific fields

`QuantSimParams` (base, shared by every `AIMETQuantizer` subclass):
```python
tie_quantizers: list[str] = ["concat"]
module_precisions: ModulePrecisions | None = None
dsp_arch: DspArchitecture | str = DspArchitecture.v79
quantsim_kwargs: dict = {}
forward_pass_callback: Optional[Callable] = None
# + inherited: dataloader, dummy_inputs, generator, quantsim
```

`LPBQParams` adds **only**: `block_size`, `block_grouping`, `symmetric`,
`decompressed_bw`, `model_preparation_path` (a convenience passthrough).

`CalibratorParams` adds **nothing** — confirming Calibrator genuinely needs
zero fields beyond the shared base.

---

## 6. NanoV4 mapping — `create_qs_model()` onto the same skeleton

File: `Nano/NanoV4/qlib/qmodule_builder.py`. NanoV4's `create_qs_model`
(`:293-320`) builds an AIMET `QuantizationSimModel` **directly**, with no
`RecipeRegistry`/`AIMETQuantizer` abstraction layer at all — but it follows
the exact same conceptual sequence.

| Generalized skeleton step | NanoV4's code | Location |
|---|---|---|
| `_prepare_model` | `create_mpp_model()` → `qti.aisw.preparer_api.prepare_model(...)` | `qmodule_builder.py:268-291`, `:88-121` |
| `_prepare_inputs` | `_get_dummy_input(generator)` → `_capture_io(...)` (a real prefix-step forward) | `qmodule_builder.py:543-548`, `:468-509` |
| `_create_quantsim` | `BuilderBase._create_quantsim()` — direct `QuantizationSimModel(model=prepared_model, quant_scheme=QuantScheme.post_training_tf, dummy_input=..., default_output_bw=16, default_param_bw=8, in_place=True, config_file=htp_config_file)` | `qmodule_builder.py:146-167` |
| Mixed precision | `TextModelBuilder._modify_quantization_scheme(quantsim, quant_params)` — hand-written, walks `quantsim.named_qmodules()`, regex-matches module names, sets bitwidth per category: RMSNorm→16-bit, attn q/k/v/o_proj→4-bit, MLP→4-bit (2-bit for `fast` model's MTP layers), lm_head→4-bit | `qmodule_builder.py:555-641` |
| `_compute_encodings` | `BuilderBase._calibrate()` → `self.model_qs.compute_encodings(self._calibration_forward_fn, {...})` | `qmodule_builder.py:367-391`, callback `:338-365` |
| `export()` | `BuilderBase._export_qs_model()` → `export_onnx(...)` + `export_lora_artifacts(...)` | `qmodule_builder.py:393-407` |

**Confirmed: the ordering matches exactly.** NanoV4's mixed-precision step
also runs *after* the base `QuantizationSimModel(...)` call (with one
uniform `default_param_bw=8`), patched on top per-module afterward — same
relative position as qairt's `_apply_module_precisions`. The only real
difference is *how* that step is expressed: qairt's is a small declarative
dispatcher (`ModulePrecisions` dataclass → regex map → generic mutation
helper); NanoV4's is a large, hand-written, Gemma4-specific imperative
function with inline regexes and hardcoded bitwidth literals per module
category.

### Config file — confirmed identical bundled JSON, different access path

NanoV4 passes `htp_config_file: str = 'htp_v81'` as a plain string,
forwarded straight into `QuantizationSimModel(..., config_file=htp_config_file)`.
AIMET's own `_get_config_file()` resolves that alias string via the exact
same `_config_file_aliases` dict (`"htp_v81" → "htp_quantsim_config_v81.json"`)
that backs qairt's `_get_htp_config_path()`'s `importlib.resources` lookup.

**Both NanoV4 and qairt resolve to the identical bundled AIMET JSON file** —
NanoV4 just passes the alias string directly; qairt wraps the same lookup
behind a `DspArchitecture` enum. NanoV4 does not define its own custom
per-op-type-defaults JSON — it relies on the same bundled HTP config for
base per-op-type symmetric/per-channel defaults, and only hardcodes its
*mixed-precision exceptions* in Python.

---

## Summary — one-paragraph mental model

Quantization, in this SDK, is a fixed five-step skeleton —
`_prepare_model` (freeze into a static graph, see `MODEL_PREPARATION_STAGE.md`)
→ `_prepare_inputs` (build a dummy trace input) → `_create_quantsim` (build
one AIMET `QuantizationSimModel` with a single global default bitwidth,
governed by a fixed bundled HTP config JSON keyed on target DSP
architecture) → apply per-module mixed-precision overrides on top → run
calibration (`compute_encodings`, which drives every quantizer's min/max
observer across every forward call the calibration loop makes, then freezes
the accumulated range into a scale/offset) → export (ONNX + encodings, in
one call, format selected by `encoding_version`). Every technique — LPBQ,
plain Calibrator, SeqMSE, AdaScale, SpinQuant — inserts exactly one
technique-specific step into this same sequence, at the same fixed point
(right after `_create_quantsim`, right before `_compute_encodings`).
NanoV4's manual `create_qs_model()` follows this identical sequence, just
expressed as direct Python calls with no `RecipeRegistry`/`AIMETQuantizer`
abstraction layer — same steps, same config file, same ordering, different
scaffolding.
