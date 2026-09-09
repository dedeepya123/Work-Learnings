# LPBQ quantization for the Gemma4 text-only decoder

What this document is: the rationale for `quantize.py`, the stage that runs after reauthoring
(`adapt.py`) and drives model preparation via qairt's real generator-based flow. It follows the
same "call qairt's internal pieces directly instead of its convenience wrapper" pattern as
`adapt.py` (see `NANOV4_TO_QAIRT.md` §4.3), applied this time to AIMET quantization rather than
reauthoring.

---

## 1. Why `quantization_config=`-driven loading is unreachable

Vanilla `AutoModelForCausalLM.from_pretrained(path, quantization_config=...)` routes through
`transformers/quantizers/base.py`'s `get_hf_quantizer()` → `HfQuantizer.preprocess_model` /
`postprocess_model`, which for qairt's LPBQ technique means `AIMETHFQuantizer`/`LPBQHFQuantizer`'s
`_process_model_before_weight_loading` / `_process_model_after_weight_loading` hooks.

We never call `.from_pretrained()` — our checkpoint is a text submodel pulled by hand out of a
multimodal safetensors file (`checkpoint.py`, see `NANOV4_TO_QAIRT.md` §2), and `qc_config` is
built via `QcAutoConfig.from_config(...)`, not `from_model_type` triggered by a HF loader. So this
whole quantizer-hook mechanism never runs for us; there is no `quantization_config=` kwarg
anywhere in this pipeline for it to key off of. Quantization has to be invoked directly on our
already-reauthored-and-adapted model, the same way reauthoring itself was invoked directly instead
of through `QcAutoModelForCausalLM.from_pretrained(..., qc_config=...)`.

---

## 2. `LPBQQuantizer.quantize()`'s auto-generator trap — resolved via `Gemma4TextGenerator`

The natural-looking call is:

```python
quantizer = LPBQQuantizer(hf_model=model, context_length=..., sequence_length=..., tokenizer=...)
result = quantizer.quantize(LPBQParams(dataloader=..., module_precisions=...))
```

This used to not be safe to call as a black box for gemma4_text. `AIMETQuantizer._prepare_model()`
(`aimet_quantizer.py`) branches on whether a `generator` was supplied:

- `generator is None` → `self.prepared_model = self.hf_model` directly. No ONNX export, no QNN-IR
  round trip.
- `generator` given → unconditionally calls `prepare_llm(generator=generator, ...)` — the
  ONNX-export → QNN-IR → PyTorch-reimport round trip built for models with a registered
  `GeneratorFactory`/`LlmIOConfig`.

The trap: `LPBQQuantizer.quantize()`'s own body, when `params.generator is None`, doesn't leave
`generator` as `None` through to `_prepare_model` — it **constructs an `LLMGenerator` itself**
first and passes that in. So calling `quantize()` always ends up on the `prepare_llm()` branch,
regardless of whether the caller supplied a generator.

Historically we couldn't use that branch: there was no `GeneratorFactory.register(...,
model_types=["gemma4_text"])` and no `LlmIOConfig` for gemma4_text anywhere in qairt, and the
generic fallback (`HybridLLMGenerator`/base `LLMGenerator`) can't build Gemma4's dual
full/sliding-window rope+mask pair — `LLMGenerationMixin.create_position_embeddings` assumes a
single `(cos, sin)` pair from a live `model.rotary_emb(config)` submodule call, not two separate
tables for two layer types.

**This is now resolved**: `Gemma4TextGenerator` (`generator.py`) is a working `LLMGenerator`
subclass purpose-built for Gemma4's shape — it overrides `_resolve_io_config` to declare the extra
`swa_attention_mask`/`swa_position_ids`/`per_layer_inputs` I/O (via
`Feature.SLIDING_WINDOW_ATTENTION`), and overrides `prepare_inputs` to build those dual mask/rope
tensors host-side via `create_attention_masks`/`create_position_embeddings` (`reauthoring.py`). It
must be constructed with `QcGemma4ForCausalLM` (`reauthoring.py`) — the outer wrapper `adapt.py`'s
`reauthor_model()` returns — not the bare `QcGemma4TextModel` decoder: `torch.onnx.export`'s
`_decide_input_format` flattens dict inputs strictly against the *traced module's own* `forward()`
named parameters, and vanilla `Gemma4ForCausalLM.forward` only names a handful of them, dropping
`swa_position_ids`/`swa_attention_mask`/`per_layer_inputs` into `**kwargs` before `self.model(...)`
is ever called. `QcGemma4ForCausalLM.forward` declares all three explicitly (mirroring
`QcLlamaForCausalLM`/`QcQwen3ForCausalLM`'s wrapper-subclassing pattern in the vendored qairt SDK),
so it's now a safe trace target — and its real output is `logits`, matching qairt's default
`IOSpecs.COMMON` output spec, so `Gemma4TextGenerator` no longer needs a `_resolve_llm_io`
output-rename override. Tracing the wrapper (rather than the bare decoder) also keeps its
`rotary_emb`/`embed_tokens`/`config` reachable at `generator.model.model.*` outside the
swap-one-level-down calibration convention — see §7.

Verified directly: `prepare_llm(generator=Gemma4TextGenerator(model=model, ...))` (where `model`
is the `QcGemma4ForCausalLM` wrapper) completes end-to-end for the full 35-layer checkpoint,
producing a `ConvertedModel`. `quantize_model()` now calls `quantizer._prepare_model(generator=
generator)` — the real branch, not the `generator=None` bypass — as its entire scope. See §7 for
exactly what changes once a real generator flows through `_prepare_model`, and what's deferred
past it.

---

## 3. `dummy_inputs` shape for AIMET tracing (historical — pre-`Gemma4TextGenerator`)

*This section describes the old `generator=None` + manually-built-tuple path that `quantize.py`
used before `Gemma4TextGenerator` existed. It no longer runs — `_prepare_model(generator=generator)`
handles tracing entirely inside `prepare_llm()`/`torch.onnx.export`, via `Gemma4TextGenerator.
prepare_inputs`'s dict, not a hand-built positional tuple. Kept for reference because the same
traceability constraints (tensor/Cache leaves only, no bare bool) will resurface once `_create_
quantsim`'s dummy inputs are designed — see §7.*

`AIMETMixin._prepare_inputs()` (`aimet_mixin.py`) treats an explicitly-passed `dummy_inputs` as
pass-through — returned as-is with no validation — becoming `assembled_dummy_inputs`, which flows
straight into AIMET's `QuantizationSimModel(model=..., dummy_input=assembled_dummy_inputs, ...)`
(`aimet_torch/_base/quantsim.py`). That constructor:

1. Wraps a non-tuple/list `dummy_input` into a 1-tuple (`if not isinstance(dummy_input, (tuple,
   list)): dummy_input = (dummy_input,)`) — so a **dict** would end up as `model(the_dict)`, a
   single positional argument bound to the model's first parameter (`input_ids`). Not what we
   want; the model needs several distinct arguments.
2. Calls `_assert_jit_traceable(model, dummy_input)`, which pytree-flattens `dummy_input` and
   raises `RuntimeError` on any leaf that isn't a `torch.Tensor` **or a `transformers.Cache`
   instance** — every other type, including a bare `bool`, fails.
3. Ultimately calls `self.model(*dummy_input)` — positional, not keyword.

The old `_build_dummy_inputs()` built a plain positional tuple matching `Gemma4ForCausalLM.
forward`'s parameter order, with a real `DynamicCache()` (no `config=`, per the KV-cache-layer trap
in `NANOV4_TO_QAIRT.md` §7.1) standing in for `past_key_values` — a `Cache` instance is explicitly
whitelisted by `_assert_jit_traceable`. This is now qairt's job internally via `prepare_llm()`.

---

## 4. The `QcGemma4TextModel` cache-required raise (still applies, differently avoided now)

`QcGemma4TextModel.forward` (`reauthoring.py`) raises `ValueError` if `use_cache` is truthy
(the default) and `past_key_values is None` — this is deliberate, not a bug we're working around
incidentally; see `NANOV4_TO_QAIRT.md` §7.1 for why (vanilla's auto-`DynamicCache(config=...)`
fallback would leave sliding-attention layers running unpatched cache behavior).

`Gemma4TextGenerator.prepare_inputs` avoids it by always returning `past_key_values=()` — an empty
legacy tuple, not `None` — so the guard's `past_key_values is None` check is false regardless of
`use_cache`. `use_cache` itself is deliberately omitted from the returned dict entirely: it's a
plain Python bool, so `torch.onnx.export`'s tracer bakes it in as a constant rather than a real
graph input, but `model_preparer`'s input-count validation (`order_inputs=True`) counts every dict
key — including non-tensor ones — against the traced graph's actual input count. Including it
caused a real "Number of onnx graph inputs (8) does not match number of dummy inputs provided (9)"
failure during generator development; omitting it is safe since the guard can never fire in this
trace path (`past_key_values` is always `()`, never `None`, here).

---

## 5. Protobuf serialization limit on the full checkpoint

Saving the ONNX graph for the full 35-layer checkpoint (real weights) hit
`google.protobuf.message.EncodeError: Failed to serialize proto` inside vendored SDK code
(`onnx_saver.py::_onnx_model_size_larger_than_max_protobuf`, which calls `onnx_model.ByteSize()`
purely to decide whether to switch to ONNX's external-data storage format). This is a
protobuf-library-level limitation with very large in-memory protos, unrelated to our model code.
**Workaround**: downgraded `protobuf` to `3.20.2` in the `qairt_env_nano` venv (from `7.35.1`) —
resolved the `EncodeError` and let ONNX saving proceed. This is a reversible environment change;
`onnx==1.19.1` officially declares `protobuf>=4.25.1`, and `qairt-dev==0.9.0` declares
`protobuf==6.31.0`, so this pins outside both declared ranges — confirmed `onnx` still imports and
the full pipeline runs correctly regardless, but if a future qairt/onnx upgrade reintroduces a hard
version check, this is the first thing to revisit.

---

## 6. `past_key_values[N]` output-buffer warnings during IR conversion

Every `prepare_llm()` run for this generator logs ~70 warnings like `Output Buffer
past_key_values[N][0/1], specified via command line, does not exist in graph` (one pair per
layer). This is expected, not a bug: `Gemma4TextGenerator.prepare_inputs` builds a **single-step**
trace with `past_key_values=()` (empty) — there is nothing for the traced graph to produce as a
real KV-cache output on a from-scratch forward pass, but the IO spec still declares the generic
70-output-slot shape (35 layers × key/value) since that's the standard LLM-generator shape. The
resulting prepared model therefore has no wired KV-cache output path — fine for model preparation,
but it means this generator/trace shape is not yet suitable for real multi-step/autoregressive use
without a cache-advancing dummy-input design (more like `inference.py`'s pattern than this
generator's).

---

## 7. What `_prepare_model(generator=generator)` changes, and what's deferred

Once a real `generator` is passed to `_prepare_model` (`aimet_quantizer.py`):

```python
self.prepared_model = prepare_llm(generator=generator, **kwargs)
del self.hf_model
self.hf_model = None
```

`self.hf_model` is deleted, and `self.prepared_model` becomes a `ConvertedModel` — a class
regenerated **op-by-op** by TorchEmitter from the IR graph (`prepare_llm()` always passes
`keep_original_model_structure=False`). It has **no** `.embed_tokens`/`.rotary_emb`/`.config` —
none of `QcGemma4TextModel`'s submodule structure survives. This is why the old
`_build_dummy_inputs`/`_calibration_forward` (which called `decoder.embed_tokens(...)`,
`decoder.rotary_emb(...)`, `decoder.config...`) were deleted outright rather than adapted: they
encode assumptions that are actively wrong against a `ConvertedModel`, and `quantize_model()`'s
scope currently stops at `_prepare_model` — `_create_quantsim`/`setup_blockwise`/
`_compute_encodings` are deferred follow-up work.

When that follow-up work happens, qairt's own generator-driven pattern (confirmed by direct
investigation of `aimet_mixin.py`/`aimet_quantizer.py`) is:

- **Dummy inputs** (`_prepare_inputs`, when no explicit `dummy_inputs` is passed): calls
  `generator.prepare_inputs(model=generator.model, ...)`, then reorders the resulting dict into a
  positional tuple filtered/ordered by `inspect.signature(self.prepared_model.forward).parameters`
  — i.e. built for the *prepared* model's signature, not the original decoder's.
- **Calibration** (`_compute_encodings`, when no explicit `forward_pass_callback` is passed):
  per batch, `_set_generator_runtime_model` swaps `generator.model` (or `generator.model.model` if
  `generator.bypass_adapted_forward` is set) to the quantsim runtime model, then calls
  `generator(**inputs)` — `LLMGenerator.forward` internally re-derives per-chunk inputs via the
  same `prepare_inputs` classmethod and calls the (now quantsim-wrapped) model. The dataloader
  just needs to yield dicts/tensors of raw `input_ids` (optionally `attention_mask`) — not
  precomputed masks/rope, since `prepare_inputs` builds those internally each time it's called.
  Get `bypass_adapted_forward` right when implementing this: it controls both which attribute gets
  swapped and which object `forward` ultimately calls.

  **The structural blocker this used to imply is now resolved at the model level.** With
  `generator.model` set to `QcGemma4ForCausalLM` (the wrapper, not the bare decoder) and
  `bypass_adapted_forward=True`, `_set_generator_runtime_model` swaps only `generator.model.model`
  — the nested `QcGemma4TextModel` — leaving the wrapper's own `.embed_tokens`/`.rotary_emb`/
  `.config` (all still the real, already-reauthored `Qc*` modules, reached via `model.model.*` in
  `Gemma4TextGenerator.prepare_inputs`) untouched and reachable for the rest of calibration. This
  is the same swap-one-level-down convention `QcLlamaForCausalLM`/`QcQwen3ForCausalLM` rely on
  (see `recipes/defaults.py`'s own docs: *"Enable bypass_adapted_forward to swap only the nested
  runtime module during calibration, preserving outer module metadata (e.g., rotary_emb) for RoPE
  pre-computation"*) — safe here for the same reason it's safe there: RoPE carries no calibrated
  weights, only a config-derived buffer, on either `QcGemma4TextRotaryEmbedding` or Llama/Qwen3's
  equivalents. Before `QcGemma4ForCausalLM` existed, `generator.model` was the bare decoder itself
  (no nested `.model`), so this swap would have destroyed the very object `prepare_inputs` needs
  `.embed_tokens`/`.rotary_emb` from — implementing `_create_quantsim`/`_compute_encodings` is
  still separate follow-up work, but it no longer has this blocker to work around.

---

## 8. Code map

| Symbol | File | Explained in |
|---|---|---|
| `quantize_model` | `quantize.py` | §2 (real generator through `_prepare_model`) |
| `Gemma4TextGenerator` | `generator.py` | §2 |
| `LPBQQuantizer._prepare_model` | qairt `aimet_quantizer.py` | §2, §7 |
| `LPBQQuantizer._prepare_inputs` / `_create_quantsim` / `_compute_encodings` | qairt `aimet_mixin.py` / `lpbq_quantizer.py` | §7 (deferred) |
| `QcGemma4TextModel.forward` cache guard | `reauthoring.py` | §4, `NANOV4_TO_QAIRT.md` §7.1 |
| Protobuf downgrade | venv (`qairt_env_nano`) | §5 |
