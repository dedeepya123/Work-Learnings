# Model Preparation — Why It Exists, What It Does, How Gemma4 Goes Through It

Continuation of: HF architecture → QAIRT adaptation (static-graph rewrite) → KV cache →
masking → RoPE. This document picks up exactly where those left off: we now have a
PyTorch model whose *architecture* has been rewritten to be static-graph-friendly
(`model_qc`). Model Preparation is the stage that actually **performs** that conversion
into a real static graph, mechanically — not just "designed to be convertible," but
actually converted.

Two levels covered, deliberately kept separate: **(A) the API internals** — what
`qti.aisw.preparer_api.prepare_model` actually does, mechanically, independent of any one
model (traced through the installed SDK under `pythonic_api/qairt/.../preparer_api/`), and
**(B) the orchestration** — how NanoV4's `qmodel.py`/`qmodule_builder.py` actually invokes
it, caches its output, and threads it into calibration/quantization for Gemma4
specifically.

---

## 1. Why does Model Preparation exist?

Start from a fact you already know cold from the adaptation work: **HTP needs a static
graph** — every tensor shape fixed, every op resolved, before any real input arrives.

Here's the part that's easy to skip past: **making the *architecture* static-graph-friendly
(the adaptation work) is necessary but not sufficient.** `model_qc` — the adapted
PyTorch model — is still, mechanically, a live PyTorch `nn.Module`. Running its
`forward()` still executes real Python control flow, real dynamic op dispatch, real
eager-mode tensor operations. Nothing has actually been "frozen" yet. The adaptations
made it *possible* to trace this model into a fixed graph without hitting a dynamic-shape
wall — they didn't perform that tracing.

**Model Preparation is the step that actually performs the freeze.** It takes the live
PyTorch model, runs it once with concrete example inputs, and produces a **new artifact** —
a graph with every shape baked in, every op represented in a hardware-portable
intermediate representation, ready for the next stages (quantization calibration,
conversion, compilation) to operate on *that* representation instead of on live PyTorch
execution.

State it as the one-sentence architectural principle, the same way you'd say it in a
review: *"Adaptation changes what the model computes and how; Preparation changes what
form the model exists in — from an executable program to a frozen graph artifact."*

```
model_qc  (adapted, but still a LIVE, dynamically-executing PyTorch program)
     │
     │   <-- Model Preparation happens HERE
     ▼
model_mpp (a NEW artifact: fixed-shape graph, re-expressed as a fresh PyTorch module
            whose forward() now just replays a frozen op sequence, 1:1 with an IR graph)
```

---

## 2. Input model vs. prepared model

| | `model_qc` (input) | `model_mpp` (output) |
|---|---|---|
| What it is | A live, hand-written PyTorch `nn.Module` (`Gemma4ForConditionalGeneration`, adapted classes) | A **freshly generated** PyTorch `nn.Module` — its `.py` source file is literally written to disk by the preparation tool, not hand-authored |
| How its `forward()` behaves | Executes real Python: `if self.config.layer_types[i] == "sliding_attention": ...`, calls into `shared_kv_states` dict lookups, branches | Replays a fixed sequence of ops, one-to-one with a frozen intermediate-representation (IR) graph — no Python branching left, because whichever branch was taken during tracing *is* the only branch that exists now |
| Where shapes live | Determined per-call, from whatever `dummy_input` shapes are passed at call time (Python is fine with that) | **Baked into the graph** — the exact shapes used *during preparation* become permanent; calling this model with different shapes is unsupported/wrong |
| Submodule structure | Rich: `self.model.layers[i].self_attn.q_proj`, named, inspectable, matches the original architecture | **Flattened.** `keep_original_model_structure=False` (confirmed in NanoV4's actual call, `qmodule_builder.py:100-119`) — the emitted model's structure mirrors the *IR graph's* op sequence, not the original class hierarchy. This is a real, consequential choice, not a detail — you lose the original module tree. |
| What it's good for | Reading, debugging, editing, architecture experimentation | Quantization calibration (state 3), and eventually conversion to a compiled on-device binary — a stable target for tools *downstream* of PyTorch |

**The one-line mental model:** `model_qc` is "the architecture, expressed as code you'd
read." `model_mpp` is "the architecture, expressed as a frozen computation graph that
happens to still be wrapped in a PyTorch `nn.Module` shell for convenience."

---

## 3. Internal transformations performed — what `prepare_model` actually does, step by step

This is the API-internals level — traced directly from
`qti/aisw/preparer_api/model_preparer.py` in the installed SDK
(`pythonic_api/qairt/2.48.40.260702/lib/python/qti/aisw/preparer_api/model_preparer.py`).
Four real stages happen inside one function call:

```
prepare_model(model, dummy_input, input_names, output_names, converter_args, ...)
     │
     │ STAGE 1 — torch2onnx: real torch.onnx.export() under the hood
     ▼
   <path>/<filename>.onnx        (an intermediate, THROWAWAY onnx file — see §4 for why "throwaway")
     │
     │ STAGE 2 — ONNX -> IR graph: runs the QAIRT converter + optimizer stack
     │           (QAIRTConverter -> QAIRTOptimizer -> QAIRTSerializer -> .dlc -> reload as IrGraph)
     │           This is real graph optimization: shape inference, op fusion, lowering
     │           into QNN's internal IR representation — the SAME converter machinery
     │           the standalone `qairt-converter` CLI tool uses.
     ▼
   IrGraph  (in-memory intermediate representation, hardware-portable, no Python left)
     │
     │ STAGE 3 — IR graph -> new PyTorch model ("the emitter" / "Preparer Pro")
     │           TorchEmitterAndConfigGenerator().prepare_model(...) walks the IR graph
     │           op-by-op and GENERATES a brand-new torch.nn.Module whose forward()
     │           replays that exact op sequence.
     ▼
   <path>/<filename>.py            (generated PyTorch source — the "prepared model" code)
   <path>/<filename>.safetensors   (weights + LWQ metadata, extracted from the IR graph)
     │
     │ STAGE 4 — load the emitted model back into memory
     ▼
   returned nn.Module   (if return_prepare_model=True)
```

Notice this directly answers "does it do ONNX export, graph optimization, op
fusion/lowering, shape inference" — **yes to all four**, and they happen in that specific
order, inside one call. Nothing about this list is optional or approximate — every one of
these four stages is a real, separate piece of machinery (the SDK's converter, its
optimizer, its serializer, its emitter), chained together automatically by `prepare_model`
so the caller only sees "PyTorch model in → PyTorch model out."

**Why "MPP" — and why this matters for how to talk about it.** The tool that does Stage 3
(the emitter) is called, verbatim inside the SDK's own code and comments, **"Model
Preparer Pro"** — `torch_emitter.py`'s own class docstring: *"User interface class for
Preparer Pro API"*; a logger named `"model preparer pro"` in `torch_utils.py`. **MPP =
"Model Preparer Pro."** NanoV4 engineers use "MPP" informally as shorthand for "the
prepared model" / "the preparation stage" generally — but the acronym's actual origin is
the SDK tool's own name, not a NanoV4-invented term. Worth having this exact answer ready,
since "what does MPP stand for" is a very natural thing to be asked and "I don't know, we
just call it that" is a weaker answer than being able to name the actual tool.

**Why the ONNX from Stage 1 is described as "throwaway":** it exists purely as the input
format the converter (Stage 2) expects — nothing downstream in NanoV4's pipeline ever
reads this particular ONNX file again. A **second, unrelated** ONNX file gets produced
much later, from the *quantized* model (§4) — don't conflate the two; they have the same
file extension and similar names but completely different roles and different points of
origin in the pipeline.

---

## 4. Call flow and artifacts generated — NanoV4's actual orchestration

This is the orchestration level — how Gemma4 specifically gets pushed through the
mechanism from §3. Three states, two transitions, owned across two classes:

```
Gemma4Context (qmodel.py)                    per-modality *Builder (qmodule_builder.py,
                                              e.g. TextModelBuilder for the text decoder)
─────────────────────────                    ──────────────────────────────────────────
create_qc_model()
  _patched_gemma4_classes()
  Gemma4ForConditionalGeneration
      .from_pretrained(...)
  self.model_qc = model            ────────▶  builder.model_qc = ctx.get_qc_text_module()
        [STATE 1]                                    │
                                                       │ copy.deepcopy(model_qc)   <- work on a COPY,
                                                       │                              never mutate state 1
                                                       ▼
                                              create_mpp_model(model_inputs)
                                                  sorted_inputs = align_inputs_with_forward(...)
                                                  converter_args = self._build_converter_args(...)
                                                  _load_or_create_mpp(model_qc_copy, ...)
                                                        │
                                                        │  IF Model.py already exists on disk:
                                                        │     load_torch_model_using_safetensors(...)  <- SKIP prepare_model entirely
                                                        │  ELSE:
                                                        ▼
                                                  _prepare_model(model_qc_copy, inputs,
                                                       model_name='Model', filename='Model',
                                                       path=prepare_path,
                                                       input_names=..., output_names=...,
                                                       onnx_export_args={"opset_version": 20},
                                                       keep_original_model_structure=False,
                                                       converter_args=converter_args,
                                                       order_inputs=True, order_outputs=True,
                                                       skipped_optimizers=[...],
                                                       return_prepare_model=True)
                                                        │
                                                        ▼  (§3's 4-stage internal pipeline runs here)
                                              prepared_model  (a fresh nn.Module)
                                              self.model_mpp = self._make_mpp_runner(prepared_model)
                                                        [STATE 2]
                                                       │
                                                       ▼
                                              create_qs_model(model_inputs, htp_config_file)
                                                  prepared_model = self.model_mpp.model   <- unwrap
                                                  quantsim = QuantizationSimModel(
                                                      model=prepared_model,
                                                      quant_scheme=post_training_tf,
                                                      dummy_input=..., default_param_bw=8,
                                                      in_place=True,           <- MUTATES prepared_model directly
                                                      config_file=htp_config_file)
                                                  self._modify_quantization_scheme(...)  <- mixed precision
                                                  self.model_qs = quantsim
                                                  self.model_mpp = None       <- drop the now-stale cached MPP
                                                        [STATE 3]
                                                       │
                                                       ▼
                                              calibrate(...)   <- AIMET compute_encodings, real calibration data
                                              export_qs_model(...)
                                                  model_qs.export(onnx_dir, 'Model', inputs, ...)
                                                        │
                                                        ▼
                                              <output>/text/base/onnx/Model.onnx          <- the REAL onnx
                                              <output>/text/base/onnx/Model_torch.encodings  <- quantization encodings
```

**Artifacts, concretely, in order of appearance:**
1. `<prepare_path>/Model.onnx` — throwaway, internal to `prepare_model`, never read again (§3)
2. `<prepare_path>/Model.py` + `Model.safetensors` — the prepared model's generated source + weights. **Also doubles as a cache**: if these already exist on disk, `_load_or_create_mpp` skips calling `prepare_model` entirely and just reloads them — preparation is expensive, so it's memoized to disk, not just in-memory.
3. `<output>/.../onnx/Model.onnx` — the *real*, downstream ONNX, exported from the **quantized** model (state 3), after calibration. This is what feeds Stage C/D of the overall pipeline (LoRA baking, ARN→ARX rewrite, per `CLAUDE.md`'s stage table) — a completely different file, from a completely different point in the pipeline, than artifact #1.
4. `Model_torch.encodings` — the calibrated quantization scale/zero-point values, sidecar to artifact #3.

**One detail worth flagging explicitly in a review, because it's a real subtlety, not a
minor implementation note:** `QuantizationSimModel(..., in_place=True)` mutates the *same
object* `self.model_mpp.model` was pointing at. So immediately after creating `model_qs`,
the code sets `self.model_mpp = None` — the cached "clean" prepared model is no longer
trustworthy (it's been mutated into a quantized one in-place), so it's discarded, and
`_load_or_create_mpp`'s disk-cache path (artifact #2) becomes the way to get a fresh,
un-quantized MPP again later if needed.

---

## 5. Before/After view of the model

**Before (model_qc):**
```python
class QcGemma4TextModel(Gemma4TextModel):
    def forward(self, input_ids, attention_mask=None, ..., **kwargs):
        ...
        for i, decoder_layer in enumerate(self.layers[:num_layers_to_run]):
            if self.config.layer_types[i] == "sliding_attention":
                cache_position = swa_cache_position
                attention_mask = swa_attention_mask
                ...
            else:
                cache_position = global_cache_position
                ...
            hidden_states = decoder_layer(hidden_states, ..., attention_mask=attention_mask, ...)
```
A real Python `for` loop, a real Python `if`, indexing into `self.layers` (a real
`nn.ModuleList` with named submodules), reading `self.config` on every call.

**After (model_mpp), conceptually** (the actual generated `.py` is auto-written, not
hand-authored, but this is what it structurally looks like):
```python
class Model(nn.Module):   # generated name, NOT "Gemma4TextModel"
    def __init__(self):
        self.op_1 = ...   # flattened, IR-graph-derived submodules — NOT self.model.layers[i].self_attn...
        self.op_2 = ...
        ...
    def forward(self, input_ids, attention_mask, past_key_0_in, past_value_0_in, ...):
        x1 = self.op_1(input_ids)
        x2 = self.op_2(x1, attention_mask)
        ...                                    # NO if/for — every op that ran during
        return logits, past_key_0_out, ...      # tracing is now a fixed, linear op sequence
```
No loop. No `if`. No `self.config` lookups. No `self.layers[i]` indexing. Every one of the
35 decoder layers' worth of ops has been **unrolled** into one flat sequence — because
during Stage 1's `torch.onnx.export`, the trace recorded exactly what ops executed for the
one concrete input it was given, and everything from there (Stages 2-3) operates on that
already-unrolled, already-concretized graph.

**This is the most important "aha" of the whole document, worth stating explicitly:**
*the `for i, decoder_layer in enumerate(...)` loop in `model_qc` does not survive into
`model_mpp` as a loop at all.* It becomes 35 copies of the same op pattern, laid out
linearly in the graph, because tracing captures *what happened*, not *the code that
describes what could happen*. The `if layer_types[i] == "sliding_attention"` branch
likewise doesn't survive as a branch — for each of the 35 positions, whichever branch was
actually taken during tracing (determined by the real `config.layer_types` list) is
permanently baked in at that position. This is the literal mechanical meaning of "static
graph": not a metaphor, but "every dynamic Python construct has been resolved to one fixed
outcome, and that outcome is now all that exists."

---

## 6. Static graph conversion and runtime readiness

Connect this back to the adaptation work directly: recall that adaptation's job was to
make sure that **when** this trace-and-freeze happens, the *right* thing gets baked in —
plain concat-cache instead of a class-split cache, masks with fixed shapes, RoPE without
wasteful duplicated ops. Model Preparation is the mechanism that performs the actual
freezing; adaptation is what made sure the *result* of that freezing is correct and
hardware-friendly. Neither one is sufficient alone:
- Preparation **without** adaptation: you'd trace the *vanilla* dynamic-shape/class-split
  logic — you'd get a static graph, but one hardwired to whatever shapes/branches your one
  example call happened to produce, likely with an inefficient/wrong-for-HTP op
  representation (e.g. the sliding-window class split baked in wrong, or `rotate_half`'s
  concat-negate baked in as real, unnecessary graph ops).
- Adaptation **without** preparation: you'd have correct, hardware-friendly *code*, but
  it's still a live, executing PyTorch program — not yet a frozen artifact anything
  downstream (quantizer, converter, compiler) can consume as a fixed graph.

**"Runtime readiness," concretely, means:** the graph now has named, fixed-shape inputs
and outputs suitable for the tools that come next. This is where the KV-cache "wiring" you
were told about in `CLAUDE.md` becomes a literal, inspectable fact rather than a phrase:
the logical PyTorch-level `past_key_values` (a list of tuples) gets flattened, at the
ONNX/graph boundary, into individually-named tensors — `past_key_{i}_in` /
`past_value_{i}_in` (inputs) and `past_key_{i}_out` / `past_value_{i}_out` (outputs), one
pair per real (non-KV-shared) layer. For Gemma4 (35 layers, 20 KV-shared → 15 real
layers), that's **15 pairs of named KV tensors** on the graph boundary, not one opaque
"cache object." This is exactly what "static graph, KV-cache wired up" means as a concrete
artifact — confirmed independently by `GENAI_BUILDER_STAGE.md`'s own inspection of a real
compiled graph's I/O listing.

**One thing preparation deliberately does NOT do:** it doesn't implement the
*recurrence* — the actual "feed step N's `past_key_i_out` back in as step N+1's
`past_key_i_in`" loop. It only guarantees the graph exposes these as first-class named
tensors so that recurrence *can* be wired externally, by the runtime driving inference.
Worth stating this boundary explicitly in a review — it avoids a common
misunderstanding ("does prepare_model make the model autoregressive?" — no, it makes the
model's I/O *shape* suitable for an external driver to make it autoregressive).

---

## 7. How this generalizes to any AI model

Nothing in §3's four-stage pipeline (export → convert/optimize → emit → reload) is
Gemma4-specific, or even LLM-specific. Any PyTorch model headed for a fixed-shape hardware
target goes through the same mechanical steps:

```
ANY torch.nn.Module + concrete example input
     │
     │ torch.onnx.export (trace-based; records exactly what ran for THAT input)
     ▼
ONNX graph (shapes now concrete, ops now graph nodes)
     │
     │ hardware-target-specific converter + optimizer (fusion, lowering, shape inference)
     ▼
IR graph (hardware-portable, no Python/framework runtime needed to describe it)
     │
     │ (optional, if you want a PyTorch object back) emitter/re-generation step
     ▼
A new, frozen "prepared" model
```

What differs model-to-model is only **which shapes get baked in, and how many named
I/O tensors the graph exposes** — a plain image classifier has one input tensor, one
output tensor, no recurrence at all; an LLM has dozens of named KV-cache tensors because
of the autoregressive recurrence. The *mechanism* (trace once, freeze, re-express) is
identical regardless.

This is also why the earlier framework document's three adaptation categories
(shape/staticness, op-representation, interface/orchestration) generalize: **any** model
being pushed through this same preparation mechanism will hit the same three questions —
what needs a fixed shape, which ops need re-expressing for the target, what needs to move
outside `forward()` — because the *preparation mechanism itself* (trace-once-freeze) is
what forces those questions, not anything unique to transformers or to Gemma4.

---

## 8. What this specifically means for Gemma4

Pulling together everything from the cache/mask/RoPE deep-dives, here's what actually gets
frozen into `model_mpp` for this specific model:

- **35 decoder layers' worth of ops, unrolled linearly** — no `for` loop survives; each
  layer's attention+MLP+norms become their own fixed slice of the graph.
- **The `layer_types[i] == "sliding_attention"` branch, resolved per-position** — 30 of
  the 35 positions get the sliding-attention op pattern baked in, 5 get the
  full-attention pattern (per Gemma4's 5:1 alternation), permanently, at trace time.
- **KV-sharing topology baked in structurally** — the 20 KV-shared layers never had
  `k_proj`/`v_proj` weights to begin with (established in the architecture notes), so
  there's nothing for tracing to even unroll there; the 15 real layers each get one
  `past_key_i_in`/`past_value_i_in` pair wired into the graph boundary.
- **The RoPE `ApplyRopeSingle` complex-multiply, unrolled per layer** — the four named
  `MulModule` instances (from the RoPE adaptation notes) become four real, individually
  distinguishable graph nodes per layer that uses them — exactly the property that made
  wrapping them as named submodules worthwhile in the first place: they survive
  preparation as separately-identifiable nodes, ready for the *next* stage (AIMET
  QuantSim, state 3) to assign each one its own calibrated quantization encoding.
- **The mask's fixed `[1,1,arn,gcl]` / `[1,1,arn,lcl]` shape becomes the graph's literal
  input tensor shape** — not a design intention anymore, but a concrete, compiled-in
  dimension in `Model.py`'s generated `forward()` signature.

---

## 9. Mapping the code flow — pythonic_api API internals vs. NanoV4 orchestration, side by side

|  | pythonic_api / SDK internals (§3) | NanoV4 orchestration (§4) |
|---|---|---|
| Entry point | `qti.aisw.preparer_api.prepare_model(...)` | `qlib/qmodule_builder.py`'s `BuilderBase.create_mpp_model()` |
| What it's given | one PyTorch model + one concrete dummy input + names | `copy.deepcopy(self.model_qc)` + `sorted_inputs` captured via `StepCaptureObserver` from a real forward pass |
| What it does with it | export→convert→optimize→emit, internally, as one atomic operation | nothing — treats `prepare_model` as an opaque black box, just supplies correctly-shaped/named arguments and unwraps the result |
| What comes back | an `nn.Module` (if `return_prepare_model=True`) | wrapped immediately in a `Traced*Runner` (e.g. `TracedTextRunner`) and stored as `self.model_mpp` |
| Caching | none — every call re-runs the full 4-stage pipeline | `qmodule_builder.py` checks `Model.py` existence first; skips `prepare_model` entirely on a cache hit |
| Relationship to quantization | none — `prepare_model` has no knowledge quantization exists | `create_qs_model` immediately consumes `self.model_mpp.model` to build a `QuantizationSimModel`; `self.model_mpp` is explicitly invalidated afterward since QuantSim mutates it in place |

The clean way to say this in review: **the SDK owns the mechanism (how a model gets
frozen); NanoV4 owns the orchestration (when to freeze, what to freeze, where to cache the
result, and how the frozen artifact feeds into everything downstream).** Reading
`qmodule_builder.py` without knowing §3's internals looks like "we call a function and get
a model back" — knowing §3 is what turns that into "we call a function that runs a
four-stage export/convert/optimize/emit pipeline and get back a structurally different
kind of model."

---

## 10. Is this stage needed for all GenAI models? Do non-GenAI models skip it?

**Every model targeting a fixed-shape/compiled hardware backend needs *some* version of
this mechanism** — trace, freeze, re-express — regardless of whether it's "GenAI" or not.
A plain CNN image classifier being compiled for the same HTP backend goes through
`torch.onnx.export` → convert/optimize → (possibly) emit, exactly like Gemma4 does. So the
*mechanism itself* is not GenAI-specific — it's compiled-inference-specific.

**What *is* specific to GenAI/autoregressive models is the *complexity of what gets
exposed on the graph boundary*, and the *supporting call plumbing* (§9's `converter_args`
built specifically for KV-cache semantics via `llm_build_preparer_converter_args`):
- A non-recurrent model (classifier, single-pass encoder) has one clean call: one
  input, one output, no state carried between calls, no cache to wire up. Preparation for
  it is comparatively trivial — export, optimize, done, no `converter_args`
  cache-tensor-layout configuration needed at all.
- A GenAI/autoregressive model has to expose **explicit recurrent state** as named
  I/O (the `past_key_i_in`/`_out` pairs) so an *external* driver can close the loop
  across calls — this is real, GenAI-specific extra work happening *within* the same
  preparation mechanism (the `converter_args` telling the converter "these tensors are
  KV-cache slots with this layout," per `air/genai_lib/llm/model_preparation_utils.py`'s
  `llm_build_preparer_converter_args`).

So the precise answer: **the preparation *stage* is universal for any
compiled-inference target; what's GenAI-specific is the *shape of the problem being
prepared for* — recurrent state needing explicit graph-boundary wiring — not the
existence of a separate preparation mechanism.** You wouldn't skip preparation for a
non-GenAI model; you'd just have a much simpler `converter_args`/input-structure to hand
it, because there's no cache recurrence to describe.

---

## Summary — the one paragraph you'd give in a review

*"Adaptation rewrote the model's architecture so it's *capable* of being frozen into a
static graph correctly. Model Preparation is the stage that actually performs that
freeze — internally, `prepare_model` traces the adapted PyTorch model through a real
`torch.onnx.export`, runs the resulting ONNX through QAIRT's converter/optimizer to get a
hardware-portable IR graph, then re-emits that IR graph as a brand-new, flattened PyTorch
module. Every `for` loop and every `if` branch in the original adapted code is gone in the
output — resolved once, permanently, into a fixed linear op sequence, with the KV-cache
recurrence exposed as 15 pairs of individually-named input/output tensors instead of one
opaque cache object. NanoV4 treats this as a cacheable, expensive black-box step — it
checks for an existing prepared model on disk before re-running it, and immediately feeds
the result into AIMET's QuantizationSimModel for calibration, at which point the prepared
model is mutated in place and its cached reference is deliberately dropped."*
