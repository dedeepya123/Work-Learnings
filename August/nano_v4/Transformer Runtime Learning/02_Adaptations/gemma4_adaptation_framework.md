# HF → HTP-Adapted Model: A Generic Reasoning Framework (Gemma4 as the worked example)

Personal notes: not "what Gemma4's adaptations are" (see the other two adaptation-reference
files) but *why* adaptation is necessary at all, and a generic framework for reasoning about
**any** model being adapted for a fixed-shape hardware backend (HTP here, but the same
reasoning applies to `torch.compile`, TensorRT, Core ML, XLA — anything that compiles a
graph ahead-of-time instead of running eager Python).

---

## 1. The starting picture — what HF orchestrates

A vanilla HF causal LM (`Gemma4TextModel.forward`) takes `input_ids` and **internally
orchestrates everything else**:
- token embeddings
- position embeddings (RoPE `cos`/`sin`), computed from `position_ids`
- attention mask(s), computed from `attention_mask` + cache state
- cache creation (if missing) and update, via live Python calls into a `Cache` object

All of this orchestration is **dynamic**: shapes grow call to call, `Cache` is a live
Python object with methods, masks are built fresh every forward pass via real
`torch.arange`/`torch.cat` calls, branching on `config.layer_types` happens in Python
control flow, not in the graph.

## 2. Why that dynamism is incompatible with a compiled HTP graph

**Can you technically get *a* graph out of a dynamic model without adaptation?** Yes —
tracing an eager PyTorch model can still produce *an* ONNX graph. What actually breaks
without adaptation is everything **downstream** of that: a static-shape compiler (HTP's
context-binary compilation) needs every tensor shape fixed ahead of time; per-op
quantization calibration needs every op to be a stable, individually-identifiable graph
node (not an inlined Python `*`); and efficient hardware execution needs ops the backend
actually has fast fixed-point kernels for (no `-inf`, no runtime transpose-inside-matmul).

So the precise claim is: **you can get a graph without adaptation, but not an efficient,
compilable, quantizable, fixed-shape hardware binary from it.** Adaptation exists to close
that gap — it's a compilation-target problem, not an "the model can't run" problem.

## 3. The core principle

> **Adaptation = keep the computed *mathematics* identical, change the *representation* —
> so the graph that results is static-shaped, individually-quantizable, and built from ops
> the target hardware runs efficiently.**

Every adaptation we've traced so far is an instance of this:
- RoPE: same complex-rotation math, `rotate_half`(concat+negate) → explicit real/imag
  split + named elementwise multiplies
- Masking: same "which tokens may attend to which" outcome, additive `-inf` →
  `where(mask==0, x, min+mask_neg)` (fixed-point-arithmetic-safe)
- KV cache write: same "store this step's K/V" outcome, unbounded `torch.cat` →
  scatter-write into a pre-allocated fixed-size buffer

None of these change *what the model computes*. All of them change *how that computation
is expressed as graph ops*.

## 4. The compile-time / runtime split — the actual generic mechanism

This is the mechanism underneath nearly every adaptation, so it's worth stating as its own
rule, generically, before looking at Gemma4 specifics:

> **For a static-shape compiled graph: every tensor's *shape* must be decided at compile
> time (from config/hyperparameters known ahead of execution); only a tensor's *contents*
> may depend on runtime inputs.**

Compile-time-known quantities, generically, for any autoregressive transformer:
- **ARN** ("prefix batch size" / chunk size — how many tokens this compiled graph
  processes per invocation; confirmed in `qargparse.py`: `'-arn', '--arn', ..., help='Prefix
  batch size (ARN)'`) — fixed per compiled graph variant (you may compile *multiple*
  graphs for different ARNs — e.g. a big-ARN "prefill" graph and an ARN=1 "decode" graph —
  but each individual graph has one fixed ARN).
- **CL** (context length — total cache capacity the graph is compiled to support) — fixed
  per compiled graph.
- **`config.layer_types`** — fixed at model-definition time, known before any input
  arrives.
- **`config.num_hidden_layers`, `num_kv_shared_layers`, `head_dim`, `global_head_dim`,
  `sliding_window`** — all fixed config values.

What is **not** known at compile time, and must be a real runtime input instead:
- **Which absolute position this call starts at** (`cache_index` — where in the KV$ buffer
  to write; supplied by the caller/runtime each step)
- **The actual token ids** and their embeddings' values
- **Actual attention-mask *content*** (which specific positions are valid vs. padding/future)

### Position embeddings — worked through your reasoning, corrected

Your reasoning: *"We know ARN → from this we can know position_ids. From position_ids we
can compute position_embeddings."* — Correct, with one precision fix: ARN alone gives you
the **shape** of `position_ids` (a length-ARN vector) and the **relative offsets within a
call** (`[0, 1, ..., ARN-1]`), not the absolute starting position. The absolute starting
position is `cache_index` (or the vanilla equivalent, `past_seen_tokens`) — a runtime
scalar. So: `position_ids = cache_index + arange(ARN)` — shape fixed at compile time
(`arange(ARN)` is the `cache_tensor` buffer from the adaptation notes), *value* depends on
the one runtime scalar `cache_index`.

Given `position_ids`, `position_embeddings = rotary_emb(position_ids)` — deterministic,
pure function, no learned parameters, so it genuinely can be precomputed once shape is
fixed. **`config.layer_types` known at compile time → build separate `(cos,sin)` tables
per layer type up front** — exactly right, and this is *why* Gemma4's model code passes
`(cos,sin)` in *per layer type*, not per layer: only 2 distinct tables exist regardless of
`num_hidden_layers`, computed from the 2 distinct `rope_theta`/`head_dim` configs.

### Masks — worked through your reasoning, corrected

Your question: *"Can we create masks at compile time? What info is needed?"* — Split this
the same way as everything else:
- **Mask *shape*** `[B, 1, Q, KV]` — `Q = ARN` (fixed), `KV` = fixed too, because for a
  fixed-capacity pre-allocated cache, `KV` is either `CL` (the full buffer, full-attention
  layers) or `sliding_window` (sliding layers) — **not** "however many tokens seen so far,"
  which is what vanilla HF computes dynamically. So mask *shape* is fully compile-time
  determined by `(ARN, CL, sliding_window, layer_type)`.
- **Mask *content*** (which of those `KV` positions are actually `0` vs `-inf`/masked-out)
  — depends on `cache_position`, a runtime value. Same split as everything else: shape at
  compile time, content at runtime.

This is also the direct answer to *"connecting causal mask and cache position — why
compile time":* the mask's shape is tied to the cache's shape (both sized off
`ARN`/`CL`/`sliding_window`), so once the cache's allocation is fixed, the mask's
allocation is automatically fixed too — they're not two independent compile-time
decisions, they're the *same* decision (buffer sizing) viewed from two different
tensors.

### Cache — the correction to your framing

Your statement: *"cache creation must happen at compile time."* More precise: **cache
*allocation* (shape, dtype, which layers even get a slot) is a compile-time decision; cache
*contents* are a runtime concern, updated every call.** This split is exactly what separates
the two adaptation strategies you've already compared:
- `reauthoring.py`'s pipeline: cache shape starts small and *grows* via `torch.cat` at
  the PyTorch-tracing stage — fine for tracing/QuantSim, but the *actual* fixed-buffer
  scatter behavior is injected by qairt's `KVCacheMapping` monkeypatch, not by
  `reauthoring.py` itself.
- `qadaptation.py`: `DynamicLayer_adapted.update()` explicitly checks whether
  `cache_position` falls inside already-allocated space (scatter) or past it (concat,
  only during trace-time graph construction) — a direct, visible expression of the
  "shape is fixed, only contents move" rule.

Which layers get a cache slot at all is also decided purely from config, at compile time:
KV-shared layers (from the `is_kv_shared_layer`/`first_kv_shared_layer_idx` logic) never
call `update()`, so they need **zero** allocated cache — a compile-time fact derivable
from `num_hidden_layers`/`num_kv_shared_layers` alone, no runtime input required.

## 5. Generalizing beyond Gemma4 — the categories

Your instinct to classify into buckets is the right move, and everything traced so far
actually resolves into three (not two) categories — the third is what your "interface
adaptation" was reaching for, made precise:

| Category | Question it answers | Gemma4 examples |
|---|---|---|
| **Shape/staticness adaptations** | "What must be a fixed size, decided before any input arrives?" | Cache pre-allocation sized by `(CL, num_kv_heads, head_dim)`; `cache_tensor` ramp buffer replacing dynamic `torch.arange`; ARN-fixed sequence length; mask shape tied to `(ARN, CL/sliding_window)` |
| **Op-representation adaptations** | "Given the shapes are fixed, which *ops* need to change so the hardware/quantizer can handle them efficiently?" | RoPE via explicit real/imag complex-multiply (no `rotate_half` concat+negate); masked-softmax / `mask_neg` instead of additive `-inf`; transposed-key storage to avoid an on-target transpose; named `MulModule`/`Add`/`Matmul` submodules so AIMET can assign a stable per-op quantization encoding |
| **Interface/orchestration adaptations** | "Which computations should move from *inside* `forward()` (dynamic, Python-driven) to *outside* it (precomputed by the caller, passed in as plain tensors)?" | Masks, position embeddings, and cache-position tensors built by the caller (`generator.py`'s `prepare_inputs`, or `qadaptation.py`'s `Gemma4ForCausalLM.forward` before calling `self.model(...)`) instead of inside `Gemma4TextModel.forward` |

This is the generic lens: for *any* model going through this kind of adaptation, ask the
same three questions in order — **(1) what needs a fixed size, (2) which ops need
re-expressing for the target hardware/quantizer, (3) what should move from inside the
model's forward pass to outside it** — and you'll land on the same shape of catalogue
regardless of which specific architecture you're adapting.

## 6. Where we go next

Next: walk the actual adaptation catalogue for Gemma4 (already covered piecewise in the
QAIRT-adaptation reference notes — RoPE, masking, cache, mask/position externalization) but
now explicitly tagging each one against this 3-category framework, one at a time, starting
with whichever category you want to dig into first.
