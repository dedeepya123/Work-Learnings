``` text
 High-level flow: HF model → on-device binary
   
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 1: HF MODEL (float, dynamic shapes, HF's own KV cache growth)     │
  │  Gemma4ForCausalLM (transformers)                                        │
  │  checkpoint.py: load_checkpoint() — loads weights, tokenizer, config     │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                  │  adapt.py: reauthor_model()
                                  │  (class-swap: Gemma4* -> QcGemma4*)
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 2: QC-ADAPTED MODEL (still float, but static-graph-shaped)        │
  │  reauthoring.py: QcGemma4ForCausalLM / QcGemma4TextModel / ...           │
  │  - fixed KV buffers (scatter-write, not torch.cat growth)                │
  │  - host-precomputed masks/RoPE passed in as real forward() args          │
  │  - Linear -> Conv2d (linear_to_conv.py)                                  │
  │  - TextModelQc wrapper: flattens output to a plain tuple for tracing     │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                  │  quantize.py: quantize_model()
                                  │  generator.py: Gemma4TextGenerator (drives prepare_llm())
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 3: MODEL PREPARATION (PyTorch -> ONNX -> QAIRT IR -> flat PyTorch)│
  │  qairt's prepare_llm() / model_preparer                                  │
  │  - torch.onnx.export traces ONE fixed shape (sequence_length=128)        │
  │  - converter+optimizer lower to QAIRT IR, emitter rebuilds flat PyTorch  │
  │  output: prepared_model (used for parity-checking, not deployed)         │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                  │  quantize.py: LPBQQuantizer.quantize()
                                  │  AIMET QuantizationSimModel + calibration
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 4: QUANTIZED EXPORT (still float VALUES, quant PARAMS attached)   │
  │  quantsim_output_v2_encodings/                                          │
  │    model.onnx        — graph structure, float32 weights (in model.data) │
  │    model.encodings   — scale/offset/bitwidth per tensor (separate JSON) │
  │  Nothing is integer yet here — encodings just describe HOW to quantize  │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                  │  builder/container.py, builder/prefix_decode.py
                                  │  qairt.gen_ai_api.GenAIBuilderFactory
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 5: GenAI BUILDER — transform / convert / compile                 │
  │                                                                           │
  │  (a) TRANSFORM  — resize_for_arn() + skip_ar_cl_conversion=True          │
  │      AR/CL resize (our custom per_layer_inputs seed rule)                │
  │      MHA -> SHA rewrite, model split (embed / decoder-layers / lm_head) │
  │                                                                           │
  │  (b) CONVERT    — per split, per ARN                                    │
  │      ONNX -> QAIRT IR -> optimize (RMSNorm fusion etc.) -> float .dlc   │
  │      apply model.encodings -> quantized .dlc (int8 weights, int16 acts) │
  │                                                                           │
  │  (c) COMPILE    — per split, both ARNs together (weight_sharing=True)   │
  │      qnn-context-binary-generator: compose graphs into ONE context      │
  │      -> split_N/model.bin  (2 graphs each: model_ar8_*, model_ar1_*)    │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                  │  container.save(path)
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 6: LLMContainer (on disk, self-contained, on-device-runnable)     │
  │  llm_container_prefix_decode/                                            │
  │  (a) TRANSFORM  — resize_for_arn() + skip_ar_cl_conversion=True          │
  │      AR/CL resize (our custom per_layer_inputs seed rule)                │
  │      MHA -> SHA rewrite, model split (embed / decoder-layers / lm_head) │
  │                                                                           │
  │  (b) CONVERT    — per split, per ARN                                    │
  │      ONNX -> QAIRT IR -> optimize (RMSNorm fusion etc.) -> float .dlc   │
  │      apply model.encodings -> quantized .dlc (int8 weights, int16 acts) │
  │                                                                           │
  │  (c) COMPILE    — per split, both ARNs together (weight_sharing=True)   │
  │      qnn-context-binary-generator: compose graphs into ONE context      │
  │      -> split_N/model.bin  (2 graphs each: model_ar8_*, model_ar1_*)    │
  └───────────────────────────────┬───────────────────────────────────────────┘
                                  │  container.save(path)
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐

  The three fundamentally different "shapes" the model passes through

  1. Semantic shape (Levels 1-2): what the model means — attention, KV cache, RoPE, as PyTorch
  ops.
  2. Graph shape (Levels 3-5): what the model is, structurally — ONNX/QAIRT IR nodes, fixed
  tensor shapes, single-head attention, split subgraphs.
                                  │  runtime/executor.py OR runtime/net_run.py
                                  ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEVEL 7: RUNTIME (two independent paths, different scope)              │
  │                                                                           │
  │  executor.py -> T2TExecutor -> Genie dialog (native or on-device adb)   │
  │    full generation loop: tokenize, mask/RoPE, prefill+decode, sample    │
  │    ✗ blocked: Genie's single kv-dim can't represent Gemma4's 2 sizes    │
  │                                                                           │
  │  net_run.py -> qnn-net-run --retrieve_context (bypasses Genie)          │
  │    runs ONE compiled split directly against real per-tensor shapes      │
  │    proves the binary itself is valid HTP hardware, not a text output    │
  └─────────────────────────────────────────────────────────────────────────┘

  The three fundamentally different "shapes" the model passes through
  
  1. Semantic shape (Levels 1-2): what the model means — attention, KV cache, RoPE, as PyTorch
  ops.
  2. Graph shape (Levels 3-5): what the model is, structurally — ONNX/QAIRT IR nodes, fixed
  tensor shapes, single-head attention, split subgraphs.
  3. Deployment shape (Levels 6-7): what actually runs — compiled DSP kernels in a context
  binary, wrapped in a config Genie's runtime knows how to drive (or doesn't, in our case).

  1. Semantic shape (Levels 1-2): what the model means — attention, KV cache, RoPE, as PyTorch
  ops.
  2. Graph shape (Levels 3-5): what the model is, structurally — ONNX/QAIRT IR nodes, fixed
  tensor shapes, single-head attention, split subgraphs.
  3. Deployment shape (Levels 6-7): what actually runs — compiled DSP kernels in a context
  binary, wrapped in a config Genie's runtime knows how to drive (or doesn't, in our case).

  Where NanoV4 does the same thing differently (recap from earlier)

  Only Level 3 onward differs mechanically — NanoV4 does value-based literal substitution
  (qchange_hardcoding.py) for AR/CL resize instead of axis-denotation semantics, and
  hand-extracts prefix/decode by output-set instead of generic residual-add splitting. From
  Level 3's convert/compile onward, both pipelines call the identical underlying SDK tools
  (qairt-converter, qairt-quantizer, qnn-context-binary-generator) — the difference is only in
  whose Python wrapper invokes them.

  Where we're actually stuck

  Level 7, and only there — everything through Level 6 is confirmed, artifact-verified, and
  correct. The break is a genuine mismatch between Gemma4's architecture (two KV head-dims) and
  Genie's dialog schema (one kv-dim field), not a defect anywhere in Levels 1-6.

```
