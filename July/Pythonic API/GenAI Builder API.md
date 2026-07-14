# GenAI Builder 
The Gen AI Builder is a Python API that automates step 2 of the typical LLM deployment workflow: 
- it takes a quantized ONNX model and compiles it into a GenAIContainer ready for on-device inference.
- The guides below cover configuration options, advanced features, and migration from notebook-based workflows

- The Gen AI Builder is a Python API that automates step 2 of the typical three-step LLM deployment workflow.
- (quantize → compile & package → deploy). It takes a quantized ONNX model (produced by step 1 of the QNN model preparation notebooks)
  and produces a GenAIContainer ready for on-device inference with a single build() call.

  NOTEBOOK / CLI PIPELINE (manual)
============================================================

  AR/CL        Split       MHA2SHA      Convert      Quantize     LoRA       Context
  Convert  --> ONNX    --> Transform --> to DLC   --> DLC      --> Import --> Binary
                                                                              Gen

  7 stages x (AR x CL x splits) = hundreds of CLI invocations


GEN AI BUILDER API (automated)
============================================================

  Factory        Configure         builder.build()
  .create()  --> set_targets    --> +---------------------------+
                 native_kv         | All 7 stages automated    |
  Auto-detects   multi_graph       | in a single call          |
  model arch     lora_config       +---------------------------+
                                             |
                                             v
                                       GenAIContainer
                                   (ready for deployment)

## Work Flow
When you call build(), the builder automates the following stages that are manual in notebooks:

- AR/CL conversion – generates ONNX models for each AR x CL combination
- ONNX splitting – partitions the model into N splits
- MHA2SHA transformation – converts multi-head to single-head attention per split
- ONNX to DLC conversion – with quantization overrides from encodings
- DLC quantization – act_bitwidth=16, bias_bitwidth=32
- LoRA graph building and import – when lora_config is set
- Context binary generation – with weight sharing and native KV format config

These 3 API calls replace the entire notebook
