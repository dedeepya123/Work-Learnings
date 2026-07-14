QAIRT Dev, also known as qairt-dev, is the Python API component of Qualcomm AI Runtime. It provides a Python interface for executing ML models on QAIRT runtimes
It mirrors select capabilities and extends the features of existing QAIRT command line tools, while also providing an intuitive pythonic API for easy integration into ML workflows.

# Features

## ONNX Model Optimization
   - To prepare and transform ONNX models for LLM deployment before conversion
   - Includes MHA→SHA conversion, sequence/context length (AR/CL) rewriting, model splitting, and custom graph transformations.
 ## Framework Model Conversion
 - Convert ONNX, Pytorch (1.x), TFLite framework models into DLC
 - Includes support for quantization and application of quantization encodings generated from AIMET.
## Compilation
- Perform Ahead-of-Time (AOT) compilation on QAIRT backends to generate optimized binaries.
- Supports compilation on HTP, HTP MCP and AIC backends.
## Model Execution

 - Execute models on python-native targets via Pybind wrappers on QAIRT APIs
 - Execute models on other targets (e.g android) via helper APIs that abstract platform-specific details
## Model Analysis
- Generate profiling reports on all supported backends
- Generate Op Trace and Qualcomm Hexagon Analysis Summary (QHAS) reports on HTP

## Gen AI Model Building and Execution

- Convert, optimize, and compile Gen AI models for on-device inference using the Gen AI Builder API with a single build() call.
- Configure model transformations, compilation options, LoRA adapters, and speculative decoding. See Configuring the Gen AI Builder for the full configuration reference.
- Perform text generation and obtain metrics via Generative AI Inference Engine (Genie 2)
- Construct Gen AI applications natively in python using simplified python bindings on Genie APIs.

Pipeline 

 Pipeline API provides orchestraction based system for Model Optimization workflows

 There are 3 layers
+----------------------------------------------------------------------+
|                      Pipeline Orchestrator                            |
|                                                                       |
|  LLMPipeline: from_pretrained() -> construct() -> generate/evaluate  |
|  Configured by: YAML Recipe                                           |
+----------------------------------+-----------------------------------+
                                   |
                                   | runs
                                   v
+----------------------------------------------------------------------+
|                           Stages                                      |
|                                                                       |
|  model_loader --> quantization_opt --> quantization --> genai_builder |
|  (thin wrappers around building blocks)                               |
+----+-----------------+--------------------+------------------+--------+
     |                 |                    |                  |
     | wraps           | wraps              | wraps            | wraps
     v                 v                    v                  v
+-----------+   +---------------+   +---------------+   +-----------+
| Model     |   | Quantization  |   | Quantization  |   | GenAI     |
| Loading   |   | Optimization  |   | Recipes       |   | Builder   |
|           |   | Techniques    |   |               |   |           |
| - Loader  |   |               |   | - LPBQ        |   | - Build   |
| - Re-     |   | - SpinQuant   |   | - GPTAQ       |   | - Export  |
|   author  |   | - SeqMSE      |   | - HF Quant    |   | - Run     |
| - Adapt   |   | - AdaScale    |   |               |   |           |
+-----------+   +---------------+   +---------------+   +-----------+
     Building Blocks (standalone, usable without the pipeline)

There are two most common ways to use the pipeline: through the HuggingFace interface (Level 1) and through the pipeline’s Python API with a YAML recipe (Level 2).

## HuggingFace interface:

- The most familiar interface for HuggingFace users. Pass a quantization_config to QcAutoModelForCausalLM.from_pretrained() and the HF quantizer lifecycle handles reauthoring, quantization, and export automatically.
 
  
