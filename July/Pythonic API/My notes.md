QAIRT Dev, also known as qairt-dev, is the Python API component of Qualcomm AI Runtime. It provides a Python interface for executing ML models on QAIRT runtimes

# Features

## ONNX Model Optimization
   - To prepare and transform ONNX models for LLM deployment before conversion
   - Includes MHA→SHA conversion, sequence/context length (AR/CL) rewriting, model splitting, and custom graph transformations.
 ## Framework Model Conversion
 Conversion and Compilation
 ## Model Execution
 
 ## Model Ananlysis

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
 
  
