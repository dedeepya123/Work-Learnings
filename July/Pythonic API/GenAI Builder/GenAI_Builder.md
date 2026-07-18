## What is GenAI Builder?
Think of GenAIBuilder as a fully automated assembly line that takes your raw/quantized HuggingFace LLM and produces device-ready deployable binaries in one builder.build() call!

Raw HF Model  ──→  GenAIBuilder.build()  ──→  Device-Ready Binaries
``` text
The Two Builder Types
GenAIBuilderFactory.create()
        │
        ├──→ GenAIBuilderHTP   ← For Qualcomm DSP/NPU (Android devices)
        │                         Input: quantized ONNX + encodings
        │
        └──→ GenAIBuilderCPU   ← For x86 Linux / CPU execution
                                  Input: raw HF model directly
```
## What Happens Inside builder.build() — Step by Step
Looking at your cache directory, each folder represents one pipeline stage:

``` text
cache/model/
│
├── arcl_*/          ← STAGE 1️: AR/CL Conversion
│
├── transform_*/     ← STAGE 2️: MHA2SHA Transformation  
│
├── convert_*/       ← STAGE 3️: ONNX → DLC Conversion
│
└── compile_*/       ← STAGE 4️: DLC → Context Binary (.bin)
    ├── ar1_cl4096_2_of_4.bin
    └── ar1_cl4096_3_of_4.bin
```
## 1️ AR/CL Conversion (arcl_* folders)
What:  Generates separate ONNX models for each 
       Auto-Regression × Context-Length combination

Why:   HTP hardware needs a FIXED input shape —
       it cannot handle dynamic shapes like CPU/GPU

Example:
  ar=1,  cl=4096  →  prefill ONNX  (processing prompt)
  ar=1,  cl=4096  →  decode  ONNX  (generating tokens)

## 2️ MHA2SHA Transformation (transform_* folders)
What:  Multi-Head Attention  →  Single-Head Attention
       per split

Why:   Hexagon HTP DSP cannot efficiently run
       multi-head attention natively.
       Single-head is hardware-optimal.

  Before:  [Q,K,V] × 32 heads fused together
  After:   [Q,K,V] × 1 head  per split  

## 3️ ONNX → DLC Conversion (convert_* folders)
What:  Converts ONNX model → Qualcomm DLC format
       (Deep Learning Container)

Why:   Qualcomm's runtime (SNPE/QNN) only 
       understands DLC — not ONNX directly

Also:  Applies your quantization encodings here!
       (.encodings file → quantization overrides)
       act_bitwidth=16, bias_bitwidth=32

## 4️ Context Binary Compilation (compile_* folders)
What:  DLC → .bin (serialized context binary)
       This is the FINAL deployable artifact!

Why:   Pre-compiles the graph for a SPECIFIC SoC
       (your SM8850 v81 DSP)

From your cache:
  ar1_cl4096_2_of_4.bin  ← split 2 compiled 
  ar1_cl4096_3_of_4.bin  ← split 3 compiled 
  
## What container.save() Produces
``` text 
serialized_output/
│
├── embedding_table.bin          ← Split 1 (embedding layer)
├── embedding_table_quant_param.json
│
├── models/
│   ├── split_0/model.bin        ← Split 2 (transformer block 1)
│   ├── split_1/model.bin        ← Split 3 (transformer block 2)
│   ├── split_2/model.bin        ← Split 4 (transformer block 3)
│   └── use_cases.json           ← AR/CL combinations supported
│
├── gen_ai_config.json           ← Genie runtime config
├── backend_extensions.json      ← HTP backend settings
├── metadata.json                ← model metadata
└── tokenizer.json               ← tokenizer

``` 
## Full API Mental Model
``` text
┌─────────────────────────────────────────────────────┐
│               GenAIBuilder API                       │
│                                                      │
│  INPUT                                               │
│  ├── HTP: quantized ONNX + .encodings               │
│  └── CPU: raw HuggingFace model                      │
│                                                      │
│  CONFIGURATION                                       │
│  ├── builder.set_targets()    ← target SoC (HTP)    │
│  ├── builder.weight_sharing   ← memory optimization │
│  └── builder.quantization_level ← CPU quant level   │
│                                                      │
│  EXECUTION                                           │
│  └── builder.build()  ← runs ALL 4 stages           │
│                                                      │
│  OUTPUT                                              │
│  └── container.save()  ← saves deployable binaries  │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│            LLMContainer + T2TExecutor                │
│                                                      │
│  LLMContainer.load()          ← load saved binaries │
│  container.get_executor(device) ← push to device    │
│  executor.generate(prompt)    ← run inference     │
└─────────────────────────────────────────────────────┘
```
##  Key Takeaways
<img width="611" height="186" alt="image" src="https://github.com/user-attachments/assets/1e59726e-572a-496f-8d14-f4a5df284099" />

The cache is just a build cache — if you re-run builder.build() with the same inputs, it reuses cached stages instead of recomputing from scratch, saving time!

