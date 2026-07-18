## What is AR (Auto-Regression)?
AR = how many tokens are processed in one forward pass

## Prefill phase (processing your prompt):

**"What is Qualcomm Snapdragon?"**
 → 7 tokens fed in TOGETHER at once
 → AR = 7 (or whatever seq_len is)

## Decode phase (generating response):
"Qualcomm" → next token → next token...
 → 1 token generated at a time
 → AR = 1

 ## Why Do We Need AR Conversion?
The Core Problem — HTP Needs FIXED Shapes!
**CPU/GPU**:   Dynamic shapes — can handle any input size
           "Give me 7 tokens? Sure!"
           "Give me 1 token?  Sure!"

**HTP/DSP**:   Static shapes ONLY — must know exact shape
           at compile time — cannot change at runtime!
So you need separate compiled graphs for each AR value:

ar=7   → one compiled graph  (for prefill)
ar=1   → another compiled graph (for decode)

## What arcl_* Stage Actually Does
ONNX model (dynamic shapes)
        │
        ▼
AR/CL Conversion Stage
        │
        ├──→ ar1_cl4096  ONNX   ← decode graph  (1 token at a time)
        └──→ ar7_cl4096  ONNX   ← prefill graph (7 tokens at once)

Each combination = separate fixed-shape ONNX 

From your cache you can see exactly this:

arcl_1966a0f6...   ← ar/cl combo 1
arcl_e1b52001...   ← ar/cl combo 2

``` text
📐 CL (Context Length) Role
|<————————————— CL = 4096 (total KV cache) ——————————————>|
|<—— past KV (already computed) ——>|<—— AR (new tokens) ——>|
```

ar1_cl4096  means:
  - Process 1 new token at a time
  - KV cache can hold up to 4096 total tokens

## Your Quantization 
"Weights still in FP32, converted to int during computation via scale/offset — correct?"

Almost correct — but more nuanced! Here's exactly what happens:

📦 What's in Your output_artifacts/
output_artifacts/
├── model.safetensors     ← weights stored as FP32  ✅ (you're right!)
└── model.encodings       ← scale + offset per layer ✅
The .encodings file looks like this per layer:


{
  "layer.0.self_attn.q_proj": {
    "weight": {
      "scale": 0.00392,    ← how to convert FP32 → INT
      "offset": 0,         ← zero point
      "bitwidth": 8        ← target precision
    }
  }
}
⚙️ What Happens at Each Stage
Stage 1 — output_artifacts/
  Weights: FP32 safetensors
  Encodings: scale/offset JSON
  ← "recipe" for quantization, not yet quantized!

Stage 2 — ONNX → DLC conversion (convert_* in cache)
  qairt-converter applies encodings to ONNX:
  FP32 weight × (1/scale) → rounded to INT8
  ← weights NOW converted to integers in DLC! ✅

Stage 3 — DLC → .bin compilation (compile_* in cache)
  HTP runtime packs quantized INT weights
  + scale/offset metadata into binary
  ← ready for hardware execution!

Stage 4 — At RUNTIME on HTP
  INT weight × scale → FP16/FP32 for compute  ← dequantize on-the-fly
  activations kept in INT16
  biases kept in INT32

``` text

Complete Quantization Flow Visualized
┌──────────────────────────────────────────────────────────┐
│  LPBQ Quantization (your quantize.py)                    │
│                                                          │
│  FP32 weights  ──calibration──→  scale + offset         │
│                                  (stored in .encodings)  │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  DLC Conversion (convert_* stage in builder)             │
│                                                          │
│  FP32 weight + scale/offset  ──→  INT8 weight in DLC    │
│  formula: int_val = round(fp32_val / scale) + offset    │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│  HTP Runtime Execution                                   │
│                                                          │
│  INT8 weight × scale  ──→  FP16 for MAC operation       │
│  activations: INT16                                      │
│  biases:      INT32                                      │
└──────────────────────────────────────────────────────────┘
```
## Summary
Question	Answer
Why AR conversion?	HTP needs fixed shapes — one graph per AR×CL combo
Why multiple arcl_* folders?	Each AR×CL combination = separate ONNX graph
Weights in FP32 in artifacts?	✅ Yes — FP32 + .encodings together = "quantization recipe"
When converted to INT?	During DLC conversion stage (convert_*)
Runtime behavior?	INT weights dequantized on-the-fly to FP16 during compute
💡 So your .encodings file is not the quantized weights — it's the recipe (scale/offset) used to convert FP32 → INT during the DLC conversion stage! 



## What Happens with Default [1, 128]?
YES — 2 Separate ONNX Graphs Are Created!

"Today, GenAIBuilderHTP builds prefill and decode graphs from a single _transformation_config, parameterized by auto_regression_number: List[int] (e.g. [1, 128])"

auto_regression_number = [1, 128]   ← DEFAULT
        │
        ├──→ AR=1   graph  ← Decode  graph (1 token at a time)
        └──→ AR=128 graph  ← Prefill graph (128 tokens at once)
  
But Your Cache Shows ONLY ar1_cl4096_* — Why?
Looking at your cache screenshot:

arcl_1966a0f6...   ← AR=1  conversion  ✅
arcl_e1b520010...  ← AR=128 conversion ✅  (2 arcl folders = 2 ARs!)

compile_*/
  ar1_cl4096_2_of_4.bin   ← only AR=1 compiled!
  ar1_cl4096_3_of_4.bin   ← only AR=1 compiled!
You have 2 arcl_* folders → both AR=1 and AR=128 were converted ✅

BUT you only see ar1_cl4096_* binaries in compile! This is because your ONNX from LPBQ quantization was exported with AR=1 shape only — the AR=128 conversion likely fell back to AR=1 or failed silently (from memory: ONNX AR=1 only build, 2026-07-18)!

## The Complete AR/CL Conversion Flow Explained
``` text
Your ONNX (dynamic shapes OR fixed AR=1)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  AR/CL Conversion Stage (skip_ar_conversion=False)  │
│                                                      │
│  auto_regression_number = [1, 128]                  │
│                                                      │
│  Loop through each AR value:                        │
│  ├── AR=1:                                          │
│  │    Take ONNX → reshape all tensors for AR=1      │
│  │    attention_mask:  [1, 1, 1,  cl]   ✅          │
│  │    input_ids:       [1, 1]            ✅          │
│  │    past_key_in:     [1, h, d, cl-1]  ✅          │
│  │    → Save as ar1_cl4096.onnx                     │
│  │                                                   │
│  └── AR=128:                                        │
│       Take ONNX → reshape all tensors for AR=128    │
│       attention_mask:  [1, 1, 128, cl]  ✅          │
│       input_ids:       [1, 128]          ✅          │
│       past_key_in:     [1, h, d, cl-128]✅          │
│       → Save as ar128_cl4096.onnx                   │
└─────────────────────────────────────────────────────┘

```

AR=128 prefill  → processes 128 tokens in one shot
               → VERY fast prefill (batched computation)
               → 10752 toks/sec prefill (Qwen2.5-1.5B on SA8797)

AR=1 decode     → processes 1 token at a time
               → standard decode speed
               → 98 toks/sec decode

