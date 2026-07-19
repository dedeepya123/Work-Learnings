## Why Split at All?

LLaMA 3.2 3B full model weights:
  FP32 = ~6GB
  INT8 = ~3GB  (after LPBQ quantization)

## Start With the Problem — Why Split at All?
LLaMA 3.2 3B full model weights:
  FP32 = ~6GB
  INT8 = ~3GB  (after LPBQ quantization)

QNN Rule: ONE context .bin = MAX 2GB
          (32-bit internal offset limit)

~3GB decoder weights → CANNOT fit in one .bin 

→ Must be split! 
2️ How Many Splits? — The Formula
num_splits = 3 + (model_params_in_GB // 2)

LLaMA 3.2 3B:
  model size ≈ 6GB (FP32)
  3 + (6 // 2) = 3 + 3 = 4 total splits 

The 3 base splits ALWAYS are:
  Base 1 → Embedding layer    (split_embedding=True)
  Base 2 → First decoder split (always need at least 1)
  Base 3 → LM Head            (split_lm_head=True)

+1 extra decoder split per 2GB of params:
  6GB // 2 = 3 → but 1 is base decoder
  → 2 extra decoder splits added
3️⃣ Your Cache — Mapped to the 4 Splits

Looking at your screenshot exactly:
``` text
cache/model/
│
├── arcl_1966a0f6...    ┐
├── arcl_e1b52001...    ┘← STAGE 1: AR/CL Conversion
│                            2 folders = AR=1 + AR=128
│                            Each AR → reshapes ONNX to fixed shapes
│
├── transform_2fdd62cb... ┐
├── transform_ec29bb0b... ┘← STAGE 2: MHA2SHA Transformation
│                            2 folders = one per AR graph
│                            Multi-Head → Single-Head per split
│
├── convert_2a7a83ff...  ┐
├── convert_30848569...  │
├── convert_ace5a228...  │← STAGE 3: ONNX → DLC Conversion
├── convert_c0b7abfd...  │   7 folders = splits × AR values
├── convert_d70146e4...  │   encodings applied here!
├── convert_dea8dd55...  │   INT8 weights packed into DLC
├── convert_fd03ed10...  ┘
│
├── compile_7f070638...  ┐
│   ├── ar1_cl4096_3_of_4.bin  ← Split 3: decoder layers 14-27
│   └── ar1_cl4096_3_of_4_cache_info.json
│
├── compile_31ca75df...  │← STAGE 4: DLC → Context Binary
│   ├── ar1_cl4096_2_of_4.bin  ← Split 2: decoder layers 0-13
│   └── ar1_cl4096_2_of_4_cache_info.json
│
└── compile_91f91a61...  ┘  ← Split 4: lm_head compiled
``` text

The 4 Splits — Full Anatomy
``` text
LLaMA 3.2 3B (28 transformer layers)
        │
        ▼  LLMSplitter (auto formula)
        │
┌───────────────────────────────────────────────┐
│  SPLIT 1/4 — Embedding                        │
│  ├── What:  token embedding Gather layer      │
│  ├── Size:  ~0.5GB                            │
│  └── Output: embedding_table.bin ✅           │
├───────────────────────────────────────────────┤
│  SPLIT 2/4 — Decoder Block 1                  │
│  ├── What:  transformer layers 0-13           │
│  ├── Size:  ~1.5GB (fits in 2GB limit ✅)     │
│  ├── Cache: ar1_cl4096_2_of_4.bin ✅          │
│  └── Final: models/split_0/model.bin ✅       │
├───────────────────────────────────────────────┤
│  SPLIT 3/4 — Decoder Block 2                  │
│  ├── What:  transformer layers 14-27          │
│  ├── Size:  ~1.5GB (fits in 2GB limit ✅)     │
│  ├── Cache: ar1_cl4096_3_of_4.bin ✅          │
│  └── Final: models/split_1/model.bin ✅       │
├───────────────────────────────────────────────┤
│  SPLIT 4/4 — LM Head                          │
│  ├── What:  final projection layer            │
│  ├── Size:  ~0.5GB                            │
│  └── Final: models/split_2/model.bin ✅       │
└───────────────────────────────────────────────┘
``` 
## What Each .bin Actually Contains
ar1_cl4096_2_of_4.bin
│
├──  Graph Section  (~50MB)
│     compiled HW op sequence for Hexagon DSP
│     "run attention → FFN → residual add..."
│     Hexagon DSP microcode instructions
│
└──  Weight Section  (~1.5GB)
      INT8 quantized weights packed contiguously
      scale + offset metadata per layer
      addressed via 32-bit offsets (max 2GB!)
## At Runtime — How Splits Execute Sequentially
Token Input
    ↓
embedding_table.bin   → looks up token embedding vector
    ↓  (embedding vector passed as input)
split_0/model.bin     → runs layers 0-13
    ↓  (hidden state passed as output→input)
split_1/model.bin     → runs layers 14-27
    ↓  (hidden state passed)
split_2/model.bin     → lm_head → logits
    ↓
Next Token Predicted! ✅

## Memory at runtime:
  - Each .bin loaded into DDR (~1.5GB each)
  - HTP streams weights DDR → compute units tile by tile
  - Activations/KV cache live in VTCM (8-16MB fast SRAM)

## Complete Mental Model Summary
Why split?        QNN 32-bit offset = 2GB max per .bin
How many?         3 + (model_GB // 2)  auto formula
LLaMA 3.2 3B?     4 splits (embed + 2 decoder + lm_head)
What is .bin?     compiled graph + INT8 weights together
VTCM role?        holds activations only (NOT weights!)
DDR role?         holds the .bin weight blobs at runtime
2GB per split?    each .bin weight section ≤ 2GB 

Your cache perfectly shows all 4 stages — 2 AR graphs created → 7 DLC conversions (splits × ARs) → 2 decoder .bin files compiled → combined into final serialized_output/ with embedding_table.bin + 3 model.bin files! 🎯 (from memory: LLaMA 3.2 3B 4-split config, 2026-07-19)
QNN Rule: ONE context .bin = MAX 2GB (32-bit internal offset limit)

2GB per split?    each .bin weight section ≤ 2GB ✅
💡 Your cache perfectly shows all 4 stages — 2 AR graphs created → 7 DLC conversions (splits × ARs) → 2 decoder .bin files compiled → combined into final serialized_output/ with embedding_table.bin + 3 model.bin files! 🎯 (from memory: LLaMA 3.2 3B 4-split config, 2026-07-19)
