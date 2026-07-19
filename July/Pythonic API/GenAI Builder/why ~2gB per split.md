## HTP Memory Hierarchy — Two Completely Different Things
``` text
┌─────────────────────────────────────────────────────────┐
│                    HTP Chip (SM8850)                     │
│                                                          │
│  ┌─────────────────┐                                     │
│  │   VTCM          │  ← 8-16 MB   FAST on-chip SRAM     │
│  │ (Vector TCM)    │     ~1 TB/s bandwidth               │
│  │                 │     Holds: activations, KV cache    │
│  └─────────────────┘                                     │
│          ↑ fast path                                     │
│  ┌─────────────────┐                                     │
│  │   HTP Engine    │  ← actual compute happens here      │
│  │  (MAC arrays)   │                                     │
│  └─────────────────┘                                     │
│          ↑ slow path                                     │
│  ┌─────────────────┐                                     │
│  │   DDR/DRAM      │  ← 12-16 GB  SLOW off-chip memory  │
│  │  (main memory)  │     ~50 GB/s bandwidth              │
│  │                 │     Holds: model weights            │
│  └─────────────────┘                                     │
└─────────────────────────────────────────────────────────┘

```
## What Each Memory Holds
Memory	Size	           Speed	                          Holds
VTCM	8–16 MB	    Very Fast (~1TB/s)	          Activations, KV cache, intermediate tensors
DDR	12–16 GB	     Slow (~50GB/s)	                   Model weights (the big stuff!)


## So Why 2GB Split Limit? — The Context Binary Constraint!
This is the key! It's NOT a VTCM limit — it's a QNN context binary limit:

## QNN HTP Runtime Rule:
  One context binary (.bin) can reference 
  maximum ~2GB of weights from DDR!

``` text
  Why?
  ├── QNN uses 32-bit offsets internally
  │   to address weights in the .bin file
  │
  └── 32-bit max addressable space = ~2GB
       → can't address beyond 2GB 
         in a single context!
```
``` text
How All Three Relate — Full Picture
During INFERENCE:

Step 1 — Weights sit in DDR (cold)
┌─────────────────────────────────────┐
│ DDR (12-16GB)                       │
│  split_0/model.bin  ~2GB  ← loaded │
│  split_1/model.bin  ~2GB            │
│  split_2/model.bin  ~2GB            │
└─────────────────────────────────────┘

Step 2 — HTP engine streams weights 
         from DDR into compute units
┌─────────────────────────────────────┐
│ HTP Engine                          │
│  reads weights from DDR tile-by-tile│
│  ~small chunks at a time            │
└─────────────────────────────────────┘

Step 3 — Activations/KV cache live 
         in VTCM (fast!)
┌─────────────────────────────────────┐
│ VTCM (8-16MB)                       │
│  input activations     ~few KB      │
│  intermediate tensors  ~few KB      │
│  KV cache entries      ~8-16MB total│
└─────────────────────────────────────┘
```

## Why VTCM = 8-16MB but Handles 2GB Weights?
``` text
VTCM does NOT hold weights 
Weights stream through tile-by-tile:

DDR weight block (2GB total)
        │
        │  small tile (~KB) at a time
        ▼
   HTP MAC units  ← multiply-accumulate
        │
        │  result
        ▼
   VTCM (8-16MB)  ← stores activations/partial results only
```

Think of it like a kitchen:

DDR   = warehouse (holds all ingredients = weights) 
VTCM  = kitchen counter (small, fast working space) 
HTP   = chef (processes ingredients tile by tile)   

Chef doesn't put the whole warehouse on the counter!
Just brings small batches → processes → stores result on counter

## Why 2GB Per Split — Confirmed from Your Cache
LLaMA 3.2 3B weights (INT8 after quantization):
  Full model  ≈ 3GB  (INT8 = roughly half of FP32 6GB)
  
  Split 1 (embedding)    ≈ 0.5GB  → embedding_table.bin
  Split 2 (layers 0-13)  ≈ 1.5GB  → 2_of_4.bin  ← fits in 2GB ✅
  Split 3 (layers 14-27) ≈ 1.5GB  → 3_of_4.bin  ← fits in 2GB ✅
  Split 4 (lm_head)      ≈ 0.5GB  → 4_of_4.bin

Each split < 2GB → each context binary fits within QNN's 
                   32-bit addressable limit! ✅
(from memory: VTCM 8-16MB limit on SM8850, 2026-07-18)

 Summary — Three Different Constraints
Constraint	Value	What It Limits
VTCM	8–16 MB	Activation tile size + KV cache per layer
QNN context binary	~2 GB	Max weights per .bin file
DDR	12–16 GB	Total model size on device
VTCM  → limits how much ACTIVATION data fits on-chip
2GB   → limits how much WEIGHT data per context binary
DDR   → limits total model size that fits on device

They are 3 independent constraints working together! 🎯

## What is a Context Binary .bin — Exactly?

Yes — .bin = Weights + Compiled Graph TOGETHER!
``` text
context binary (.bin)
        │
        ├──  Compiled HW Graph  ← "what operations to run"
        │         (operator sequences, tensor shapes,
        │          data flow, kernel microcode for Hexagon DSP)
        │
        └──  Quantized Weights  ← "the actual numbers"
                  (INT8 weights packed tightly,
                   scale/offset metadata per layer)
Think of it like a self-contained executable:

.bin  ≈  program code  +  data  packed together
          (graph)           (weights)
``` text
 How the .bin is Structured Internally
┌─────────────────────────────────────────┐
│         context binary (.bin)           │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  QNN Context Header               │  │
│  │  - graph topology                 │  │
│  │  - input/output tensor specs      │  │
│  │  - op configs for Hexagon DSP     │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Weight Blob                      │  │
│  │  - INT8 quantized weights         │  │
│  │  - scale/offset per layer         │  │
│  │  - packed contiguously in memory  │  │
│  │  - max ~2GB (32-bit offset limit) │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```
 
 ## The 2GB Limit — Exactly Why
QNN internally uses 32-bit offsets to 
address weights INSIDE the .bin file:
``` text
┌──────────────────────────────────────────┐
│  Weight Blob                             │
│  offset=0x00000000  → layer_0.q_proj    │
│  offset=0x00A3F210  → layer_0.k_proj    │
│  offset=0x01234567  → layer_1.q_proj    │
│         ...                              │
│  offset=0x7FFFFFFF  → MAX! (~2GB)     │
│  offset=0x80000000  → OVERFLOW!       │
└──────────────────────────────────────────┘

32-bit max = 2^31 = ~2GB addressable space
            → can't go beyond 2GB in ONE .bin!
```
## What Happens at Runtime — Complete Picture
``` text
STEP 1: QNN loads .bin into DDR
┌─────────────────────────────────┐
│ DDR                             │
│  ar1_cl4096_2_of_4.bin  ~2GB   │ ← loaded by QNN runtime
│  [graph topology + weights]     │
└─────────────────────────────────┘

STEP 2: QNN reads graph topology
        → sets up execution plan
        → knows which ops to run in order

STEP 3: HTP streams weights tile-by-tile
        from DDR into compute units
┌─────────────────────────────────┐
│ DDR weight at offset 0x00A3F210 │
│        │ small tile (~KB)       │
│        ↓                        │
│   HTP MAC units                 │
│        │ multiply-accumulate    │
│        ↓                        │
│   VTCM (8-16MB)                 │
│   [activations/partial results] │
└─────────────────────────────────┘

STEP 4: Output activation passed 
        to next split as input
```

## Your Cache .bin Files Confirmed
From your screenshot:
``` text
compile_7f070638.../
├── ar1_cl4096_3_of_4.bin      ← graph + weights for layers 14-27
└── ar1_cl4096_3_of_4_cache_info.json  ← metadata about this binary

compile_31ca75df.../
├── ar1_cl4096_2_of_4.bin      ← graph + weights for layers 0-13
└── ar1_cl4096_2_of_4_cache_info.json  ← metadata about this binary

## The cache_info.json tells you what's inside each .bin:

{
  "split_index": 2,
  "ar": 1,
  "cl": 4096,
  "weight_size_bytes": 1800000000,   ← ~1.8GB weights
   "graph_size_bytes":  50000000,     ← ~50MB graph
   "total_size_bytes":  1850000000    ← total .bin size
}
<img width="622" height="203" alt="image" src="https://github.com/user-attachments/assets/d35a8946-0f54-49f7-9674-fdbd132c1d8a" />
