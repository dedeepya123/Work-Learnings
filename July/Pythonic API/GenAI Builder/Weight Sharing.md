## Why Do AR=1 and AR=128 Have Identical Weights?

This is the key insight:

AR=1  graph (decode):    Q_proj weight = [[0.23, -0.14, ...]]  ← SAME!
AR=128 graph (prefill):  Q_proj weight = [[0.23, -0.14, ...]]  ← SAME!

Why? Because AR conversion ONLY changes:
  ✅ Input/output tensor SHAPES
  ✅ Attention mask dimensions
  ✅ KV cache dimensions
  
  Does NOT change weights — weights are identical!
  Weights are just numbers — they don't care about seq_len!

Weight Sharing OFF vs ON — Visualized
weight_sharing = False
split_0/model.bin  (per split, per AR — SEPARATE .bin files!)
│
├── ar1_cl4096_2_of_4.bin
│   ├── Graph section (AR=1  ops)    ← different
│   └── Weight blob  ~1.5GB          ← DUPLICATE copy!
│
└── ar128_cl4096_2_of_4.bin
    ├── Graph section (AR=128 ops)   ← different
    └── Weight blob  ~1.5GB          ← DUPLICATE copy!

Total storage = 1.5 + 1.5 = 3GB per split!  wasteful
weight_sharing = True (default in production!)

split_0/model.bin  (ONE .bin for ALL ARs of this split!)
│
├── Graph section AR=1   ops  ← AR=1  graph topology
├── Graph section AR=128 ops  ← AR=128 graph topology
│
└── Shared Weight Blob  ~1.5GB  ← ONE copy shared by BOTH! ✅

Total storage = 1.5GB only! ← half the size! ✅

## What Happens Internally During Compilation
``` text
builder.build()
        │
        ├── transform()  ← AR=1 ONNX + AR=128 ONNX (separate graphs)
        │
        ├── convert()    ← ONNX → DLC for each AR separately
        │                   ar1_cl4096_2_of_4.dlc
        │                   ar128_cl4096_2_of_4.dlc
        │
        └── compile()    ← KEY STEP!
                            CompilerModes.WEIGHT_SHARING
                            │
                            ├── takes ar1_cl4096_2_of_4.dlc
                            ├── takes ar128_cl4096_2_of_4.dlc
                            │
                            └── QNN HTP detects identical weights
                                → stores weights ONCE
                                → stores 2 graph sections
                                → outputs ONE model.bin
``` 

## What's Inside ONE Weight-Shared .bin
models/split_0/model.bin  (weight shared!)
┌────────────────────────────────────────────┐
│  Context Binary Header                     │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Graph 1: AR=1  (decode graph)       │  │
│  │  - op sequence for seq_len=1         │  │
│  │  - tensor shape refs for AR=1        │  │
│  │  - weight POINTERS → shared blob ↓  │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Graph 2: AR=128 (prefill graph)     │  │
│  │  - op sequence for seq_len=128       │  │
│  │  - tensor shape refs for AR=128      │  │
│  │  - weight POINTERS → shared blob ↓  │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Shared Weight Blob  (~1.5GB)        │  │
│  │  - q_proj INT8 weights for all heads │  │
│  │  - k_proj INT8 weights               │  │
│  │  - v_proj INT8 weights               │  │
│  │  - ffn weights etc.                  │  │
│  │  ← BOTH graphs point here! ✅        │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘

## At Runtime — How HTP Switches Between Graphs
Genie runtime:
  
  Prefill phase → select Graph 2 (AR=128) from .bin
    → load weights from shared blob
    → process 128 tokens at once 
  
  Decode phase → select Graph 1 (AR=1) from .bin
    → load weights from SAME shared blob
    → process 1 token at a time 

Weights loaded ONCE into DDR → used by both graphs! 

``` text
"graph ar128_cl8192_scl512_1_of_3 is loaded"    ← prefill
"Successfully copied shared weights."            ← shared blob 
"graph ar1_cl8192_1_of_3 is loaded"             ← decode
← same weights already in DDR — no reload!
```

<img width="605" height="178" alt="image" src="https://github.com/user-attachments/assets/4965652d-b3fc-4316-9c76-3820512baf86" />

**Weight Sharing is WITHIN a Single Split, Between AR Graphs**

## Weight Sharing scope:

AR=1   graph of split_2  ←→  AR=128 graph of split_2  (SAME split, different AR)
AR=1   graph of split_3  ←→  AR=128 graph of split_3  (SAME split, different AR)

split_2 weights  ←→  split_3 weights  (DIFFERENT splits — NO sharing!)

## Visualized — Your 4 Splits
``` text
Split 1/4 → embedding_table.bin      ← only 1 graph, no AR sharing needed
                                      
Split 2/4 → model.bin                ← weight shared .bin!
  ├── Graph: ar1_cl4096_2_of_4        (decode)
  ├── Graph: ar128_cl4096_2_of_4      (prefill)
  └── Shared Weight Blob ~1.5GB    ← ONE copy, BOTH graphs point here

Split 3/4 → model.bin                ← weight shared .bin!
  ├── Graph: ar1_cl4096_3_of_4        (decode)
  ├── Graph: ar128_cl4096_3_of_4      (prefill)
  └── Shared Weight Blob ~1.5GB    ← ONE copy, BOTH graphs point here

Split 4/4 → model.bin                ← only 1 graph (lm_head decode only)

```
## Why NOT Across Splits?

Each split has COMPLETELY DIFFERENT weights:
  split_2 = transformer layers 0-13   ← q_proj, k_proj layers 0-13
  split_3 = transformer layers 14-27  ← q_proj, k_proj layers 14-27

  These weights are DIFFERENT numbers!
  → Nothing to share across splits! 
  
  Weight sharing only works when
  two graphs have IDENTICAL weights
  → AR=1 and AR=128 of SAME split
    have 100% identical weights 

## Storage Impact — Your LLaMA 3.2 3B
WITHOUT weight sharing:
  split_2: ar1_2_of_4.bin  ~1.5GB
         + ar128_2_of_4.bin ~1.5GB  = 3GB
  split_3: ar1_3_of_4.bin  ~1.5GB
         + ar128_3_of_4.bin ~1.5GB  = 3GB
  Total = 6GB 

WITH weight sharing (your case):
  split_2: model.bin = 1.5GB ✅  (AR=1 + AR=128 graphs + 1 shared blob)
  split_3: model.bin = 1.5GB ✅
  Total = 3GB ✅  ← 50% storage saving!

## Clean Summary

## Simple rule: 
Weight sharing = same layers, different input shapes (AR). 
Each split has its own unique layers → no cross-split sharing possible! (from memory: weight_sharing=True config, 2026-07-20)

<img width="616" height="122" alt="image" src="https://github.com/user-attachments/assets/d8da0420-b9b0-4b38-a516-5879f32b1c03" />
