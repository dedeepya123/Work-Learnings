# MHA2SHA Transformation — Why, What & How Internally
## Why Do We Need MHA2SHA at All?

- As multi-head attention works with large tensors, running transformers on HTP with single NSP is difficult.
- Hence we split Multi-Head Attention into a series of Single Head Attention.
- The operations are mutually exclusive and hence work fine for HTP."

The Core Problem:
Standard MHA (Multi-Head Attention):
  LLaMA 3.2 3B = 32 heads
  
  ALL 32 heads computed TOGETHER as one big fused operation:
  Q: [1, 32, seq_len, head_dim]  ← all 32 heads in one tensor!
  K: [1, 32, seq_len, head_dim]
  V: [1, 32, seq_len, head_dim]
  
  QK matmul:  [1, 32, seq_len, seq_len]  ← HUGE tensor!
  
  At seq_len=4096:
  [1, 32, 4096, 4096] = 32 × 4096 × 4096 = 536M elements
  In FP16 = 1GB just for this ONE tensor! 
  
  VTCM = 8-16MB → CANNOT hold this! 

**The Solution — Split Into 32 Separate Single-Head Operations**

## AFTER MHA2SHA:

head_0:   Q[1,1,seq,dim] × K[1,1,seq,dim] → [1,1,seq,seq]   tiny!
head_1:   Q[1,1,seq,dim] × K[1,1,seq,dim] → [1,1,seq,seq]   tiny!
...
head_31:  Q[1,1,seq,dim] × K[1,1,seq,dim] → [1,1,seq,seq]   tiny!
          ↓
     Concat all 32 head outputs
          ↓
     Same result as MHA!  Mathematically equivalent!
Each single-head tensor = 1/32 of original size → fits in VTCM! 

## What Changes in the ONNX Graph — Internally
From official MHA2SHA Details and Usage Confluence:
``` text
BEFORE (MHA):
┌─────────────────────────────────────────────────┐
│  Q_proj (MatMul) → reshape [B,32,T,d]           │
│  K_proj (MatMul) → reshape [B,32,T,d]           │
│  V_proj (MatMul) → reshape [B,32,T,d]           │
│         ↓                                        │
│  QK_matmul: Q×K → [B,32,T,T]   ← HUGE!         │
│  Scale + Softmax → [B,32,T,T]                   │
│  QKV_matmul: attn×V → [B,32,T,d]               │
└─────────────────────────────────────────────────┘

AFTER (32× SHA):
┌─────────────────────────────────────────────────┐
│  For head_i in range(32):                        │
│    Q_i = Slice(Q_proj, head=i) → [B,1,T,d]     │
│    K_i = Slice(K_proj, head=i) → [B,1,T,d]     │
│    V_i = Slice(V_proj, head=i) → [B,1,T,d]     │
│    QK_i = Q_i × K_i → [B,1,T,T]  ← tiny!    │
│    attn_i = Softmax(QK_i)                       │
│    out_i = attn_i × V_i → [B,1,T,d]            │
│                                                  │
│  Concat(out_0,...,out_31) → [B,32,T,d]        │
└─────────────────────────────────────────────────┘
```
## How the Transformer Stage Does It — Confirmed from Your Cache

Your two transform_* folders = MHA2SHA applied per AR graph:

transform_2fdd62cb...   ← MHA2SHA for AR=1  graph 
transform_ec29bb0b...   ← MHA2SHA for AR=128 graph 

**Each split of each AR graph goes through this transformation:**

ar1_cl4096_2_of_4.onnx (14 transformer layers, each with MHA)
        │
        ▼ MHA2SHA transformer
        │
        ├── Layer 0:  MHA(32 heads) → 32 × SHA ops + Concat
        ├── Layer 1:  MHA(32 heads) → 32 × SHA ops + Concat
        │   ...
        └── Layer 13: MHA(32 heads) → 32 × SHA ops + Concat
        
ar1_cl4096_2_of_4_sha.onnx  ← output with SHA 
## What Also Changes — Encodings Propagation

MHA has ONE encoding per weight:
  q_proj.weight → scale: 0.00392

After MHA2SHA, need 32 SEPARATE encodings:
  head_0.q_proj.weight → scale: 0.00392
  head_1.q_proj.weight → scale: 0.00391
  ...
  head_31.q_proj.weight → scale: 0.00389

## mha2sha_optimizer automatically propagates
encodings from MHA → SHA 
new .encodings file generated alongside new .onnx!

Also: Other Transformations Run ALONGSIDE MHA2SHA

<img width="617" height="197" alt="image" src="https://github.com/user-attachments/assets/e9aece28-d4e7-473e-8778-34f095d2981b" />


<img width="572" height="158" alt="image" src="https://github.com/user-attachments/assets/1d6b2fca-f60a-439f-92e9-98273b3f353f" />

## Summary

- MHA2SHA is an HTP optimization that breaks fused multi-head attention into sequential single-head operations
- Needed when the intermediate QK attention tensor [BS, heads, seq, seq] is too large to fit in VTCM (8-16MB), which is always the case for large LLMs during prefill.
- Skip it only when heads are few, sequence is tiny, or the model has non-standard attention patterns."
