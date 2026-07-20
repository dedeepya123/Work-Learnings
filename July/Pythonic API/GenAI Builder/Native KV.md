## What is Native KV Format?
It's about how Key-Value cache tensors are laid out in memory — the shape/layout of K and V tensors optimized specifically for HTP hardware!

## The Standard (Non-Native) KV Layout
``` text
Standard HuggingFace KV cache layout:
  Key:   [batch, num_heads, seq_len, head_dim]
         [  1,     32,       4096,    128   ]

  Meaning: 
    - 32 heads
    - each head has 4096 sequence positions
    - each position has 128 dimensional vector
    - stored HEAD-FIRST → head dimension is innermost
```

## Why is This Bad for HTP?

When doing attention computation:
  Q × K^T  ← needs to TRANSPOSE K!

  K:      [1, 32, 4096, 128]
  K^T:    [1, 32, 128,  4096]  ← transpose needed!

  Transposing at RUNTIME on HTP:
  → Requires extra memory movement
  → Data layout mismatch with HTP vector units
  → HTP vector engine prefers COLUMN-MAJOR access
  → SLOW! 

## Native KV Format = Transposed Layout Pre-Stored
``` text
Native KV format (transposed_key_cache=True):
  Key stored as:   [batch, num_heads, head_dim, seq_len]
                   [  1,     32,       128,     4096  ]
                   ← ALREADY TRANSPOSED at store time!

  Value stored as: [batch, num_heads, seq_len, head_dim]
                   [  1,     32,       4096,    128   ]
                   ← same as standard (V doesn't need transpose!)
```

### How This Helps at Inference Time
WITHOUT native KV (standard):

  TOKEN arrives:
  1. Compute new K vector [1, 32, 1, 128]
  2. Append to K cache   [1, 32, seq, 128]
  3. TRANSPOSE K cache   [1, 32, 128, seq]  ← extra op at runtime! ❌
  4. Q × K^T             ← attention compute
  5. VTCM holds transposed copy temporarily

WITH native KV (transposed_key_cache=True):

  TOKEN arrives:
  1. Compute new K vector [1, 32, 128, 1]   ← stored transposed immediately!
  2. Append to K cache   [1, 32, 128, seq]  ← already in right layout! ✅
  3. Q × K              ← NO transpose needed! direct compute!
  4. VTCM never needs transposed copy
##  Memory Layout Visualized
### Standard K cache (non-native):
``` text
┌─────────────────────────────────────────┐
│  head_0: [pos_0[d0,d1..d127],           │
│           pos_1[d0,d1..d127],           │
│           ...                           │
│           pos_4095[d0,d1..d127]]        │
│                                         │
│  To do Q×K^T → must transpose all this  │
│  across seq dimension → expensive!    │
└─────────────────────────────────────────┘

Native K cache (transposed, head_dim first):
┌─────────────────────────────────────────┐
│  head_0: [dim_0[pos_0, pos_1..pos_4095],│
│           dim_1[pos_0, pos_1..pos_4095],│
│           ...                           │
│           dim_127[pos_0..pos_4095]]     │
│                                         │
│  Q×K → direct matrix multiply! ✅      │
│  HTP vector units read contiguous cols  │
└─────────────────────────────────────────┘
```

## Where is it Set — Your Code

### In QcAutoConfig (model config level):
qc_config = QcAutoConfig.from_pretrained(
    model_path,
    model_config_overrides={
        "return_new_key_value_only": True,
        "transposed_key_cache": True,    #  HERE — tells model to output
                                          #    K in transposed layout
        "input_tokens_per_inference": 128,
    },
)

# In GenAI Builder (compilation level):
builder.native_kv_format = True          #  HERE — tells compiler to
                                          #    compile KV ops expecting
                                          #    transposed K layout
## Critical — Both Must Match!
``` text
MISMATCH case:
  qc_config: transposed_key_cache = True   ← model outputs K transposed
  builder:   native_kv_format = False      ← compiler expects standard layout
                    ↓
  Q × K → WRONG RESULT!  (shapes don't align!)
```
``` text
CORRECT case:
  qc_config: transposed_key_cache = True   
  builder:   native_kv_format = True       
             (both agree on layout!)
 How it Flows Through YOUR Pipeline
QUANTIZATION (quantize.py):
  transposed_key_cache=True in qc_config
  → ONNX exported with K cache shape [1, 32, 128, 4096] 
  → encodings generated for this layout
```

## CONVERSION (builder convert stage):
  qairt-converter sees K tensors in transposed layout
  → DLC stores K in [head_dim, seq] order 

## COMPILATION (builder compile stage):
  native_kv_format=True
  → qnn-context-binary-generator compiles attention
    ops KNOWING K is already transposed
  → NO runtime transpose op inserted! 
  → direct matmul kernel used 

RUNTIME (Genie on SM8850):
  K cache stored as [1, 32, 128, 4096]
  Q × K → direct HTP vector matmul 
  No extra transpose overhead!

native_kv=False --> Runtime transpose at every decode step --> ~10-15% slower decode
native_kv=True --> No transpose — direct HTP  -- > matmul	Faster decode!

## summary
Native KV format means the Key cache is stored pre-transposed in memory matching exactly what HTP's matrix multiply unit expects — so at every decode step Genie can directly multiply Q×K without any runtime reshape/transpose overhead, 
saving time and VTCM at every single token generation step!" 
