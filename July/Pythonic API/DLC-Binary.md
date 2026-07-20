## DLC → Context Binary Compilation — AOT Deeply Explained

### What is AOT (Ahead-of-Time) Compilation?

"AOT = Ahead Of Time — models should be pre-compiled on developer machines (e.g. x86) instead of on devices. 
Pre-compiled models will be delivered to end-user devices, pre-compiled and optimized for different SoCs."

``` text
JIT (Just-in-Time):               AOT (Ahead-of-Time):
  Model compiled AT RUNTIME          Model compiled OFFLINE
  on the device                      on your Linux x86 machine
        │                                    │
        │ First inference is slow!          │ Compilation done ONCE
        │ (must compile first)              │ upfront!
        ▼                                    ▼
   Fast subsequent runs              Fast from first inference
```

Your builder.build() = AOT compilation — you compiled on your x86 Linux machine, now the .bin files are ready to run instantly on SM8850 without any compilation overhead! 

## The Internal Tool — qnn-context-binary-generator

# What _run_compile_tasks() calls internally per split:

qnn-context-binary-generator \
    --backend  libQnnHtp.so \                  # ← HTP backend
    --model    libQnnModelDlc.so \             # ← DLC loader library
    --dlc_path ar1_cl4096_2_of_4.dlc,\
               ar128_cl4096_2_of_4.dlc \       # ← BOTH AR graphs together!
    --binary_file ar1_cl4096_2_of_4 \          # ← output name
    --config_file HtpConfigFile_API.json \     # ← HTP backend config
    --data_format_config data_format.json \    # ← native KV format config
    --output_dir ./compile_31ca75df...         # ← your cache folder ✅

## What Happens INSIDE qnn-context-binary-generator — Stage by Stage
``` text
Starting compilation
[ 2%]  shape_inference         ← infer all tensor shapes
[ 6%]  configure_incremental_ops
[ 8%]  prequant_gt             ← pre-quantization graph transforms
[10%]  dry run validation      ← validate all ops supported on HTP
[14%]  apply_aimet_quant_json  ← apply AIMET encodings (your LPBQ!)
[18%]  unsup_op_gt             ← check unsupported ops
[22%]  superop_gt              ← fuse ops into HTP super-ops
[28%]  offload_gt              ← decide which ops go to HTP vs CPU
[32%]  layout_gt               ← set optimal tensor layouts (NHWC etc)
[36%]  buffer_optimizations    ← optimize memory buffers
[40%]  post_layout_quant       ← post-layout quantization
[44%]  finalize_tensor_layouts ← lock in all tensor layouts
[53%]  populate_execution_table← build execution schedule
[59%]  configure_tensor_meta   ← set tensor metadata
[63%]  merge_duplicate_params  ← weight deduplication (WEIGHT SHARING!)
[67%]  topology_validation     ← validate full graph topology
[71%]  op_validation           ← validate each op config
[75%]  memory_planning_v2      ← VTCM allocation planning
[79%]  generate_cache_op       ← generate HTP cache operations
[85%]  target artifacts generation ← generate Hexagon DSP microcode
[91%]  target optimization     ← DSP-specific optimizations
[100%] DONE → ar1_cl4096_2_of_4.bin
```
## Key AOT Compilation Steps — Deeply Explained
### Step 1️ — Op Validation (dry run)
Each DLC op checked against SM8850 HTP v81 op registry:
  MatMul INT8×INT8 →  supported on v81
  Conv2D INT8     →  supported
  Float32 MatMul  →  rejected! Must be quantized!

This is WHY you sometimes see:
"QnnBackend_validateOpConfig failed" ← op not supported on this SoC!
### Step 2️ — Super-Op Fusion
HTP has hardware-fused "super-ops":
  MatMul + BiasAdd + ReLU → ONE fused HTP super-op 
  QK_MatMul + Scale + Softmax → fused attention super-op 

Why? Avoids round-tripping through DDR between ops!
Keeps intermediate results in VTCM registers 
### Step 3️ — VTCM Memory Planning
VTCM = 8-16MB on SM8850
  Planner allocates VTCM tiles per op:
  ├── activation tensors:    ~KB per op
  ├── intermediate results:  ~KB
  └── KV cache entries:      allocated in DDR (too big for VTCM)

Goal: maximize VTCM reuse, minimize DDR traffic

### Step 4️ — Weight Sharing (merge_duplicate_params)
Takes AR=1 DLC + AR=128 DLC together:
  Detects identical weight tensors across both graphs
  → Stores ONE copy in weight blob
  → Both graph sections point to same offset 
  = 50% storage reduction!
### Step 5️ — Hexagon DSP Microcode Generation
Final step: generates actual Hexagon v81 machine code!
  QNN graph ops → HVX (Hexagon Vector eXtensions) instructions
  HVX = SIMD vector processor in Hexagon DSP

<img width="602" height="164" alt="image" src="https://github.com/user-attachments/assets/5be9f566-560d-4aec-869f-74d83c5c63bb" />
``` text
quantized DLC (INT8 weights + graph)
        │
        ▼  qnn-context-binary-generator (AOT compiler)
        │
        ├── Op validation   → checks SM8850 v81 support
        ├── Super-op fusion → hardware-optimized fused ops
        ├── VTCM planning   → memory allocation for 8-16MB VTCM
        ├── Weight sharing  → AR=1 + AR=128 share ONE weight blob
        └── HVX microcode  → actual Hexagon DSP instructions
        │
        ▼
    ar1_cl4096_2_of_4.bin  ← self-contained executable
```
[HVX microcode + INT8 weights + scale/offsets] Ready to run INSTANTLY on SM8850 HTP 

**AOT = "compile once offline, run forever on device" — this is exactly what your builder.build() did, and why your .bin files load instantly on SM8850 HTP without any warm-up compilation!**


## After conversion:
  Weights → INT8  stored in DLC file (on disk)
  
## After compilation (.bin):
  Weights → INT8 ✅ packed into .bin weight blob
  
At runtime:
  .bin loaded → DDR ✅ (weights NOW in DDR!)
  
### Key point — weights are NOT in DDR during compilation! They go to DDR only at runtime when .bin is loaded!
``` text
Three Stages — Where Weights Live
STAGE         WHERE WEIGHTS LIVE        FORMAT
─────────────────────────────────────────────────
After LPBQ    output_artifacts/          FP32 safetensors
quantization  model.safetensors          + scale/offset
              (on disk)                  in .encodings

After DLC     convert_*/                 INT8 packed
conversion    ar1_cl4096_2_of_4.dlc      inside DLC
              (on disk)                  format

After AOT     compile_*/                 INT8 packed
compilation   ar1_cl4096_2_of_4.bin      inside .bin
              (on disk)                  weight blob

At RUNTIME    DDR (on device)            INT8 loaded
              when QNN loads .bin        from DDR
```
##  So What Does Compilation ACTUALLY Do?
Compilation is NOT just mapping graph to weights — it does several critical things:
``` text
DLC (graph + INT8 weights, generic format)
        │
        ▼ qnn-context-binary-generator (AOT compilation)
        │
        │  1️ GRAPH COMPILATION
        │     Generic DLC ops → Hexagon v81 DSP microcode
        │     MatMul op → specific HTP instruction sequence
        │     Attention → fused HTP super-op
        │     (Like compiling C++ → machine code for specific CPU!)
        │
        │  2️ MEMORY PLANNING
        │     Decides WHERE each tensor lives:
        │     Activations → VTCM (fast!)
        │     Weights     → DDR  (large!)
        │     KV cache    → DDR  (pre-allocated region)
        │     Intermediate→ VTCM (tile-by-tile)
        │
        │  3️ WEIGHT PACKING + ADDRESSING
        │     Packs INT8 weights contiguously
        │     Assigns 32-bit offsets to each weight
        │     (this IS the 2GB limit reason!)
        │     offset_q_proj_layer0 = 0x00000000
        │     offset_k_proj_layer0 = 0x00A3F210
        │     offset_v_proj_layer0 = 0x01234567
        │     ...max 0x7FFFFFFF = 2GB 
        │
        │  4️ WEIGHT SHARING (merge_duplicate_params)
        │     AR=1 graph weights = AR=128 graph weights
        │     → deduplicate → ONE shared weight blob
        │     → both graphs point to SAME offsets 
        │
        │  5️ SoC-SPECIFIC OPTIMIZATION
        │     v81 specific tile sizes
        │     VTCM allocation per layer
        │     HTP vector width optimization
        │
        ▼
ar1_cl4096_2_of_4.bin
  ├── Hexagon v81 microcode  ← compiled graph 
  ├── Memory plan            ← what goes where 
  ├── Weight offset table    ← graph→weight mapping 
  └── INT8 weight blob       ← packed weights 
```
  e.g. INT8 MatMul → VMPYWEH + VRMPYWEH HVX instructions
  These are the actual hardware instructions!
  Packed into the .bin graph section → ready to execute!


DLC file  ≈  C++ source code
             (portable, generic, not yet optimized)
             
.bin file ≈  compiled binary executable
             (machine code for SPECIFIC CPU = v81 Hexagon DSP)
             
AOT compilation ≈  g++ -O3 -march=native source.cpp -o binary
                   (compile ONCE offline → run fast anywhere!)

Without AOT:    Device compiles at first run → slow first inference!
With AOT:       Pre-compiled → instant first inference! 

## what comilation Does?
<img width="625" height="179" alt="image" src="https://github.com/user-attachments/assets/995ee93d-f8ef-45b1-b3fb-ec112348aa54" />

compilation maps graph to weights in memory" — is correct for the weight packing + offset addressing step (Step 3 above)! But compilation also does graph-to-microcode compilation, memory planning, and SoC-specific optimizations that are equally important!
