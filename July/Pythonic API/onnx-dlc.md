# ONNX → DLC Conversion Stage — Deeply Explained!

The Internal API — qairt-converter + qairt-quantizer
## 
STAGE 3: convert_* folders in your cache
        │
        ├── STEP A: qairt-converter   ← ONNX → DLC (with encodings applied)
        └── STEP B: qairt-quantizer   ← DLC → Quantized DLC (act/bias bitwidth)

## Step A — qairt-converter (ONNX → DLC)
Confirmed from real internal build scripts (build_qwen_1_5b_ar1_model.sh, build_phi3.5_ar128_model.sh):


# What the builder calls internally for each split:
``` text 
qairt-converter \
    --input_network ar1_cl4096_2_of_4.onnx \           # ← your SHA ONNX (post MHA2SHA)
    --quantization_overrides ar1_cl4096_2_of_4.encodings \  # ← your LPBQ encodings!
    --output_path ar1_cl4096_2_of_4.dlc \
    --source_model_input_layout "past_key_0_in" NHWC \  # ← KV cache layout
    --source_model_input_layout "past_value_0_in" NHWC \
    ... (one per KV head)

## What qairt-converter Does:
``` text
ONNX graph (floating point ops)
        │
        ├── Reads --quantization_overrides (.encodings)
        │   → Applies scale/offset per layer FROM your LPBQ calibration
        │   → Converts FP32 weights → INT8 weights in DLC
        │
        ├── Sets input/output layouts for KV cache tensors (NHWC)
        │
        └── Outputs: ar1_cl4096_2_of_4.dlc  ← Qualcomm DLC format
```

## Step B — qairt-quantizer (DLC → Quantized DLC)

What the builder calls after conversion:
qairt-quantizer \
    --input_dlc ar1_cl4096_2_of_4.dlc \
    --output_dlc ar1_cl4096_2_of_4_quantized.dlc \
    --act_bitwidth 16 \    # ← activations = INT16
    --bias_bitwidth 32     # ← biases = INT32
What qairt-quantizer Does:
``` text
DLC (weights already INT8 from converter)
        │
        ├── Sets act_bitwidth=16  → all activations = INT16
        ├── Sets bias_bitwidth=32 → all biases = INT32
        │
        └── Outputs: ar1_cl4096_2_of_4_quantized.dlc  ← Final quantized DLC
```
<img width="609" height="110" alt="image" src="https://github.com/user-attachments/assets/e74bd975-4d34-4e1d-988d-53427e8e4e1c" />


After both steps — final DLC precision:
Weights:      INT8   ← from LPBQ encodings (qairt-converter)
Activations:  INT16  ← from qairt-quantizer (act_bitwidth=16)
Biases:       INT32  ← from qairt-quantizer (bias_bitwidth=32)
KV Cache:     INT8   ← from LPBQ encodings (kv cache tensors)

``` text
For each split (e.g. ar1_cl4096_2_of_4):

transform_*/ar1_cl4096_2_of_4_sha.onnx    ← input (post MHA2SHA)
transform_*/ar1_cl4096_2_of_4_sha.encodings ← input (propagated encodings)
        │
        ▼ qairt-converter --quantization_overrides
        │
convert_30848569.../ar1_cl4096_2_of_4.dlc     ← Step A output
        │
        ▼ qairt-quantizer --act_bitwidth 16 --bias_bitwidth 32
        │
convert_30848569.../ar1_cl4096_2_of_4_quantized.dlc  ← Step B output 
        │
        ▼  (feeds into compile stage)
compile_31ca75df.../ar1_cl4096_2_of_4.bin    ← Final context binary!
```

<img width="598" height="112" alt="image" src="https://github.com/user-attachments/assets/fbcf2c95-45af-4751-8dd4-6e9282985f46" />
