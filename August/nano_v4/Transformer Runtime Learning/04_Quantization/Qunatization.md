# Quantization — Quick Notes

## 1. Why Quantize?

Quantization represents high-precision FP values using a finite set of lower-precision discrete levels.

Benefits:
- Lower model memory
- Lower memory bandwidth
- Lower compute/energy cost
- Potential accelerator performance benefits

Tradeoff:

```text
lower precision → efficiency ↑
                → numerical accuracy may ↓
```

**Model preparation changes the representation of computation.**  
**Quantization changes the numerical representation of tensors.**

---

## 2. Quantization Math

Common affine mapping:

```text
q = round(x / scale) + zero_point

x ≈ (q - zero_point) * scale
```

- `scale` → spacing between representable FP values
- `zero_point / offset` → aligns FP zero with integer domain
- `bitwidth` → number of available quantized levels

Given FP range `[x_min, x_max]` and integer range `[q_min, q_max]`:

```text
scale = (x_max - x_min) / (q_max - q_min)
```

Symmetric quantization approximately centers the range around zero.

Granularity:
- **Per-tensor** → one encoding for the whole tensor
- **Per-channel** → separate encodings for channels

---

## 3. Quantization vs QuantSim vs Calibration

### Quantization
The mathematical reduction from high precision to lower precision.

### Quantization Simulation
AIMET `QuantizationSimModel` configures quantization behavior around the model so quantization effects can be simulated while still executing in PyTorch.

Conceptual model:

```text
FP tensor
   ↓
Quantize
   ↓
integer representation
   ↓
Dequantize
   ↓
approximate FP tensor
```

Q/DQ is the mental model; don't assume AIMET literally implements every case as standalone Q/DQ modules.

### Calibration

Finds suitable quantization parameters by observing model tensors during representative execution.

```text
model execution
      ↓
tensor statistics
      ↓
encoding
(scale + offset + bitwidth)
```

---

## 4. Quantization Flow

Generic AIMET flow:

```text
validation
    ↓
_prepare_model()
    ↓
_prepare_inputs()
    ↓
_create_quantsim()
    ↓
technique-specific step
    ↓
_compute_encodings()
    ↓
QuantizationResult
```

`_create_quantsim()` establishes the base quantization configuration.

Global defaults are applied first; module-specific precision overrides are applied afterward.

```text
global defaults
      ↓
QuantSim construction
      ↓
module-specific overrides
```

---

## 5. Quantization Configuration

HTP config controls behavior such as:

- Activation granularity
- Weight granularity
- Symmetric/asymmetric behavior
- Operation-specific quantization
- Fusion/supergroup behavior

Observed HTP configuration:

```text
Activations → generally per-tensor
Weights     → per-channel for selected ops
Bias        → generally unquantized
```

Selected weight ops include:

```text
Conv
ConvTranspose
Gemm
MatMul
PRelu
```

Module-specific precision can override defaults for things such as:

```text
lm_head
embedding
KV cache
RMSNorm
custom module/regex
```

---

## 6. Quantization Scheme

Current/default flow:

```text
QuantScheme.post_training_tf
        ↓
min/max based observation
```

Other available schemes include enhanced and percentile variants.

For the current flow, think:

```text
representative execution
        ↓
running min/max
        ↓
final encoding
```

---

## 7. Calibration Internals

Core flow:

```text
_compute_encodings()
        ↓
QuantizationSimModel.compute_encodings()
        ↓
aimet_nn.compute_encodings()
        ↓
enable statistics collection
        ↓
forward_pass_callback()
        ↓
model execution
        ↓
observer sees FP tensor values
        ↓
collect_stats()
        ↓
merge_stats()
        ↓
statistics accumulate across batches
        ↓
compute_encodings()
        ↓
final scale + offset
```

Important:

**Calibration does NOT produce one final encoding per batch and average them.**

Instead:

```text
Batch 1 ──→ statistics ─┐
Batch 2 ──→ statistics ─┤
Batch 3 ──→ statistics ─┤
                         ↓
                  accumulated stats
                         ↓
                   final encoding
```

---

## 8. Weight vs Activation Encodings

### Weights

Weights already exist:

```text
stored weight
     ↓
inspect values
     ↓
derive encoding
```

Calibration data is generally not needed just to observe weight values.

### Activations

Activations are produced during execution:

```text
input
  ↓
model
  ↓
activation
  ↓
observe statistics
  ↓
derive encoding
```

Therefore representative calibration data is important for activation encodings.

| | Weights | Activations |
|---|---|---|
| Already stored? | Yes | No |
| Need model execution to observe? | Generally no | Yes |
| Need encoding? | Yes | Yes |
| Calibration data for range? | Generally no | Yes |

---

## 9. Gemma4 Quantization Stage

Generic pipeline is bypassed for Gemma4 because the calibration configuration contains a live Python `DataLoader`, which cannot pass through the generic YAML-serializable configuration path.

Instead:

```python
quantizer = quantize_model(
    input.model.qc_model,
    qc_config,
    input.tokenizer,
    prepare_path=config.prepare_path,
    export_path=config.export_path,
)
```

Input:

```text
TextModelQc / qc_model
        +
qc_config
        +
tokenizer
```

Output:

```text
quantizer.quantsim.model
quantizer.quantsim
quantizer
encodings
exported model
```

---

## 10. Export

AIMET supports:

```text
v1
v2
all
```

Current Gemma4 flow uses:

```text
export_format = "v2"
encoding_version = "2.0.0"
```

Output path convention:

```text
<export_path>_v2_encodings/
    model.onnx
    model.encodings
```

Gemma4 uses custom ONNX export arguments because its KV-cache ownership differs across layers; generic automatic export assumptions do not match the architecture.

---

## Core Mental Model

```text
Prepared model
      ↓
still floating-point computation
      ↓
QuantSim
      ↓
configure quantizers
      ↓
Calibration
      ↓
run representative data
      ↓
observe tensor statistics
      ↓
derive encodings
      ↓
quantized behavior can be simulated/exported
```

### One-line summary

> **Quantization reduces numerical precision; QuantSim models that behavior; calibration observes real tensor values and determines the encoding parameters needed to represent them.**
