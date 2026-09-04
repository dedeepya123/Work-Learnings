# Model Preparation — Lesson 3: ONNX

## Core idea

**ONNX = standardized computational graph representation.**

```text
PyTorch model
    ↓
ONNX graph
    ↓
QAIRT IR
```

## What ONNX contains

* Inputs / outputs
* Operator nodes
* Tensor connections
* Weights / constants (initializers)
* Shapes / dtypes / graph metadata

Example:

```text
x → MatMul → Add → ReLU → y
```

## What changes from PyTorch

```text
model_qc
Python + PyTorch control flow
        ↓
      ONNX
explicit tensor computation graph
```

Python loops/branches are no longer the main representation.

## Gemma4 example

ONNX can expose:

```text
input_ids
attention_mask
swa_attention_mask
past_key_0
past_value_0
...
past_key_14
past_value_14
```

and explicit operations for all 35 layers.

Cache abstraction → explicit tensor I/O.

## Why ONNX?

QAIRT can consume an explicit standardized computation graph instead of understanding the original Python model implementation.

## Important distinction

```text
model_qc  = PyTorch program
ONNX      = standardized graph
QAIRT IR  = QAIRT internal graph
model_mpp = generated prepared PyTorch representation
```

ONNX is an **intermediate representation**, not the final hardware executable.

## Boundary questions

At every representation boundary ask:

1. What representation am I in?
2. What information does it contain?
3. What changed/lost?
4. What was gained?
5. Why does the next stage need it?

## Actual preparation flow

```text
model_qc
   ↓ torch.onnx.export
Model.onnx          ← internal/temporary
   ↓ QAIRT Converter
QAIRT IR
   ↓ QAIRT Optimizer
optimized IR
   ↓ MPP
model_mpp
```

**Mental model:**

> PyTorch describes *how the program is written*; ONNX describes *the computation as an explicit graph*.
