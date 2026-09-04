# Model Preparation — Lessons 3–5 Quick Reference

## Representation ladder

```text
model_qc
  ↓ export
ONNX
  ↓ convert
QAIRT IR
  ↓ optimize
Optimized QAIRT IR
  ↓ MPP
model_mpp
```

## ONNX

**Standardized computational graph.**

Contains:

* Inputs / outputs
* Operator nodes
* Tensor connections
* Weights/constants
* Shapes/dtypes

Think: **"What computation does the model perform?"**

## QAIRT IR

**QAIRT's internal graph representation.**

Why:

* QAIRT owns/understands the representation
* Enables graph analysis and transformation
* Supports QAIRT-specific shape/operation/target reasoning

Think: **"How does QAIRT represent this computation?"**

## QAIRT Optimizer

Transforms:

```text
QAIRT IR → Optimized QAIRT IR
```

Typical goals:

* Remove redundant work
* Constant folding
* Simplify graph
* Fuse operations
* Lower/replace operations
* Improve target suitability
* Reduce unnecessary data movement

Think: **"Can this computation be represented better?"**

Optimization ≠ simply reducing node count.

## MPP

Runs after graph preparation/optimization.

```text
Optimized QAIRT IR
       ↓
    MPP/emitter
       ↓
Model.py + Model.safetensors
       ↓
model_mpp
```

`model_mpp` represents/replays the prepared graph; it is **not simply the original `model_qc` structure**.

## Key distinction

```text
Adaptation = change the PyTorch program
Preparation = capture + convert + optimize + emit graph representations
```

## Boundary questions

At every boundary:

1. What representation am I in?
2. What does it contain?
3. What changed/lost?
4. Why does the next stage need it?

**Core mental model:**

> Program → Portable graph → QAIRT graph → Optimized graph → Generated prepared model
