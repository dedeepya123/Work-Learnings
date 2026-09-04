# Model Preparation — Lessons 3–6 Quick Reference

## Representation flow

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

## `model_qc`

Adapted PyTorch **program**.

Contains:

* Python control flow
* Module hierarchy
* Model architecture
* Config-driven logic
* PyTorch operations

## ONNX

Standardized **computation graph**.

Think:

> What computation does the model perform?

## QAIRT IR

QAIRT's **internal graph representation**.

Think:

> How does QAIRT represent this computation?

Enables QAIRT-specific analysis, transformation and optimization.

## Optimized QAIRT IR

Same intended computation, transformed for better:

* efficiency
* simplicity
* target suitability
* data movement
* operation execution

## `model_mpp`

**Generated PyTorch representation of the prepared graph.**

Not:

* a copy of `model_qc`
* a reconstruction of Gemma classes
* the final hardware executable

Conceptually:

```text
Optimized graph
      ↓
MPP emitter
      ↓
Model.py + Model.safetensors
      ↓
model_mpp
```

With:

```text
keep_original_model_structure=False
```

the generated model can be flattened into a fixed sequence of operations.

## Key distinction

```text
model_qc = original/adapted PyTorch program

model_mpp = generated PyTorch representation
            of prepared computation
```

Both are `nn.Module`, but they have different roles.

## Cache transformation

```text
DynamicCache / Python abstraction
        ↓
explicit graph tensor I/O
        ↓
past_key_i / past_value_i
past_key_i_out / past_value_i_out
```

## Core mental model

> The pipeline preserves the **computation**, not the original Python class structure.

```text
Python program
    ↓
portable graph
    ↓
QAIRT graph
    ↓
optimized graph
    ↓
generated PyTorch graph representation
```

## Summary
"The adapted PyTorch model is first captured into an explicit ONNX computation graph. QAIRT converts that into its own IR so it can analyze and optimize the graph. After preparation, MPP emits a new PyTorch module that represents that prepared graph. So the final PyTorch model isn't a reconstruction of the original model; it's a generated representation of the already-prepared computation."



``` text
model_qc = ...

means:

adapted model/program.

Then:

copy.deepcopy(model_qc)

protects the original.

Then:

_prepare_model(...)

does the preparation pipeline.

Inside that:

torch.onnx.export()
        ↓
QAIRTConverter
        ↓
QAIRTOptimizer
        ↓
TorchEmitterAndConfigGenerator
        ↓
model_mpp

And:

keep_original_model_structure=False

basically tells the emitter:

Don't try to preserve the original Gemma Python hierarchy; generate the prepared graph representation.

That's why the generated model can be much flatter than the original Gemma classes.
```
