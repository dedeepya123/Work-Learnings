# Lesson 1 — What Is Model Adaptation?

## 1. Core Idea

Model adaptation transforms the representation of an existing
HF/PyTorch model so that its computation can be represented in a
target runtime-compatible form.

Adaptation should preserve the intended model semantics.

Conceptually:

HF Model
    ↓
Adaptation
    ↓
Runtime-compatible Model

---

## 2. Model Semantics vs Representation

Model semantics describe WHAT the model computes.

Example:

Y = XWᵀ + b

Representation describes HOW that computation is expressed.

HF:

nn.Linear(...)

Runtime-oriented:

Gemma4QuantizableLinear(...)

The mathematical operation can remain the same while the
representation changes.

---

## 3. Why Adaptation Is Needed

A model being executable in PyTorch does not automatically mean
that its computation is represented in the form required by a
target runtime.

Adaptation can provide:

- runtime-compatible operators
- runtime-aware modules
- compiler-visible computation
- required dtype/shape behavior
- runtime-specific interfaces
- support for specialized execution paths

The exact reason must be determined for each adaptation.

---

## 4. Adaptation Is Not Quantization

Adaptation, preparation, quantization and compilation are separate
concepts.

High-level flow:

HF Model
    ↓
Adaptation
    ↓
Runtime-compatible model
    ↓
Preparation
    ↓
Quantization
    ↓
Compilation
    ↓
Binary
    ↓
Target hardware

This lesson focuses only on adaptation.

---

## 5. Types of Adaptation

### Replacement

Original:

nn.Linear

Adapted:

Runtime-specific Linear

The conceptual operation remains the same.

### Wrapping

An existing computation is wrapped with runtime-specific behavior
or metadata.

### Restructuring

The computation graph is reorganized into a form better suited for
the target runtime while preserving intended semantics.

---

## 6. Adaptation Thinking Framework

For every adaptation ask:

1. What did HF originally have?
2. What computation does it perform?
3. What did we replace/change?
4. Why did we change it?
5. What remains mathematically/semantically the same?
6. How do tensor shapes change or remain unchanged?
7. Why does the target runtime care about this operation?

---

## 7. Gemma4 Example

Original conceptual computation:

hidden_states
    ↓
Q projection
    ↓
query_states

HF representation may use a standard Linear.

Gemma4 runtime-oriented code uses:

Gemma4QuantizableLinear

The important distinction:

MODEL SEMANTICS:
hidden_states → Q

IMPLEMENTATION:
standard PyTorch representation → runtime-aware representation

---

## 8. Key Mental Model

Original model
    ↓
What computation is being performed?
    ↓
HF/PyTorch representation
    ↓
Adaptation
    ├── replace
    ├── wrap
    └── restructure
    ↓
Runtime-compatible representation
    ↓
Same intended model semantics

---

## 9. Golden Rule

Always understand the original computation first.

Then ask:

"Why does the target runtime need this computation represented
differently?"

Do not memorize adaptations.

Derive them from:
original computation → runtime requirement → adaptation.
