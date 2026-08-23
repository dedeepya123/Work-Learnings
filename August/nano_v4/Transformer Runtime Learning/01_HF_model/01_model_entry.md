# Lesson 1 — HF Gemma4 Model Entry

## Goal

Understand how configuration, HF classes, PyTorch modules,
checkpoint weights, and forward execution combine to form
the actual Gemma4 text CausalLM.

---

## 1. The four pieces

CONFIGURATION
→ describes the architecture

MODEL CLASSES
→ define how the architecture is represented

PYTORCH MODULE TREE
→ actual instantiated model objects

CHECKPOINT
→ learned parameter tensors

Together:

configuration
    ↓
model construction
    ↓
PyTorch module hierarchy
    +
checkpoint parameters
    ↓
loaded pretrained model
    ↓
forward()
    ↓
tensor computation

---

## 2. CausalLM composition

Conceptually:
``` text
Gemma4ForCausalLM
    ├── Gemma4TextModel
    │     ├── embedding
    │     ├── decoder layers
    │     └── final norm
    │
    └── lm_head
```
TextModel produces hidden states.

CausalLM adds the language-modeling head
to produce vocabulary logits.

---

## 3. Important execution distinction

__init__()
→ constructs the model/object graph

forward()
→ executes the computation

Configuration
→ provides information used during construction

Checkpoint
→ provides learned parameter values

---

## 4. input_ids → inputs_embeds

Conceptual execution path:

input_ids [B,S]
    ↓
CausalLM.forward()
    ↓
TextModel.forward()
    ↓
embedding
    ↓
inputs_embeds [B,S,H]

For Gemma4 text:

H = 1536

---

## 5. Configuration → tensor reasoning

Example:

hidden_size = 1536
    ↓
embedding dimension
    ↓
hidden-state dimension
    ↓
decoder input/output interface

Example:

num_hidden_layers = 35
    ↓
model construction
    ↓
35 decoder-layer objects
    ↓
forward iterates through them

---

## 6. Important Gemma4 clues

Config:

num_attention_heads = 8
num_key_value_heads = 1
head_dim = 256
hidden_size = 1536

These values indicate that attention cannot be assumed
to follow a simple standard MHA layout.

Therefore inspect actual Gemma4 attention implementation
before making assumptions about tensor shapes.

---

## 7. CraftModule

CraftModule inherits from nn.Module.

It provides a common abstraction for primitive operations
such as Matmul, Add, Sin, Cos, Softmax, etc.

Conceptually:

operation
    ↓
CraftModule wrapper
    ↓
PyTorch operation
    ↓
optional output fake quantization

CraftModule is a toolchain/implementation abstraction,
not a Transformer architectural component.

---

## 8. Core mental model
``` text
Architecture concept
    ↓
configuration
    ↓
Python object construction
    ↓
PyTorch module hierarchy
    ↓
forward()
    ↓
tensor computation
    ↓
model state
    ↓
runtime representation
```
This connection is more important than memorizing
individual HF classes.
