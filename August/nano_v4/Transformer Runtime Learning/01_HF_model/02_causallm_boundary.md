# Lesson 2 — Gemma4ForCausalLM Boundary

## Goal

Understand what Gemma4ForCausalLM owns, what it delegates to
Gemma4TextModel, and how hidden states become logits.

---

# 1. Mental Model

Gemma4ForCausalLM
│
├── Gemma4TextModel       ← Transformer backbone
│
├── lm_head               ← hidden → vocabulary
│
└── lm_head_outs          ← post-LM-head/custom processing

Execution:

model(...)
    ↓
Gemma4ForCausalLM.forward()
    ↓
Gemma4TextModel.forward()
    ↓
last_hidden_state
    ↓
select required positions
    ↓
lm_head
    ↓
lm_head_outs
    ↓
logits/output
