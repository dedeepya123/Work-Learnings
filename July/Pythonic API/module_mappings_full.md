# Forward Pass
A Llama model is not just one block. It is a stack of pieces:

1. the top-level model wrapper
2. the base model container
3. the decoder layers
4. the attention blocks inside each decoder layer
5. the rotary embedding helper
6. the config object that carries runtime flags

The QC re-authoring path swaps each of those pieces with a QAIRT-aware version so the same pretrained Llama can run in a QC/HTP-friendly way.

---

## 1. The full Llama object structure

In vanilla Hugging Face, the model looks like this conceptually:

- `LlamaForCausalLM`
  - contains `self.model`
    - `LlamaModel`
      - contains `self.embed_tokens`
      - contains `self.layers` = many `LlamaDecoderLayer`
        - each `LlamaDecoderLayer` contains `self.self_attn`
          - which is a `LlamaAttention`
      - contains `self.norm`
      - contains `self.rotary_emb`
  - contains `self.lm_head`

So the forward pass is:

- embed input ids
- pass through many decoder layers
- each decoder layer runs self-attention
- normalize final hidden states
- project to vocabulary logits

That is the normal HF path.

---

## 2. What QC re-authoring changes

When QAIRT re-authoring is applied, the same conceptual structure still exists, but several components are swapped to QAIRT versions:

- `LlamaConfig` → `QcLlamaConfig`
- `LlamaRotaryEmbedding` → `QcLlamaRotaryEmbedding`
- `LlamaForCausalLM` → `QcLlamaForCausalLM`

And internally, the attention block is also swapped:

- `LlamaAttention` → `QcLlamaAttention`

So the model still has the same architecture, but the implementation of several pieces is adapted for QC/HTP execution.

---

## 3. Step-by-step forward pass in the normal Llama model

Let’s walk through it like a teaching example.

### Step A: input enters the model

You call:

- `model(input_ids=...)`

The top-level `LlamaForCausalLM.forward(...)` receives the tokens.

At this point, the model does not yet compute logits.  
It first passes the request into the base model:

- `self.model(...)`

---

### Step B: embedding layer

Inside `LlamaModel.forward(...)`:

- `input_ids` are converted to embeddings with `self.embed_tokens`
- this gives a tensor of shape roughly:
  - `[batch, seq_len, hidden_size]`

So the model starts with token vectors.

This is the “meaning” stage: tokens become dense representations.

---

### Step C: position info is created

The model then builds:

- `position_ids`
- `cache_position`
- a causal mask

This is important because Llama is autoregressive.

So before attention happens, the model knows:

- where each token sits in the sequence
- which past tokens are available
- which positions are allowed to attend to which earlier tokens

This is the context setup for attention.

---

### Step D: rotary embeddings are prepared

The model calls:

- `self.rotary_emb(hidden_states, position_ids)`

This produces the RoPE components used by attention.

In normal HF:
- this is handled by `LlamaRotaryEmbedding`

In QC:
- it becomes `QcLlamaRotaryEmbedding`

That swap matters because QAIRT wants more control over how position embeddings are handled during export/runtime.

---

### Step E: each decoder layer runs

The model loops through all decoder layers:

- for each layer:
  - run layernorm
  - run self-attention
  - add residual
  - run MLP
  - add residual again

So the real computation happens in the decoder stack.

---

### Step F: self-attention inside each decoder layer

Inside each `LlamaDecoderLayer`, the code calls:

- `self.self_attn(...)`

That attention block is where the model compares the current token representation to past context.

In normal HF:
- `self.self_attn` is a `LlamaAttention`

In QC:
- it is a `QcLlamaAttention`

That is the most important layer-level swap.

---

## 4. What happens inside the attention block

Inside attention:

- query, key, value projections are computed
- the tensors are reshaped for multi-head attention
- rotary embeddings are applied
- past key/value cache is updated
- attention scores are computed
- output is projected back to hidden size

This is the key computation that allows the model to “look at” prior tokens.

In QC mode, this same logical operation is performed, but the implementation is adapted so it supports:

- QC config flags
- custom cache behavior
- custom masking logic
- transposed key-cache support
- runtime-friendly execution semantics

That is why the attention swap is so central.

---

## 5. What happens after attention

After all decoder layers finish:

- you get the final hidden representation
- the model applies `self.norm`
- then the LM head converts hidden states to token logits

So the full flow is:

- tokens → embeddings
- embeddings → many decoder layers
- decoder layers → hidden states
- hidden states → vocabulary logits

That is the model’s forward pass.

---

## 6. Where the QC pieces fit into that same flow

Now let’s map the QC components to the same flow.

### Config
- `LlamaConfig` becomes `QcLlamaConfig`
- this carries extra flags the QC attention path needs

### Rotary embedding
- `LlamaRotaryEmbedding` becomes `QcLlamaRotaryEmbedding`
- this adapts the positional embedding handling

### Top-level model wrapper
- `LlamaForCausalLM` becomes `QcLlamaForCausalLM`
- this preserves the same model structure but adds QC-specific runtime behavior

### Attention block
- `LlamaAttention` becomes `QcLlamaAttention`
- this is where the actual attention computation gets adapted

So the QC path is not replacing the whole model with something unrelated.  
It is adapting the same forward pipeline so it is suitable for QAIRT/HTP execution.

---

## 7. Why this is useful for QC

The reason is practical:

- Hugging Face Llama is built for general-purpose PyTorch execution
- QAIRT needs a model that can be exported and run on HTP with specific assumptions
- the re-authoring layers make the model compatible with those assumptions

In other words:

- the model still learns and predicts the same way
- but the runtime path is adjusted for QC hardware and execution constraints

---

## 8. The easiest mental model

Think of it like this:

- normal Llama = a general-purpose reference implementation
- QC Llama = the same reference implementation, but with QC-specific adapters inserted at the right places

The backbone is the same, but the execution path is customized.

---

## 9. Very short summary

A full forward pass in Llama is:

- embed tokens
- build position/context state
- run rotary embedding
- pass through decoder layers
- each decoder layer runs attention
- produce final hidden states
- project to logits

The QC re-authoring path changes the implementation of the config, rotary embedding, top-level model wrapper, and attention block so that this same forward flow becomes QAIRT/HTP-friendly.
