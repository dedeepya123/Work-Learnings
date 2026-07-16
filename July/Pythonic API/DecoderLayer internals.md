The call path is clear now. I’m aligning it with the QC replacement so you can see the same layer before and after the swap.

Yes — this is the clearest way to see it.

Let’s take one decoder layer and follow the call chain.

---

## 1. The normal Hugging Face path

In vanilla Transformers, the model has a decoder layer like this:

- `LlamaDecoderLayer`
  - contains `self.self_attn`
  - `self.self_attn` is a `LlamaAttention`

So when the model runs one layer, the flow is:

1. `LlamaModel.forward(...)`
2. loop over decoder layers
3. each decoder layer calls:
   - `self.self_attn(...)`
4. that attention block computes attention and returns an updated hidden state

So the important line is this conceptual call:

- `LlamaDecoderLayer -> LlamaAttention`

---

## 2. What the decoder layer actually does

In the HF implementation, one decoder layer does:

- normalize the input
- run self-attention
- add residual
- run MLP
- add residual again

The relevant part is the attention part:

- it calls `self.self_attn(...)`
- passes:
  - `hidden_states`
  - `attention_mask`
  - `position_ids`
  - `past_key_values`
  - `cache_position`
  - `position_embeddings`

That is the exact handoff from the decoder layer to the attention module.

---

## 3. What changes under QAIRT re-authoring

After module replacement, the same decoder layer still exists, but its attention submodule is no longer the vanilla HF one.

The swap is:

- `LlamaAttention` → `QcLlamaAttention`

So the call path becomes conceptually:

- `LlamaDecoderLayer -> QcLlamaAttention`

The decoder layer itself does not need to change its code.  
It still calls `self.self_attn(...)` in the same way.

That is the subtle but important part:

- the parent module stays the same
- the child module object changes class
- the call site is unchanged

That is why module replacement is so elegant.

---

## 4. Why the decoder layer does not need to change

Because the decoder layer only cares that `self.self_attn` is callable and accepts the same arguments.

So from the decoder layer’s perspective:

- before re-authoring: `self.self_attn` is a `LlamaAttention`
- after re-authoring: `self.self_attn` is a `QcLlamaAttention`

The call remains:

- `hidden_states, _ = self.self_attn(...)`

The difference is inside the implementation of that module.

---

## 5. What the QC attention implementation changes

The vanilla `LlamaAttention.forward(...)` uses the standard HF attention path:

- projects to Q/K/V
- applies rotary embeddings
- updates cache through the HF cache interface
- sends the tensors to a generic attention backend

The QC version, `QcLlamaAttention.forward(...)`, does the same high-level task but with QC-specific behavior:

- it can use QC-style cache handling
- it can support transposed key cache
- it can use custom RoPE logic
- it can apply QC-specific masking / attention logic
- it is tailored for HTP-oriented execution

So the decoder layer still “gets attention output,” but the internal logic is now QAIRT-adapted.

---

## 6. The easiest way to picture it

Think of it like this:

- before:
  - decoder layer asks “attention block, compute attention for me”
  - attention block answers using standard HF logic

- after:
  - decoder layer still asks the same thing
  - but the attention block now answers using a QC-specific implementation

The surrounding architecture is unchanged; only the implementation behind that call is adapted.

---

## 7. Why this matters for QC

This is the most important practical point:

- the model can stay structurally the same
- but the attention block can now behave in a way that is compatible with the QAIRT runtime

That is exactly what the re-authoring pipeline is trying to achieve.

---

## 8. The simple summary

One decoder layer behaves like this:

- input comes in
- it is normalized
- it calls self-attention
- the attention module computes context
- the result is added as a residual
- then MLP runs

With re-authoring:

- the same decoder layer calls the same interface
- but the attention module is now `QcLlamaAttention`
- so the attention computation uses QC-adapted logic

That is the heart of the module-mapping story.

If you want, the next step can be a “before/after pseudocode” view of one decoder layer, showing the exact same call flow but with the HF class and QC class side by side.
