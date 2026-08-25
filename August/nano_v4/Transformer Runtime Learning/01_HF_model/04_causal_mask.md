# Lesson 4 — Causal Attention Mask

## 1. Purpose

A causal mask controls which K/V positions each query is allowed to attend to.

For causal attention:

    key_position <= query_position

is allowed, while future positions are blocked.

The mask is not a learned model operation. It is control information for attention.

---

## 2. Attention context

Attention is conceptually:

    Attention(Q, K, V)
      = softmax(QKᵀ / sqrt(d) + mask) V

The mask controls visibility.

---

## 3. `_preprocess_mask_arguments()`

This helper determines the geometry needed to construct the mask.

Important outputs:

- `q_length` — number of current query positions
- `kv_length` — number of K/V positions available
- `q_offset` — logical starting position of the current queries
- `kv_offset` — logical starting position of the K/V region

---

## 4. Where information comes from

### `inputs_embeds`

Shape:

    [batch, query_length, hidden_size]

Used to determine:

- batch size
- query length
- dtype
- device

Therefore:

    q_length = inputs_embeds.shape[1]

### `past_key_values`

Provides information about previously processed tokens.

    q_offset = past_key_values.get_seq_length()

It also provides layer-specific KV geometry:

    kv_length, kv_offset =
        past_key_values.get_mask_sizes(q_length, layer_idx)

This is especially important for Gemma4 because it has hybrid
sliding-attention and full-attention layers.

---

## 5. Prefill

Example:

    input sequence = 128 tokens

    q_length  = 128
    kv_length = 128
    q_offset  = 0

The causal relationship is:

        K0 K1 K2 ... K127
    Q0   ✓  ✗  ✗ ... ✗
    Q1   ✓  ✓  ✗ ... ✗
    ...
    Q127 ✓  ✓  ✓ ... ✓

---

## 6. Decode

Suppose 100 tokens are already cached and one new token is processed.

    q_length  = 1
    q_offset  = 100
    kv_length ≈ 101   # full attention, conceptually

The query corresponds to logical position 100 and can attend to
the cached history plus itself.

Decode still has causal semantics, but an explicit causal mask
may not need to be materialized. Some attention backends can
represent causality implicitly.

---

## 7. `attention_mask`

A normal 2D attention mask generally represents valid/padded tokens.

Example:

    [1, 1, 1, 1, 0, 0]

This is different from the causal mask.

Causal mask:
    controls future visibility.

Padding mask:
    controls invalid/padded positions.

They can be combined.

A pre-created 4D mask can also be passed directly, in which case
mask creation can exit early.

---

## 8. Mask construction

High-level flow:

    inputs_embeds
          +
    past_key_values
          +
    attention_mask
          +
    position_ids
          +
    config
          |
          v
    _preprocess_mask_arguments()
          |
          v
    q_length / kv_length / offsets
          |
          v
    causal rule
          +
    optional constraints
          |
          v
    attention backend mask interface
          |
          v
    mask / implicit causal semantics

---

## 9. Important systems insight

The semantic requirement is:

    "future positions must not be attended to."

The implementation does NOT have to be a giant explicit mask tensor.

Depending on the backend, causality may be represented through:

- explicit mask tensor
- `is_causal`
- BlockMask
- specialized attention-kernel behavior

Therefore:

    semantic mask != necessarily materialized mask tensor

This distinction becomes important when lowering HF -> ONNX -> QAIRT IR -> compiled binary.

---

## 10. Key mental model

Inputs tell us:

    "How many new queries am I processing?"

Cache tells us:

    "How far into the sequence have I already processed,
     and what K/V positions are relevant for this layer?"

Together they determine:

    "What attention positions exist and which ones are legal?"

---

## 11. Gemma4-specific point

Gemma4 text has:

- 35 decoder layers
- sliding-attention layers
- full-attention layers
- sliding window = 512

Therefore KV/mask geometry cannot always be assumed to be identical
for every layer.



How does Gemma4 create its causal attention mask?"

You should be able to say:

"At the HF level, the model first determines the geometry of the attention operation. inputs_embeds gives the current query length, while the cache gives the number of previously processed positions and, for Gemma4's hybrid sliding/full-attention layers, the effective KV sizes for the particular layer. _preprocess_mask_arguments() derives q_length, kv_length, and the corresponding offsets. The causal mask is then constructed from these positions, optionally combined with padding or other constraints, and passed through an attention-backend-specific mask interface. During decode, the query length is typically one while the K/V side includes the cached history, so an explicit causal mask may be unnecessary for some backends—the causal semantics can be represented implicitly by the attention kernel."

`past_key_values.get_mask_sizes(..., layer_idx)` abstracts this
layer-specific behavior.
