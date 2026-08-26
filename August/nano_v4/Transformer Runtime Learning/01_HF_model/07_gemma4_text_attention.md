# Gemma4TextAttention — Notes & Q&A

Personal study notes on `Gemma4TextAttention.forward` (vanilla `modeling_gemma4.py`, mirrored
with QAIRT/HTP adaptations in `reauthoring.py`).

---

## Forward pass trace

**Inputs:** `hidden_states` `[B,S,H]`, `position_embeddings` (per-layer `(cos,sin)`),
`attention_mask`, `shared_kv_states` (dict), `past_key_values`, `cache_position`

### 1. Query projection + RoPE

```
q_proj: [B,S,H] → [B,S,n_q_heads*head_dim] → view → [B,S,n_q_heads,head_dim]
q_norm (RMSNorm over head_dim)
apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
transpose(1,2) → [B,n_q_heads,S,head_dim]
```

`apply_rotary_pos_emb`:
- `cos`/`sin` are duplicated to full `head_dim`: `emb = cat(freqs, freqs)` → `cos = emb.cos()`
- `rotate_half(x) = cat(-x[second_half], x[first_half])`
- `result = x*cos + rotate_half(x)*sin`

Q and K are both rotated (their dot product becomes relative-position-dependent).
**V is never rotated** — it's never dotted against anything, only weight-summed by
`softmax(QKᵀ)`.

### 2. Key/Value — branches on `is_kv_shared_layer`

Checkpoint config: `num_hidden_layers=35`, `num_kv_shared_layers=20`
→ `first_kv_shared_layer_idx = 35 - 20 = 15`
→ layers **0–14**: real (compute own K/V)
→ layers **15–34**: shared (borrow K/V; `k_proj`/`v_proj`/`k_norm`/`v_norm` weights aren't
  even allocated for these layers)

Sharing target = nearest earlier **real** layer of the **same** `layer_type` (sliding
borrows from a sliding source, full borrows from a full source — never cross-type).

For this checkpoint: all sliding borrowers → **layer 13**, all full borrowers → **layer 14**.

```python
if not is_kv_shared_layer:
    k = k_proj(hidden_states) → k_norm → rope(k) → transpose(1,2)
    v = v_proj(hidden_states) → v_norm → transpose(1,2)      # no rope on V
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, layer_idx, cache_kwargs)
        # update() returns the FULL accumulated (all-history) k/v, even for a
        # sliding-window cache layer (which only KEEPS a truncated window
        # internally for its own next call)
    if store_full_length_kv:          # True only for layer 13 and layer 14
        shared_kv_states[layer_idx] = (k, v)      # publish for borrowers
else:
    k, v = shared_kv_states[kv_shared_layer_index]    # just read, no compute
```

`shared_kv_states` is reset to `{}` at the **start of every forward call** — it's a
same-forward-pass relay dict, not a structure persisted across decode steps. Borrower
layers still apply their **own** `attention_mask` (their own sliding/full window logic)
against the borrowed k/v — sharing the tensor does not mean sharing the masking behavior.

### 3. Attention

```
attn_output, attn_weights = attention_interface(q, k, v, attention_mask, scaling=..., sliding_window=...)
attn_output.reshape → [B,S,n_q_heads*head_dim]
o_proj(attn_output) → [B,S,H]
```

**Output:** `attn_output` `[B,S,H]`, `attn_weights` (optional)

---

## Q&A

### Q1. RoPE math — how does "for every 2 dims we apply a rotation" map to this code?

The code uses the **rotate-half** layout, not the adjacent-pair layout from the RoPE paper
diagrams. It pairs dim `i` (first half) with dim `i + d/2` (second half), instead of
adjacent dims `(x0,x1), (x2,x3), ...`.

Treat each pair `(x1_i, x2_i)` as a complex number `z_i = x1_i + i·x2_i`. Rotating by angle
`θ_i` means multiplying by `e^{iθ_i} = cos θ_i + i sin θ_i`:

```
z_i' = z_i · e^{iθ_i} = (x1_i·cos_i − x2_i·sin_i) + i·(x1_i·sin_i + x2_i·cos_i)
```

Expand `x*cos + rotate_half(x)*sin` elementwise (recall `cos_i == cos_{i+d/2}` since
cos/sin are duplicated across both halves):

```
new_x1_i = x1_i·cos_i − x2_i·sin_i    ← real part of z_i'
new_x2_i = x2_i·cos_i + x1_i·sin_i    ← imag part of z_i'
```

Exact match — `rotate_half` + duplicated cos/sin is a vectorized way to rotate every
`(x1_i, x2_i)` pair by `θ_i`, without a loop or a complex dtype.

`θ_i` comes from `compute_default_rope_parameters`:
```python
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))   # d/2 distinct frequencies
freqs = inv_freq_expanded @ position_ids_expanded              # θ_{i,pos} = pos * inv_freq_i
```
Lower `i` → lower frequency → captures long-range position; higher `i` → faster rotation →
fine-grained local position.

**QAIRT-side note:** the reauthored code (`QcApplyRopeSingle` in `reauthoring.py`)
reimplements the same complex-multiply explicitly on real/imag halves, with **half-sized**
(non-duplicated) cos/sin, avoiding `rotate_half`'s concat+negate — friendlier for a
fixed-function HTP graph. Same math, restructured for hardware lowering.

### Q2. Why is RoPE applied to Q and K but not V?

RoPE's job is to make the attention **score** `Q·Kᵀ` depend on relative position, not
absolute position. If Q at position `m` is rotated by `θ·m` and K at position `n` by `θ·n`,
their dot product depends only on `(m − n)` — the rotation angles cancel/subtract inside
the inner product. This is a property of the dot product between two rotated vectors, so
it only matters for the two tensors that get dotted together for scoring: **Q and K**.

V never appears in a dot product for scoring — it's the payload that gets weight-summed by
the (already position-aware) attention weights: `attn_output = softmax(QKᵀ)·V`. Rotating V
would add computation for no benefit (no second rotated tensor to dot it against) and would
actively corrupt its content direction, since nothing downstream undoes that rotation.

### Q3. What does `store_full_length_kv` actually do, and why do layers 13/14 need it?

It is **not** about what the cache keeps for its own next step — it's about what a source
layer **publishes** into `shared_kv_states` for its borrowers to read.

Key detail: `key_states`/`value_states` get reassigned to whatever `past_key_values.update()`
**returns** — and both `DynamicLayer.update()` and `DynamicSlidingWindowLayer.update()`
return the **full concatenated (all-history) tensor**, even though `DynamicSlidingWindowLayer`
only *keeps* a truncated window internally for its own next call:

```python
# DynamicSlidingWindowLayer.update()
full_key_states = torch.cat([self.keys, key_states], dim=-2)     # everything so far
self.keys = full_key_states[:, :, -self.sliding_window + 1:, :]  # only THIS kept for next step
return full_key_states, full_value_states                          # but THIS is returned/used now
```

So `store_full_length_kv=True` layers (13 and 14) publish the complete, untruncated,
this-step K/V — the exact tensor they themselves use for their own attention this step.

**Trace, one forward step (20-token prefill):**
- **Layer 13** (sliding, real): computes k/v → cache-updates (accumulates full history) →
  gets back all 20 tokens' K/V from `update()` → publishes into `shared_kv_states[13]`.
- **Layer 15** (sliding, shared, borrower): skips k_proj/v_proj/cache entirely → reads
  `shared_kv_states[13]` → runs its own Q against it, with its own sliding-window mask
  (which restricts which of those 20 tokens it may attend to). Correct windowed *behavior*
  comes from the mask, not from what's stored in the relay dict.
- **Layer 19** (full, shared, borrower): reads `shared_kv_states[14]` — a different pool,
  since it's a different `layer_type`.

Why "full length" and not "windowed": the code doesn't rely on the coincidence that all
sliding borrowers apply the same window mask anyway — it publishes whatever `update()`
naturally returns (the real, full computation), which stays correct regardless of masking
details, and which a full-attention borrower genuinely needs (long-range K/V).

**QAIRT-specific note:** since the on-device pipeline bans `DynamicCache(config=...)`
(see the KV-cache trap notes), every real layer in the actual runtime is a plain
`DynamicLayer` anyway — there's no windowed-vs-full distinction at the *cache* level at
all; sliding-window behavior is entirely delegated to `swa_attention_mask`. The vanilla
logic above is still what the `store_full_length_kv` flag/naming was designed around.

### Q4. Why doesn't cross-layer KV sharing degrade quality much? Is this a Gemma4 invention?

**Not a Gemma4 invention.** Google's own Gemma 3n already shipped this idea (as "KV
sharing") before Gemma4. It also traces to academic work on cross-layer attention (CLA) —
sharing K/V projections across groups of adjacent layers to cut KV-cache memory — and to
production-serving writeups from companies like Character.AI, who converged on the same
idea independently for the same reason. Gemma4 tunes *which*/*how many* layers share; it
didn't invent the mechanism.

**Why it doesn't hurt much:**
- Only K/V (the attention "lookup index") is shared — Q, attention weights, attention
  output, and the MLP are all still computed fresh, per layer, every layer.
- By layer 13–14 of 35, hidden states have already been mixed through many rounds of
  self-attention + MLP; the K/V projection of a nearby depth tends to be highly correlated
  with what a slightly-later layer would have computed itself.
- Sharing is **same-type-only** (sliding↔sliding, full↔full) — preserves *what kind of
  lookup* is being reused; a full-attention borrower still gets genuinely long-range K/V,
  a sliding borrower still gets local-context K/V.
- The model is trained/fine-tuned **with** this sharing scheme already in place — later
  layers learn to work well given borrowed K/V, this isn't a post-hoc ablation.

**Where the savings come from (two separate wins):**
1. **Parameters** — shared layers never allocate `k_proj`/`v_proj`/`k_norm`/`v_norm` at all.
2. **Compute + KV-cache memory at inference** — shared layers skip K/V projection matmuls
   every step and skip writing to the cache; the cache only needs to physically store
   entries for the "real" layers (15 of 35 here), cutting cache memory roughly by the
   shared fraction — important for long-context/on-device inference where cache memory is
   often the binding constraint.
