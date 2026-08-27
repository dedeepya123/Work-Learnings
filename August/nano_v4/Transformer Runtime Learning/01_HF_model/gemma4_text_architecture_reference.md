# Gemma4 Text-Only Model — Architecture Reference

Scope: vanilla HF `Gemma4TextConfig` / `Gemma4TextModel` / `Gemma4ForCausalLM`
(`transformers/models/gemma4/modeling_gemma4.py`). Text-only decoder path — vision/audio
submodels and the QAIRT/HTP-adapted (`Qc*`) classes are out of scope here (see the
KV-cache and mask notes for QAIRT-specific deviations).

---

## 1. `Gemma4ForCausalLM` — top level

**Role:** transformer backbone + LM head.

```
input_ids [B,S]
   │
   ▼
Gemma4TextModel  →  last_hidden_state [B,S,H]   (already final-normed, see §2)
   │
   ▼
lm_head (Linear, H → vocab_size)   ── weight TIED to embed_tokens (tie_word_embeddings=True;
   │                                   checkpoint only stores the embedding, lm_head reuses it
   │                                   — see checkpoint.py's manual tie-back on load)
   ▼
logits [B,S,vocab_size]
   │
   ▼ (only if config.final_logit_softcapping is set)
logits = tanh(logits / softcap) * softcap        ── squashes extreme logits into a bounded
                                                     range before the LM loss/sampling sees them
   ▼
output: CausalLMOutputWithPast(logits, past_key_values, ...)
```

**Note on "normalize":** the final RMSNorm happens *inside* `Gemma4TextModel`, at the end of
the decoder-layer loop — not as a separate step the CausalLM head performs. By the time
`last_hidden_state` reaches `lm_head`, it's already normalized.

---

## 2. `Gemma4TextModel` — the backbone

**Prepares inputs, builds per-forward-call state, runs the decoder stack.**

1. **Token embeddings**
   `inputs_embeds = embed_tokens(input_ids)` — scaled by `embed_scale = sqrt(hidden_size)`
   inside `Gemma4QuantizableEmbedding` (a Gemma-family convention since Gemma1/2; comment
   in code flags that this scale factor gets downcast imprecisely in bf16 — a known,
   accepted quirk, not a bug).

2. **Per-Layer Embeddings (PLE)**, only if `hidden_size_per_layer_input > 0`:
   `per_layer_inputs = get_per_layer_inputs(input_ids, inputs_embeds)` then
   `project_per_layer_inputs(inputs_embeds, per_layer_inputs)` — produces one extra
   per-layer residual-injection tensor, sliced per layer inside the decoder loop (§3).

3. **Cache** — if `use_cache=True` and no `past_key_values` was passed in:
   `DynamicCache(config=self.config)` is built. This walks `config.layer_types` and
   constructs one cache-layer object per real (non-KV-shared) decoder layer:
   `DynamicSlidingWindowLayer` for `"sliding_attention"`, `DynamicLayer` for
   `"full_attention"`. (Full mechanics: see the KV-cache-mechanics notes.)

4. **`shared_kv_states = {}`** — a fresh, empty relay dict, reset every forward call. Layers
   that own their K/V publish into it; KV-shared layers read from it instead of computing.

5. **Position embeddings, per layer type** — RoPE is computed **once per unique
   `layer_type`**, not once per layer (since sliding and full layers use different RoPE
   theta / head_dim, per config): `position_embeddings[layer_type] = rotary_emb(hidden_states, position_ids, layer_type)`.
   Every layer of that type reuses the same `(cos, sin)` pair.

6. **Causal mask, per layer type** — a dict, not a single tensor:
   ```python
   causal_mask_mapping = {
       "full_attention": create_causal_mask(...),
       "sliding_attention": create_sliding_window_causal_mask(...),
   }
   ```
   Each decoder layer looks up the mask matching its own `layer_types[i]`. (Full mechanics
   — shapes, offsets, prefill vs. decode — see the causal-mask notes.)

7. **Decoder loop** — calls `Gemma4TextDecoderLayer` once per layer, `num_hidden_layers`
   times, threading `hidden_states`, that layer's `per_layer_input` slice, the shared
   `shared_kv_states` dict (mutated in place across iterations), that layer-type's
   `(position_embeddings, attention_mask)`, and `past_key_values`.

8. **Final norm** — `hidden_states = self.norm(hidden_states)` (RMSNorm) after the loop —
   this is the "normalize" step, done here, not in `Gemma4ForCausalLM`.

**Output:** `last_hidden_state` (post-final-norm), `past_key_values` (the same cache object,
now mutated with every real layer's new entries).

---

## 3. `Gemma4TextDecoderLayer` — the actual residual structure

This is the part most worth tightening: Gemma4 uses a **"sandwich norm"** pattern — normalize
*before* a sub-block AND normalize its *output* again before adding the residual — not the
single-pre-norm pattern from a vanilla GPT-style block. It does this **twice** (attention,
then MLP), plus an optional **third** sandwich for PLE injection, plus one unconditional
final scalar gate.

```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)              ← pre-norm, before attention
hidden_states, _ = self_attn(hidden_states, ...)
hidden_states = post_attention_layernorm(hidden_states)     ← post-norm, on attention's OUTPUT
hidden_states = residual + hidden_states                    ← residual add #1

residual = hidden_states
hidden_states = pre_feedforward_layernorm(hidden_states)    ← pre-norm, before MLP
hidden_states = mlp(hidden_states)
   │
   ├─ if enable_moe_block (config-gated, off by default):
   │     router selects top_k experts → weighted expert outputs,
   │     combined with the dense mlp() output via two more norms
   │     (post_feedforward_layernorm_1 / _2) — parallel MoE path, not a replacement for mlp()
   ▼
hidden_states = post_feedforward_layernorm(hidden_states)   ← post-norm, on MLP's OUTPUT
hidden_states = residual + hidden_states                    ← residual add #2

if hidden_size_per_layer_input (PLE enabled):
    residual = hidden_states
    hidden_states = per_layer_input_gate(hidden_states)      ← project down to PLE dim
    hidden_states = act_fn(hidden_states)
    hidden_states = mul(hidden_states, per_layer_input)      ← elementwise gate by this layer's PLE slice
    hidden_states = per_layer_projection(hidden_states)       ← project back up to hidden_size
    hidden_states = post_per_layer_input_norm(hidden_states)
    hidden_states = residual + hidden_states                  ← residual add #3

hidden_states = hidden_states * layer_scalar    ← UNCONDITIONAL final gate, every layer,
                                                    regardless of PLE. layer_scalar is a
                                                    registered BUFFER (torch.ones(1) default,
                                                    loaded from checkpoint like a weight, but
                                                    not a gradient-trained nn.Parameter by
                                                    construction)
return hidden_states
```

Four norm-pairs total per layer (or three, if PLE is off): `input_layernorm` /
`post_attention_layernorm` around attention, `pre_feedforward_layernorm` /
`post_feedforward_layernorm` around the MLP, and (if PLE) `post_per_layer_input_norm` around
the PLE injection (PLE has no separate "pre" norm — it reads the already-normalized residual
stream from the previous stage directly).

**Why this matters over a simpler "one residual per layer" mental model:** the *whole
layer's* output only exists because of up to three separate residual streams merging (attn,
mlp, PLE) plus a final learned/loaded scalar multiply — not one add at the very end.

---

## 4. `Gemma4TextAttention` — full detail, including KV sharing

```
Inputs: hidden_states [B,S,H], position_embeddings=(cos,sin) for this layer's type,
        attention_mask for this layer's type, shared_kv_states (dict, mutated),
        past_key_values, cache_position

Q:
  q_proj(hidden_states) → view [B,S,n_q_heads,head_dim]
  q_norm (RMSNorm over head_dim)      ← QK-norm, stabilizes attention without needing the
                                         classic 1/sqrt(head_dim) scaling (see note below)
  apply_rotary_pos_emb(q, cos, sin)
  transpose(1,2) → [B,n_q_heads,S,head_dim]

K, V — branches on is_kv_shared_layer:
  if is_kv_shared_layer:
      k, v = shared_kv_states[kv_shared_layer_index]     ← just read, no compute at all,
                                                             no k_proj/v_proj weights even exist
  else:
      k = k_proj(hidden_states) → k_norm → apply_rotary_pos_emb(k, cos, sin) → transpose(1,2)
      v = v_proj(hidden_states) → v_norm → transpose(1,2)      ← V is normalized but NEVER
                                                                   rope'd (rope only matters
                                                                   for the Q·K dot product)
      if past_key_values is not None:
          k, v = past_key_values.update(k, v, layer_idx, cache_kwargs)   ← cache append
      if store_full_length_kv:        ← true only for the two "source" layers
          shared_kv_states[layer_idx] = (k, v)                            ← publish for borrowers

Attention interface (eager / sdpa / flash — see §5)
  attn_output, attn_weights = attention_interface(q, k, v, attention_mask, scaling=1.0, ...)
  attn_output.reshape → [B,S,n_q_heads*head_dim]

o_proj(attn_output) → [B,S,H]

Output: attn_output [B,S,H], attn_weights (optional)
```

**KV sharing recap** (checkpoint: `num_hidden_layers=35`, `num_kv_shared_layers=20` →
layers 0–14 real, 15–34 shared; sliding borrowers → layer 13, full borrowers → layer 14;
same-type-only sharing). Full derivation: see the `Gemma4TextAttention` notes file.

---

## 5. Attention interface (`eager` / `sdpa` / `flash_attention`)

```
Inputs: query, key, value [B,H,S,d] (or [B,kv_heads,S,d] for k/v before repeat), causal_mask [B,1,Q,KV]

repeat_kv(key, value, n_rep)          ← GQA broadcast: num_key_value_heads (4 in this
                                          checkpoint) repeated up to num_attention_heads (8)
                                          — happens HERE, inside the interface, never inside
                                          the cache itself
attn_weights = (Q @ K.transpose(-2,-1)) * scaling      ← scaling = 1.0 in Gemma4, NOT the
                                                            usual 1/sqrt(head_dim)! q_norm/
                                                            k_norm already keep magnitudes in
                                                            a stable range, so the classic
                                                            scale-down isn't needed on top
attn_weights = attn_weights + causal_mask               ← additive: 0 (allowed) / -inf (masked)
attn_weights = softmax(attn_weights, dim=-1)
attn_output = attn_weights @ value

return attn_output, attn_weights
```

Three interchangeable implementations exist (`eager`, `sdpa`, `flash_attention_2`/`3`,
`flex_attention`) — same math, different kernel/memory-layout tradeoffs. Which one runs is
chosen via `config._attn_implementation`; the QAIRT/HTP path uses its own `eager`-style
reimplementation (`qc_gemma4_eager_attention_forward`) because HTP needs the plain-tensor-op
form, not a fused kernel call.

---

## 6. `Gemma4TextMLP`

```
Inputs: normalized hidden_states (from pre_feedforward_layernorm)

gate = act_fn(gate_proj(hidden_states))     ← act_fn = gelu_pytorch_tanh by default
up   = up_proj(hidden_states)
fused = gate * up                            ← elementwise, NOT a matmul despite the name
                                                 "mul" in code — gating, same shape both sides
down_proj(fused)

Output: transformed hidden_states, same shape as input
```

This is a **gated MLP** — same family as Llama's SwiGLU, but Gemma uses **GeLU** as the
gate activation instead of SiLU, so it's more precisely a "GeGLU" variant. Worth having that
exact word ready in an interview rather than saying "SwiGLU" by reflex.

**Ties back to KV sharing:** if `config.use_double_wide_mlp` is set, `intermediate_size` is
**doubled** specifically for KV-shared layers (`is_kv_shared_layer` check inside
`Gemma4TextMLP.__init__`). Reading: a layer that skipped computing its own K/V gets some of
that "spent" capacity back via a wider MLP — a deliberate compute-budget rebalancing, not
incidental.

---

## 7. Non-vanilla design choices worth naming outright (interview quick-recall)

| Feature | One-line reason |
|---|---|
| **QK-norm** (`q_norm`/`k_norm`) | RMSNorm on Q/K before RoPE — stabilizes attention magnitude; also *why* `scaling=1.0` instead of `1/sqrt(head_dim)` is safe here |
| **Alternating sliding/full attention** | 4:1 (or 5:1) sliding-to-full ratio; different RoPE theta *and* different `head_dim`/`global_head_dim` per type |
| **Cross-layer KV sharing** | Back ~57% of layers (20/35 here) reuse an earlier same-type layer's K/V — cuts cache memory + compute; same-type-only sharing to preserve local-vs-global lookup semantics |
| **Per-Layer Embeddings (PLE)** | A second, cheaper embedding table injecting a token-identity residual signal at every layer, gated per layer |
| **Tied + scaled embeddings** | `lm_head` weight = `embed_tokens` weight (transposed use); embeddings scaled by `sqrt(hidden_size)` at lookup time |
| **Final logit softcapping** | `tanh(logits/cap)*cap` — bounds extreme logits before loss/sampling |
| **`layer_scalar` buffer** | Per-layer learned-or-loaded scalar gate on the whole layer's output, applied unconditionally after everything else |
| **GeGLU MLP** | Gated MLP using GeLU (not SiLU/SwiGLU) as the gate activation |
| **Optional MoE block** | Runs *alongside* the dense MLP (not instead of it) when `enable_moe_block=True`, combined via extra norms |
