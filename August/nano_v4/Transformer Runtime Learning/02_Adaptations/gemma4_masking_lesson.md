# Whiteboard Lesson: position_ids, cache_position, Masking, Local/Global Attention

A from-scratch mental-model build for reading production Gemma4 masking/caching code (HF +
Qualcomm/QAIRT adaptation), each concept run through: intuition → toy example → tensor
shapes → pseudocode → HF Gemma4 code → Qualcomm mapping.

Companion to: `gemma4_causal_mask_notes.md`, `gemma4_kv_cache_mechanics_notes.md`,
`gemma4_masking_theory_to_adaptation_mentor_notes.md`,
`gemma4_kv_cache_theory_to_adaptation_mentor_notes.md`, `gemma4_adaptation_framework_notes.md`.
This file is the "first principles, concept-by-concept" version — read this one first if
starting cold; read the others for full code traces and worked shape/value tables.

---

## Lesson 1: `position_ids` — "where am I in the sequence?"

### Intuition
Attention is a set operation — no built-in sense of order. RoPE injects order by rotating
each token's Q/K vector by an angle proportional to that token's **absolute position in
the sequence**. Before anything else, every token needs to know: *"what is my absolute
position number?"* That number is `position_ids`.

### Toy example
`"The cat sat"` — first thing the model has ever seen:
```
token:        The   cat   sat
position_ids:  0     1     2
```
Now generating the 4th token `"down"` after that (cache already holds 3 tokens):
```
token:        down
position_ids:  3        ← NOT 0. Continues from where the sequence actually is.
```

### Tensor shape
`position_ids: [B, S]` — one integer per token per batch item. Examples above:
`[1, 3]` then `[1, 1]`.

### Pseudocode
```
position_ids[i] = (number of tokens already processed before this call) + i
```

### HF Gemma4 code
```python
# Gemma4TextModel.forward (vanilla)
if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    position_ids = torch.arange(inputs_embeds.shape[1], device=...) + past_seen_tokens
    position_ids = position_ids.unsqueeze(0)
```
`past_seen_tokens` = "how many tokens already exist." `arange(S)` = "my index within *this*
call." Sum = absolute position. Feeds `rotary_emb(hidden_states, position_ids, layer_type)`
to produce `(cos, sin)`.

### Qualcomm mapping
On a compiled graph, `torch.arange(S) + past_seen_tokens` can't be a live op — `S` is fixed
(`= ARN`), but `past_seen_tokens` is a **runtime scalar**. Solution: precompute the ramp
`[0,1,...,ARN-1]` once as a buffer (`cache_tensor` in `qadaptation.py`; or
`qgenerator.py:364-366`: `torch.arange(kv_glb, kv_glb+arn)`), feed only the scalar starting
point (`cache_index`/`kv_glb`) as a real input. Same formula, split into
"fixed ramp (compile time) + scalar offset (runtime)."

**Checkpoint:** `position_ids` answers *"what angle do I rotate by."* Nothing to do with
the cache yet.

---

## Lesson 2: Prefill vs Decode — the two regimes that create all this complexity

### Intuition
Every downstream confusion (`cache_position`, offsets, mask shapes) exists *because*
generation happens in two distinct modes, and production code handles both with the same
code path:
- **Prefill**: a whole prompt already exists (say 20 tokens). Process all 20 at once — no
  cache to read from yet, you're *building* it.
- **Decode**: cache already holds K/V for prior tokens. You get exactly 1 new token, must
  attend it against everything cached, then add it to the cache.

### Toy example
Prompt = `"The cat sat on"` (4 tokens). Generating `"the"`, then `"mat"`.
```
Call 1 (PREFILL):  input = [The, cat, sat, on]   (4 tokens at once)
Call 2 (DECODE):   input = [the]                  (1 token)
Call 3 (DECODE):   input = [mat]                   (1 token)
```

### Tensor shapes
| | Query length (`Q`) | Already cached |
|---|---|---|
| Prefill | `Q = 4` | 0 |
| Decode step 1 | `Q = 1` | 4 |
| Decode step 2 | `Q = 1` | 5 |

### Pseudocode
```
if nothing cached yet:
    Q = len(prompt)              # prefill
else:
    Q = 1                         # decode
KV = (tokens already cached) + Q  # what attention needs to see
```

### HF Gemma4 code
No explicit `if prefill: ... else: ...` — it's implicit in *shapes*. `Gemma4TextModel.forward`
just does `inputs_embeds.shape[1]` for `Q`, `past_key_values.get_seq_length()` for however
many are cached. Empty cache → `get_seq_length()==0` → that's "prefill," by shape alone,
not by an explicit flag. This is exactly what confuses people reading the code — the *same
code* runs both regimes; only the runtime values differ.

### Qualcomm mapping
A static graph generally **cannot** serve both regimes as the same invocation — shapes
must be fixed. The pipeline compiles **separate graphs**: a big-ARN prefill graph and a
small/ARN=1 decode graph (see `GENAI_BUILDER_STAGE.md`'s prefix/decode split; `qmodel.py`'s
`Model_Builder` producing separate prefix/decode containers). What HF does "implicitly by
runtime shape," Qualcomm's pipeline does "explicitly by compiling two different graphs."

**Checkpoint:** now you know *why* two regimes exist. Next: the cache needs a way to know
"where does the new data go" — a different question from `position_ids`.

---

## Lesson 3: `cache_position` — "where do I *write* into the cache?"

### Intuition
`position_ids` answers "what angle do I rotate by" (semantic, per-token). `cache_position`
answers a completely different question: **"at what index/offset in the KV cache buffer
should this token's K/V be stored?"** — a *storage* question. In vanilla eager HF these two
numbers happen to be numerically identical (both just count "how many tokens came
before"), which is exactly why people conflate them. The moment you touch a
compiled/static-buffer implementation (Qualcomm), they diverge — that divergence is the
whole reason this concept needs its own name.

### Toy example
Same 4-token prefill, decode `"the"`:
```
Prefill:  cache_position = [0, 1, 2, 3]     ← write new K/V into slots 0,1,2,3
Decode:   cache_position = [4]              ← write new K/V into slot 4
```
Identical to `position_ids` so far. Divergence case — **alignment padding**
(real, in `qgenerator.py`: `write_base_glb = _align_up(kv_glb)`). Say writes must land on
8-aligned boundaries. After 4 real tokens, real count `kv_glb=4`, but write offset rounds
up to `write_base_glb=8`:
```
kv_glb (real tokens so far)             = 4
write_base_glb (where next write lands) = 8   ← NOT the same number!
cache_position for next call = [8]             ← writes to slot 8, not slot 4
position_ids for next call   = [4]             ← still rotates by angle for position 4!
```
This is the concrete case where `cache_position != position_ids` — exactly why production
code carries both as separate named things instead of one.

### Tensor shape
`cache_position: [Q]` (or `[B, Q]`) — same *length* as `position_ids`, different
*meaning*: indexes into the **cache buffer**, not the **RoPE angle table**.

### Pseudocode
```
cache_position[i] = write_offset + i     # WHERE to write — may include alignment padding
position_ids[i]   = real_position + i    # what angle to ROTATE by — always the true position
# Simple/vanilla implementations: write_offset == real_position → they look identical.
# Implementations with buffer alignment/scatter semantics: they can differ.
```

### HF Gemma4 code
Vanilla HF barely surfaces `cache_position` as its own concept for Gemma4 — folded into
`past_key_values.update()`'s bookkeeping, mostly overlapping with position tracking. It
becomes explicit and separate specifically in a cache implementation doing index-based
writes rather than pure append — exactly the QAIRT adaptations:
```python
# qadaptation.py, Gemma4TextAttention.forward:
cache_kwargs = {"cache_position": cache_position, "transposed_key_cache": ..., ...}
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
```
```python
# DynamicLayer_adapted.update() — cache_position used as literal SCATTER indices
indices = cache_position.view(1,1,1,-1).expand(...)
value_cache = self.values.scatter(dim=-2, index=indices, src=value_states)
```
The tell: `.scatter(index=cache_position, ...)` means "where to write," full stop — not
"what angle."

### Qualcomm mapping
`qadaptation.py`'s `cache_tensor` buffer builds `cache_position = cache_index + cache_tensor`
(fixed ramp + scalar). `qgenerator.py` tracks **two separate numbers** explicitly:
`kv_glb`/`kv_swa` (real past-token counts, feed masking/position_ids) vs.
`write_base_glb`/`write_base_swa` (aligned write offsets, feed cache_position/scatter).
`qgenerator.py:325-326` (`write_base_glb = _align_up(kv_glb)`) is the single clearest line
in the whole pipeline showing these are genuinely different numbers, not two names for one
thing.

**Checkpoint:** three distinct numbers now in play: `position_ids` (rotation angle),
`cache_position` (write offset), and implicitly a third — "how many valid tokens exist so
far" (`kv_glb`/`get_seq_length()`) — which feeds the **mask**, not the cache write, and not
RoPE. That third one is next.

---

## Lesson 4: Causal Masking — the core concept

### Intuition
Attention scores every query against every key. Causal masking zeroes out (or `-inf`s)
scores where a query would look at a key *after* it in time — a decoder must not see the
future. Separately, **padding masking** hides key slots that don't correspond to real
tokens at all (unwritten/reserved cache slots). Production code often conflates these two
into one tensor — keep them conceptually separate first.

### Toy example
4 real tokens, no padding, plain causal:
```
        kv=0  kv=1  kv=2  kv=3
q=0      ✓     ✗     ✗     ✗
q=1      ✓     ✓     ✗     ✗
q=2      ✓     ✓     ✓     ✗
q=3      ✓     ✓     ✓     ✓
```
Padding case: same 4 tokens, KV buffer has capacity for 6 slots (2 unwritten):
```
        kv=0  kv=1  kv=2  kv=3  kv=4  kv=5
q=3      ✓     ✓     ✓     ✓     ✗     ✗     ← kv=4,5 don't exist yet, forbidden regardless of causal rule
```
Both rules ("no future" + "no unwritten slots") apply simultaneously — final mask is their
intersection (AND).

### Tensor shape
`[B, 1, Q, KV]` — `1` = head axis (masking is head-agnostic, broadcasts across all heads).
`Q` = new query tokens this call, `KV` = key slots to compare against (vanilla HF: grows
every call; Qualcomm: fixed to buffer capacity).

### Pseudocode
```
for each query position q, key position kv:
    causal_ok  = (kv <= q)               # can't see the future
    exists_ok  = (kv < num_real_tokens)   # can't see unwritten slots
    mask[q, kv] = 0        if causal_ok AND exists_ok
    mask[q, kv] = -inf     otherwise
```

### HF Gemma4 code
```python
# transformers/masking_utils.py
def causal_mask_function(batch_idx, head_idx, q_idx, kv_idx) -> bool:
    return kv_idx <= q_idx                      # the "no future" rule, as a predicate

def create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids, ...):
    ...
    kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)   # <- from the CACHE
    ...
    mask = mask_function(*broadcast over arange(q_length)+q_offset, arange(kv_length)+kv_offset)
```
Note: the padding/"exists" concern in vanilla HF usually isn't separate — HF's cache
*only ever contains real tokens* (grows exactly by however many tokens exist, no
pre-allocated unwritten slack). So vanilla HF's causal mask alone suffices — there's no
"unwritten slot" case, since the tensor never has unwritten slots. **This assumption
breaks in the Qualcomm adaptation** (fixed-size buffer, mostly unwritten most of the time)
— which is why the adapted mask-building code explicitly tracks "which slots are real" as
a *second* condition, not just "no future."

### Qualcomm mapping
```python
# qgenerator.py, build_text() — literally builds both conditions explicitly, per row
causal_mask = torch.full((1,1,arn,gcl), mask_neg)          # start: EVERYTHING forbidden (handles "doesn't exist yet")
for r in range(num_real):
    past_glb = glb_cols < kv_glb                             # "this key already existed before this call"
    new_glb  = (glb_cols >= write_base_glb) & (glb_cols <= write_base_glb + r)   # causal rule, applied only to the new region
    causal_mask[0, 0, r, past_glb | new_glb] = 0.0            # flip to ALLOWED
```
Read literally: "start with everything forbidden" already handles the padding/exists rule
(unwritten slots default to forbidden, never touched). `past_glb | new_glb` OR encodes the
causal rule *on top of* that — `new_glb`'s `<= write_base_glb + r` is the "no future" rule,
restricted to the newly-written region (anything in `past_glb` is unconditionally real and
unconditionally in the past, by construction). Two rules, one mask, built by
"default-deny then selectively allow" instead of HF's "compute a boolean predicate over a
known-fully-real range."

**Checkpoint — the reframe that should click:** vanilla HF's mask only ever needs the
causal rule because its cache never has "fake" slots. Qualcomm's mask needs causal **and**
existence rules because its cache is a fixed buffer that's mostly unwritten. This single
fact explains almost every extra line of complexity in `qgenerator.py`'s mask code versus
`masking_utils.py`'s.

---

## Lesson 5: Offsets — connecting `cache_position`/`kv_glb` to the mask's shape

### Intuition
A mask tensor's indices always start at `0` (`torch.arange(Q)`, `torch.arange(KV)`) — but
the *true* sequence positions being compared usually aren't `0`-based (decode step 500
isn't asking "is kv_idx ≤ 0", it's asking "is kv_idx ≤ 500"). An **offset** is the number
added to a local `0..N-1` index to recover the true absolute position, so the causal
*comparison* runs on real numbers, not accidentally-reset-to-0 local ones.

### Toy example
Decode step, 1 new token, 20 tokens already cached:
```
Local index (what arange gives you):  q_idx_local = 0
q_offset = 20   (20 tokens already existed)
True q_idx = q_offset + q_idx_local = 20
```
Forgetting the offset and comparing `kv_idx <= 0` instead of `kv_idx <= 20` would wrongly
forbid the new token from seeing almost its entire history.

### Tensor shapes
Offsets are scalars (or 1-element per-batch tensors), not full tensors — they shift a
`torch.arange(...)` before it's used in the predicate.

### Pseudocode
```
q_arange  = arange(Q)  + q_offset      # local index -> true position
kv_arange = arange(KV) + kv_offset
mask = causal_predicate(q_arange, kv_arange)   # now comparing REAL positions
```

### HF Gemma4 code
```python
# masking_utils.py, _preprocess_mask_arguments
q_offset = past_key_values.get_seq_length()                  # "how many tokens came before" — from the cache
kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)   # cache tells you both
```
Full-attention (`DynamicLayer`): `kv_offset` always `0` (nothing evicted, oldest key always
position 0). Sliding (`DynamicSlidingWindowLayer`): `kv_offset` shifts forward once the
window fills — the oldest *surviving* key genuinely isn't at position 0 anymore.

### Qualcomm mapping
`qgenerator.py` doesn't use the words "q_offset"/"kv_offset" but computes the *exact same
quantities* under different names: `kv_glb`/`kv_swa` **are** the `q_offset`/`kv_offset`
equivalents (real-position counters). `write_base_glb`/`write_base_swa` are a *third*,
Qualcomm-specific quantity vanilla HF never needs (alignment-adjusted write offset).
Generalization: **offsets exist in both worlds to answer "what's the true absolute
position," but Qualcomm needs an extra offset because its writes can land at
aligned-but-not-literally-sequential positions.**

---

## Lesson 6: Local (sliding) vs Global (full) attention layers

### Intuition
Full attention: "I can see everything causally before me, no matter how long ago."
Sliding attention: "I can only see a fixed-size trailing window — anything older is
invisible, even though it's causally valid." Gemma4 alternates these (5 sliding : 1 full)
as a compute/memory tradeoff — for masking/caching purposes, the important thing: **these
are two different rules, requiring two different masks and (in vanilla HF) two different
cache-layer classes.**

### Toy example
Window = 4, at query position 10:
```
Full attention   at q=10: allowed kv ∈ {0,1,...,10}          (everything so far)
Sliding attention at q=10 (window=4): allowed kv ∈ {7,8,9,10} (only trailing 4)
```

### Tensor shapes
Both masks `[B,1,Q,KV]` — but `KV` for full-attention layers tracks toward
`context_length`, while `KV` for sliding layers tracks toward (or caps at)
`sliding_window`. Two **separate mask tensors** exist per model, selected per-layer by
`config.layer_types[i]`.

### Pseudocode
```
full_mask[q,kv]    = 0 if kv <= q else -inf
sliding_mask[q,kv] = 0 if (kv <= q) AND (kv > q - window) else -inf
```

### HF Gemma4 code
```python
causal_mask_mapping = {
    "full_attention":    create_causal_mask(**mask_kwargs),
    "sliding_attention":  create_sliding_window_causal_mask(**mask_kwargs),
}
...
for i, decoder_layer in enumerate(self.layers):
    mask = causal_mask_mapping[self.config.layer_types[i]]     # per-layer SELECT
```
Cache side: vanilla HF's `DynamicCache(config=...)` builds **different cache-layer
classes** per type: `DynamicSlidingWindowLayer` (self-truncates every `update()`) for
sliding, `DynamicLayer` (never truncates) for full — exactly the class split behind the
KV-cache trap (qairt's monkeypatch only reaches `DynamicLayer`).

### Qualcomm mapping
Same two-mask idea, two parallel code blocks in `qgenerator.py`/`qadaptation.py`:
`attention_mask`/`gcl`/`kv_glb`/`write_base_glb` for full, `swa_attention_mask`/`lcl`/
`kv_swa`/`write_base_swa` for sliding — mirrored variable names, computed side-by-side in
the same per-row loop. But **the cache class itself is unified** (`DynamicLayer_adapted`,
no sliding-specific subclass) — the type distinction lives *entirely* in which
mask/buffer-size gets used, never in the cache-object's own logic. Biggest structural
difference from vanilla HF's per-type-class approach — a deliberate simplification (one
patchable class, correctness delegated to the mask).

---

## Lesson 7: Tying it together — the four questions

Quick anchor, since the cache mechanics themselves are covered in depth elsewhere: the
cache stores K/V so old tokens' projections aren't recomputed every step. `cache_position`
(Lesson 3) says *where* to write into it. `position_ids` (Lesson 1) says what RoPE angle a
token gets, independent of cache mechanics. The mask (Lessons 4-6) says which of the
cache's *contents* — real or padding, full or windowed — a given query may actually look
at. All four concepts answer four different questions about the same moment in time —
whenever one of these names shows up in code, ask **"which of the four questions is this
variable answering?"**

| Concept | Question it answers |
|---|---|
| `position_ids` | What RoPE angle do I rotate by? |
| `cache_position` | Where in the cache buffer do I write? |
| `kv_glb` / `get_seq_length()` | How many real tokens exist so far? (feeds the mask) |
| the mask itself | Given what exists, what am I *allowed to look at*? |

---

## Lesson 8: Prefill vs Decode, revisited — full vocabulary

**Prefill** (4-token prompt, empty cache):
```
position_ids   = [0,1,2,3]           # RoPE angles
cache_position = [0,1,2,3]           # write offsets (no alignment gap yet)
kv_glb (before) = 0                   # nothing existed
mask: causal_mask[q, kv] = 0 for kv<=q, else mask_neg   # standard triangle, restricted to written region
```

**Decode step** (5th token, cache now holds 4):
```
position_ids   = [4]                 # true 5th position
cache_position = [4]  (or [8] if alignment padded — this is where they could diverge)
kv_glb (before) = 4
mask: allow kv in {0,1,2,3,4}, forbid rest
```

---

## Suggested next exercise

Trace a real ARN=521 prefill → first decode step by hand, filling in every one of these
variables (`position_ids`, `cache_position`, `kv_glb`, `write_base_glb`, mask shape and
which columns flip to 0) with real numbers — the fastest way to convert "followed the
lesson" into "can independently read this code."
