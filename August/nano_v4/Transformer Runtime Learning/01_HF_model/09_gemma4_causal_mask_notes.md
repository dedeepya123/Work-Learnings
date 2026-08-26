# Causal Mask Creation — Full vs Sliding Attention (Gemma4 / HF `masking_utils.py`)

Personal study notes: how the attention mask is built per layer type, for both prefill and
decode, with the exact numbers traced through. Code: `transformers/masking_utils.py`,
`transformers/cache_utils.py`.

---

## 1. Mask shape

**`[B, 1, Q, KV]`** — confirmed directly from the docstrings:

- `sdpa_mask` (masking_utils.py:384): *"Create a 4D boolean mask of shape
  `(batch_size, 1, query_length, kv_length)`"*
- `eager_mask` (masking_utils.py:560): *"Create a 4D float mask of shape
  `(batch_size, 1, query_length, kv_length)`"* — `0.0` where allowed, `-inf` (dtype min)
  where forbidden. This is the version used by eager attention (what the QAIRT/HTP path
  uses).

The `1` is the **head dimension** — one mask shared across all attention heads, since
masking is a token-visibility rule (who can look at whom), not a per-head concept. It
broadcasts against `attn_weights` of shape `[B, H, Q, KV]` at apply time.

`KV` is **not always equal to `Q`**:
- Prefill with an empty cache: `KV == Q` (no history yet).
- Decode: `KV = (tokens already cached) + Q`, or — for sliding-window layers, once the
  window has filled — `KV` pins to the window size regardless of how long generation runs.

---

## 2. Offsets — what `q_offset` / `kv_offset` actually mean

This is the part that's easy to gloss over, so slow down here.

`torch.arange` always starts counting from `0`. But the *actual* position of a query or key
token in the overall sequence is usually **not** `0` — e.g. on a decode step after an
8-token prefill, the new query token is at absolute position `8`, not position `0`.

Offsets are simply: **"how much do I need to shift a local `0..N-1` index range to get the
true absolute sequence position?"**

```python
q_arange  = torch.arange(q_length)  + q_offset    # local query index  -> absolute position
kv_arange = torch.arange(kv_length) + kv_offset    # local key/value index -> absolute position
```

- **`q_offset`** = number of tokens already processed before this call =
  `past_key_values.get_seq_length()`. Prefill → `0` (nothing processed yet). Decode step N
  → however many tokens came before (grows every step).
- **`kv_offset`** = absolute position of the **first** key/value entry that will appear in
  this step's `kv_arange`. For full attention this is always `0` (the oldest token is
  always position 0 — nothing is ever evicted). For sliding attention, once the window
  fills, the oldest *available* key is no longer position 0 — it's been evicted — so
  `kv_offset` has to shift forward to track "where the oldest still-cached token actually
  sits in the real sequence."

Why this matters concretely: the mask predicate (`causal_mask_function`,
`sliding_window_overlay`) only knows how to compare **raw index values** like
`kv_idx <= q_idx`. If those indices weren't shifted to true absolute positions first, a
sliding-window layer whose cache only holds the *last 3* tokens would wrongly think those 3
tokens are at positions `0,1,2` instead of their real positions (e.g. `5,6,7`) — the
causal/window comparison would be comparing the wrong numbers entirely. Offsets are what
keep the predicate's arithmetic anchored to true sequence positions even though the
*tensor* holding those tokens is always small and re-indexed from 0 internally.

Where offsets come from, per layer type (`cache_utils.py`):
```python
# DynamicLayer.get_mask_sizes
def get_mask_sizes(self, query_length):
    kv_offset = 0                                    # nothing ever evicted -> oldest is always position 0
    kv_length = self.get_seq_length() + query_length
    return kv_length, kv_offset

# DynamicSlidingWindowLayer.get_mask_sizes
def get_mask_sizes(self, query_length):
    is_full = self.cumulative_length >= self.sliding_window
    kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)   # oldest surviving token's true position
    kv_length = (self.sliding_window - 1 + query_length) if is_full else (self.cumulative_length + query_length)
    return kv_length, kv_offset
```
And the overall `q_offset` used across both types comes from `_preprocess_mask_arguments`
(masking_utils.py:846): `q_offset = past_key_values.get_seq_length()`.

---

## 3. The mask-function pattern — small boolean predicates, then broadcast

Everything is built from tiny predicates over `(batch_idx, head_idx, q_idx, kv_idx)`
(these are the **absolute**, offset-adjusted positions from above), then vectorized:

```python
def causal_mask_function(batch_idx, head_idx, q_idx, kv_idx) -> bool:
    return kv_idx <= q_idx                       # standard causal: can't see the future

def sliding_window_overlay(sliding_window) -> Callable:
    def inner_mask(batch_idx, head_idx, q_idx, kv_idx) -> bool:
        return kv_idx > q_idx - sliding_window   # can't see further back than the window
    return inner_mask

def sliding_window_causal_mask_function(sliding_window) -> Callable:
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)   # AND of both
```

- **Full attention** (`create_causal_mask`) uses `causal_mask_function` alone.
- **Sliding attention** (`create_sliding_window_causal_mask`) uses the **AND** of the
  sliding overlay and the causal function — both must hold.

`sdpa_mask`/`eager_mask` vectorize the predicate over full ranges and broadcast to `[B,1,Q,KV]`:
```python
q_arange = torch.arange(q_length) + q_offset
kv_arange = torch.arange(kv_length) + kv_offset
attention_mask = mask_function(batch_arange, head_arange, q_arange, kv_arange)   # -> [B,1,Q,KV]
```
`eager_mask` then converts bool → float: `0.0` (allowed) / `dtype_min` ≈ `-inf` (forbidden).

---

## 4. Worked trace — full attention (window irrelevant), 8-token prompt + 2 decode steps

**Prefill** (`q_length=8`, cache empty → `get_seq_length()=0`):
- `kv_length = 0 + 8 = 8`, `kv_offset = 0`
- `q_arange = [0..7] + 0 = [0..7]`, `kv_arange = [0..7] + 0 = [0..7]`
- Mask shape `[1,1,8,8]`. Predicate `kv_idx <= q_idx` → standard lower-triangular.
  Row `q_idx=3` allows `kv_idx ∈ {0,1,2,3}`.

**Decode step 1** (`q_length=1`, cache now has 8 stored → `get_seq_length()=8`):
- `kv_length = 8 + 1 = 9`, `kv_offset = 0`
- `q_arange = [0] + 8 = [8]`, `kv_arange = [0..8] + 0 = [0..8]`
- Mask shape `[1,1,1,9]`. Row for `q_idx=8`: `kv_idx <= 8` → **all 9** positions allowed.

**Decode step 2** (`get_seq_length()=9`):
- `kv_length=10`, `kv_offset=0`, `q_arange=[9]`, `kv_arange=[0..9]` → all 10 allowed.

**Pattern:** full attention's mask is always "everything up to and including current
position." `kv_offset` never moves (nothing is ever evicted); `kv_length` simply grows by 1
every step.

---

## 5. Worked trace — sliding attention, `sliding_window=4`, same 8-token prompt

Note: `get_mask_sizes()` is called using the cache's state **before** this step's
`update()` runs (mask is built, then attention runs, then the cache updates/truncates for
next time) — so `cumulative_length` below reflects the *pre-update* value at each step.

**Prefill** (`q_length=8`, `cumulative_length=0` going in):
- `is_full = 0 >= 4` → `False`
- `kv_offset = max(0 - 4 + 1, 0) = 0`
- `kv_length = 0 + 8 = 8` (not-full branch: `cumulative_length + query_length`)
- `q_arange = [0..7] + 0`, `kv_arange = [0..7] + 0`
- predicate = AND(`kv_idx > q_idx - 4`, `kv_idx <= q_idx`)
  - Row `q_idx=0`: `kv_idx > -4` AND `kv_idx<=0` → `{0}`
  - Row `q_idx=3`: `kv_idx > -1` AND `kv_idx<=3` → `{0,1,2,3}` (window not full yet — behaves like plain causal)
  - Row `q_idx=7`: `kv_idx > 3` AND `kv_idx<=7` → `{4,5,6,7}` — **window kicks in**: position 7 can no longer see positions 0–3.

This matches HF's own `sdpa_mask` docstring example exactly (same shape pattern, window=3
there). **During prefill, the mask alone enforces sliding-window visibility** — the cache
tensor itself still physically contains the entire prompt at this point (per the KV-cache
mechanics notes); nothing has been evicted yet mid-call.

*After* prefill, the cache truncates: `self.keys` now holds only the last `4-1=3` tokens
(positions 5,6,7). `cumulative_length` becomes `8`.

**Decode step 1** (`q_length=1`, pre-update state: `cumulative_length=8`):
- `is_full = 8 >= 4` → `True`
- `kv_offset = max(8 - 4 + 1, 0) = 5`
- `kv_length = 4 - 1 + 1 = 4` (full branch: `sliding_window - 1 + query_length`)
- `q_offset` (from `past_key_values.get_seq_length()` = `cumulative_length` = `8`) → `q_arange = [8]`
- `kv_arange = [0..3] + 5 = [5,6,7,8]`
- predicate for `q_idx=8`: `kv_idx > 8-4=4` AND `kv_idx<=8` → all of `{5,6,7,8}` pass →
  mask `[1,1,1,4]`, **all True**.

Cross-check against the actual K/V tensor at this step (cache-mechanics notes): stored
`[5,6,7]` + new `[8]` → returned tensor is exactly `[5,6,7,8]` — matches `kv_arange`
exactly. Mask says "all 4 allowed," correct: with window=4, a query at position 8 should
see positions 5–8 inclusive, exactly what's physically present.

**Decode step 2** (`cumulative_length=9` pre-update):
- `is_full=True`, `kv_offset = max(9-4+1,0)=6`, `kv_length=4`
- `q_arange=[9]`, `kv_arange=[6,7,8,9]`
- predicate: `kv_idx > 9-4=5` AND `kv_idx<=9` → all pass → mask `[1,1,1,4]`, all True.

Matches the K/V tensor `[6,7,8,9]` from the cache trace.

**Key observation:** once `is_full=True`, `kv_length` is **constant at `sliding_window`**
every step (it stops growing) — this is the steady-state regime where mask shape and cache
size are both pinned at the window size, and the mask ends up "trivially all-True" because
the offset/length bookkeeping already excludes anything outside the window *before* the
predicate even has to reject anything.

---

## 6. Applying the mask during attention compute — identical for prefill/decode

This part doesn't distinguish prefill vs. decode, or full vs. sliding — it's the same
masked-softmax regardless, e.g. `eager_attention_forward` / `qc_gemma4_eager_attention_forward`:

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling   # [B, H, Q, KV]
attn_weights = attn_weights + attention_mask                                # additive; [B,1,Q,KV] broadcasts over H
attn_weights = softmax(attn_weights, dim=-1)                                # -inf -> 0 probability after softmax
attn_output = torch.matmul(attn_weights, value_states)
```

The mask is **added**, not multiplied, before softmax: `exp(-inf) = 0`, so forbidden
positions contribute exactly zero probability mass. Since the mask has no head axis
(`[B,1,Q,KV]`) but `attn_weights` does (`[B,H,Q,KV]`), it broadcasts across every head —
confirming that masking is head-agnostic: which tokens may attend to which is a property of
position and layer type, not of which head is computing the score.

---

## 7. Summary table

| | Full attention (`causal_mask_function`) | Sliding attention (`sliding_window_causal_mask_function`) |
|---|---|---|
| `kv_offset` | always `0` (nothing evicted) | `0` while filling; then `cumulative_length - window + 1` once full — tracks the true position of the oldest surviving cached token |
| `kv_length` | `seen_so_far + q_length` — grows every step | `cumulative_length + q_length` while filling; pins to `sliding_window` once full |
| Predicate | `kv_idx <= q_idx` | `kv_idx <= q_idx` **AND** `kv_idx > q_idx - window` |
| Prefill enforcement | mask = plain lower triangle | mask alone enforces the window (cache tensor still holds full prompt at this point) |
| Decode (steady state) enforcement | mask = "everything so far" | mask ends up all-True — window is already enforced by *what's physically in the cache* (offset/length bookkeeping matches exactly what storage kept) |

**One-line takeaway:** offsets exist to translate a cache/mask tensor's always-`0`-based
local indexing back into true absolute sequence positions, so the causal/window predicates
compare real positions rather than accidentally-reset-to-0 local ones; without correct
offsets, a sliding-window layer's small re-indexed cache tensor would make the mask
compare the wrong numbers entirely.
