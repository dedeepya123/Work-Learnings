# KV Cache Mechanics — Full vs Sliding Attention (Gemma4 / HF `DynamicCache`)

Personal study notes: how `DynamicLayer` vs `DynamicSlidingWindowLayer` actually store and
return key/value tensors, worked through with concrete numbers. Code: `transformers/cache_utils.py`.

---

## 1. Lazy initialization — what it means

```python
def __init__(self):
    self.keys: torch.Tensor | None = None
    self.values: torch.Tensor | None = None
    self.is_initialized = False

def lazy_initialization(self, key_states, value_states) -> None:
    self.dtype, self.device = key_states.dtype, key_states.device
    self.keys = torch.tensor([], dtype=self.dtype, device=self.device)   # shape [0]
    self.values = torch.tensor([], dtype=self.dtype, device=self.device)
    self.is_initialized = True
```

`DynamicCache()` allocates nothing up front (`self.keys = None`) — dtype/device (and, for
some construction paths, even how many layers exist) aren't known yet. The first `update()`
call swaps `None` for an **empty 1-D tensor of shape `[0]`**, purely so `torch.cat` has
something to concatenate against.

Empirically: `torch.cat([torch.tensor([]), real_4d_tensor], dim=-2)` returns the real 4D
tensor unchanged — concatenating an empty tensor is a no-op. That's what lets
`lazy_initialization` + the first `cat` collapse into "just start with the real tensor,"
without a special-cased first branch in `update()`.

---

## 2. Full attention (`DynamicLayer`) — worked example

Setup: `B=1`, `seq=8`, prefill computes all 8 positions in one call, one layer.

```
Q: [1, num_q_heads, 8, head_dim]
K: [1, num_kv_heads, 8, head_dim]
V: [1, num_kv_heads, 8, head_dim]
```

```python
def update(self, key_states, value_states, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
    self.keys = torch.cat([self.keys, key_states], dim=-2)      # cat([0], [1,h,8,d]) -> [1,h,8,d]
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values
```

After prefill: `self.keys.shape == [1, num_kv_heads, 8, head_dim]`. **Nothing is ever
dropped** — hence "Dynamic": it just keeps growing. Next decode step (1 new token):
`cat([1,h,8,d], [1,h,1,d]) -> [1,h,9,d]`, stored and returned as-is. Unbounded growth.

---

## 3. Sliding window (`DynamicSlidingWindowLayer`) — worked example, window=4, seq=12

```python
class DynamicSlidingWindowLayer(DynamicLayer):
    def __init__(self, sliding_window):
        self.sliding_window = sliding_window          # = 4
        self.cumulative_length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]

        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]   # keep only window-1
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
        return full_key_states, full_value_states                           # return everything
```

**Step A — prefill, all 12 tokens in one `update()` call:**
- Not initialized → lazy-init → `self.keys = tensor([])`
- `cumulative_length = 0 + 12 = 12`
- `full_key_states = cat([empty], [1,h,12,d]) = [1,h,12,d]`
- `self.keys = full_key_states[:, :, -3:, :]` → **keeps only the last 3** (positions 9,10,11;
  `-sliding_window + 1 = -3`)
- **Returns** `full_key_states` → shape `[1,h,12,d]` — the complete 12, handed to attention

After this single call, `self.keys` holds only 3 tokens — not 4, not 12. That's fine:
attention this step used the **returned** 12-token tensor, masked per query position by
`sliding_window_overlay` (query at position 11 sees `kv` where `11-4 < kv ≤ 11` → positions
8,9,10,11 — all present in the returned tensor; the mask hides the rest). The cache's job
at this instant is "hand back everything computed"; the **mask** enforces who may look at what.

**Why keep exactly `window - 1`, not `window`:** it's sized for the *next* call to top back
up to a full window. Next arrival (1 new token during decode) concatenates onto these 3:
`3 + 1 = 4 = sliding_window`. Stored buffer is `window - 1` so that "stored + next arrival"
lands exactly at `window`, never over.

**Step B — decode step, 1 new token (position 12):**
- Already initialized, skip lazy-init
- `cumulative_length = 12 + 1 = 13`
- `full_key_states = cat([9,10,11 stored], [12 new]) = [1,h,4,d]` (positions 9,10,11,12)
- `self.keys = full_key_states[:, :, -3:, :]` → keep last 3 → positions **10,11,12**
- **Returns** the 4-token tensor `[9,10,11,12]`

Query at position 12: mask allows `12-4 < kv ≤ 12` → kv in `(8,12]` → positions 9,10,11,12 —
exactly what's in the returned tensor. Match.

**Step C — next decode step (position 13):**
- `cat([10,11,12 stored], [13 new]) = [10,11,12,13]`
- store last 3 → `[11,12,13]`
- return `[10,11,12,13]`

From here on, steady state: cache holds 3, incoming adds 1, returns exactly `window` (4),
stores back `window - 1` (3).

---

## 4. Full vs sliding — side by side

| | Full attention (`DynamicLayer`) | Sliding attention (`DynamicSlidingWindowLayer`, window=4) |
|---|---|---|
| What's **stored** after prefill(12) | all 12 | only last 3 (`window - 1`) |
| What's **returned** at any step | everything stored | everything just computed **this call** (stored + incoming), *before* truncation |
| Growth pattern | unbounded, grows every step | stored buffer caps at `window - 1`; returned tensor caps at `window` once steady-state hits |
| Who enforces "can't see too far back" | mask only (tensor holds everything) | **both** mask (during prefill, before storage catches up) **and** storage truncation (steady-state decode) |

**One-line takeaway:** during prefill, sliding-window correctness is carried entirely by the
mask (the returned tensor still contains the whole prompt); it's only from the first
post-prefill decode step onward that the *storage* truncation itself starts limiting what's
even physically available to return.

---

## 5. Tying back to attention shapes generally

- Cache tensors are always `[B, num_kv_heads, kv_seq_len_so_far, head_dim]` — **not**
  `num_q_heads` (GQA: Gemma4 has `num_attention_heads=8`, `num_key_value_heads=4` in this
  checkpoint's config; KV heads get `repeat_kv`'d up to match Q heads only inside
  `eager_attention_forward`, never inside the cache itself).
- At any single step, for *every* layer type, the K/V tensor handed to attention is the
  **full history seen so far by that call** — windowing is a masking concept for full vs.
  sliding *inside a single step's attention computation*, and (for sliding layers) also a
  storage-truncation concept for *what persists into the next step*. These are two
  different mechanisms answering two different questions ("who can I look at right now" vs.
  "what do I keep around for later"), and it's easy to conflate them if you only look at the
  mask or only look at the cache in isolation.
