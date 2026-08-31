# Worked Trace: Mask Construction, HF vs QC, Full vs Sliding Attention

One concrete example run through all four combinations (HF-full, HF-sliding, QC-full,
QC-sliding), hand-computed from the actual verified formulas in `masking_utils.py`
(vanilla HF) and `qgenerator.py` (QAIRT adaptation), so the logical equivalence — and the
one structural difference that matters — is visible in real numbers, not just prose.

**Setup.** 6 tokens generated as: prefill 3 tokens (positions 0,1,2) → decode position 3 →
decode position 4 → decode position 5. `sliding_window = 4`.
QC fixed buffers: `gcl=12` (global/full-attention buffer), `lcl=8` (local/sliding buffer)
— both deliberately larger than the 6 tokens we'll ever write, so cache eviction (a
separate upstream mechanism, not covered here) never triggers. **Simplification for this
trace:** no write-alignment padding — `write_base_glb == kv_glb` and
`write_base_swa == kv_swa` exactly, isolating the full-vs-sliding / HF-vs-QC comparison
from the alignment-offset complexity already covered in the whiteboard lesson (Lesson 3).

---

## Part A — Full (global) attention

### A1. HF — `DynamicLayer` + `create_causal_mask`

Formula (`masking_utils.py`, `DynamicLayer.get_mask_sizes`):
```
kv_offset = 0
kv_length = get_seq_length() + query_length
q_offset  = get_seq_length()
predicate: kv_idx <= q_idx
```

| Call | `get_seq_length()` (before) | `q_offset` | `kv_length` | `q_arange` | `kv_arange` | Allowed kv per row |
|---|---|---|---|---|---|---|
| Prefill (3 tok) | 0 | 0 | 3 | `[0,1,2]` | `[0,1,2]` | row0:{0} row1:{0,1} row2:{0,1,2} |
| Decode pos3 | 3 | 3 | 4 | `[3]` | `[0,1,2,3]` | row3:{0,1,2,3} |
| Decode pos4 | 4 | 4 | 5 | `[4]` | `[0,1,2,3,4]` | row4:{0,1,2,3,4} |
| Decode pos5 | 5 | 5 | 6 | `[5]` | `[0,1,2,3,4,5]` | row5:{0,1,2,3,4,5} |

Mask **shape grows** every call: `[1,1,3,3]` → `[1,1,1,4]` → `[1,1,1,5]` → `[1,1,1,6]`.
Every row sees **everything up to itself** — the defining property of full attention:
nothing is ever excluded once written.

### A2. QC — `torch.full(gcl) + per-row loop` (`qgenerator.py:346,351-354`)

Formula:
```python
causal_mask = torch.full((1,1,arn,gcl), mask_neg)      # start ALL masked
for r in range(num_real):
    past_glb = glb_cols < kv_glb
    new_glb  = (glb_cols >= write_base_glb) & (glb_cols <= write_base_glb + r)
    causal_mask[0,0,r, past_glb | new_glb] = 0.0
```

| Call | `kv_glb` (before) | `write_base_glb` | Row(s), cols flipped to `0.0` | Cols still `mask_neg` |
|---|---|---|---|---|
| Prefill (r=0,1,2) | 0 | 0 | r=0:{0} · r=1:{0,1} · r=2:{0,1,2} | r=0: 1-11 · r=1: 2-11 · r=2: 3-11 |
| Decode pos3 (r=0) | 3 | 3 | {0,1,2,3} | 4-11 |
| Decode pos4 (r=0) | 4 | 4 | {0,1,2,3,4} | 5-11 |
| Decode pos5 (r=0) | 5 | 5 | {0,1,2,3,4,5} | 6-11 |

Mask **shape stays `[1,1,arn,12]` every single call** — only which columns are `0.0` vs
`mask_neg` changes.

### A3. Side-by-side grid — full attention, decode step at position 5

```
HF   (shape [1,1,1,6]):     kv:  0    1    2    3    4    5
                            q=5: ✓    ✓    ✓    ✓    ✓    ✓        (nothing beyond col 5 EXISTS)

QC   (shape [1,1,1,12]):    kv:  0    1    2    3    4    5    6    7    8    9   10   11
                            q=5: ✓    ✓    ✓    ✓    ✓    ✓    ✗    ✗    ✗    ✗    ✗    ✗
                                 └──────── real, written ────────┘  └─── unwritten slack ───┘
```
**Identical decision** on the 6 real positions. QC just carries 6 extra always-masked
columns because its buffer is allocated wider than currently needed.

---

## Part B — Sliding (local) attention, `sliding_window = 4`

### B1. HF — `DynamicSlidingWindowLayer` + `create_sliding_window_causal_mask`

Formula (`masking_utils.py` + `DynamicSlidingWindowLayer.get_mask_sizes`, evaluated using
**pre-update** `cumulative_length` each call):
```
is_full   = cumulative_length >= sliding_window
kv_offset = max(cumulative_length - sliding_window + 1, 0)
kv_length = (sliding_window - 1 + query_length) if is_full else (cumulative_length + query_length)
q_offset  = cumulative_length
predicate: (kv_idx <= q_idx) AND (kv_idx > q_idx - sliding_window)
```

| Call | `cumulative_length` (before) | `is_full` | `kv_offset` | `kv_length` | `kv_arange` | Allowed kv |
|---|---|---|---|---|---|---|
| Prefill r=0 | 0 | False | 0 | 0+1=1 → *(see note)* | | |
| Prefill (all 3, one call) | 0 | False | 0 | 0+3=3 | `[0,1,2]` | row0:{0} row1:{0,1} row2:{0,1,2} (triangle — window not yet exceeded) |
| Decode pos3 | 3 | **False** (3<4) | 0 | 3+1=4 | `[0,1,2,3]` | row3: `kv>3-4=-1` & `kv<=3` → {0,1,2,3} — all 4, window fits exactly |
| Decode pos4 | 4 | **True** (4≥4) | max(4-4+1,0)=1 | 4-1+1=4 | `[1,2,3,4]` | row4: `kv>0` & `kv<=4` → {1,2,3,4} — **0 excluded, window now active** |
| Decode pos5 | 5 | True | max(5-4+1,0)=2 | 4 | `[2,3,4,5]` | row5: `kv>1` & `kv<=5` → {2,3,4,5} — **0,1 excluded** |

*(Prefill's `kv_length` note: computed once for the whole 3-token call, not per-row — the
table above uses the single real prefill call.)*

Underlying cache-buffer contents at each point (from `DynamicSlidingWindowLayer.update()`,
self-truncating to `sliding_window - 1 = 3` stored):
```
after prefill:      stored = [0,1,2]        (3 tokens, buffer not yet over capacity)
after decode pos3:  stored = [1,2,3]        (4 total seen, keeps last 3)
after decode pos4:  stored = [2,3,4]
after decode pos5:  stored = [3,4,5]
```
Notice: **position 0's actual K/V tensor no longer physically exists** in HF's cache after
the pos4 step — it was `torch.cat`-and-sliced away. The mask excludes it because it's
*gone*, not (only) because a rule says to ignore it.

### B2. QC — `torch.full(lcl) + per-row loop` (`qgenerator.py:347,356-361`)

Formula:
```python
window_lo = max(0, kv_swa + r - win + 1)
past_swa  = (swa_cols >= window_lo) & (swa_cols < kv_swa)
new_swa   = (swa_cols >= write_base_swa) & (swa_cols <= write_base_swa + r)
sliding_mask[0,0,r, past_swa | new_swa] = 0.0
```

| Call | `kv_swa` (before) | `write_base_swa` | `window_lo` (r=0) | Cols flipped to `0.0` | Cols `mask_neg` |
|---|---|---|---|---|---|
| Prefill r=0 | 0 | 0 | max(0,0-3)=0 | past_swa: none (kv_swa=0) · new_swa: {0} → **{0}** | 1-7 |
| Prefill r=1 | 0 | 0 | max(0,1-3)=0 | new_swa: {0,1} → **{0,1}** | 2-7 |
| Prefill r=2 | 0 | 0 | max(0,2-3)=0 | new_swa: {0,1,2} → **{0,1,2}** | 3-7 |
| Decode pos3 (r=0) | 3 | 3 | max(0,3-3)=0 | past_swa: cols in [0,3)={0,1,2} · new_swa:{3} → **{0,1,2,3}** | 4-7 |
| Decode pos4 (r=0) | 4 | 4 | max(0,1)=1 | past_swa: cols in [1,4)={1,2,3} · new_swa:{4} → **{1,2,3,4}** | 0, 5-7 |
| Decode pos5 (r=0) | 5 | 5 | max(0,2)=2 | past_swa: cols in [2,5)={2,3,4} · new_swa:{5} → **{2,3,4,5}** | 0,1, 6-7 |

Mask shape: **`[1,1,arn,8]` every call, never changes.**

### B3. Side-by-side grid — sliding attention, decode step at position 4 (the "aha" step)

```
HF   (shape [1,1,1,4]):     kv:  1    2    3    4
                            q=4: ✓    ✓    ✓    ✓        (position 0's tensor is GONE — cat-and-sliced away)

QC   (shape [1,1,1,8]):     kv:  0    1    2    3    4    5    6    7
                            q=4: ✗    ✓    ✓    ✓    ✓    ✗    ✗    ✗
                                 ↑                        └── unwritten slack ──┘
                            position 0's K/V is STILL PHYSICALLY IN THE BUFFER
                            (never evicted/truncated) — the MASK alone hides it.
```

**This is the one structural difference that actually matters, made concrete with real
numbers:** HF enforces the sliding window partly via **cache-buffer truncation** (old data
is deleted, so it literally can't be attended to) — the mask only has to handle the
*currently-fitting* window. QC enforces the sliding window **entirely via the mask** — the
cache-adaptation class (`DynamicLayer_adapted`) never truncates anything; column 0's real
K/V sits in the buffer, unchanged, forever (or until an eviction mechanism elsewhere
overwrites it) — it's `sliding_mask[...,0] = mask_neg` alone doing the exclusion work that
HF split between "delete it" and "mask it."

---

## Part C — Full data-flow diagram, this example, one decode step (position 4)

```
                         HF (vanilla)                          QC (adapted)
                    ┌─────────────────────┐            ┌──────────────────────────┐
 INPUT              │ token @ position 4    │            │ token @ position 4         │
                    └──────────┬──────────┘            └─────────────┬────────────┘
                               │                                       │
 position_ids       [4]  (past_seen_tokens=4 + 0)           [4]  (kv_glb/kv_swa=4 + cache_tensor[0])
                               │                                       │
 cache write        DynamicLayer: cat -> grows to        DynamicLayer_adapted: SCATTER
 (full)             len 5                                  into gcl=12 buffer at col 4
                    DynamicSlidingWindowLayer: cat,        (SAME class/mechanism as full —
 cache write        then keep-last-3 truncate ->           no truncation at all)
 (sliding)          stored = [2,3,4]                       into lcl=8 buffer at col 4
                               │                                       │
 mask build         create_causal_mask /                  torch.full(mask_neg) then
                    create_sliding_window_causal_mask       per-row flip, using
                    — live predicate over                   kv_glb/kv_swa (read from
                    get_seq_length()/get_mask_sizes()        unpadded_past_kv shapes)
                    INSIDE forward()                         OUTSIDE the model (qgenerator.py)
                               │                                       │
 mask shape         full: [1,1,1,5]  sliding: [1,1,1,4]    full: [1,1,1,12]  sliding: [1,1,1,8]
 (this call)         (both EXACTLY match real content)      (both FIXED, padded with mask_neg
                                                              past the real content)
                               │                                       │
 attention          attn_weights += mask                   attn_weights += mask   OR
                    (additive only)                         where(mask==0, x, min+mask_neg)
                                                              (masked-softmax, the default here)
                               │                                       │
 RESULT             identical attention distribution over the 4 logically-visible positions {1,2,3,4}
                    — confirmed by the grids in B3.
```

---

## Summary — what's identical, what's different

| | Identical between HF and QC | Different |
|---|---|---|
| **Full attention** | Exact same allowed-set at every step (verified A3) | Mask shape: HF grows; QC fixed at `gcl` |
| **Sliding attention** | Exact same allowed-set at every step (verified B3) | *How* the window is enforced: HF partly via cache truncation (data deleted) + partly via mask; QC entirely via mask (data never deleted) |
| **Underlying math** | Same causal + (for sliding) same window rule, same `mask_neg`-style suppression | Representation only — this whole document is one big instance of "same math, different representation" |

**One sentence for a design review:** *"We validated, position by position, that the
fixed-buffer masking scheme produces bit-for-bit the same attention visibility as HF's
dynamically-growing/truncating cache — the only behavioral difference is that our sliding
buffer physically retains old K/V entries the window has logically moved past; masking
alone (not cache truncation) is what hides them, and we've confirmed that substitution is
exact."*
