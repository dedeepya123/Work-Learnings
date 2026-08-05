# The DynamicCache(config=...) trap — what, how, why

What this document is: a plain explanation of the test in
`models/gemma4_text/test_cache_config_trap.py`, which proves — by actually running the model,
not just by reading code — why our reauthored Gemma4 model requires `DynamicCache()` built with
no `config=` argument, and what goes wrong if you build it the "normal" way instead.

---

## 1. What problem this test is even about

To generate text, a decoder model needs a KV cache (see main pipeline doc for the full
explanation of what a KV cache is). `transformers.cache_utils.DynamicCache` is HF's generic
cache container. It can be built two ways:

```python
DynamicCache()                    # every layer -> plain DynamicLayer
DynamicCache(config=model.config) # layer class chosen per-layer from config.layer_types
```

Gemma4 has two kinds of attention layers, alternating: `full_attention` and `sliding_attention`
(sliding-window attention only looks back a fixed number of tokens instead of the whole
history). When you pass `config=`, HF gives `full_attention` layers the plain `DynamicLayer`
class, but gives `sliding_attention` layers a **different** class,
`DynamicSlidingWindowLayer`.

qairt's HTP adaptation patches KV-cache behavior (transposed-key layout, scatter writes,
clamp-based fake-quant) by monkey-patching methods **directly onto the `DynamicLayer` class
only**. It has no knowledge of `DynamicSlidingWindowLayer` at all — that class defines its own
`update()` method, which means it never runs the patched code, no matter what.

So: **if a `DynamicCache(config=...)` ever gets built for this model, every sliding-attention
layer silently runs unpatched, vanilla HF cache code — while every full-attention layer runs the
qairt-adapted code.** Same model, two different behaviors depending on which layer you're in.

This test exists to actually *show* that happening, instead of just asserting it in prose.

---

## 2. Why one forward call isn't enough to see it

The first instinct is: build both caches, run the model once, compare. That doesn't work, and
the test's first draft found out why the hard way.

On the very first call to any cache layer, the cache is empty. Both the patched
`DynamicLayer.update()` and the vanilla `DynamicSlidingWindowLayer.update()` handle an empty
cache the same way — they just store whatever they're given as the first entry. There's no
concatenation, no scatter, nothing that depends on the previous contents. So on call 1, patched
and unpatched code look identical from the outside: same shape, no error, no visible difference.

**The divergence only appears once there's already something in the cache to combine the new
write with** — i.e., on a *second* call. That's exactly what happens in real decoding: call 1 is
the prefill (the whole prompt at once), call 2+ are one-token-at-a-time decode steps, each of
which has to combine a new token's key/value with everything already cached.

So the test does exactly that: one prefill call over the real prompt, then one manual decode
step with a single made-up next token — deliberately reproducing the "second call" shape of a
real generation loop, without needing a full `.generate()` loop to do it.

---

## 3. What the test actually does, step by step

File: `models/gemma4_text/test_cache_config_trap.py`

1. Build the real reauthored model the normal way (`load_checkpoint` → `build_qc_config` →
   `reauthor_model` — the exact same pipeline `main.py` uses).
2. Pick one `full_attention` layer index and one `sliding_attention` layer index from
   `model.config.layer_types`, so the comparison is layer-type-specific, not a guess.
3. **Case 1** — build `DynamicCache()` (no `config=`). Run prefill, then one decode step. Print
   the cache layer classes used, and the cached-key shape at both chosen layers.
4. **Case 2** — build `DynamicCache(config=model.config)`. Print the cache layer classes
   *before* running anything (you can already see the split: `DynamicLayer` for full-attention,
   `DynamicSlidingWindowLayer` for sliding-attention). Then run the same prefill + decode step,
   wrapped in a `try/except`, since — as it turns out — this doesn't just misbehave quietly.

---

## 4. What actually happened when it was run

**Case 1** (`DynamicCache()`, no config): both layers came out as plain `DynamicLayer`. Both
produced the expected shape `(batch, num_kv_heads, head_dim, 1)` — key transposed, only the
newest token's slice kept (because `return_new_key_value_only=True` is set in our config; see
§5). Clean, consistent, no surprises.

**Case 2** (`DynamicCache(config=model.config)`): the prefill call (call 1) succeeded silently —
per §2, empty-cache first-writes look the same either way. The decode step (call 2) then
**crashed**:

```
RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 16 but got size 1
```

This is a real, reproducible crash — not a subtle silent-divergence bug, at least not for this
model. It's arguably a better outcome to have discovered: a loud crash on the very first decode
step is impossible to miss, whereas the "silently different behavior per layer" framing in the
main pipeline doc describes the *general* risk this class of bug represents (which could easily
be a silent numerical difference for a different model/config combination, rather than a hard
crash).

---

## 5. Why it crashes specifically — the actual mechanism

Three facts have to line up to produce this exact `RuntimeError`:

1. **`transposed_key_cache=True`** is set in our Qc config. Our adapted attention code
   (`QcGemma4TextAttention.forward` in `reauthoring.py`) unconditionally transposes the key
   tensor's last two dimensions before handing it to the cache's `update()` — this is the
   on-device-friendly layout qairt's patched `DynamicLayer.update()` expects and handles
   correctly.
2. **Vanilla `DynamicSlidingWindowLayer.update()`** (the one that actually runs for
   sliding-attention layers in Case 2, because it shadows the patch) was never written expecting
   a transposed layout. It assumes the sequence axis is at dimension `-2` and does
   `torch.cat([self.keys, key_states], dim=-2)` to grow the cache.
3. Because keys are transposed, dimension `-2` is actually the *head_dim* axis, not the
   *sequence* axis, from vanilla's point of view. On call 1 there's nothing to concatenate
   against yet, so this mismatch causes no error. On call 2, `torch.cat` tries to concatenate
   along an axis where the two tensors' sizes don't actually correspond to the same thing —
   PyTorch catches the resulting shape mismatch and raises.

In short: qairt's patch and our adapted attention code agree on a transposed-key convention.
Vanilla cache code was never told about that convention. `DynamicSlidingWindowLayer` runs
vanilla code (because the patch doesn't reach it) while receiving transposed-convention data
(because the rest of the model doesn't know it's talking to unpatched code) — and those two
mismatched assumptions collide the moment there's a second write to reconcile against the first.

---

## 6. What this confirms about the main pipeline

This is the concrete, run-and-observe confirmation of something the main pipeline
(`NANOV4_TO_QAIRT.md`) states as a design decision: `QcGemma4TextModel.forward` raises a
`ValueError` up front if `use_cache=True` and no `past_key_values` was explicitly passed in —
refusing to let a `config=`-built cache get auto-constructed internally (which is exactly what
vanilla Gemma4 code, and `.generate()` by extension, would otherwise do on your behalf). Every
caller — `main.py`'s smoke test, this test, and any future multi-step generation loop — must
build `DynamicCache()` with no `config=` and pass it in explicitly, so every layer uniformly gets
the plain, patchable `DynamicLayer`, regardless of whether that layer is full- or
sliding-attention.

---

## 7. Where the code lives

| What | File |
|---|---|
| The test itself | `models/gemma4_text/test_cache_config_trap.py` |
| The guard this test justifies | `QcGemma4TextModel.forward`, `models/gemma4_text/reauthoring.py` |
| Where keys get transposed before caching | `QcGemma4TextAttention.forward`, `models/gemma4_text/reauthoring.py` |
| qairt's cache patch (`DynamicLayer` only) | `KVCacheMapping` / `_qc_dynamic_layer_update`, qairt package (`llm/loader/htp_mappings.py`, `llm/models/utils.py`) |
| Vanilla class that shadows the patch | `DynamicSlidingWindowLayer`, `transformers/cache_utils.py` |
