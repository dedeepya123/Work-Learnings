## What “DynamicLayer patching” means

This is a separate re-authoring step from module replacement.

- Module replacement swaps the model’s neural modules, such as attention layers.
- DynamicLayer patching changes how the KV cache behaves during generation.

So instead of changing the model architecture, QAIRT changes the cache behavior that the model uses to store past keys and values.

---

## The natural Transformers way

In normal Hugging Face decoding, the model uses a cache object like `DynamicCache` for past key/value states.

During each attention step, the code does something conceptually like:

- `past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)`

That call updates the cache with the new tokens.

And when the model needs the current cache length, it may call:

- `past_key_values.get_seq_length()`

This is the standard, generic behavior:
- append new KV entries
- keep them in a simple, general-purpose structure
- work for most Hugging Face inference paths

---

## What QAIRT changes

QAIRT patches the underlying cache helper methods on the Transformers cache class so that the same call sites still work, but the implementation is QC-aware.

In this code, the patch replaces:

- `update`
- `get_seq_length`

with QC-specific implementations from qairt/experimental/pipeline/torch/llm/models/utils.py and applies them through qairt/experimental/pipeline/torch/llm/loader/htp_mappings.py.

So the model still calls the same methods, but now those methods behave differently.

---

## Why this is done

The reason is simple: the QC/HTP path wants a cache strategy that is better suited to backend execution.

The QC implementation can support things like:

- transposed key cache layout
- scatter-based cache updates
- returning only newly added KV pairs when useful
- more efficient handling of cache positions

That is different from the generic Hugging Face cache behavior, which is designed for broad compatibility rather than backend-specific optimization.

---

## How it differs from the natural way

Here is the clean comparison.

<img width="236" height="297" alt="image" src="https://github.com/user-attachments/assets/3dbb67e9-b612-46da-8b9f-2a3032c9a040" />


---

## Why patching instead of rewriting everything

This is important.

QAIRT does not rebuild the whole generation stack.

It keeps the same model and the same cache API:

- `update(...)`
- `get_seq_length()`

and only changes the implementation behind those methods.

That means:

- the model code does not need to be rewritten
- the cache interface remains compatible
- the runtime behavior becomes QC-aware

This is much cleaner than introducing a completely different cache subsystem everywhere.

---

## The key mental model

Think of it like this:

- normal Hugging Face cache = “generic memory for past tokens”
- QAIRT cache = “the same memory, but with QC-specific update logic”

The model still uses the cache in the same conceptual way, but the cache behaves more efficiently for the target backend.

---

## One-line summary

DynamicLayer patching is the step where QAIRT keeps the normal KV-cache interface but swaps in QC-optimized cache update and length-query behavior so generation works better on the HTP path.

If you want, I can next show you the exact difference between:
- normal `DynamicCache.update(...)`



Yes — here is the exact difference in the most concrete form DynamicCache.update(...)
QC-patched QcDynamicLayer.update(...)

## 1. Normal Transformers behavior

In standard Hugging Face, the cache update is conceptually:

- take the new key/value tensors
- append them into the layer cache
- return the updated cache

In simple form:

```python
def update(key_states, value_states, layer_idx, cache_kwargs):
    # append new tokens to the existing cache
    self.keys[layer_idx] = concat(self.keys[layer_idx], key_states)
    self.values[layer_idx] = concat(self.values[layer_idx], value_states)
    return self.keys[layer_idx], self.values[layer_idx]
```

And for length:

```python
def get_seq_length():
    return len(self.values[layer_idx])
```

This is generic and works for most Hugging Face inference paths.

---

## 2. QAIRT-patched behavior

QAIRT replaces those methods with QC-aware implementations.

The patched update logic is not just “append.” It can:

- handle transposed key cache layout
- use scatter-based writes
- optionally store only the newly added tokens
- choose between concatenation and efficient in-place placement depending on the cache position

So the logic becomes more like:

```python
def update(key_states, value_states, cache_kwargs):
    if cache_position is None or cache is empty:
        # start or grow by concatenation
        new_keys = concat(existing_keys, key_states)
        new_values = concat(existing_values, value_states)
    else:
        # use scatter-based write for efficient partial updates
        write into the correct positions
        update only the relevant slots

    if return_new_key_value_only:
        store only new tokens
    else:
        store full concatenated cache

    return updated_keys, updated_values
```

That is the exact difference.

---

## 3. The important conceptual difference

### Natural HF cache update
- “keep the history in a simple, standard way”

### QAIRT patched update
- “keep the history, but use a backend-friendly update strategy”

So the difference is not “cache exists vs no cache.”  
The difference is:

- how the cache is written
- how the cache is grown
- how positions are handled
- how the layout is optimized

---

## 4. What the QAIRT code is doing specifically

In the QAIRT implementation, the patch logic is driven by config flags such as:

- `transposed_key_cache`
- `return_new_key_value_only`
- `perform_scatter_kv_cache_update`

So the update routine changes behavior depending on those flags.

That means the cache update is no longer just a generic append.  
It becomes a configuration-driven, backend-oriented update path.

---

## 5. The exact effect on `get_seq_length`

In normal HF:

```python
def get_seq_length():
    return current_length_of_cache
```

In QAIRT:

```python
def get_seq_length():
    if values is None or empty:
        return 0
    return values.shape[-2]
```

This is still conceptually the same result — “how many tokens are cached” — but it is implemented against the QC cache structure directly.

---

## 6. The cleanest way to say it

The natural cache path is:

- generic, standard, compatible

The QAIRT-patched cache path is:

- same interface, but optimized for QC/HTP-style runtime behavior

---

## 7. One-line summary

The exact difference is that normal Transformers cache updates are generic append-based operations, while QAIRT’s patched DynamicLayer cache updates are QC-aware, layout-aware, and can use optimized scatter-based writes and selective cache storage.

If you want, I can next show you the same comparison as a tiny side-by-side Python-like snippet with “HF vs Qc” for both `update()` and `get_seq_length()`.
- QC-patched `QcDynamicLayer.update(...)`

in a small pseudo-code example.

Sure — here is the side-by-side picture.

## Natural Hugging Face cache behavior

In standard Transformers, the cache update is basically “append the new K/V tokens to the existing cache.”

```python
def update(key_states, value_states, layer_idx, cache_kwargs):
    # standard behavior
    self.keys[layer_idx] = concat(self.keys[layer_idx], key_states)
    self.values[layer_idx] = concat(self.values[layer_idx], value_states)
    return self.keys[layer_idx], self.values[layer_idx]
```

And the length query is:

```python
def get_seq_length():
    return len(self.values[layer_idx])
```

So the behavior is:
- simple
- generic
- append-based
- intended for broad compatibility

---

## QAIRT-patched QC cache behavior

In QAIRT, the same methods are replaced with QC-aware versions.

```python
def update(key_states, value_states, cache_kwargs):
    # QC-aware behavior
    if cache_position is None or cache is empty:
        # grow by concatenation
        key_cache = concat(self.keys, key_states)
        value_cache = concat(self.values, value_states)
    else:
        # use scatter-based writes for partial updates
        write into the correct positions using cache_position

    if return_new_key_value_only:
        self.keys = key_states
        self.values = value_states
    else:
        self.keys = key_cache
        self.values = value_cache

    return self.keys, self.values
```

And the length query becomes:

```python
def get_seq_length():
    if self.values is None or self.values.numel() == 0:
        return 0
    return self.values.shape[-2]
```

So the behavior is:
- still conceptually “maintain past K/V”
- but now with QC-specific handling for layout and writes

---

## The key difference

### HF
- “append new tokens to a generic cache”

### QAIRT
- “update the cache using QC-friendly rules, possibly with transposed layout or scatter writes”

That is the exact difference.

---

## In one sentence

Natural Transformers cache handling is generic append-based KV storage, while QAIRT’s patched cache handling is a backend-aware KV update path that can use optimized layout and selective write strategies.
