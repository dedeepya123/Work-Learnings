Difference between normal LlamaAttention and QcLlamaAttention

- Normal LlamaAttention is the stock Hugging Face implementation.
- QcLlamaAttention is a QAIRT-specific adaptation of that same block for the HTP backend.
- It keeps the same overall role — “compute attention for a decoder layer” — but changes the execution path and the config contract.

1. Same high-level purpose, different implementation
Normal Hugging Face LlamaAttention:

uses the standard Transformers attention path
relies on the generic attention_interface
is designed for broad compatibility with Hugging Face training/inference

QcLlamaAttention:
- is a custom forward path tailored for QAIRT / Qualcomm execution
- is optimized for HTP-oriented inference behavior
- adds QC-specific logic around cache handling and RoPE

2. The biggest difference: cache and key handling
In normal LlamaAttention, the forward path is fairly standard:

project to query/key/value
apply RoPE
update cache via past_key_values.update(...)
compute attention with the standard attention interface
In QcLlamaAttention, the forward path adds explicit QC behavior:
supports transposed_key_cache
changes how key states are arranged before attention computation
uses QC-specific cache update logic through self._update_cache_kwargs_with_qc_config(...)
can use special masked-softmax behavior when enabled by config

This is the main reason QAIRT swaps it in.

3. RoPE handling is slightly different
In Hugging Face:

it uses apply_rotary_pos_emb(...) directly
In QAIRT:

it still uses rotary embeddings, but it has an extra branch:
if the RoPE shapes are not compatible, it falls back to a custom apply_rope(...)
it logs QC-specific operations
That makes it more robust for the target runtime and config combinations.

4. It uses a different config expectation
Normal LlamaAttention expects a plain LlamaConfig.

QcLlamaAttention expects a QcLlamaConfig, which is a subclass of LlamaConfig plus QcConfigMixin.

That means it can understand additional flags like:

transposed_key_cache
enable_masked_softmax
input_tokens_per_inference
other QC backend knobs

So the module swap is not only about code logic; it is also about enabling new config semantics.

5. The attention math path is made more explicit
In stock Transformers, the attention block delegates to an abstraction:

attention_interface = eager_attention_forward or another backend-specific attention function
In QcLlamaAttention, the code computes the attention weights more directly inside the module:

it builds attn_weights explicitly
uses torch.matmul(...)
optionally uses transposed keys
optionally applies custom masking
That makes the behavior more predictable for QAIRT’s compilation/export pipeline.

6. Why QAIRT wants this version
The reason is not “it is mathematically different” in the broad sense.
It is that QAIRT wants the module to be:

easier to map to Qualcomm execution
compatible with QC cache semantics
compatible with specific export/compile assumptions
able to expose extra runtime flags
So QcLlamaAttention is basically a “backend-adapted” version of the same attention block

Side-by-side summary
Aspect	Normal LlamaAttention	QcLlamaAttention
Purpose	Standard HF attention	QAIRT/HPT-adapted attention
Config type	LlamaConfig	QcLlamaConfig
Cache path	Standard HF cache update	QC-aware cache update
Key layout	Standard key/value layout	Optional transposed-key behavior
RoPE	Standard HF rotary embedding path	QC-aware/custom fallback path
Masking	Standard attention interface	Explicit QC masking path
Goal	General compatibility	HTP/backend-specific execution

QcLlamaAttention is not a totally different architecture; it is the same Llama attention idea, but rewritten so QAIRT can control cache behavior, key layout, and runtime-specific attention logic for HTP execution.

