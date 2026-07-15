QcAutoModelForCausalLM re-authoring really does these three distinct internal jobs:
- Module replacement
- DynamicLayer patching
- Config flag setting

1. Module replacement

This is the main transformation step.

- It uses TransformersModuleMapping.apply(model, qc_config=qc_config)
- Internally, ModuleMappingTransform.apply() walks model.modules()
- For each module, it checks if type(module) is in a mapping table
- If yes, it replaces module.__class__ with the QC version
- So it does not build a new model graph — it changes the class of existing modules in-place.

## What gets replaced
For each supported model type, there is a mappings.py file that defines a dictionary MAPPINGS
Examples:

### For Llama:

LlamaAttention → QcLlamaAttention
LlamaForCausalLM → QcLlamaForCausalLM
LlamaRotaryEmbedding → QcLlamaRotaryEmbedding
LlamaConfig → QcLlamaConfig

### For Phi3:

Phi3Attention → QcPhiAttention
Phi3MLP → Phi3UnpackedMLP
Phi3Model → QcPhi3Model
Phi3RotaryEmbedding → QcPhiPhiRotaryEmbedding
Phi3Config → QcPhiConfig

### For Qwen3:

Qwen3Attention → QcQwen3Attention
Qwen3ForCausalLM → QcQwen3ForCausalLM
Qwen3RotaryEmbedding → QcQwen3RotaryEmbedding
Qwen3Config → QcQwen3Config

### What init_fn means
Some replacements need extra work because the original class and QC class have different attributes.

Example:

Phi3Attention has fused qkv_proj
QcPhiAttention uses separate q_proj, k_proj, v_proj

After swapping __class__, an init_fn runs to unpack the fused projection into the QC layout without reloading weights

So module replacement can also:
- swap the config class
- swap attention / rotary / LM classes
- run initialization hooks to keep weights valid

2. DynamicLayer patching for KV cache management

This is about the runtime cache implementation, not structural module replacement.

What happens:

KVCacheMapping.apply(model) patches transformers.cache_utils.DynamicLayer
It replaces:
DynamicLayer.update
DynamicLayer.get_seq_length
with QC versions from QcDynamicLayer

Why this matters:
- HuggingFace uses DynamicLayer for KV cache updates
QC needs different cache behavior:
- transposed_key_cache
- perform_scatter_kv_cache_update
return_new_key_value_only

So instead of changing the whole cache system, it patches the methods used by the existing runtime

The patched methods do things like:

use self.keys / self.values
concatenate or scatter new KV values
support transposed-key layout
compute the cached sequence length from the current values tensor
This is the “KV cache management” part.

3. Config flag setting

- After replacement, the model config gets adjusted for the QC runtime.
- Internally this is done in TransformersModuleMapping._set_config_flags(model).

From the code, it at least sets:

model.config._attn_implementation = "eager"
model.config._attn_implementation_internal = "eager"
This is a small but important step:

it ensures the attention implementation path used by the model is compatible with the QC rewrites
it can also preserve config fields needed by QC attention layers
So “config flag setting” means:

make sure the model’s config is in the right mode for QC attention
enable backend-compatible runtime behavior
