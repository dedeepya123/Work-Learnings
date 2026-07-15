# QCAuto Config 

## What QcAutoConfig does

- QcAutoConfig is a helper factory for building Qualcomm-enhanced transformer configs from HuggingFace configs.

### Main purpose
Converts a standard HF config into a Qc*Config class (e.g. QcLlamaConfig) for the correct model type.

Lets you pass Qualcomm-specific config options (qc_kwargs) without manually instantiating the Qc config class.

### How it works
It provides three factory methods:
1. from_pretrained(pretrained_model_name_or_path, **qc_kwargs)
   - Loads the HF config from a model path or Hub ID.
   - Converts it into the matching Qc config.
   - returns a new Qc config instance (e.g. ``QcLlamaConfig``) with all base config attributes and the requested Qc parameters applied.
2. from_config(config, **qc_kwargs)
  - Takes an existing HF config object.
  - Uses config.model_type to select the right Qc config class.
  - Applies Qualcomm-specific settings.

3. from_model_type(model_type, base_config, **qc_kwargs)
     - Uses an explicit model type string.
     - Finds the mapping from HF config class to Qc config class.
     - Creates the Qc config from the HF config data.

### Why it exists

- Avoids Unrecognized configuration class errors when loading models with Qualcomm-specific config classes.
- Enables QcAutoModelForCausalLM.from_pretrained(..., qc_config=...) to work cleanly.
- Supports both built-in model mappings and plugin-registered model mappings.

### Behavior details
- Copies all standard HF config attributes into the new Qc config.
- Sets QC-specific fields only if they exist on the Qc config.
- Logs warnings for unknown QC-specific keys.
- Can merge model_config_overrides and qc_kwargs.

### Summary
QcAutoConfig builds the correct Qualcomm-aware config from HuggingFace model config metadata, making Qc model loading easier and safer.



QcAutoConfig.from_pretrained(...) does two things:

- Loads the HuggingFace base config for the model. Wraps it into the matching Qc*Config class.Applies QC-specific settings.

## What model_config_overrides is

- Optional dict for standard HuggingFace config fields.
- These are the same keys you would normally set on a pretrained model config.

Example keys:
num_hidden_layers
hidden_size
num_attention_heads
max_position_embeddings

The code does:
- if key is not QC-specific, set it directly on the base HF config
- if key is QC-specific, move it into qc_kwargs automatically

So model_config_overrides is mainly for normal HF config fields, but it also supports QC keys as a convenience.

## What **qc_kwargs is

- QC-specific keyword args.
These are attributes from QcConfigMixin.QC_BACKING_ATTRS.

### Common examples:
transposed_key_cache=True
perform_scatter_kv_cache_update=True
input_tokens_per_inference=512
num_logits_to_keep=1
enable_masked_softmax=True
Those are the extra Qualcomm-specific config flags added to the QC config.

## Key difference
model_config_overrides: modifies the original HF config first.
qc_kwargs: configures only the QC-specific fields on the resulting Qc*Config.
qc_kwargs takes precedence for QC-specific keys if the same key is also present in model_config_overrides.

Example

- num_hidden_layers and max_position_embeddings are standard HF config changes.
- transposed_key_cache, perform_scatter_kv_cache_update, input_tokens_per_inference are QC-specific.

Because transposed_key_cache is also passed explicitly as a qc_kwargs, that explicit value wins.


Summary
model_config_overrides = normal HuggingFace config fields (plus QC keys if you want to pass them there too)
**qc_kwargs = Qualcomm-specific config flags
from_pretrained(...) is the convenient one-step loader for both kinds of overrides


## What transposed_key_cache=True means
- This QC flag changes how attention key cache tensors are stored and used.
- Normally key cache is kept with shape like [batch, heads, seq_len, head_dim].
- With transposed_key_cache=True, the key tensor is stored/transposed into [batch, heads, head_dim, seq_len].
- That means attention uses: torch.matmul(query, key) directly instead of torch.matmul(query, key.transpose(...))
So setting it to True tells the QC runtime to use a key-cache layout optimized for Qualcomm/HTP execution, which can improve memory access and performance.

## What input_tokens_per_inference means

- This QC flag defines a fixed number of input tokens the model expects during inference.
- If set, the model registers a cache_tensor buffer of length input_tokens_per_inference.
- That buffer is used by the QC re-authoring/mapping code to build static cache position data.
- It also influences inference preparation logic where sequence length may be overridden to match this fixed number.

Example : input_tokens_per_inference=512 means “this model will use 512 input tokens per inference step for cache setup”.
It is useful when the backend expects a known static token count for cache indexing.
