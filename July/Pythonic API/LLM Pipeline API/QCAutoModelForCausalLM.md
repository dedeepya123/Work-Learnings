# QCAutoModelForCausalLM

Qc-enhanced AutoModelForCausalLM with internal re-authoring.
This class provides an interface for loading models with Qc-specific optimizations for Qualcomm hardware.
The class automatically applies:
    1. Module replacement (HF modules → Qc-optimized modules)
    2. DynamicLayer patching for KV cache management
    3. Config flag setting (use_cache, padding_side, etc.)

## What QcAutoModelForCausalLM does
QcAutoModelForCausalLM is a wrapper around HuggingFace AutoModelForCausalLM that:
- loads a model from a Hub ID or local path
- optionally re-authors it into Qualcomm-optimized modules
- optionally applies QC config settings via qc_config

## Key behavior
- It is not instantiated directly; use QcAutoModelForCausalLM.from_pretrained(...)
- It loads the model with standard HF code: AutoModelForCausalLM.from_pretrained(...)

If model_reauthoring=True (default), it then: replaces standard HF transformer modules with QC-optimized equivalents 
patches KV-cache handling for Qualcomm backend applies any QC config provided via qc_config

## What qc_config does there

If qc_config is passed and no config is already in pretrained_kwargs: it converts qc_config to a plain HF base config
passes that base config into AutoModelForCausalLM.from_pretrained(...) this avoids “unrecognized config class” errors the QC config is still kept for later re-authoring

## When re-authoring is skipped

If model_reauthoring=False, it simply returns the vanilla HF model without QC module replacement.

## summary
QcAutoModelForCausalLM is a convenient entry point for:

loading causal LM models applying Qualcomm-specific module and cache rewrites integrating QcAutoConfig settings safely during load


## What “re-authoring” does
 Re-authoring means transforming a loaded HuggingFace causal LM into a Qualcomm-optimized version.

### The two main steps
- QcAutoModelForCausalLM._reauthor(model, qc_config) does:

1. TransformersModuleMapping.apply(model, qc_config=qc_config) loads the model-specific replacement mappings for your model type
replaces the model’s config with a QC-aware config

- if qc_config is provided, it assigns that config and copies QC-specific values otherwise it swaps the config class to the QC subclass
replaces submodules in the model graph using MAPPINGS

e.g. HF attention/LM modules → QC-optimized modules
sets config flags like _attn_implementation = "eager"

2. KVCacheMapping.apply(model)

- patches HuggingFace DynamicLayer cache methods
- replaces DynamicLayer.update and DynamicLayer.get_seq_length with QC implementations
- This makes KV cache updates use the Qualcomm-compatible cache logic

## What gets changed

- model.config becomes a QC-enhanced config class retains any QC-specific settings from qc_config 
model submodules.

- HuggingFace transformer blocks are swapped for QC versions

e.g. custom attention, rotary embedding, and LM wrappers
KV cache behavior

- cache update logic is monkey-patched globally for DynamicLayer
this affects how past key/value cache is stored and updated

## Why it matters
Re-authoring is the step that prepares a model for the QAIRT Qualcomm backend by:

using QC-specific module implementations
applying QC config flags
enabling QC-friendly KV cache behavior
If you set model_reauthoring=False, none of the above QC transformations happen and you keep a vanilla HF model.

## Summary
re-authoring = replace HF modules + replace config + patch cache logic for Qualcomm execution.


## For a supported model like Llama:

LlamaAttention → QcLlamaAttention
LlamaForCausalLM → QcLlamaForCausalLM
LlamaRotaryEmbedding → QcLlamaRotaryEmbedding
That means the main attention/cache path and config behavior become QC-optimized.

After re-authoring: the model is not a vanilla HF model anymore the supported transformer pieces are replaced with QC versions
but “all modules” is only true for the supported mapped modules, not arbitrary custom layers outside that mapping system
