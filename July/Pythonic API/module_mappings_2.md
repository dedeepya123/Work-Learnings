Yes — these three swaps are the “model-level” part of the re-authoring story, and they are all there for the same reason: make the Hugging Face Llama model behave like a QAIRT/HTP-friendly model without rebuilding it from scratch.

1. LlamaForCausalLM → QcLlamaForCausalLM
What the original class does: LlamaForCausalLM is the full top-level causal-LM wrapper. It owns the embedding layer, decoder layers, LM head, and the overall forward pass.

## What changes in the QC version:
QcLlamaForCausalLM is still a Llama causal LM, but it is adapted for QAIRT.
It adds QC-specific behavior such as:
registering a cache_tensor buffer when input_tokens_per_inference is set
using a QAIRT-friendly forward/output path for export/runtime
being compatible with the re-authoring pipeline
Why we do it:

The top-level model needs to be aware of QC-specific runtime state.
The attention module and config may need extra buffers/flags, and the model wrapper is the right place to provide them.
This makes the model “HTP-ready” while keeping the original architecture and pretrained weights intact.

## In short:

original = standard Hugging Face full model wrapper
QC version = same model wrapper, but with QAIRT-specific runtime hooks

# 2. LlamaRotaryEmbedding → QcLlamaRotaryEmbedding
What the original class does:
LlamaRotaryEmbedding is the helper that computes or provides rotary position embeddings for the attention block.
What changes in the QC version:
QcLlamaRotaryEmbedding is a small adaptation of that helper.
Its key change is that it can return precomputed position embeddings directly when the input is already a tuple/list of (cos, sin) tensors.
That avoids unnecessary recomputation and makes it more compatible with QC-style execution paths.
Why we do it:

Some QAIRT paths work with precomputed embeddings or custom position handling.
This makes the rope logic more flexible and predictable for export/inference.
It is a small but important compatibility layer for the attention path.
In short:

original = standard HF rotary embedding helper
QC version = same role, but more compatible with QAIRT’s attention/runtime expectations


3. LlamaConfig → QcLlamaConfig
What the original class does:

LlamaConfig holds all the model hyperparameters and standard Hugging Face config values.
What changes in the QC version:

QcLlamaConfig inherits from LlamaConfig and also mixes in QcConfigMixin.
That means it can carry extra QAIRT/QC-specific settings such as:
transposed_key_cache
input_tokens_per_inference
other backend-specific flags
Why we do it:
The QC attention module and QC model wrapper need extra knobs that the standard HF config does not provide.
Without this, the replaced modules would not have access to the runtime flags they need.
So the config swap is what allows the QC modules to “see” the extra behavior they were built for.

In short:

original = plain HF config
QC version = same config, plus QC-specific fields and behavior

Why all three are swapped together
These are not three independent changes. They form one coherent adaptation stack:

LlamaConfig provides the extra runtime flags
LlamaAttention uses those flags
LlamaRotaryEmbedding helps produce the right positional inputs
LlamaForCausalLM is the top-level model that hosts everything and adds QC runtime state
So the replacement is really about making the whole Llama stack “QAIRT-aware” while keeping the same underlying pretrained model structure.



Think of it like this:

Normal Llama = stock Hugging Face model
Re-authored Llama = same model, but with:
QC-friendly attention
QC-friendly rotary embedding
QC-friendly config
QC-friendly top-level model wrapper
That is why the mapping in qairt/experimental/pipeline/torch/llm/models/llama/mappings.py includes all three classes.


QcLlamaRotaryEmbedding is a small adaptation of that helper.
Its key change is that it can return precomputed position embeddings directly when the input is already a tuple/list of (cos, sin) tensors.
That avoids unnecessary recomputation and makes it more compatible with QC-style execution paths.
