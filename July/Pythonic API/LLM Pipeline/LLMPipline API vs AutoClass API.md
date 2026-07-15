LLMPipeline.from_pretrained(...) is a higher-level pipeline loader.
QcAutoModelForCausalLM.from_pretrained(...) is the lower-level model loader that actually loads and re-authors the model.

They are related, but not the same.

What QcAutoConfig does
Builds a Qualcomm-specific config object only.
Example: QcAutoConfig.from_pretrained("model", transposed_key_cache=True)
It does not load a model by itself.
It prepares qc_config to pass into QcAutoModelForCausalLM.
What QcAutoModelForCausalLM.from_pretrained(...) does
Loads a model via HuggingFace AutoModelForCausalLM.from_pretrained(...)
Optionally applies QC re-authoring:
module replacement
KV cache patching
config flag setting
Uses qc_config if provided to preserve QC-specific settings
Returns a single model object
This is the direct loader for a model.

What LLMPipeline.from_pretrained(...) does
This is a pipeline abstraction that does more than just model loading:

Builds LLMPipelineConfig from a recipe file or dict
Creates a pipeline object with stage definitions
Runs the first pipeline stage, usually model_loader
That stage loads the model and tokenizer
It may also:
apply default HTP adaptations
move the model to CPU or GPU based on pipeline execution environment
save pipeline/manifest state for resume
handle plugin-based model mappings
manage downstream stage inputs/outputs
So LLMPipeline.from_pretrained(...) is not just model loading — it is recipe-driven pipeline initialization.

How they differ
QcAutoModelForCausalLM.from_pretrained
direct model loader
use when you just want a model
does re-authoring if requested
does not know about pipeline recipes or stages
LLMPipeline.from_pretrained
pipeline-level entry point
loads recipe/config, constructs stages, runs model_loader
eventually calls QcAutoModelForCausalLM.from_pretrained inside ModelLoadingStage
can also load tokenizer, apply adaptations, and move model to device
supports resume/caching via pipeline manifest
Internal connection
Inside ModelLoadingStage._execute():

if config.model_reauthoring is true:
it calls QcAutoModelForCausalLM.from_pretrained(..., model_reauthoring=True, qc_config=...)
else:
it still calls QcAutoModelForCausalLM.from_pretrained(..., model_reauthoring=False)
So the pipeline uses the same lower-level loader, but only as part of its stage execution.

When to use which
Use QcAutoConfig + QcAutoModelForCausalLM when you want direct model loading and QC re-authoring.
Use LLMPipeline.from_pretrained when you want the full pipeline flow, recipe-driven configuration, tokenizer loading, and stage management.
That is the key differenc
