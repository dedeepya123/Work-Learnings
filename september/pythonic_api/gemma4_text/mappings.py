
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextMLP,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
)
from transformers.models.gemma4.quantization_gemma4 import Gemma4QuantizableEmbedding

from qairt.experimental.pipeline.torch.llm.loader.htp_mappings import TransformersModuleMapping

from .reauthoring import (
    QcGemma4ForCausalLM,
    QcGemma4QuantizableEmbedding,
    QcGemma4RMSNorm,
    QcGemma4TextAttention,
    QcGemma4TextConfig,
    QcGemma4TextDecoderLayer,
    QcGemma4TextMLP,
    QcGemma4TextModel,
    QcGemma4TextRotaryEmbedding,
    _init_qc_gemma4_causal_lm_cache_tensor,
    _init_qc_gemma4_quantizable_embedding,
    _init_qc_gemma4_text_attention,
    _init_qc_gemma4_text_mlp_act_fn,
    _revert_qc_gemma4_causal_lm_cache_tensor,
)

MAPPINGS = {
    Gemma4TextConfig: QcGemma4TextConfig,
    Gemma4ForCausalLM: QcGemma4ForCausalLM,
    Gemma4TextAttention: QcGemma4TextAttention,
    Gemma4TextDecoderLayer: QcGemma4TextDecoderLayer,
    Gemma4TextModel: QcGemma4TextModel,
    Gemma4RMSNorm: QcGemma4RMSNorm,
    Gemma4TextMLP: QcGemma4TextMLP,
    Gemma4TextRotaryEmbedding: QcGemma4TextRotaryEmbedding,
    Gemma4QuantizableEmbedding: QcGemma4QuantizableEmbedding,
}


INIT_MAPPINGS: dict = {
    QcGemma4TextMLP: _init_qc_gemma4_text_mlp_act_fn,
    QcGemma4TextAttention: _init_qc_gemma4_text_attention,
    QcGemma4QuantizableEmbedding: _init_qc_gemma4_quantizable_embedding,
    QcGemma4ForCausalLM: _init_qc_gemma4_causal_lm_cache_tensor,
}
REVERT_INIT_MAPPINGS: dict = {
    QcGemma4ForCausalLM: _revert_qc_gemma4_causal_lm_cache_tensor,
}

for _from_cls, _to_cls in MAPPINGS.items():
    TransformersModuleMapping.register(
        _from_cls,
        _to_cls,
        init_fn=INIT_MAPPINGS.get(_to_cls),
        revert_init_fn=REVERT_INIT_MAPPINGS.get(_to_cls),
    )
