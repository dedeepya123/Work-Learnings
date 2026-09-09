
from transformers import Gemma4TextConfig

from qairt.experimental.pipeline.torch.llm.loader.auto_classes import QcAutoConfig


def build_qc_config(text_config: Gemma4TextConfig):
    return QcAutoConfig.from_config(
        text_config,

        return_new_key_value_only=True,
        transposed_key_cache=True,
        perform_scatter_kv_cache_update=True,
        input_tokens_per_inference=128,

        num_logits_to_keep=0,
        mask_neg=-200,
        context_length=text_config.max_position_embeddings,
        num_layers_to_run=None,
        pad_to_left=False,
        modified_sliding_window=None,
        enable_masked_softmax=True,
        kv_clip_only=True,
    )
