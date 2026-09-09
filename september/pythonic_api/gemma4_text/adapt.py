
from qairt.api.configs.common import BackendType
from qairt.experimental.pipeline.torch.common.adaptations.adapter import Adapter
from qairt.experimental.pipeline.torch.llm.loader.auto_classes import QcAutoModelForCausalLM

import models.gemma4_text.mappings
from models.gemma4_text.linear_to_conv import replace_linears_with_convs as gemma4_replace_linears_with_convs


_EXTENDED_CONFIG_ATTRS = (
    "sliding_window_pattern", "mask_neg", "context_length",
    "num_layers_to_run", "pad_to_left", "modified_sliding_window",
)


def reauthor_model(model, qc_config):

    model = QcAutoModelForCausalLM._reauthor(model, qc_config=qc_config)


    for key in _EXTENDED_CONFIG_ATTRS:
        setattr(model.config, key, getattr(qc_config, key))


    model = Adapter.apply_adaptations(
        model,
        backend=BackendType.HTP,
        model_type="LLM",
        apply_default_adaptations=False,
        adaptations=[gemma4_replace_linears_with_convs],
    )

    model.eval()
    return model
