
from models.gemma4_text.reauthoring import QcGemma4TextAttention, QcGemma4TextModel


def assert_reauthored(model) -> None:
    assert isinstance(model.model, QcGemma4TextModel), f"model.model is {type(model.model)}"
    assert all(
        isinstance(layer.self_attn, QcGemma4TextAttention) for layer in model.model.layers
    ), "not every layer's self_attn was swapped"
    assert model.config.transposed_key_cache is True
    assert model.config.mask_neg == -200
