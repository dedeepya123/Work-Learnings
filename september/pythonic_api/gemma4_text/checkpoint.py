
from dataclasses import dataclass
from typing import Optional

from safetensors import safe_open
from transformers import AutoTokenizer, Gemma4Config, Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerBase
from transformers.models.gemma4.config import CraftConfig


LANGUAGE_MODEL_PREFIX = "model.language_model."
TOP_LEVEL_RENAMES = {
    "lm_head_out.": "lm_head_outs.",
    "logit_div.": "div.",
    "logit_tanh.": "tanh.",
    "logit_mul.": "mul.",
}


@dataclass
class GemmaCheckpoint:
    model: Gemma4ForCausalLM
    text_config: Gemma4TextConfig
    tokenizer: PreTrainedTokenizerBase
    missing_keys: list
    unexpected_keys: list


def _remap_key(key: str) -> Optional[str]:
    if key.startswith(LANGUAGE_MODEL_PREFIX):
        return "model." + key[len(LANGUAGE_MODEL_PREFIX):]
    for old_prefix, new_prefix in TOP_LEVEL_RENAMES.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return None


def load_checkpoint(model_path: str) -> GemmaCheckpoint:
    config = Gemma4Config.from_pretrained(model_path)
    text_config = config.text_config


    text_config.craft_config = CraftConfig(
        use_craft=False,
        int_precision=None,
        use_optional_craft_sfq=False,
        use_lora_craft_sfq=False,
    )

    model = Gemma4ForCausalLM(text_config)
    model_keys = set(model.state_dict().keys())

    state_dict = {}
    with safe_open(model_path + "/model.safetensors", framework="pt") as f:
        for key in f.keys():
            new_key = _remap_key(key)
            if new_key is not None and new_key in model_keys:
                state_dict[new_key] = f.get_tensor(key)


        if "model.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
        if "model.embed_tokens.weight_scale" in state_dict:
            state_dict["lm_head.weight_scale"] = state_dict["model.embed_tokens.weight_scale"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return GemmaCheckpoint(
        model=model,
        text_config=text_config,
        tokenizer=tokenizer,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )
