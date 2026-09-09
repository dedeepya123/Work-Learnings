
import os

import torch
from torch.utils.data import DataLoader, Dataset

from aimet_torch.onnx_utils import OnnxExportApiArgs
from qairt.api.configs.common import DspArchitecture
from qairt.experimental.pipeline.torch.llm.quantization.techniques.bases.definitions import (
    LPBQParams,
    ModulePrecisions,
    Precisions,
)
from qairt.experimental.pipeline.torch.llm.quantization.techniques.lpbq.lpbq_quantizer import (
    LPBQQuantizer,
)

from models.gemma4_text.generator import Gemma4TextGenerator
from models.gemma4_text.reauthoring import TextModelQc


PREPARE_PATH = os.path.join(os.getcwd(), "nano/prepared_model")


EXPORT_PATH = os.path.join(os.getcwd(), "nano/quantsim_output")

NUM_KV_LAYERS = 15


CALIBRATION_PROMPTS = [
    "What is the capital of France?",
    "Write a short greeting.",
]


class PromptDataset(Dataset):

    def __init__(self, tokenizer, prompts: list[str]):
        self._tokenizer = tokenizer
        self._prompts = prompts

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(self, idx: int) -> dict:
        messages = [{"role": "user", "content": self._prompts[idx]}]
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
        )


        return {"input_ids": input_ids.squeeze(0)}


def _build_onnx_export_args() -> OnnxExportApiArgs:
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids_cos",
        "position_ids_sin",
    ]
    for i in range(NUM_KV_LAYERS):
        if (i + 1) % 5 == 0:
            input_names += [f"past_key_{i}_in", f"past_value_{i}_in"]
        else:
            input_names += [f"swa_key_{i}_in", f"swa_value_{i}_in"]
        
    input_names += [
        "per_layer_inputs",
        "swa_position_ids_cos",
        "swa_position_ids_sin",
        "swa_attention_mask",
        "cache_index",
        "swa_cache_index",
    ]

    output_names = ["logits"]
    for i in range(NUM_KV_LAYERS):

        if (i + 1) % 5 == 0:
            output_names += [f"past_key_{i}_out", f"past_value_{i}_out"]
        else:
            output_names += [f"swa_key_{i}_out", f"swa_value_{i}_out"]

    return OnnxExportApiArgs(input_names=input_names, output_names=output_names, opset_version=20)


def quantize_model(
    model, qc_config, tokenizer, prepare_path: str = PREPARE_PATH, export_path: str = EXPORT_PATH
) -> LPBQQuantizer:

    sequence_length = qc_config.input_tokens_per_inference


    text_model_qc = TextModelQc(model)

    generator = Gemma4TextGenerator(
        model=text_model_qc,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        context_length=qc_config.context_length,


        config=qc_config,
    )
    quantizer = LPBQQuantizer(
        hf_model=text_model_qc,
        context_length=qc_config.context_length,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(PromptDataset(tokenizer, CALIBRATION_PROMPTS), batch_size=1)

    params = LPBQParams(
        dataloader=dataloader,
        generator=generator,
        model_preparation_path=prepare_path,
        dsp_arch=DspArchitecture.v81,
        module_precisions=ModulePrecisions(
            rms_norm=Precisions(weight_precision="int16"),
        ),
        quantsim_kwargs={"default_param_bw": 8, "in_place": True},
    )

    with torch.no_grad():
        quantizer.quantize(params)
        quantizer.export(
            export_path,
            filename_prefix="model",
            onnx_export_args=_build_onnx_export_args(),
        )

    return quantizer
