
import os

from models.gemma4_text.builder.container import (
    MODEL_PATH,
    new_builder_for_resized,
    resize_for_arn,
)
from models.gemma4_text.checkpoint import load_checkpoint
from models.gemma4_text.qc_config import build_qc_config

CONTAINER_OUTPUT_PATH = os.path.join(os.getcwd(), "nano/llm_container_prefix_decode_new")

DECODE_ARN = 1
PREFIX_ARN = 8
CONTEXT_LENGTH = 512


def build_prefix_and_decode() -> object:
    ckpt = load_checkpoint(MODEL_PATH)
    qc_config = build_qc_config(ckpt.text_config)

    prefix_resized = resize_for_arn(PREFIX_ARN, CONTEXT_LENGTH, f"model_ar{PREFIX_ARN}_cl{CONTEXT_LENGTH}")
    decode_resized = resize_for_arn(DECODE_ARN, CONTEXT_LENGTH, f"model_ar{DECODE_ARN}_cl{CONTEXT_LENGTH}")

    builder = new_builder_for_resized(prefix_resized, qc_config)
    builder.set_transformation_options(options={"context_length": [CONTEXT_LENGTH]})


    builder.attach_model_for_arn(
        arn=DECODE_ARN,
        model_path=decode_resized.onnx_path,
        encodings_path=decode_resized.encodings_path,
    )
    builder.weight_sharing = True

    container = builder.build()
    container.save(CONTAINER_OUTPUT_PATH, exist_ok=True)
    return container


if __name__ == "__main__":
    container = build_prefix_and_decode()
    print("prefix+decode container ->", type(container).__name__)
