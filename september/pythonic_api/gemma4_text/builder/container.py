
import os

from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory
from qairt.optimizer.onnx import (
    AxisDenotationConfig,
    AxisDenotationSeedRule,
    ExportedFiles,
    GraphContext,
    change_seq_and_context_length,
)
from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation

from models.gemma4_text.builder.gen_ai_config import build_gen_ai_config_dict
from models.gemma4_text.checkpoint import load_checkpoint
from models.gemma4_text.qc_config import build_qc_config

MODEL_PATH = "/prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/Nano/NanoV4/data/nano_v4_fast"

EXPORT_DIR = os.path.join(os.getcwd(), "nano/quantsim_output_v2_encodings")
ONNX_MODEL_PATH = os.path.join(EXPORT_DIR, "model.onnx")
ENCODINGS_PATH = os.path.join(EXPORT_DIR, "model.encodings")

RESIZED_DIR = os.path.join(os.getcwd(), "nano/resized_onnx_new")
CONTAINER_OUTPUT_PATH = os.path.join(os.getcwd(), "nano/llm_container")


ARN = 8
CONTEXT_LENGTH = 512


CHIPSET = "chipset:SM8850"


PER_LAYER_INPUTS_SEED_RULE = AxisDenotationSeedRule(
    name_pattern=r"per_layer_inputs",
    denotations=[
        AxisDenotation.BATCH,
        AxisDenotation.SEQ_LENGTH,
        AxisDenotation.UNKNOWN,
        AxisDenotation.UNKNOWN,
    ],
)


def resize_for_arn(
    ar: int,
    cl: int,
    tag: str,
    onnx_path: str = ONNX_MODEL_PATH,
    encodings_path: str = ENCODINGS_PATH,
    resized_dir: str = RESIZED_DIR,
) -> ExportedFiles:
    ctx = GraphContext.from_files(onnx_path, encodings_path=encodings_path)
    axis_config = AxisDenotationConfig(custom_seed_rules=[PER_LAYER_INPUTS_SEED_RULE])
    change_seq_and_context_length(ctx, ar, cl, axis_denotation_config=axis_config)
    return ctx.export(resized_dir, prefix=tag)


def new_builder_for_resized(resized: ExportedFiles, qc_config) -> object:
    builder = GenAIBuilderFactory.create(
        pretrained_model_path=str(resized.onnx_path),
        backend_type="HTP",
        tokenizer_path=MODEL_PATH,
        config_dict=build_gen_ai_config_dict(qc_config),
    )


    builder._transformation_config.model_transformer_config.arn_cl_options.skip_ar_cl_conversion = True
    builder.set_targets([CHIPSET])
    return builder


def build_one_container(ar: int, cl: int, tag: str, output_path: str) -> object:
    ckpt = load_checkpoint(MODEL_PATH)
    qc_config = build_qc_config(ckpt.text_config)

    resized = resize_for_arn(ar, cl, tag)
    builder = new_builder_for_resized(resized, qc_config)
    builder.set_transformation_options(options={"context_length": [cl]})

    container = builder.build()
    container.save(output_path, exist_ok=True)
    return container


def build_container() -> object:
    return build_one_container(ARN, CONTEXT_LENGTH, f"model_ar{ARN}_cl{CONTEXT_LENGTH}", CONTAINER_OUTPUT_PATH)


if __name__ == "__main__":
    container = build_container()
    print("build_container check passed: container ->", type(container).__name__)
