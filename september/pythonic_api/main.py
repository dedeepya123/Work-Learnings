"""Load checkpoint, reauthor for QAIRT HTP, verify, and quantize."""

from models.gemma4_text.adapt import reauthor_model
from models.gemma4_text.checkpoint import load_checkpoint
from models.gemma4_text.qc_config import build_qc_config
from models.gemma4_text.quantize import quantize_model
from models.gemma4_text.verify import assert_reauthored

MODEL_PATH = "/prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/Nano/NanoV4/data/nano_v4_fast"


def main() -> None:
    ckpt = load_checkpoint(MODEL_PATH)
    print(f"Missing keys: {ckpt.missing_keys}")
    print(f"Unexpected keys: {ckpt.unexpected_keys}")
    print("text model loaded..!!!")

    qc_config = build_qc_config(ckpt.text_config)
    model = reauthor_model(ckpt.model, qc_config)
    # print(model)
    assert_reauthored(model)
    print("reauthoring check passed: model.model ->", type(model.model).__name__)
    print("reauthoring check passed: self_attn ->", type(model.model.layers[0].self_attn).__name__)

    quantizer = quantize_model(model, qc_config, ckpt.tokenizer)
    print("prepare_model check passed: prepared_model ->", type(quantizer.prepared_model).__name__)

    # run_test(model, ckpt.tokenizer, [{"role": "user", "content": "What is the capital of France?"}])


if __name__ == "__main__":
    main()


