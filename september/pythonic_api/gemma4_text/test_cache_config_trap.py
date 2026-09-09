
import torch
from transformers.cache_utils import DynamicCache

from models.gemma4_text.adapt import reauthor_model
from models.gemma4_text.checkpoint import load_checkpoint
from models.gemma4_text.qc_config import build_qc_config
from models.gemma4_text.reauthoring import create_position_embeddings

MODEL_PATH = "/prj/qct/aisw_scratch/lv/local_dev/users/dlekkala/Nano/NanoV4/data/nano_v4_fast"


def _build_model():
    ckpt = load_checkpoint(MODEL_PATH)
    qc_config = build_qc_config(ckpt.text_config)
    model = reauthor_model(ckpt.model, qc_config)
    return model, ckpt.tokenizer


def _forward_step(model, input_ids, start_pos, past_key_values):
    position_ids = torch.arange(start_pos, start_pos + input_ids.shape[1]).unsqueeze(0)
    position_embeddings = create_position_embeddings(model.model.rotary_emb, position_ids)
    with torch.no_grad():
        model(
            input_ids=input_ids,
            position_ids=position_embeddings,
            past_key_values=past_key_values,
            use_cache=True,
        )


def _run_prefill_then_decode(model, tokenizer, past_key_values):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    prompt_len = inputs["input_ids"].shape[1]


    _forward_step(model, inputs["input_ids"], start_pos=0, past_key_values=past_key_values)


    next_token = torch.tensor([[1234]])
    _forward_step(model, next_token, start_pos=prompt_len, past_key_values=past_key_values)


def _report_layer(cache, idx, layer_type, per_layer_head_dim):
    layer = cache.layers[idx]
    print(f"layer {idx} ({layer_type}, {type(layer).__name__}) cached key shape:",
          tuple(layer.keys.shape), f"-- per-layer head_dim={per_layer_head_dim}")


def main() -> None:
    model, tokenizer = _build_model()
    layer_types = model.config.layer_types
    full_idx = layer_types.index("full_attention")
    sliding_idx = layer_types.index("sliding_attention")
    full_head_dim = model.model.layers[full_idx].self_attn.head_dim
    sliding_head_dim = model.model.layers[sliding_idx].self_attn.head_dim

    print("layer_types (first 6):", layer_types[:6])
    print(f"comparing layer {full_idx} (full_attention, head_dim={full_head_dim}) vs "
          f"layer {sliding_idx} (sliding_attention, head_dim={sliding_head_dim})")
    print("running: 1 prefill call + 1 decode-step call, so the cache is written to twice")
    print("(return_new_key_value_only=True is set, so a healthy cached-key shape after any call")
    print(" is (batch, num_kv_heads, head_dim, 1) -- only the newest token's key, transposed)")
    print()

    print("=== Case 1: DynamicCache() -- no config, what our pipeline actually requires ===")
    cache_ok = DynamicCache()
    _run_prefill_then_decode(model, tokenizer, cache_ok)
    print("cache layer classes used:", {type(l).__name__ for l in cache_ok.layers})
    _report_layer(cache_ok, full_idx, "full_attention", full_head_dim)
    _report_layer(cache_ok, sliding_idx, "sliding_attention", sliding_head_dim)
    print("-> both layers used the patched DynamicLayer; both shapes show head_dim transposed to dim -2.")
    print()

    print("=== Case 2: DynamicCache(config=model.config) -- the trap ===")
    cache_bad = DynamicCache(config=model.config)
    print("cache layer classes BEFORE forward:", {type(l).__name__ for l in cache_bad.layers})
    try:
        _run_prefill_then_decode(model, tokenizer, cache_bad)
    except RuntimeError as e:
        print("CRASHED on the decode step (call 2) with:")
        print(" ", e)
        print()
        print("Why: DynamicSlidingWindowLayer defines its own update(), which shadows the patched")
        print("DynamicLayer.update() qairt installed -- so sliding-attention layers run 100% vanilla")
        print("cache code. Vanilla update() assumes keys are NOT transposed (seq axis at dim -2), but")
        print("our adapted attention always hands it transposed keys (transposed_key_cache=True).")
        print("On call 1 (empty cache) this mismatch is silent -- the first-write path just stores")
        print("whatever shape it's given. On call 2, vanilla's torch.cat along dim -2 collides with")
        print("the transposed axis and the shapes no longer line up -> RuntimeError.")
        return


    _report_layer(cache_bad, full_idx, "full_attention", full_head_dim)
    _report_layer(cache_bad, sliding_idx, "sliding_attention", sliding_head_dim)
    print("-> full_attention still used (patched) DynamicLayer.")
    print("-> sliding_attention used DynamicSlidingWindowLayer -- if its shape looks non-transposed")
    print("   compared to Case 1's sliding_attention shape, the patch silently never reached it.")


if __name__ == "__main__":
    main()
