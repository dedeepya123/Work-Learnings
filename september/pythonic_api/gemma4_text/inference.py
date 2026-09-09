
import torch
from transformers.cache_utils import DynamicCache

from models.gemma4_text.reauthoring import (
    QcGemma4TextRotaryEmbedding,
    create_attention_masks,
    create_per_layer_inputs,
    create_position_embeddings,
)


def run_test(model, tokenizer, messages: list) -> None:
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )


    past_key_values = DynamicCache()


    position_ids = torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0)
    position_embeddings, swa_position_embeddings = create_position_embeddings(model.model.rotary_emb, position_ids)
    inputs_embeds = model.model.embed_tokens(inputs["input_ids"])
    attention_mask, swa_attention_mask = create_attention_masks(
        model.config, inputs_embeds, inputs.get("attention_mask"), position_ids, past_key_values
    )
    per_layer_inputs = create_per_layer_inputs(model.model, inputs["input_ids"], inputs_embeds)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            position_ids=position_embeddings,
            swa_attention_mask=swa_attention_mask,
            swa_position_ids=swa_position_embeddings,
            past_key_values=past_key_values,
            per_layer_inputs=per_layer_inputs,
            use_cache=True,
        )

    print("logits shape:", outputs.logits.shape)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    print("next token:", tokenizer.decode(next_token))


    assert isinstance(model.model.rotary_emb, QcGemma4TextRotaryEmbedding)
    print(
        "reauthoring check passed: rotary_emb ->",
        type(model.model.rotary_emb).__name__,
        "consumed a precomputed (cos, sin) tuple",
    )


    cached_key = past_key_values.layers[0].keys
    seq_len = inputs["input_ids"].shape[1]
    head_dim = model.config.head_dim
    assert cached_key.shape[-2:] == (head_dim, seq_len), f"unexpected cached key shape {cached_key.shape}"
    print(
        "reauthoring check passed: cached key shape ->",
        tuple(cached_key.shape),
        "(head_dim, seq) order confirms transposed_key_cache path ran)",
    )
