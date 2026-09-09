
from typing import Optional

import torch

from qairt.experimental.pipeline.torch.llm.generation.generator import LLMGenerator
from qairt.experimental.pipeline.torch.llm.utils.llm_io import Feature, IOType, LlmIOConfig

from models.gemma4_text.reauthoring import create_per_layer_inputs, create_position_embeddings


class Gemma4TextGenerator(LLMGenerator):

    @classmethod
    def _configure_io(cls, config) -> None:

        num_kv_owning_layers = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        cls.io_config = LlmIOConfig(
            num_hidden_layers=num_kv_owning_layers,
            io_type=IOType.HF,


            output_io_type=IOType.GENIE,
            features=(Feature.SLIDING_WINDOW_ATTENTION,),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self)._configure_io(self.config)


        type(self)._host_decoder = self.model.model

    @classmethod
    def prepare_inputs(
        cls,
        model,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values,
        sequence_length: int,
        context_length: int,
        **kwargs,
    ) -> dict:
        decoder = getattr(model, "model", None) or cls._host_decoder
        config = decoder.config
        device = next(decoder.parameters()).device
        dtype = next(decoder.parameters()).dtype
        pad_token = config.pad_token_id or 0

        if input_ids is None:
            input_ids = torch.zeros((1, sequence_length), dtype=torch.long, device=device)


        num_real = input_ids.shape[1]
        if num_real < sequence_length:
            pad = torch.full((1, sequence_length - num_real), pad_token, dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, pad], dim=1)

        gcl = context_length
        lcl = config.modified_sliding_window or config.sliding_window
        swp = config.sliding_window_pattern
        num_ltr = config.num_layers_to_run or config.num_hidden_layers
        full_kv_layers = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        num_kv = min(num_ltr, full_kv_layers)
        layer_types = config.layer_types[:num_kv]
        has_global = num_kv >= swp
        num_kv_heads = config.num_key_value_heads
        global_head_dim = config.global_head_dim
        head_dim = config.head_dim
        mask_neg = float(config.mask_neg)


        past_key_values = tuple(
            (
                torch.zeros((1, num_kv_heads, global_head_dim, gcl), dtype=dtype, device=device)
                if layer_type == "full_attention"
                else torch.zeros((1, num_kv_heads, head_dim, lcl), dtype=dtype, device=device),
                torch.zeros((1, num_kv_heads, gcl, global_head_dim), dtype=dtype, device=device)
                if layer_type == "full_attention"
                else torch.zeros((1, num_kv_heads, lcl, head_dim), dtype=dtype, device=device),
            )
            for layer_type in layer_types
        )


        def _align_up(x: int) -> int:
            a = 32
            m = ((x + a - 1) // a) * a
            return m if m != 0 else a

        write_base_glb = _align_up(0)
        write_base_swa = _align_up(0)
        assert write_base_swa + sequence_length <= lcl, (
            f"SWA write base {write_base_swa} + sequence_length {sequence_length} exceeds lcl {lcl}"
        )


        sliding_mask = torch.full((1, 1, sequence_length, lcl), mask_neg, device=device)
        cols_swa = torch.arange(lcl, device=device)
        for r in range(num_real):
            sliding_mask[0, 0, r, (cols_swa >= write_base_swa) & (cols_swa <= write_base_swa + r)] = 0.0


        position_ids = torch.arange(sequence_length, dtype=torch.long, device=device).unsqueeze(0)
        inputs_embeds = decoder.embed_tokens(input_ids)
        position_embeddings, swa_position_embeddings = create_position_embeddings(decoder.rotary_emb, position_ids)
        per_layer_inputs = create_per_layer_inputs(decoder, input_ids, inputs_embeds)


        out = {"input_ids": input_ids}

        if has_global:
            assert write_base_glb + sequence_length <= gcl, (
                f"global write base {write_base_glb} + sequence_length {sequence_length} exceeds gcl {gcl}"
            )
            causal_mask = torch.full((1, 1, sequence_length, gcl), mask_neg, device=device)
            cols_glb = torch.arange(gcl, device=device)
            for r in range(num_real):
                causal_mask[0, 0, r, (cols_glb >= write_base_glb) & (cols_glb <= write_base_glb + r)] = 0.0
            out["attention_mask"] = causal_mask
            out["position_ids"] = position_embeddings

        out["past_key_values"] = past_key_values
        out["per_layer_inputs"] = per_layer_inputs
        out["swa_position_ids"] = swa_position_embeddings
        out["swa_attention_mask"] = sliding_mask

        if has_global:
            out["cache_index"] = torch.tensor([write_base_glb], dtype=torch.int64, device=device)

        out["swa_cache_index"] = torch.tensor([write_base_swa], dtype=torch.int64, device=device)

        return out
