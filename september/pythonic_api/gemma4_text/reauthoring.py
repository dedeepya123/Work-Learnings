
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Gemma4TextConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Act2FN,
    Gemma4ForCausalLM,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextMLP,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
    repeat_kv,
)
from transformers.models.gemma4.quantization_gemma4 import Gemma4QuantizableEmbedding, fake_quant_activation
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from qairt.experimental.pipeline.torch.llm.models.utils import QcAttentionMixin, QcConfigMixin

from .quant_utils import clip_only_fake_quant


class QcGemma4TextConfig(Gemma4TextConfig, QcConfigMixin):


    QC_BACKING_ATTRS = {
        **QcConfigMixin.QC_BACKING_ATTRS,
        "sliding_window_pattern": None,
        "mask_neg": -200,
        "context_length": None,
        "num_layers_to_run": None,
        "pad_to_left": False,
        "modified_sliding_window": None,
        "kv_clip_only": True,
    }

    def __init__(self, *args, **kwargs):
        Gemma4TextConfig.__init__(self, *args, **kwargs)
        QcConfigMixin.__init__(self, **kwargs)

    @property
    def sliding_window_pattern(self) -> int:
        if getattr(self, "_sliding_window_pattern", None) is None:
            self._sliding_window_pattern = self.layer_types.index("full_attention") + 1
        return self._sliding_window_pattern

    @sliding_window_pattern.setter
    def sliding_window_pattern(self, value: Optional[int]):
        self._sliding_window_pattern = value

    @property
    def mask_neg(self) -> int:
        if not hasattr(self, "_mask_neg"):
            self._mask_neg = -200
        return self._mask_neg

    @mask_neg.setter
    def mask_neg(self, value: int):
        self._mask_neg = value

    @property
    def context_length(self) -> Optional[int]:
        if not hasattr(self, "_context_length"):
            self._context_length = None
        return self._context_length

    @context_length.setter
    def context_length(self, value: Optional[int]):
        self._context_length = value

    @property
    def num_layers_to_run(self) -> Optional[int]:
        if not hasattr(self, "_num_layers_to_run"):
            self._num_layers_to_run = None
        return self._num_layers_to_run

    @num_layers_to_run.setter
    def num_layers_to_run(self, value: Optional[int]):
        self._num_layers_to_run = value

    @property
    def pad_to_left(self) -> bool:
        if not hasattr(self, "_pad_to_left"):
            self._pad_to_left = False
        return self._pad_to_left

    @pad_to_left.setter
    def pad_to_left(self, value: bool):
        self._pad_to_left = value

    @property
    def modified_sliding_window(self) -> Optional[int]:
        if not hasattr(self, "_modified_sliding_window"):
            self._modified_sliding_window = None
        return self._modified_sliding_window

    @modified_sliding_window.setter
    def modified_sliding_window(self, value: Optional[int]):
        self._modified_sliding_window = value

    @property
    def kv_clip_only(self) -> bool:
        if not hasattr(self, "_kv_clip_only"):
            self._kv_clip_only = True
        return self._kv_clip_only

    @kv_clip_only.setter
    def kv_clip_only(self, value: bool):
        self._kv_clip_only = value


class QcApplyRopeSingle(nn.Module):

    def forward(
        self, x_real: torch.Tensor, x_im: torch.Tensor, rope_vals: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        rope_real, rope_im = rope_vals
        x_prod_real = x_real * rope_real - x_im * rope_im
        x_prod_im = x_real * rope_im + x_im * rope_real
        return torch.cat((x_prod_real, x_prod_im), dim=-1)


def qc_gemma4_eager_attention_forward(
    module: "QcGemma4TextAttention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if module.config.transposed_key_cache:
        attn_weights = torch.matmul(query, key_states) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        if module.enable_masked_softmax:
            attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
            attn_weights = torch.where(
                attention_mask == 0, attn_weights, attn_weights_min + module.config.mask_neg
            )
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QcGemma4TextAttention(QcAttentionMixin, Gemma4TextAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        shared_kv_states: dict,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = query_states.transpose(1, 2)
        q_half = query_states.shape[-1] // 2
        query_states = self.apply_rope_fn(
            query_states[..., :q_half], query_states[..., q_half:], position_embeddings
        )

        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = (
                self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
            )

            key_states = self.k_norm(key_states)
            key_states = key_states.transpose(1, 2)
            k_half = key_states.shape[-1] // 2
            key_states = self.apply_rope_fn(
                key_states[..., :k_half], key_states[..., k_half:], position_embeddings
            )


            if self.config.transposed_key_cache:
                key_states = key_states.transpose(2, 3)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

            if self.k_cache_scale is not None and self.k_cache_num_bits is not None:
                key_states = self._kv_fake_quant_fn(
                    key_states, self.k_cache_scale, int(self.k_cache_num_bits.item())
                )
            if self.v_cache_scale is not None and self.v_cache_num_bits is not None:
                value_states = self._kv_fake_quant_fn(
                    value_states, self.v_cache_scale, int(self.v_cache_num_bits.item())
                )

        if past_key_values is not None and not self.is_kv_shared_layer:
            cache_kwargs = {"cache_position": kwargs.get("cache_position")}
            cache_kwargs = self._update_cache_kwargs_with_qc_config(cache_kwargs)
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = key_states, value_states

        attn_output, attn_weights = qc_gemma4_eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, scaling=self.scaling, **kwargs
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _init_qc_gemma4_text_attention(module: "QcGemma4TextAttention") -> None:
    module.apply_rope_fn = QcApplyRopeSingle()
    module.enable_masked_softmax = module.config.enable_masked_softmax
    module._kv_fake_quant_fn = (
        clip_only_fake_quant if module.config.kv_clip_only else fake_quant_activation
    )


class QcGemma4TextDecoderLayer(Gemma4TextDecoderLayer):

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.add1(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.add2(residual, hidden_states)

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = self.mul1(hidden_states, per_layer_input)
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = self.add3(residual, hidden_states)

        hidden_states = self.mul2(hidden_states, self.layer_scalar)
        return hidden_states


class QcGemma4TextModel(Gemma4TextModel):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        swa_position_ids=None,
        swa_attention_mask=None,
        cache_position: Optional[torch.LongTensor] = None,
        swa_cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)


        if use_cache and past_key_values is None:
            raise ValueError(
                "QcGemma4TextModel requires an explicit past_key_values=DynamicCache() "
                "(no config=) when use_cache is True; the vanilla auto-construction path "
                "is not compatible with KVCacheMapping's patched cache layers."
            )


        return_legacy_cache = False
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            legacy_past_key_values = past_key_values
            past_key_values = DynamicCache()
            for layer_idx, (key_states, value_states) in enumerate(legacy_past_key_values):
                past_key_values.update(key_states, value_states, layer_idx)


        if swa_cache_position is None:
            layer_idx = (
                0
                if self.config.sliding_window_pattern == self.config.num_hidden_layers
                else self.config.sliding_window_pattern
            )
            past_seen_tokens = (
                past_key_values.get_seq_length(layer_idx=layer_idx) if past_key_values is not None else 0
            )
            swa_cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        hidden_states = self.embed_outs(inputs_embeds)


        if isinstance(position_ids, (tuple, list)) or position_ids is None:
            position_embeddings = position_ids
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids, "full_attention")

        if isinstance(swa_position_ids, (tuple, list)):
            swa_position_embeddings = swa_position_ids
        else:
            swa_position_embeddings = self.rotary_emb(hidden_states, swa_position_ids, "sliding_attention")

        global_causal_mask = attention_mask
        global_cache_position = cache_position
        global_position_embeddings = position_embeddings

        shared_kv_states = {}
        num_layers_to_run = getattr(self.config, "num_layers_to_run", None) or self.config.num_hidden_layers
        for i, decoder_layer in enumerate(self.layers[:num_layers_to_run]):

            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            if self.config.layer_types[i] == "sliding_attention":
                cache_position = swa_cache_position
                attention_mask = swa_attention_mask
                position_embeddings = swa_position_embeddings
            else:
                cache_position = global_cache_position
                attention_mask = global_causal_mask
                position_embeddings = global_position_embeddings

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        if return_legacy_cache:
            past_key_values = tuple((layer.keys, layer.values) for layer in past_key_values.layers)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class QcGemma4ForCausalLM(Gemma4ForCausalLM):


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        swa_position_ids=None,
        swa_attention_mask=None,
        cache_position: Optional[torch.LongTensor] = None,
        swa_cache_position: Optional[torch.LongTensor] = None,
        cache_index: Optional[torch.Tensor] = None,
        swa_cache_index: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), (
                "QcGemma4ForCausalLM has no \"cache_tensor\"; check that "
                "config.input_tokens_per_inference is set."
            )
            cache_position = cache_index + self.cache_tensor
        if swa_cache_index is not None:
            assert hasattr(self, "cache_tensor"), (
                "QcGemma4ForCausalLM has no \"cache_tensor\"; check that "
                "config.input_tokens_per_inference is set."
            )
            swa_cache_position = swa_cache_index + self.cache_tensor

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            use_cache=use_cache,
            swa_position_ids=swa_position_ids,
            swa_attention_mask=swa_attention_mask,
            cache_position=cache_position,
            swa_cache_position=swa_cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state


        logits = self.lm_head(hidden_states)
        logits = self.lm_head_outs(logits)
        if self.config.final_logit_softcapping is not None:
            logits = self.div(logits, self.config.final_logit_softcapping)
            logits = self.tanh(logits)
            logits = self.mul(logits, self.config.final_logit_softcapping)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _init_qc_gemma4_causal_lm_cache_tensor(module: "QcGemma4ForCausalLM") -> None:
    if module.config.input_tokens_per_inference is not None:
        module.register_buffer(
            "cache_tensor",
            torch.arange(module.config.input_tokens_per_inference),
            persistent=False,
        )


def _revert_qc_gemma4_causal_lm_cache_tensor(module: "QcGemma4ForCausalLM") -> None:
    if hasattr(module, "cache_tensor"):
        delattr(module, "cache_tensor")


class TextModelQc(nn.Module):

    def __init__(self, qc_model: "QcGemma4ForCausalLM") -> None:
        super().__init__()
        self.qc_model = qc_model

    @property
    def model(self):
        return self.qc_model.model

    @property
    def config(self):
        return self.qc_model.config

    @property
    def device(self) -> torch.device:
        return next(self.qc_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.qc_model.parameters()).dtype

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        per_layer_inputs=None,
        swa_position_ids=None,
        swa_attention_mask=None,
        cache_index=None,
        swa_cache_index=None,
    ):
        outputs = self.qc_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            per_layer_inputs=per_layer_inputs,
            swa_position_ids=swa_position_ids,
            swa_attention_mask=swa_attention_mask,
            cache_index=cache_index,
            swa_cache_index=swa_cache_index,
        )
        flat_kv = [t for pair in outputs.past_key_values for t in pair]
        return tuple([outputs.logits, *flat_kv])


class QcGemma4QuantizableEmbedding(Gemma4QuantizableEmbedding):

    def dequantize(self) -> None:
        if self.weight_scale is not None:
            weight_scale = self.weight_scale
            stretch_factor = self.embedding_dim // weight_scale.shape[-1]
            if stretch_factor > 1:
                weight_scale = weight_scale.repeat_interleave(stretch_factor, dim=-1)
            self.weight = nn.Parameter(self.weight.to(dtype=weight_scale.dtype) * weight_scale)
            self.weight_scale = None


def _init_qc_gemma4_quantizable_embedding(module: "QcGemma4QuantizableEmbedding") -> None:
    module.dequantize()


class QcGemma4RMSNorm(Gemma4RMSNorm):

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states / torch.sqrt(mean_squared)


class QcGemma4TextMLP(Gemma4TextMLP):
    pass


def _init_qc_gemma4_text_mlp_act_fn(module: "QcGemma4TextMLP") -> None:
    module.act_fn = Act2FN("gelu", craft_config=module.config.craft_config)


class QcGemma4TextRotaryEmbedding(Gemma4TextRotaryEmbedding):
    pass


def create_position_embeddings(
    rotary_emb: "QcGemma4TextRotaryEmbedding",
    position_ids: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    x = torch.ones(1, device=position_ids.device, dtype=dtype)
    config = rotary_emb.config

    def _rope_vals(layer_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = config.head_dim if layer_type == "sliding_attention" else config.global_head_dim
        cos, sin = rotary_emb(x, position_ids, layer_type)
        cos, sin = cos.unsqueeze(dim=1), sin.unsqueeze(dim=1)
        return cos[..., : dim // 2], sin[..., : dim // 2]

    full_position_embeddings = _rope_vals("full_attention")
    swa_position_embeddings = _rope_vals("sliding_attention")
    return full_position_embeddings, swa_position_embeddings


def create_per_layer_inputs(
    decoder: "QcGemma4TextModel", input_ids: torch.Tensor, inputs_embeds: torch.Tensor
) -> Optional[torch.Tensor]:
    if not decoder.config.hidden_size_per_layer_input:
        return None
    return decoder.get_per_layer_inputs(input_ids, inputs_embeds)


def create_attention_masks(
    config: "QcGemma4TextConfig",
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    past_key_values=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    causal_mask = create_causal_mask(**mask_kwargs)
    sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)
    return causal_mask, sliding_mask
