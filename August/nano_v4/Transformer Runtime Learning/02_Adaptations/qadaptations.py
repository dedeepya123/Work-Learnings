# !/usr/bin/env python3
# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

""" This file provides adaptations to the Gemma4 model. These adaptations are being done to optimize the model execution on the HTP backend. """


from typing import Any, Dict, Optional, Tuple, Union
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property

import torch
from torch import nn
import torch.nn.functional as F

from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import repeat_kv
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4MtpCrossAttention as Gemma4MtpCrossAttention_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AssistantDecoderLayer as Gemma4AssistantDecoderLayer_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AssistantModel as Gemma4AssistantModel_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention as Gemma4TextAttention_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer as Gemma4TextDecoderLayer_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as Gemma4TextModel_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model as Gemma4Model_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM as Gemma4ForCausalLM_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as Gemma4ForConditionalGeneration_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4ModelOutputWithPast as Gemma4ModelOutputWithPast_original, Gemma4CausalLMOutputWithPast as  Gemma4CausalLMOutputWithPast_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel as Gemma4VisionModel_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder as Gemma4VisionPatchEmbedder_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoder as Gemma4VisionEncoder_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionAttention as Gemma4VisionAttention_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding as Gemma4VisionRotaryEmbedding_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler as Gemma4VisionPooler_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionMLP as Gemma4VisionMLP_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP as Gemma4TextMLP_original
from transformers.models.gemma4.modeling_gemma4 import Act2FN
from transformers.models.gemma4.quantization_gemma4 import Gemma4QuantizableEinsum as Gemma4QuantizableEinsum_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel as Gemma4AudioModel_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioAttention as Gemma4AudioAttention_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModelOutput, Gemma4AudioSubSampleConvProjectionOutput
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioFeedForward as Gemma4AudioFeedForward_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioLayer as Gemma4AudioLayer_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioLightConv1d as Gemma4AudioLightConv1d_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioSubSampleConvProjection as Gemma4AudioSubSampleConvProjection_original
from transformers.models.gemma4.modeling_gemma4 import Conv1d, Mul
from transformers.models.gemma4.quantization_gemma4 import Gemma4QuantizableEmbedding as Gemma4QuantizableEmbedding_original
from transformers.models.gemma4.quantization_gemma4 import Gemma4QuantizableLinear as Gemma4QuantizableLinear_original
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm as Gemma4RMSNorm_original

from transformers.utils import logging

from transformers.models.gemma4.quantization_gemma4 import fake_quant_activation

from air.nanov4.utils import fake_quant, EinsumOp, AudioModelHelpers
from qlib.qadaptation_flags import AdaptationFlags

logger = logging.get_logger(__name__)


def save_tensor(tensor, name):
    import numpy as np
    prefix = 'qc'
    np.save(f'workspace/compare/{prefix}/{name}.npy', tensor.numpy())
    pass


def to_legacy_cache(cache):
    """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
    backward compatibility."""

    if isinstance(cache, tuple):
        return cache
        
    legacy_cache = ()
    for layer in cache.layers:
        legacy_cache += ((layer.keys, layer.values),)
    return legacy_cache

def from_legacy_cache(past_key_values):
    """
    Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
    backward compatibility.
    """
    cache = DynamicCache_adapted()

    if past_key_values is None:
        logger.warning_once("past_key_values should not be None in from_legacy_cache()")
    if past_key_values is not None:
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
    return cache


@dataclass
class Gemma4ModelOutputWithPast(Gemma4ModelOutputWithPast_original):
    past_kv_local: Tuple[torch.Tensor] | None = None
    past_kv_global: Tuple[torch.Tensor] | None = None
    audio_subsample_state: Tuple[torch.Tensor] | None = None
    audio_past_key_values: Tuple[Tuple[torch.Tensor]] | None = None
    audio_lconv1d_state: Tuple[torch.Tensor] | None = None


    @classmethod
    def from_tuple(cls, outputs, has_audio=False) -> "Gemma4ModelOutputWithPast":
        result = cls()
        result.last_hidden_state = outputs[0]
        result.past_key_values = outputs[1]
        result.past_kv_local = outputs[2]
        result.past_kv_global = outputs[3]
        if has_audio:
            result.audio_subsample_state = outputs[4]
            result.audio_past_key_values = outputs[5]
            result.audio_lconv1d_state = outputs[6]

        return result
        

@dataclass
class Gemma4CausalLMOutputWithPast(Gemma4CausalLMOutputWithPast_original):
    assistant_logits: torch.Tensor | None = None
    assistant_state: torch.Tensor | None = None
    assistant_indices: torch.Tensor | None = None
    audio_subsample_state: torch.Tensor | None = None
    audio_past_key_values: torch.Tensor | None = None
    audio_lconv1d_state: torch.Tensor | None = None


    def to_tuple(self) -> Tuple:
        result = (self.loss, self.logits, self.past_key_values, self.hidden_states, self.attentions, 
                  self.assistant_logits, self.assistant_state, self.assistant_indices,
                  self.audio_subsample_state, self.audio_past_key_values, self.audio_lconv1d_state)
        return tuple(val for val in result if val is not None)


class MulModule(nn.Module):
    def forward(self, a, b):
        return a * b


## ADAPTATION_1: we apply the rope separately to query and key states for on-target efficiency
## Creating a separate class because we want to uniquely identify the EleMul operations in QuantSim
class ApplyRopeSingle(nn.Module):
    '''
    Based on FacebookResearch's llama, provided by Carl
    '''
    def __init__(self):
        super().__init__()
        self.mul_x_real_rope_real = MulModule()
        self.mul_x_im_rope_im = MulModule()
        self.mul_x_real_rope_im = MulModule()
        self.mul_x_im_rope_real = MulModule()

    def forward(self, x_real, x_im, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
        rope_real = rope_vals[0]  # shape should be 1, 1, seqlen, head_dim/2
        rope_im = rope_vals[1]  # shape should be 1, 1, seqlen, head_dim/2

        x_prod_real = self.mul_x_real_rope_real(x_real, rope_real) - self.mul_x_im_rope_im(x_im, rope_im)
        x_prod_im = self.mul_x_real_rope_im(x_real, rope_im) + self.mul_x_im_rope_real(x_im, rope_real)

        # TODO: HF need to uses different interleaving
        x = torch.cat((x_prod_real, x_prod_im), dim=3).view(*x_real.shape[:-1], -1)
        return x


def _apply_rope_single(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    raise RuntimeError("_apply_rope_single is deprecated; use ApplyRopeSingle module instead")

def _apply_rope_multidim(x, rope_vals: Tuple[torch.Tensor, torch.Tensor]):
    num_input_channels = x.shape[-1]
    c = x.shape[-1] // 4
    x_r0, x_i0, x_r1, x_i1 = torch.split(x, [c, c, c, c], dim=-1)
    rope_real_parts, rope_im_parts = rope_vals[0], rope_vals[1]

    _rope_single = ApplyRopeSingle()

    y_parts = [
        _rope_single(
            x_real=x_r,
            x_im=x_i,
            rope_vals=(rope_real_parts[k][None, :, :, :], rope_im_parts[k][None, :, :, :])
        )
        for k, (x_r, x_i) in enumerate([(x_r0, x_i0), (x_r1, x_i1)])
    ]
    result = torch.cat(y_parts, dim=-1)
    return result

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float | int = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    transposed_key_cache = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5
    

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # ADAPTATION_2: We send the transposed key cache to avoid the transpose inside matmul, it is expensive on target
    if transposed_key_cache:
        attn_weights = torch.matmul(query, key_states) * scaling
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap


    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask

        if attention_mask.shape[-1] != value_states.shape[-2]:#[:, :, :, : value_states.shape[-2]], value_states.shape[-2] is same as attention_mask length for static graph
        # The following section of code will only run when we want to trace the adapted model when creating the MPP model (we will pass KV$ for num_sequential_layers-1), 
        # otherwise the last dimension will have shape mismatch between the attn_weights (=ARN) and the attention_mask (=CL)
            causal_mask = attention_mask[:, :, :, :value_states.shape[-2]]

        if module.enable_masked_softmax:
            attn_weights_min, _ = torch.min(attn_weights, dim=-1, keepdim=True)
            minus_value = getattr(module.config, 'mask_neg', -200)
            attn_weights = torch.where(causal_mask==0, attn_weights, attn_weights_min + minus_value)
        else:
            attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma4MtpCrossAttention(Gemma4MtpCrossAttention_original):

    def __init__(self, config, layer_idx):
        super(Gemma4MtpCrossAttention, self).__init__(config, layer_idx)
        self.enable_masked_softmax = getattr(config, 'adaptations', AdaptationFlags()).enable_masked_softmax
        self.apply_rope_fn = ApplyRopeSingle()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[1:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # cos, sin = position_embeddings

        # 1. Project Query using Drafter weights
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = query_states.transpose(1, 2)
        _q_half = query_states.shape[-1] // 2
        query_states = self.apply_rope_fn(query_states[..., :_q_half], query_states[..., _q_half:], position_embeddings)
        # query_states = self.apply_rope_fn(query_states, position_embeddings)

        # 2. Read-only extraction from Backbone Cache
        if past_key_values is not None:
            # last_sliding_idx = getattr(self.config, "last_backbone_sliding_idx", None)
            # last_full_idx = getattr(self.config, "last_backbone_full_idx", None)

            # if self.is_sliding:
            #     target_layer_idx = last_sliding_idx
            # else:
            #     target_layer_idx = last_full_idx

            # if target_layer_idx is None:
            #     raise ValueError(f"Target backbone layer index not set in config for layer {self.layer_idx}")

            # if hasattr(past_key_values, "layers"):
            #     key_states = past_key_values.layers[target_layer_idx].keys
            #     value_states = past_key_values.layers[target_layer_idx].values
            # else:
            #     key_states, value_states = past_key_values[target_layer_idx]

            # # Device placement
            # key_states = key_states.to(query_states.device)
            # value_states = value_states.to(query_states.device)

            # # Note: We assume the cache already has RoPE applied if needed,
            # # and we do not update the cache.

            key_states, value_states = past_key_values

        else:
            raise ValueError("past_key_values must be provided for MTP cross-attention")

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            transposed_key_cache=True,
            **kwargs,
        )

        attn_output = attn_output.reshape(1, *input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma4AssistantDecoderLayer(Gemma4AssistantDecoderLayer_original):

    def __init__(self, config, layer_idx):
        super(Gemma4AssistantDecoderLayer, self).__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        # position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.add1(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.add2(residual, hidden_states)

        hidden_states = self.mul1(hidden_states, self.layer_scalar)
        return hidden_states

class Gemma4AssistantModel(Gemma4AssistantModel_original):

    def __init__(self, config):
        super(Gemma4AssistantModel, self).__init__(config)
        self.matmul_centroid1 = EinsumOp()

    def _apply_centroid_decoding(
        self,
        hidden_states: torch.Tensor,  # [B, L, D]
    ) -> torch.Tensor:
        """Apply centroid decoding by skipping full lm_head computation."""
        _, _, b, d = hidden_states.shape
        # 1. Compute centroid scores: [B, L, D] @ [D, C] -> [B, L, C]
        centroid_logits = self.matmul_centroid1(hidden_states, self.centroids.T)

        # 2. Find threshold from top-k centroids
        _, top_k_indices = torch.topk(centroid_logits, k=self.centroid_intermediate_top_k, dim=-1)  # [B, L, top_k]
        
        # selected_tokens = self.canonical_positions_per_cluster[top_k_indices].reshape(1, 1, top_k_indices.shape[-2], -1)
        # selected_weights = self.lm_head.weight[:, :, 0, 0][selected_tokens].squeeze(0)
        # selected_logits = self.lm_head_einsum(hidden_states, selected_weights, einsum_str="blnd,bntd->blnt")

        # output = self.mask.scatter(dim=3, index=selected_tokens, src=selected_logits)

        # return output

        # 3. Get token indices for these centroids
        # batch_size, _, seq_len, _ = hidden_states.shape

        # selected_weights = self.lm_head.weight.view(
        #     self.config.num_centroids,
        #     self.vocab_size_per_centroid,
        #     self.hidden_size,
        # )[top_k_indices]  # [B, L, top_k, K, D]
        # selected_weights = selected_weights.view(
        #     batch_size, seq_len, -1, self.hidden_size)  # [B, L, top_k * K, D]

        # selected_scales = None
        # if hasattr(self.lm_head, "weight_scale") and self.lm_head.weight_scale is not None:
        #     selected_scales = self.lm_head.weight_scale.view(
        #         self.config.num_centroids,
        #         self.vocab_size_per_centroid,
        #         1,
        #     )[top_k_indices]  # [B, L, top_k, K, 1]
        #     selected_scales = selected_scales.view(
        #         batch_size, seq_len, -1)  # [B, L, top_k * K]

        # selected_logits = self.lm_head_einsum(
        #     "bld,blnd->bln",
        #     hidden_states,
        #     selected_weights,
        #     # weight_scale=selected_scales,
        # )  # [B, L, top_k * K]
        # selected_logits = torch.matmul(hidden_states, selected_weights.permute(0,1,3,2))
        # selected_logits = self.lm_head_einsum_out(selected_logits)

        return self.lm_head(hidden_states), top_k_indices


    def forward(self,
        embedding: torch.Tensor,
        state: torch.Tensor,
        # position_ids: torch.LongTensor = None,
        attention_mask: dict[str, torch.Tensor] | None = None,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        # past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        past_kv_local: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_kv_global: tuple[torch.Tensor, torch.Tensor] | None = None,
        swa_attention_mask: torch.Tensor = None,
        swa_position_embeddings: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the drafter.

        Args:
            embedding: Token embedding from backbone's embed_tokens.
                Shape: [batch, seq_len, backbone_hidden_size]
            state: Projected activation from previous drafter step
                (or backbone's final activation for the first step).
                Shape: [batch, seq_len, backbone_hidden_size]
            position_ids: Position indices.
                Shape: [batch, seq_len]
            attention_mask: Dict mapping layer_type → mask tensor.
            position_embeddings: Dict mapping layer_type → (cos, sin) RoPE tensors.
            past_key_values: KV cache (backbone's, read-only for drafter).
            cache_position: Cache position indices.

        Returns:
            logits: Token prediction logits.
                Shape: [batch, seq_len, vocab_size]
            projected_state: Post-projected activation for next step.
                Shape: [batch, seq_len, backbone_hidden_size]
        """
        # Step 1: Concatenate embedding and state
        drafter_input = torch.cat([embedding, state], dim=-1)
        # drafter_input: [B, L, 2 * backbone_d]

        # Step 2: Pre-projection
        hidden_states = self.pre_projection(drafter_input)
        # hidden_states: [B, L, drafter_d]

        # if position_embeddings is None:
        #     position_embeddings = {}
        #     for layer_type in set(self.config.layer_types):
        #         position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        num_mtp_layers_to_run = getattr(self.config, 'num_mtp_layers_to_run', len(self.layers))
        # Step 3: Pass through drafter transformer layers
        for layer in self.layers[:num_mtp_layers_to_run]:
            layer_type = layer.self_attn.layer_type
            hidden_states = layer(
                hidden_states,
                attention_mask=swa_attention_mask if layer_type == "sliding_attention" else attention_mask,
                position_embeddings=swa_position_embeddings if layer_type == "sliding_attention" else position_embeddings,
                # position_ids=position_ids,
                past_key_values=past_kv_local if layer_type == "sliding_attention" else past_kv_global,
                use_cache=False,  # Drafter does NOT write to cache
                cache_position=cache_position,
            )

        # Step 4: Final norm
        hidden_states = self.final_norm(hidden_states)

        # Step 5: Post-projection for state propagation to next step
        projected_state = self.post_projection(hidden_states)

        # Step 6: Logits via drafter's own lm_head OR centroid decoding
        if self.config.use_centroid_embedder and hasattr(self, "centroids") and self.centroids is not None:
            logits, indices = self._apply_centroid_decoding(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        return logits, projected_state, indices


class Gemma4TextAttention(Gemma4TextAttention_original):
    
    def __init__(self, config, layer_idx: int):
        super(Gemma4TextAttention, self).__init__(config, layer_idx)
        self.apply_rope_fn = ApplyRopeSingle()
        adaptations = getattr(config, 'adaptations', AdaptationFlags())
        self.enable_masked_softmax = adaptations.enable_masked_softmax
        if adaptations.kv_clip_only:
            self._kv_fake_quant_fn = lambda x, bits, scale: fake_quant(x, bits, scale)
        else:
            self._kv_fake_quant_fn = lambda x, bits, scale: fake_quant_activation(x, scale, bits)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        past_key_values: Optional[Cache] = None,
        # QC adapted args
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

        transposed_key_cache = self.config.transposed_key_cache if hasattr(self.config,
                                                                           'transposed_key_cache') else False
                                                                           
        return_new_key_value_only = self.config.return_new_key_value_only if hasattr(self.config,
                                                                                     'return_new_key_value_only') else False
        input_shape = hidden_states.shape[1:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        n, h, w, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)

        query_states = query_states.transpose(1, 2)
        _q_half = query_states.shape[-1] // 2
        query_states = self.apply_rope_fn(query_states[..., :_q_half], query_states[..., _q_half:], position_embeddings)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
        # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
        # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.kv_shared_layer_index]
            # Device of past layer may be different from current one
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = key_states.transpose(1, 2)
            _k_half = key_states.shape[-1] // 2
            key_states = self.apply_rope_fn(key_states[..., :_k_half], key_states[..., _k_half:], position_embeddings)

            if transposed_key_cache:
                key_states = key_states.transpose(2, 3)

            value_states = self.v_norm(value_states)

            value_states = value_states.transpose(1, 2)
            
            if self.k_cache_scale is not None and self.k_cache_num_bits is not None:
                key_states = self._kv_fake_quant_fn(
                    key_states, int(self.k_cache_num_bits.item()), self.k_cache_scale
                )
            if self.v_cache_scale is not None and self.v_cache_num_bits is not None:
                value_states = self._kv_fake_quant_fn(
                    value_states, int(self.v_cache_num_bits.item()), self.v_cache_scale
                )

        # ADAPTATION_3: We require to redefine the attention class since we need to pass additional cache_kwargs (specifically the transposed key cache and the return_new_key_value_only)
        if past_key_values is not None and not self.is_kv_shared_layer:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "cache_position": cache_position,
                "transposed_key_cache": transposed_key_cache,
                "num_key_value_heads": self.config.num_key_value_heads,
                "return_new_key_value_only": return_new_key_value_only,
                "head_dim": self.head_dim
            }

            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
        
        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = key_states, value_states

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            transposed_key_cache=transposed_key_cache,
            **kwargs,
        )

        attn_output = attn_output.reshape(n, h, w, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, key_states, value_states

class Gemma4TextDecoderLayer(Gemma4TextDecoderLayer_original):

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        # QC adapted args
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, key_states, value_states = self.self_attn(
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

            # Take hidden states before MLP here
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            # Combine mlp and moe outputs
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
        return hidden_states, key_states, value_states


class Gemma4TextModel(Gemma4TextModel_original):

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the context-aware component of PLE and combine with token-identity.

        Projects `inputs_embeds` through `per_layer_model_projection` (Linear), scales by
        `1/sqrt(hidden_size)`, reshapes to `[batch, seq, num_layers, ple_dim]`, and normalizes
        with `per_layer_projection_norm` (RMSNorm).

        If `per_layer_inputs` (the token-identity component from `get_per_layer_inputs()`)
        is provided, combines both: `(context_projection + token_identity) * (1/sqrt(2))`.
        If `per_layer_inputs` is None (e.g. for multimodal inputs where input_ids are not
        available), returns just the context projection.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call project_per_layer_inputs() from a model initialized with a config that does not"
                f" support per-layer embeddings. {self.config}"
            )

        per_layer_projection = self.per_layer_model_projection_mul(self.per_layer_model_projection(inputs_embeds), self.per_layer_model_projection_scale)
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[1:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return self.mul(self.add(per_layer_projection, per_layer_inputs), self.per_layer_input_scale)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        # QC adapted args
        swa_position_ids: Optional[torch.LongTensor] = None,
        swa_attention_mask: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        swa_cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (`torch.Tensor` of shape `(batch_size, sequence_length, num_hidden_layers, hidden_size_per_layer_input)`, *optional*):
            Pre-computed per-layer input embeddings. When provided, these are used directly instead of being
            computed from `input_ids` via `get_per_layer_inputs()`. This is primarily used by the multimodal
            model (`Gemma4Model`) which pre-computes per-layer inputs from the original `input_ids` *before*
            merging multimodal soft tokens into `inputs_embeds` — at which point the original token ids are
            no longer recoverable.
        """
        # ADAPATATION 4: We need to return the output from the model in dict format otehrwise jit trace cannot trace the outputs correctly and they are not seen in the onnx graph.
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                assert input_ids is not None, f"input_ids cannot be None if per_layer_inputs is not provided"
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
                per_layer_inputs = self.ple_outs(per_layer_inputs)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None:
            raise ValueError("use_cache is True but past_key_values are not provided")
            # past_key_values = DynamicCache(config=self.config)

        # ADAPTAION 5: We convert the past_key_values which flow into the model in the tuple format into the Cache object format. Even though we create a dynamic cache object here, 
        # the past_key_values tuple already contains the hybrid KV cache information i.e the sliding window layer have the KV cache of the size sliding_window.
        return_legacy_cache = False
        if past_key_values is not None and not isinstance(past_key_values, Cache):
            return_legacy_cache = True

            past_key_values = from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

        # if position_ids is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        #     position_ids = position_ids.unsqueeze(0)
        
        if swa_cache_position is None:
            # Ideally if the cache_position is None, we expect user to not pass index for both global and the local layers. [so both the ifs are either executed or not executed, for readability kept sep]
            # if swa_cache_position (for local layer) is None, compute like HF does, note not using the global layer idx here.
            layer_idx = 0 if self.config.sliding_window_pattern == self.config.num_hidden_layers else self.config.sliding_window_pattern
            past_seen_tokens = past_key_values.get_seq_length(layer_idx=layer_idx) if past_key_values is not None else 0
            swa_cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        
        # if swa_position_ids is None:
        #     swa_position_ids = swa_cache_position.unsqueeze(0)

        # ADAPATATION 6: Remove attn mask creation from inside model

        # embed positions
        hidden_states = self.embed_outs(inputs_embeds)
        
        if isinstance(position_ids, (tuple, list)) or position_ids is None: # position_ids is None when the Model is smaller than the sliding window pattern
            position_embeddings = position_ids
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids, "full_attention")

        if isinstance(swa_position_ids, (tuple, list)):
            swa_position_embeddings = swa_position_ids
        else:
            swa_position_embeddings = self.rotary_emb(hidden_states, swa_position_ids, "sliding_attention")
        
        ###### Inferring the sliding causal mask from the global causal mask. ######
        global_causal_mask = attention_mask

        ###### Inferring the sliding cache position from the global cache position. ######
        global_cache_position = cache_position

        global_position_embeddings = position_embeddings

        # Initialize as empty dict - it will be filled in the right layers
        shared_kv_states = {}

        # Determine how many layers to run
        num_layers_to_run = getattr(self.config, 'num_layers_to_run', len(self.layers))
        
        # Last local/global KV, to be used by MTP model
        global_k, global_v = None, None
        swa_k, swa_v = None, None
        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: num_layers_to_run]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            if self.config.layer_types[i] == "sliding_attention":
                cache_position = swa_cache_position
                attention_mask = swa_attention_mask
                position_embeddings = swa_position_embeddings
            else:
                cache_position = global_cache_position
                attention_mask = global_causal_mask
                position_embeddings = global_position_embeddings

            hidden_states, key_states, value_states = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

            if self.config.layer_types[i] == "sliding_attention":
                swa_k, swa_v = key_states, value_states
            else:
                global_k, global_v = key_states, value_states
        
        hidden_states = self.norm(hidden_states)

        if return_legacy_cache:
            past_key_values = to_legacy_cache(past_key_values)

        if not return_dict:
            return tuple([hidden_states, past_key_values, (swa_k, swa_v), (global_k, global_v)])

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Gemma4Model(Gemma4Model_original):

    def __init__(self, config):
        super(Gemma4Model, self).__init__(config)

        self.language_model.__class__ = Gemma4TextModel
        
        self.post_init()
        if getattr(config.text_config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.text_config.input_tokens_per_inference),
                                 persistent=False)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        image_pooling_idx: torch.Tensor | None = None,
        vision_output_length: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple:
        r"""
        vision_output_length (`int`, *optional*):
            The number of soft tokens the vision encoder should output. If not provided, defaults to
            `Gemma4VisionConfig.default_output_length`.
        """
        # Filter kwargs to only pass supported arguments to the vision tower
        vision_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("output_attentions", "output_hidden_states", "output_length_overrides")
        }
        last_hidden_state, hidden_states, attentions = self.vision_tower(
            pixel_values=pixel_values, image_position_ids=image_position_ids, image_pooling_idx=image_pooling_idx,
            output_length=vision_output_length, **vision_kwargs
        )
        pooler_output = self.embed_vision(inputs_embeds=last_hidden_state)

        return pooler_output

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_subsample_state: Optional[list[torch.Tensor]] = None,
        audio_past_key_values: Optional[list[torch.Tensor]] = None,
        audio_lconv1d_state: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Gemma4AudioModelOutput:
        r"""
        input_features (`torch.FloatTensor]` of shape `(num_images, seq_length, num_features)`):
            The tensors corresponding to the input audio.
        input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        """
        if self.audio_tower is None:
            raise ValueError(
                "Audio features were requested, but the model was initialized without an audio_config. "
                "Cannot process audio without an audio tower and audio embedder."
            )

        chunk_size = kwargs.pop("streaming_chunk_size_tokens", None)
        if chunk_size is None:
            chunk_size = getattr(self.config.audio_config, "streaming_chunk_size_tokens", None)
        
        assert input_features.shape[-2] % (12 * 4) == 0, 'The padded audio input should be a multiple of 12 and 4'

        audio_state = []
        if audio_subsample_state is not None:
            audio_state.append(audio_subsample_state)
            for (k_cache, v_cache), lconv1d_state in zip(audio_past_key_values, audio_lconv1d_state):
                audio_state.append([
                    {
                        "prev_k": k_cache,
                        "prev_v": v_cache,
                    },
                    lconv1d_state
                ])

        audio_outputs = self.audio_tower(
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            streaming_chunk_size_tokens=chunk_size,
            state=audio_state,
            **kwargs,
        )

        audio_outputs.pooler_output = self.embed_audio(inputs_embeds=audio_outputs.last_hidden_state)

        return audio_outputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        # QC adapted args
        swa_position_ids: Optional[torch.LongTensor] = None,
        swa_attention_mask: Optional[Cache] = None,
        per_layer_inputs: Optional[torch.FloatTensor] = None,
        cache_index: Optional[torch.LongTensor] = None,
        swa_cache_index: Optional[torch.LongTensor] = None,
        image_pooling_idx: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        audio_subsample_state: Optional[list[torch.Tensor]] = None,
        audio_past_key_values: Optional[list[torch.Tensor]] = None,
        audio_lconv1d_state: Optional[list[torch.Tensor]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Gemma4ModelOutputWithPast:
        r"""
        input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcGemma4Model doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            cache_position = cache_index + self.cache_tensor
        else:
            cache_position = None

        swa_cache_position = None
        # swa_cache_index determines the starting index where the new KV$ should be scattered for the local/ sliding window layer.
        if swa_cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcGemma4Model doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            # cache_position for local layer
            swa_cache_position = swa_cache_index + self.cache_tensor
        
        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        multimodal_mask = image_mask | video_mask | audio_mask

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # if self.config.get_text_config().hidden_size_per_layer_input:
        #     pad_embedding = self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]
        #     llm_inputs_embeds = torch.where(multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
        #     per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
        # else:
        #     per_layer_inputs = None

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values=pixel_values, image_position_ids=image_position_ids, image_pooling_idx=image_pooling_idx, return_dict=True)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            image_mask = image_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            
            # Use cumsum to track which image feature row to use at each position
            # For positions with image tokens, this gives: 0, 1, 2, 3, ..., N-1
            # For non-image positions, this stays at the previous value
            image_indices = torch.cumsum(image_mask.squeeze(-1).long(), dim=1) - 1  # [batch, seq_len]
            # Clamp to valid range [0, num_image_features-1] to handle edge cases
            image_indices = torch.clamp(image_indices, 0, image_features.shape[1] - 1)
            
            # Gather image features based on indices: [batch, seq_len, hidden_dim]
            # Expand indices to match hidden dimension
            image_indices_expanded = image_indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
            selected_image_features = torch.gather(image_features, 1, image_indices_expanded)
            # Replace embeddings at image positions using where
            # Only positions where image_mask is True will be replaced with image features
            inputs_embeds = torch.where(image_mask, selected_image_features, inputs_embeds)

        # Merge text and audio
        audio_output_states = tuple()
        if input_features is not None and input_features_mask is not None:
            audio_output = self.get_audio_features(input_features, input_features_mask, audio_attention_mask, audio_subsample_state, 
                                                   audio_past_key_values, audio_lconv1d_state, return_dict=True)
            audio_features = audio_output.pooler_output.squeeze(1)
            audio_mask_from_encoder = audio_output.attention_mask  # True = valid
            # audio_features = audio_features[audio_mask_from_encoder].unsqueeze(0)
            n_audio_tokens = audio_mask.sum()
            audio_mask = audio_mask.unsqueeze(-1)
            audio_indices = torch.cumsum(audio_mask.squeeze(-1).long(), dim=1) - 1
            audio_indices = torch.clamp(audio_indices, 0, audio_features.shape[1] - 1)
            audio_indices_expanded = audio_indices.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
            selected_audio_features = torch.gather(audio_features, 1, audio_indices_expanded)
            inputs_embeds = torch.where(audio_mask, selected_audio_features, inputs_embeds)

            audio_output_states = AudioModelHelpers.flatten_output_audio_states(audio_output.past_key_values)

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.language_model(
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            # QC adapted args
            swa_position_ids=swa_position_ids,
            swa_attention_mask=swa_attention_mask,
            # cache_index=cache_index,
            # swa_cache_index=swa_cache_index,
            cache_position=cache_position,
            swa_cache_position=swa_cache_position,
            **kwargs,
        )

        if not return_dict:
            return outputs + audio_output_states

        return Gemma4ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )


class Gemma4ForCausalLM(Gemma4ForCausalLM_original):
    
    def __init__(self, config):
        super(Gemma4ForCausalLM, self).__init__(config)
        self.model = Gemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        if getattr(config, "input_tokens_per_inference", None) is not None:
            self.register_buffer(name='cache_tensor', tensor=torch.arange(config.input_tokens_per_inference),
                                 persistent=False)
            # self.cache_tensor = torch.arange(config.input_tokens_per_inference)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        swa_position_ids: Optional[torch.LongTensor] = None,
        swa_attention_mask: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.FloatTensor] = None,
        cache_index: Optional[torch.Tensor] = None,
        swa_cache_index: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **lm_kwargs,
    ) -> Gemma4CausalLMOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcGemma4Model doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
        
            cache_position = cache_index + self.cache_tensor

        swa_cache_position = None
        # swa_cache_index determines the starting index where the new KV$ should be scattered for the local/ sliding window layer.
        if swa_cache_index is not None:
            assert hasattr(self, "cache_tensor"), "QcGemma4Model doesn't have attribute \"cache_tensor\", " \
                                                  "check if \"input_tokens_per_inference\" is specified in model config"
            # cache_position for local layer
            swa_cache_position = swa_cache_index + self.cache_tensor

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.model(
            input_ids=input_ids,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            swa_position_ids=swa_position_ids,
            past_key_values=past_key_values,
            swa_attention_mask=swa_attention_mask,
            cache_index=cache_index,
            swa_cache_index=swa_cache_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            swa_cache_position=swa_cache_position,
            **lm_kwargs,
        )

        if not isinstance(outputs, tuple):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if (final_logit_softcapping := self.config.final_logit_softcapping) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Gemma4ForConditionalGeneration(Gemma4ForConditionalGeneration_original):

    def __init__(self, config):
        super(Gemma4ForConditionalGeneration, self).__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Optional[Cache] = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: torch.LongTensor | None = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = None,
        # QC adapted args
        swa_position_ids: Optional[torch.LongTensor] = None,
        swa_attention_mask: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.FloatTensor] = None,
        cache_index: Optional[torch.Tensor] = None,
        swa_cache_index: Optional[torch.Tensor] = None,
        image_pooling_idx: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        audio_subsample_state: Optional[list[torch.Tensor]] = None,
        audio_past_key_values: Optional[list[torch.Tensor]] = None,
        audio_lconv1d_state: Optional[list[torch.Tensor]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Gemma4CausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        logits_to_keep = logits_to_keep if logits_to_keep else getattr(self.config, "logits_to_keep", 0)
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        has_audio = input_features is not None

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            # QC adapted args
            per_layer_inputs=per_layer_inputs,
            swa_position_ids=swa_position_ids,
            swa_attention_mask=swa_attention_mask,
            cache_index=cache_index,
            swa_cache_index=swa_cache_index,
            image_pooling_idx=image_pooling_idx,
            audio_attention_mask=audio_attention_mask,
            audio_subsample_state=audio_subsample_state,
            audio_past_key_values=audio_past_key_values,
            audio_lconv1d_state=audio_lconv1d_state,
            return_dict=return_dict,
            **kwargs,
        )

        if isinstance(outputs, tuple):
            outputs = Gemma4ModelOutputWithPast.from_tuple(outputs, has_audio=has_audio)

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        predicted_tokens = logits.argmax(dim=-1)
        embedding = self.model.get_input_embeddings()(predicted_tokens)
        state = hidden_states
        assistant_logits, assistant_state, assistant_indices = self.assistant_model(
            embedding=embedding,
            state=state,
            past_kv_local=outputs.past_kv_local,
            past_kv_global=outputs.past_kv_global,
            position_embeddings=position_ids,
            swa_position_embeddings=swa_position_ids,
            attention_mask=attention_mask,
            swa_attention_mask=swa_attention_mask,
        )

        loss = None
        if not return_dict:
            output = (logits, outputs.last_hidden_state, outputs.past_key_values, assistant_logits, assistant_state, assistant_indices)
            if has_audio:
                output += (outputs.audio_subsample_state, outputs.audio_past_key_values, outputs.audio_lconv1d_state)
            return (loss,) + output if loss is not None else output


        return Gemma4CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
            assistant_logits=assistant_logits,
            assistant_state=assistant_state,
            assistant_indices=assistant_indices,
            audio_subsample_state=outputs.audio_subsample_state,
            audio_past_key_values=outputs.audio_past_key_values,
            audio_lconv1d_state=outputs.audio_lconv1d_state,
        )



class Gemma4VisionRotaryEmbedding(Gemma4VisionRotaryEmbedding_original):
    
    
    def calc_rotary_for_maxpos(self):
        inv_freq = self.inv_freq
        device = inv_freq.device

        # Use same maximum range of position_ids
        position_ids = torch.arange(0, self.config.max_position_embeddings, device=device).unsqueeze(0)


        inv_freq_expanded = self.inv_freq[:, None].float()
        dim_freqs = (inv_freq_expanded.float() @ position_ids.float()).transpose(0, 1)
        all_embs = {'cos': dim_freqs.cos() * self.attention_scaling, 'sin': dim_freqs.sin() * self.attention_scaling}

        return all_embs

    @torch.no_grad()
    def from_cached_embeddings(self, cached_embeddings: torch.Tensor, position_ids: torch.Tensor):
        position_ids = position_ids.clamp(min=0, max=self.config.max_position_embeddings - 1) # To account for padding positions which are set to -1
        return torch.nn.functional.embedding(position_ids, cached_embeddings.to(position_ids.device))

    @torch.no_grad()
    def forward(self, x, position_ids):

        all_embs = self.calc_rotary_for_maxpos()
        cos_parts = []
        sin_parts = []

        for d in range(2):
            pos = position_ids[:, :, d].clamp(min=0, max=self.config.max_position_embeddings - 1)
            cos_parts.append(self.cos_outs(torch.nn.functional.embedding(pos, all_embs['cos'].to(pos.device)).to(x.dtype)))
            sin_parts.append(self.sin_outs(torch.nn.functional.embedding(pos, all_embs['sin'].to(pos.device)).to(x.dtype)))

        return cos_parts, sin_parts

class Gemma4VisionMLP(Gemma4VisionMLP_original):
    """Vision MLP with erf-based GeLU replacing the default gelu_pytorch_tanh.

    MPP/HTP only supports erf-based GeLU; this adaptation makes QC match MPP behaviour
    instead of silently diverging at the QC vs MPP parity checkpoint.
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__(config)
        self.act_fn = Act2FN("gelu", craft_config=config.craft_config)


class Gemma4TextMLP(Gemma4TextMLP_original):
    """Text MLP with erf-based GeLU replacing the default gelu_pytorch_tanh.

    See Gemma4VisionMLP for rationale.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.act_fn = Act2FN("gelu", craft_config=config.craft_config)


class Gemma4VisionAttention(Gemma4VisionAttention_original):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.enable_masked_softmax = getattr(self.config, "enable_masked_softmax", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        n, h, w, _ = hidden_states.shape
        input_shape = hidden_states.shape[1:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        untiled_shape = (n, 1, h*w, -1)
        
        assert self.config.num_attention_heads == self.config.num_key_value_heads, "Current implementation only supports Q heads = KV heads"
        attn_output_all, attn_weights_all = [], []
        for head in range(self.config.num_attention_heads):
            query_states = getattr(self, f"q_proj_{head}")(hidden_states) #.view(hidden_shape)
            query_states = self.q_norm(query_states)
            # query_states = query_states.transpose(1, 2)
            query_states = _apply_rope_multidim(query_states, position_embeddings)

            key_states = getattr(self, f"k_proj_{head}")(hidden_states) #.view(hidden_shape)
            key_states = self.k_norm(key_states)
            # key_states = key_states.transpose(1, 2)
            key_states = _apply_rope_multidim(key_states, position_embeddings)
            key_states = key_states.reshape(untiled_shape)

            value_states = getattr(self, f"v_proj_{head}")(hidden_states) #.view(hidden_shape)
            value_states = self.v_norm(value_states)
            # value_states = value_states.transpose(1, 2)
            value_states = value_states.reshape(untiled_shape)

            if self.config.adaptations.vision_attention_forward:
                attention_interface: Callable = eager_attention_forward
            else:
                attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                    'sdpa', eager_attention_forward
                )
                attention_mask = attention_mask.bool() if attention_mask is not None else None

            attn_output_h, attn_weights_h = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output_all.append(attn_output_h.transpose(1, 2))
            attn_weights_all.append(attn_weights_h.reshape(1, 1, h*w, h*w))
        
        attn_output = torch.cat(attn_output_all, dim=3)
        attn_weights = torch.cat(attn_weights_all, dim=1)
        attn_output = attn_output.reshape(n, h, w, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma4VisionModel(Gemma4VisionModel_original):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__(config)
        self.tiling_size = config.tiling_size

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor,
        image_pooling_idx: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` or `list[torch.FloatTensor]`):
            The images to encode. Either a single `[batch, channels, height, width]` tensor
            (all images same size) or a list of `[1, channels, height, width]` tensors (different sizes).
        pixel_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        L = int(pixel_values.shape[1])
        assert L % self.tiling_size == 0, f"Pixel values shape {pixel_values.shape} at position 1 cannot be tiled using tile size {self.tiling_size}"
        pixel_values        = pixel_values.reshape(self.tiling_size, L // self.tiling_size, pixel_values.shape[2])
        image_position_ids  = image_position_ids.reshape(self.tiling_size, L // self.tiling_size, image_position_ids.shape[2])
        image_pooling_idx   = image_pooling_idx.reshape(self.tiling_size, L // self.tiling_size)

        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = L // (pooling_kernel_size * pooling_kernel_size)

        valid_positions = torch.where(image_position_ids[:, :, 0] >= 0, 1.0, 0.0).to(torch.float32)
        inputs_embeds = self.patch_embedder(pixel_values=pixel_values, pixel_position_ids=image_position_ids, padding_positions=valid_positions)
        N, H, C = inputs_embeds.shape
        inputs_embeds = inputs_embeds.reshape(1, N, H, C)
        _last_hidden_state, _hidden_states, _attentions = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=valid_positions,
            pixel_position_ids=image_position_ids,
            **kwargs,
        )
        _last_hidden_state  = _last_hidden_state.reshape(1, L, C)
        image_position_ids  = image_position_ids.reshape(1, L, image_position_ids.shape[2])
        image_pooling_idx   = image_pooling_idx.reshape(1, L)
        valid_positions     = valid_positions.reshape(1, L)

        hidden_states, pooler_mask = self.pooler(
            hidden_states=_last_hidden_state,
            pixel_position_ids=image_position_ids,
            padding_positions=valid_positions,
            image_pooling_idx=image_pooling_idx,
            output_length=output_length,
        )

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return hidden_states, _hidden_states, _attentions


class Gemma4VisionPatchEmbedder(Gemma4VisionPatchEmbedder_original):
    def split_pe_table(self):
        size = self.position_embedding_size
        self.pe_table_0 = nn.Embedding(size, self.hidden_size)
        self.pe_table_0.weight.data = self.position_embedding_table[0, :size].data
        self.pe_table_1 = nn.Embedding(size, self.hidden_size)
        self.pe_table_1.weight.data = self.position_embedding_table[1, :size].data

    def _position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
        """Prepare patch positions map for matmul with positon embedding table."""
        
        pixel_position_ids = pixel_position_ids.clamp(min=0, max=self.position_embedding_size - 1)
        pixel_position_ids_0 = pixel_position_ids[:, :, 0]
        pixel_position_ids_1 = pixel_position_ids[:, :, 1]
        position_embeddings = self.pe_table_0(pixel_position_ids_0) + self.pe_table_1(pixel_position_ids_1)
        position_embeddings = torch.where(padding_positions.unsqueeze(-1) > 0, position_embeddings, torch.tensor(0.0, dtype=position_embeddings.dtype, device=position_embeddings.device))
        return position_embeddings


class Gemma4VisionEncoder(Gemma4VisionEncoder_original):
    def forward(self, inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.LongTensor | None = None,
        **kwargs
        ):

        attention_mask = attention_mask.reshape(1, 1, 1, -1)  # [1, seq] -> [1, 1, 1, seq], broadcasts across query dim

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)

        # decoder layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                **kwargs,
            )

        return hidden_states, (), ()


class Gemma4VisionPooler(Gemma4VisionPooler_original):
    """Scaling and optional spatial pooling for vision encodings"""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__(config)
        self.pooling_kernel_size = config.pooling_kernel_size

    def _avg_pool_by_positions(
        self, hidden_states: torch.Tensor, image_pooling_idx: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        2D spatial pooling according to patch positions.
        Pools the input tokens by averaging patches within a `k^2` grid, where `k` is determined by the ratio between
        input and output lengths
        """
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        assert k == self.pooling_kernel_size, "Kernel pooling size doesn't match expected value"
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )

        # Can't perform 'index_select' on dim==1 because of MPP bug
        eye = torch.eye(length, dtype=torch.float32, device=hidden_states.device) / (self.pooling_kernel_size ** 2)
        weights = torch.index_select(eye, 0, image_pooling_idx.squeeze(0)).unsqueeze(0)
        weights = weights.permute(0,2,1)
        _weights = F.one_hot(image_pooling_idx.long(), length).float() / k_squared
        assert  torch.allclose(_weights.transpose(1, 2), weights), f"The adaptation for one-hot in VisionPooler is not correct: {torch.abs(weights - _weights.transpose(1, 2)).max()}"
        mask = (weights.sum(dim = 2) > 0)
        assert (torch.logical_not((_weights == 0).all(dim=1)) == mask).all(), "The adaptation for mask in VisionPooler is not correct"
        output = self.matmul(weights.to(hidden_states.dtype), hidden_states)

        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        image_pooling_idx: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]}). Change the value of `num_soft_tokens` when processing."
            )

        # Padded positions are not fed to LLM anyway, so pooling them wont make a difference.
        hidden_states = torch.where(padding_positions.unsqueeze(-1) > 0, hidden_states, 0.0)

        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, image_pooling_idx, output_length
            )

        hidden_states = self.mul(hidden_states, self.root_hidden_size)
        return hidden_states, padding_positions


def unfold_along_dim1(x, frame_len, frame_step):
    B, T, C, F = x.shape
    assert (B * T) % frame_step == 0, "The audio length must be divisible by frame_step"
    x = x.reshape((B * T)//frame_step, frame_step, C, F)

    assert frame_len // frame_step == 2, "frame_length should be twice frame_step for the unfold adaptation to be correct"
    x_unfolded = torch.cat([x[:-1], x[1:]], dim=1)

    return x_unfolded


class Gemma4AudioAttention(Gemma4AudioAttention_original):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.value_states_mul = Mul(config.craft_config)
        self.key_states_mul = Mul(config.craft_config)

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts overlapping context windows of `context_size` for every block, strided by `chunk_size`."""
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        # Unfold formula is floor((seq_length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        # We have kernel_size = self.context_size, stride=self.chunk_size, dilation=1, padding = 0
        # From the formula this reduces to floor((seq_length - self.context_size) / self.chunk_size + 1)
        # If y = floor((seq_length - self.context_size) / self.chunk_size + 1), then for for any x in [0, self.chunk_size),
        # we have floor((seq_length + x - self.context_size) / self.chunk_size + 1) = y
        # So we can ignore padding as long as padding length is below self.chunk_size
        padding_length = self.max_future_horizon + self.chunk_size - 1
        _hidden_states = F.pad(hidden_states, 
                               (0, 0, 0, 0, 0, padding_length)).unfold(1, self.context_size, self.chunk_size)
        _hidden_states = torch.movedim(_hidden_states, -1, 2)
        if padding_length >= self.chunk_size:
            hidden_states = F.pad(
                hidden_states, (0, 0, 0, 0, 0, padding_length)
            )

        hidden_states = unfold_along_dim1(hidden_states, self.context_size, self.chunk_size)

        assert torch.allclose(_hidden_states, hidden_states.unsqueeze(0), atol=1e-6), "Unfold adaptation in Gemma4AudioAttention is wrong"
        return hidden_states.contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative position shift for blocked attention. See appendix B of https://huggingface.co/papers/1901.02860."""
        num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(num_heads, num_blocks, block_size, context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
        state: dict[str, torch.Tensor] | None = None,
        is_streaming: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, _, seq_length, _ = hidden_states.shape
        assert batch_size == 1, "Only batch size 1 is supported for adaptations"

        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        per_dim_scale_sp = self.softplus_per_dim_scale(self.per_dim_scale)

        query_states = self.mul_query_states1(query_states, self.q_scale)
        query_states = self.mul_query_states2(query_states, per_dim_scale_sp)
        key_states = self.mul_key_states(key_states, self.k_scale)

        if state is None:
            key_states_cat = F.pad(key_states, (0, 0, 0, 0, self.max_past_horizon, 0))
            value_states_cat = F.pad(value_states, (0, 0, 0, 0, self.max_past_horizon, 0))
        else:
            prev_k = state["prev_k"]
            prev_v = state["prev_v"]
            key_states_cat = torch.cat([prev_k, key_states], dim=1)
            value_states_cat = torch.cat([prev_v, value_states], dim=1)

        query_states_blocked = self._convert_to_block(query_states, is_streaming=(is_streaming if is_streaming is not None else (state is not None)))
        num_blocks = query_states_blocked.shape[1]
        query_states_blocked = query_states_blocked.squeeze(0)

        # When num_blocks == 1 (e.g., during streaming inference with a single chunk),
        # we pad the context to context_size to prevent shape mismatches in attention
        # operations.
        if num_blocks == 1:
            pad_len = self.context_size - key_states_cat.shape[1]
            if pad_len > 0:
                key_states_blocked = F.pad(key_states_cat, (0, 0, 0, 0, 0, pad_len))
                value_states_blocked = F.pad(value_states_cat, (0, 0, 0, 0, 0, pad_len))
            else:
                key_states_blocked = key_states_cat
                value_states_blocked = value_states_cat
        else:
            key_states_blocked = self._extract_block_context(key_states_cat)
            value_states_blocked = self._extract_block_context(value_states_cat)
            key_states_blocked = key_states_blocked[-num_blocks:]
            value_states_blocked = value_states_blocked[-num_blocks:]
        # Dummy node for MatMul second input to be 8bit
        key_states_blocked = self.key_states_mul(key_states_blocked, 1 - 1e-6)
        value_states_blocked = self.value_states_mul(value_states_blocked, 1 - 1e-6)

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states_blocked.dtype)

        queries = query_states_blocked.permute(2, 0, 1, 3)
        matrix_ac = self.matmul_ac(queries, key_states_blocked.permute(2, 0, 3, 1))

        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = self.matmul_bd(queries_flat, relative_key_states.permute(1, 2, 0))
        matrix_bd = matrix_bd.reshape(self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = self.add(matrix_ac, matrix_bd)
        attn_weights = self.div(attn_weights, self.softcap)
        attn_weights = self.tanh(attn_weights)
        attn_weights = self.mul_logits(attn_weights, self.softcap)

        attn_weights = torch.where(attention_mask > 0, attn_weights, self.config.attention_invalid_logits_value)

        attn_weights = self.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states_blocked.dtype)
        attn_output = self.matmul_output(attn_weights, value_states_blocked.permute(2, 0, 1, 3))
        attn_output = attn_output.permute(1, 2, 0, 3).reshape(batch_size, 1, num_blocks * self.chunk_size, -1)

        attn_output = attn_output[:, :, :seq_length].contiguous()

        new_state = {
            "prev_k": key_states_cat[:, -self.max_past_horizon:, :, :].contiguous(),
            "prev_v": value_states_cat[:, -self.max_past_horizon:, :, :].contiguous(),
        }

        attn_output = self.post(attn_output.to(dtype=hidden_states.dtype))

        return attn_output, attn_weights, new_state

class Gemma4AudioFeedForward(Gemma4AudioFeedForward_original):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = self.post_layer_norm(hidden_states)

        return self.add(self.mul(hidden_states, self.post_layer_scale), residual)


class Gemma4AudioCausalConv1d(Conv1d):

    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding = self.left_pad
        batch_size, _one, channels, _ = x.shape
        if state is None:
            conv_padding = torch.zeros((batch_size, _one, channels, padding), dtype=x.dtype, device=x.device)
        else:
            conv_padding = state
            _, _, _, state_length = conv_padding.shape
            assert state_length == padding, f"State shape {conv_padding.shape} does not match padding {padding}"

        # x is [B, 1, C, T]
        x_cat = torch.cat([conv_padding, x], dim=3)

        out = super().forward(x_cat)

        new_state = x_cat[:, :, :, x_cat.shape[2] - padding:]

        return out, new_state

class Gemma4AudioLightConv1d(Gemma4AudioLightConv1d_original):
    def forward(self, hidden_states: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = self.glu(hidden_states, dim=-1)

        # Conv1d takes only 3D input, so have to resize. TODO: Check if we can use Conv2D instead
        hidden_states = hidden_states.squeeze(1)
        hidden_states, new_state = self.depthwise_conv1d(hidden_states.transpose(1, 2), state=state)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.unsqueeze(1)

        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        return self.add(residual, hidden_states), new_state


class Gemma4AudioLayer(Gemma4AudioLayer_original):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None,
        position_embeddings: torch.Tensor,
        state: tuple[dict[str, torch.Tensor], torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, tuple[dict[str, torch.Tensor], torch.Tensor]]:
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = self.norm_pre_attn(hidden_states)

        is_streaming = kwargs.get("is_streaming", None)
        hidden_states, _, new_attn_state = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            state=state[0] if state is not None else None,
            is_streaming=is_streaming,
        )

        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = self.add(hidden_states, residual)

        hidden_states, new_lconv_state = self.lconv1d(
            hidden_states,
            state=state[1] if state is not None else None,
        )
        hidden_states = self.feed_forward2(hidden_states)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states, (new_attn_state, new_lconv_state)


class Gemma4AudioSubSampleConvProjection(Gemma4AudioSubSampleConvProjection_original):
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        *,
        is_last: bool = False,
    ) -> Gemma4AudioSubSampleConvProjectionOutput:
        hidden_states = input_features
        state0, state1 = (None, None) if state is None else state

        hidden_states, mask, new_state0 = self.layer0(hidden_states, input_features_mask, state=state0, is_last=is_last)
        hidden_states, mask, new_state1 = self.layer1(hidden_states, mask, state=state1, is_last=is_last)

        batch_size, _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, 1, seq_len, -1)
        return Gemma4AudioSubSampleConvProjectionOutput(
            hidden_states=self.input_proj_linear(hidden_states),
            mask=mask,
            state=(new_state0, new_state1)
        )

class Gemma4AudioModel(Gemma4AudioModel_original):
    def __init__(self, config):
        import copy
        config = copy.deepcopy(config)
        super().__init__(config)

    def _forward_chunk(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: Sequence | None = None,
        is_last: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4AudioModelOutput:
        subsample_state = state[0] if state is not None else None
        hidden_states, output_mask, new_subsample_state = self.subsample_conv_projection(
            input_features, input_features_mask, state=subsample_state, is_last=is_last
        )
        position_embeddings = self.rel_pos_enc(hidden_states)

        new_layer_states = []
        layer_states = state[1:] if state is not None else None
        for i, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, new_layer_state = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                state=layer_states[i] if layer_states is not None else None,
                **kwargs,
            )
            new_layer_states.append(new_layer_state)

        hidden_states = self.output_proj(hidden_states)

        past_key_values = [new_subsample_state] + new_layer_states
        return Gemma4AudioModelOutput(last_hidden_state=hidden_states, attention_mask=output_mask, past_key_values=past_key_values)

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        streaming_chunk_size_tokens: int | None = None,
        state: tuple | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4AudioModelOutput:
        """Encodes audio features to soft tokens.

        This method supports two modes of operation:
        1. **One-shot mode** (Default): Set `streaming_chunk_size_tokens` to `None`. Processes the full sequence at once.
        2. **Streaming mode**: Set `streaming_chunk_size_tokens` to the desired output chunk size (e.g., 12). Pass the full sequence to `forward`. It will handle chunking and state management internally.

        Args:
            input_features (`torch.Tensor`):
                Input audio features of shape `(batch_size, seq_len, feature_dim)`.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask of shape `(batch_size, seq_len)`.
            streaming_chunk_size_tokens (`int`, *optional*):
                Chunk size for internal streaming.
            **kwargs:
                Additional keyword arguments.

        Returns:
            `Gemma4AudioModelOutput`: The model output containing the encoded audio features and attention mask.
        """
        # Always use One-shot mode but while accepting state input
        return self._forward_chunk(input_features, input_features_mask, attention_mask, state=state, is_last=False, **kwargs)


class Gemma4QuantizableEinsum(Gemma4QuantizableEinsum_original):
    
    def forward(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        einsum_str: str,
    ) -> torch.Tensor:
        if self.input_bits is not None:
            input = fake_quant(input, self.input_bits, self.input_scale)
        
        assert einsum_str == "blnd,bntd->blnt", f"Unexpected einsum string: {einsum_str}"
        # b, l, n, d = input.shape
        # _, _, t, _ = weights.shape
        input = input.permute(0, 2, 1, 3)
        weights = weights.permute(0, 1, 3, 2)
        out = torch.matmul(input, weights)
        out = out.permute(0, 2, 1, 3)

        if self.output_bits is not None:
            out = fake_quant(out, self.output_bits, self.output_scale)
        
        return out


class Gemma4QuantizableEmbedding(Gemma4QuantizableEmbedding_original):
    def __init__(
      self,
      config,
      num_embeddings: int,
      embedding_dim: int,
      padding_idx: int,
      embed_scale: float = 1.0,
      num_weight_scales: int = 1,
      device: torch.device | str | None = None,
      dtype: torch.dtype | None = None,
  ) -> None:
        super().__init__(config, num_embeddings, embedding_dim, padding_idx, 
                         embed_scale, num_weight_scales=num_weight_scales,
                         device=device, dtype=dtype)
        self.register_buffer("embed_scale", torch.tensor(self.embed_scale).reshape(1, 1, 1, 1), persistent=False)

    def dequantize(self):
        if self.weight_scale is not None:
            weight_scale = self.weight_scale
            stretch_factor = self.embedding_dim // weight_scale.shape[-1]
            if stretch_factor > 1:
                weight_scale = weight_scale.repeat_interleave(stretch_factor, dim=-1)
            self.weight = nn.Parameter(
                self.weight.to(dtype=weight_scale.dtype) * weight_scale,
            )
            self.weight_scale = None


class Gemma4RMSNorm(Gemma4RMSNorm_original):

    def _norm(self, x):
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x / torch.sqrt(mean_squared)


class Gemma4QuantizableLinear(Gemma4QuantizableLinear_original):
    def get_weight(self):
        if self.weight_scale is not None:
            return self.weight * self.weight_scale
        else:
            return self.weight

    def get_bias(self):
        return None



class DynamicCache_adapted(Cache):
    def __init__(self):
        super().__init__(
            layer_class_to_replicate=DynamicLayer_adapted,
            offloading=False,
            offload_only_non_sliding=False,
        )

class DynamicLayer_adapted(DynamicLayer):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache as:  https://github.com/huggingface/transformers/blob/d79b2d981f28b2730d402244ac3c2e9a8c054eee/src/transformers/cache_utils.py#L98
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            return self.keys , self.values
        else:
            return_new_key_value_only = cache_kwargs.get('return_new_key_value_only', False)
            transposed_key_cache = cache_kwargs.get('transposed_key_cache', False)
            cache_position = cache_kwargs.get('cache_position')
            num_key_value_heads = cache_kwargs.get('num_key_value_heads')
            head_dim = cache_kwargs.get('head_dim')
            key_cat_dim = -1 if transposed_key_cache else -2
            # if the size of past key cache passed is smaller in value than the last position where the new kv is to be inserted
            # [in case when Cache position determined automatically by HF] (Ctx_len+ARN), then we want to perform concat and not do scattering.
            if self.values.shape[-2] <= cache_position[-1]:
                key_cache = torch.cat([self.keys, key_states], dim=key_cat_dim)
                value_cache = torch.cat([self.values, value_states], dim=-2)
            else:
                # the cache_position passed in as model i/p by user is a 1d tensor reflecting the positions
                # from valid_kv_end to valid_kv_end+ARN, we convert this into the indices for scattering. [# bsz, num_key_value_heads, head_dim, seq_len]-> works for transposed keys
                indices = cache_position.view(1, 1, 1, -1).expand(value_states.shape[0], num_key_value_heads, head_dim, cache_position.shape[-1])
                value_cache = self.values.scatter(dim=-2, index=indices.transpose(-1,-2), src=value_states)

                indices = indices.transpose(-1, -2) if key_cat_dim == -2 else indices

                key_cache = self.keys.scatter(dim=key_cat_dim, index=indices, src=key_states)

            if return_new_key_value_only:
                self.keys = key_states
                self.values = value_states
            else:
                self.keys = key_cache
                self.values = value_cache
            return key_cache, value_cache

    def get_seq_length(self, cache_position=None) -> int:
        # NOTE: Starting from transformers>=4.54, DynamicCache::get_seq_length now internally uses DynamicLayer::get_seq_length, so we need to update this method accordingly
        # https://github.com/huggingface/transformers/blob/9c641dc16154964e5ffc0c13e9ec6aaffa295ed6/src/transformers/cache_utils.py#L1210
        if self.values is None or self.values.numel() == 0:
            return 0
        return self.values.shape[-2]


def DynamicLayer_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return DynamicLayer_adapted.update(self, key_states, value_states, cache_kwargs)


def DynamicLayer_get_seq_length(self, cache_position=None) -> int:
    return DynamicLayer_adapted.get_seq_length(self, cache_position)


def update_attr(cls, attr_name, new_attr):
    attr_backup_name = f'_original_{attr_name}'
    if hasattr(cls, attr_name):
        if not hasattr(cls, attr_backup_name):
            setattr(cls, attr_backup_name, getattr(cls, attr_name))
            setattr(cls, attr_name, new_attr)
        return True
    return False
