from ..auto import AutoModel
# common_typos_disable
# isort: skip_file
# pylint: disable=all
# pytype: skip-file
# refex: disable=pytotw.034
# fmt: off
import copy
import math
from collections.abc import Callable, Sequence

from ...generation.candidate_generator import AssistedCandidateGenerator
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import LogitsProcessorList
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_experts_implementation, use_kernelized_func
from ...masking_utils import (
    create_bidirectional_mask,
    create_causal_mask,
    create_masks_for_generate,
    create_sliding_window_causal_mask,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from .configuration_gemma4 import Gemma4AudioConfig, Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig, Gemma4AssistantConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...generation.utils import GenerationMixin
from .quantization_gemma4 import Gemma4QuantizableEmbedding, Gemma4QuantizableLinear, Gemma4QuantizableScaledLinear, fake_quant_activation, Gemma4QuantizableEinsum
from . import config as gemma4_config
from . import static_fake_quant


logger = logging.get_logger(__name__)


class CraftModule(nn.Module):
  """A base class for basic operations with CRAFT quantization."""

  def __init__(self, craft_config: gemma4_config.CraftConfig | None = None):
    super().__init__()
    self.craft_config = craft_config
    self.output_sfq = static_fake_quant.StaticFakeQuant(
        craft_config=craft_config
    )

  def apply_output_sfq(self, x: torch.Tensor) -> torch.Tensor:
    return self.output_sfq(x)

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return self.apply_output_sfq(x)


class Matmul(CraftModule):
  """Matmul with CRAFT."""

  def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(x, w)
    return self.apply_output_sfq(logits)


class Div(CraftModule):
  """Divide with CRAFT."""

  def forward(self, x: torch.Tensor, v: float) -> torch.Tensor:
    result = x / v
    return self.apply_output_sfq(result)


class Concat(CraftModule):
  """Concatenate with CRAFT."""

  def forward(self, x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    result = torch.concat((x, y), dim=dim)
    return self.apply_output_sfq(result)


class Pow(CraftModule):
  """Power with CRAFT."""

  def forward(self, x: torch.Tensor, v: float) -> torch.Tensor:
    result = torch.pow(x, v)
    return self.apply_output_sfq(result)


class Sqrt(CraftModule):
  """Sqrt with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = torch.sqrt(x)
    return self.apply_output_sfq(result)


class Mul(CraftModule):
  """Multiply with CRAFT."""

  def forward(self, x: Any, v: Any) -> torch.Tensor:
    result = x * v
    return self.apply_output_sfq(result)


class Add(CraftModule):
  """Add with CRAFT."""

  def forward(self, x: Any, v: Any) -> torch.Tensor:
    result = x + v
    return self.apply_output_sfq(result)


class Sub(CraftModule):
  """Subtract with CRAFT."""

  def forward(self, x: Any, v: Any) -> torch.Tensor:
    result = x - v
    return self.apply_output_sfq(result)


class Neg(CraftModule):
  """Negate with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = x.neg()
    return self.apply_output_sfq(result)


class Sin(CraftModule):
  """Sin with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = torch.sin(x)
    return self.apply_output_sfq(result)


class Cos(CraftModule):
  """Cos with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = torch.cos(x)
    return self.apply_output_sfq(result)


class Tanh(CraftModule):
  """Tanh with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = torch.tanh(x)
    return self.apply_output_sfq(result)


class BMM(CraftModule):
  """BMM with CRAFT."""

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.bmm(x, y)
    return self.apply_output_sfq(result)


class Softplus(CraftModule):
  """Softplus with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = torch.nn.functional.softplus(x)
    return self.apply_output_sfq(result)


class Softmax(CraftModule):
  """Softmax with CRAFT."""

  def forward(self, x: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
    result = torch.nn.functional.softmax(x, dim=dim, dtype=dtype)
    return self.apply_output_sfq(result)


class ReLU(CraftModule):
  """ReLU with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = torch.relu(x)
    return self.apply_output_sfq(result)


class GLU(CraftModule):
  """GLU with CRAFT."""

  def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
    result = torch.nn.functional.glu(x, dim=dim)
    return self.apply_output_sfq(result)


class SiLU(CraftModule):
  """SiLU with CRAFT."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = torch.nn.functional.silu(x)
    return self.apply_output_sfq(result)


class Linear(nn.Linear):
  """Linear with CRAFT."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      bias: bool = True,
      craft_config: gemma4_config.CraftConfig | None = None
    ):
    super().__init__(in_features, out_features, bias)
    self.craft_config = craft_config
    self.output_sfq = static_fake_quant.StaticFakeQuant(
        craft_config=craft_config
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = super().forward(x)
    return self.output_sfq(result)


class Conv1d(nn.Conv1d):
  """Conv1d with CRAFT."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int = 1,
      padding: int = 0,
      groups: int = 1,
      bias: bool = True,
      craft_config: gemma4_config.CraftConfig | None = None,
  ):
    super().__init__(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
    )
    self.craft_config = craft_config
    self.output_sfq = static_fake_quant.StaticFakeQuant(
        craft_config=craft_config
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = super().forward(x)
    return self.output_sfq(result)



class Conv2d(nn.Conv2d):
  """Conv2d with CRAFT."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: tuple[int, int],
      stride: tuple[int, int],
      padding: tuple[int, int],
      bias: bool = True,
      craft_config: gemma4_config.CraftConfig | None = None,
  ):
    super().__init__(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    self.craft_config = craft_config
    self.output_sfq = static_fake_quant.StaticFakeQuant(
        craft_config=craft_config
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = super().forward(x)
    return self.output_sfq(result)


class Act2FN(CraftModule):
  """Act2FN with CRAFT."""

  def __init__(self, act_fn_name: str, craft_config: gemma4_config.CraftConfig | None = None):
    super().__init__(craft_config=craft_config)
    self.act_fn = ACT2FN[act_fn_name]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = self.act_fn(x)
    return self.apply_output_sfq(result)


class RotaryEmbeddingOperator(nn.Module):
    """Rotary embedding operator with CRAFT."""

    def __init__(self, craft_config: gemma4_config.CraftConfig | None = None):
        super().__init__()
        self.craft_config = craft_config
        self.cos_mul = Mul(craft_config=craft_config)
        self.sin_mul = Mul(craft_config=craft_config)
        self.add = Add(craft_config=craft_config)
        self.cat = Concat(craft_config=craft_config)
        self.neg = Neg(craft_config=craft_config)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        neg_x2 = self.neg(x2)
        return self.cat(neg_x2, x1, dim=-1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_rot = self.cos_mul(x, cos)
        y_rot = self.sin_mul(self.rotate_half(x), sin)
        return self.add(x_rot, y_rot)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Gemma4 outputs, with hidden states and attentions.
    """
)
class Gemma4ModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None

    audio_hidden_states: torch.FloatTensor | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Gemma4 causal language model (or autoregressive) outputs.
    """
)
class Gemma4CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None

    audio_hidden_states: torch.FloatTensor | None = None


@dataclass
@auto_docstring
class Gemma4AudioModelOutput(BaseModelOutputWithPooling):
    r"""
    Attributes:
        attention_mask (`torch.BoolTensor`, *optional*):
            A torch.BoolTensor of shape `(batch_size, num_frames)`. True for valid positions, False for padding.
        past_key_values (`list[tuple[dict[str, torch.Tensor], torch.Tensor]]`, *optional*):
            List containing the state from the subsampling layer and the KV cache states from each attention layer.
    """

    attention_mask: torch.BoolTensor | None = None
    past_key_values: list[tuple[dict[str, torch.Tensor], torch.Tensor]] | None = None


class Gemma4ClippableLinear(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig | Gemma4AudioConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)

        hidden_states = self.linear(hidden_states)

        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)

        return hidden_states


class Gemma4RMSNorm(CraftModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        with_scale: bool = True,
        craft_config: gemma4_config.CraftConfig | None = None):
        super().__init__(craft_config=craft_config)
        # eps is not exactly 1e-6 in bfloat16
        self.eps_val = eps
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32), persistent=False)
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to addess compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return self.apply_output_sfq(normed_output.type_as(hidden_states))


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding for the audio encoder.

    Produces position embeddings of shape [1, 2*context_size - 1, hidden_size] with
    concatenated [sin..., cos...] layout matching the original Gemma4 convention.
    """

    inv_timescales: torch.Tensor

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
        self.register_buffer("inv_timescales", inv_timescales.unsqueeze(0).unsqueeze(0), persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(12, -1, -1, device=hidden_states.device)
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


class Gemma4AudioAttention(nn.Module):
    """Chunked local attention with relative position bias"""

    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = 1.0

        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.q_proj = Gemma4QuantizableLinear(config, config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Gemma4QuantizableLinear(config, config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = Gemma4QuantizableLinear(config, config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.post = Gemma4QuantizableLinear(config, config.hidden_size, config.hidden_size, bias=False)

        self.relative_k_proj = Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False, craft_config=config.craft_config)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.add = Add(config.craft_config)
        self.div = Div(config.craft_config)
        self.mul_logits = Mul(config.craft_config)
        self.tanh = Tanh(config.craft_config)
        self.softplus_per_dim_scale = Softplus(config.craft_config)
        self.softmax = Softmax(config.craft_config)
        self.mul_query_states1 = Mul(config.craft_config)
        self.mul_query_states2 = Mul(config.craft_config)
        self.mul_key_states = Mul(config.craft_config)
        self.matmul_ac = Matmul(config.craft_config)
        self.matmul_bd = Matmul(config.craft_config)
        self.matmul_output = Matmul(config.craft_config)

        self.register_buffer("softcap", torch.tensor(self.attention_logits_soft_cap), persistent=False)

    def _convert_to_block(self, hidden_states: torch.Tensor, is_streaming: bool = False) -> torch.Tensor:
        """Splits a `(batch_size, seq_len, num_heads, head_dim)` tensor into non-overlapping blocks of `chunk_size` along the sequence dim.

        In streaming mode, sequence length must be a multiple of chunk size.

        Args:
            hidden_states: Input tensor.
            is_streaming: Whether we are in streaming mode.
        """
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        if is_streaming and pad != 0:
            raise ValueError(f"Sequence length {seq_len} must be a multiple of chunk size {self.chunk_size} in streaming mode")
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim).contiguous()

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts overlapping context windows of `context_size` for every block, strided by `chunk_size`."""
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        hidden_states = F.pad(
            hidden_states, (0, 0, 0, 0, 0, self.max_future_horizon + self.chunk_size - 1)
        )
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        hidden_states = torch.movedim(hidden_states, -1, 2)
        return hidden_states.contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative position shift for blocked attention. See appendix B of https://huggingface.co/papers/1901.02860."""
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
        state: dict[str, torch.Tensor] | None = None,
        is_streaming: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
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

        # When num_blocks == 1 (e.g., during streaming inference with a single chunk),
        # we pad the context to context_size to prevent shape mismatches in attention
        # operations.
        if num_blocks == 1:
            pad_len = self.context_size - key_states_cat.shape[1]
            if pad_len > 0:
                padded_k = F.pad(key_states_cat, (0, 0, 0, 0, 0, pad_len))
                padded_v = F.pad(value_states_cat, (0, 0, 0, 0, 0, pad_len))
            else:
                padded_k = key_states_cat
                padded_v = value_states_cat
            key_states_blocked = padded_k.unsqueeze(1)
            value_states_blocked = padded_v.unsqueeze(1)
        else:
            key_states_blocked = self._extract_block_context(key_states_cat)
            value_states_blocked = self._extract_block_context(value_states_cat)
            key_states_blocked = key_states_blocked[:, -num_blocks:]
            value_states_blocked = value_states_blocked[:, -num_blocks:]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states_blocked.dtype)

        queries = query_states_blocked.permute(0, 3, 1, 2, 4)
        matrix_ac = self.matmul_ac(queries, key_states_blocked.permute(0, 3, 1, 4, 2))

        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = self.matmul_bd(queries_flat, relative_key_states.permute(1, 2, 0))
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = self.add(matrix_ac, matrix_bd)
        attn_weights = self.div(attn_weights, self.softcap)
        attn_weights = self.tanh(attn_weights)
        attn_weights = self.mul_logits(attn_weights, self.softcap)

        # Form a sliding window mask to limit attention to past and future horizons.
        # q_indices are indices within the current query chunk [0, chunk_size-1].
        # o_indices are indices within the context window [0, context_size-1].
        q_indices = torch.arange(self.chunk_size, device=queries.device).view(1, 1, 1, self.chunk_size, 1)
        o_indices = torch.arange(self.context_size, device=queries.device).view(1, 1, 1, 1, self.context_size)
        sliding_window_mask = (o_indices >= q_indices) & (o_indices <= q_indices + self.max_past_horizon + self.max_future_horizon)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                combined_mask = attention_mask & sliding_window_mask
            else:
                # Assume float mask where 0.0 is valid and -inf is invalid
                combined_mask = torch.isclose(attention_mask, torch.tensor(0.0, device=attention_mask.device), atol=1e-7) & sliding_window_mask
        else:
            combined_mask = sliding_window_mask

        attn_weights = attn_weights.masked_fill(
            ~combined_mask, self.config.attention_invalid_logits_value
        )

        attn_weights = self.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states_blocked.dtype)
        attn_output = self.matmul_output(attn_weights, value_states_blocked.permute(0, 3, 1, 2, 4))
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)

        attn_output = attn_output[:, :seq_length].contiguous()

        new_state = {
            "prev_k": key_states_cat[:, -self.max_past_horizon:, :, :].contiguous(),
            "prev_v": value_states_cat[:, -self.max_past_horizon:, :, :].contiguous(),
        }

        attn_output = self.post(attn_output.to(dtype=hidden_states.dtype))

        return attn_output, attn_weights, new_state


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, in_channels: int, out_channels: int, norm_eps: float):
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(0, 1),  # Pad freq dim, handle time dim manually
            bias=False,
            craft_config=config.craft_config,
        )
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, elementwise_affine=True, bias=False)
        self.act = ReLU(craft_config=config.craft_config)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None, state: torch.Tensor | None = None, *, is_last: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Forward pass for the subsampling convolution layer.

        This method supports streaming by accepting and returning state for carry-over
        between chunks. It handles manual padding on the time dimension to ensure
        causality and parity with non-streaming execution.

        Args:
            hidden_states: Input tensor of shape (batch, channels, time, freq).
            mask: Optional attention mask tensor.
            state: Optional state tensor from the previous chunk for carry-over.
                Shape should be (batch, channels, state_len, freq).
            is_last: Whether this is the last chunk in a streaming sequence.
                If True, adds right-side padding to match non-streaming behavior.

        Returns:
            A tuple containing:
                - hidden_states: Output tensor after convolution, normalization, and activation.
                - mask: Subsampled and sliced mask to match the output features length.
                - new_state: State tensor to be passed to the next chunk.
        """
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]

        # hidden_states shape: (batch, channels, time, freq)
        if state is not None:
            left_tensor = state
        else:
            # First chunk, pad 1 on left (time) with zeros
            left_tensor = torch.zeros(
                (hidden_states.shape[0], hidden_states.shape[1], 1, hidden_states.shape[3]),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )

        tensors_to_cat = [left_tensor, hidden_states]

        if is_last:
            # Pad 1 on right (time) with zeros
            pad_right = torch.zeros(
                (hidden_states.shape[0], hidden_states.shape[1], 1, hidden_states.shape[3]),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            tensors_to_cat.append(pad_right)

        combined_input = torch.cat(tensors_to_cat, dim=2)

        conv_out = self.conv(combined_input.to(self.conv.weight.dtype))
        # Calculate new state
        outputs = conv_out.shape[2]
        keep_from = outputs * 2  # stride is 2
        new_state = combined_input[:, :, keep_from:, :]

        hidden_states = self.act(self.norm(conv_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())

        if mask is not None:
            mask = mask[:, ::2][:, :hidden_states.shape[2]]

        return hidden_states, mask, new_state


@dataclass
class Gemma4AudioSubSampleConvProjectionOutput:
    """Output type for Gemma4AudioSubSampleConvProjection.

    Attributes:
        hidden_states: Projected hidden states of shape (batch, seq_len, hidden_size).
        mask: Subsampled mask of shape (batch, seq_len).
        state: Tuple of states from layer0 and layer1 for carry-over.
    """
    hidden_states: torch.Tensor
    mask: torch.Tensor | None
    state: tuple[torch.Tensor | None, torch.Tensor | None]

    def __iter__(self):
        return iter((self.hidden_states, self.mask, self.state))


class Gemma4AudioSubSampleConvProjection(CraftModule):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            config=config,
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            config=config,
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
        )
        proj_input_dim = (config.subsampling_conv_channels[0] // 4) * config.subsampling_conv_channels[1]
        self.input_proj_linear = Linear(proj_input_dim, config.hidden_size, bias=False, craft_config=config.craft_config)

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        state: tuple[torch.Tensor | None, torch.Tensor | None] | None = None,
        *,
        is_last: bool = False,
    ) -> Gemma4AudioSubSampleConvProjectionOutput:
        hidden_states = input_features.unsqueeze(1)
        state0, state1 = (None, None) if state is None else state
        hidden_states, mask, new_state0 = self.layer0(hidden_states, input_features_mask, state=state0, is_last=is_last)
        hidden_states, mask, new_state1 = self.layer1(hidden_states, mask, state=state1, is_last=is_last)

        batch_size, _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        return Gemma4AudioSubSampleConvProjectionOutput(
            hidden_states=self.input_proj_linear(hidden_states),
            mask=mask,
            state=(new_state0, new_state1)
        )


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_1 = Gemma4QuantizableLinear(config, config.hidden_size, config.hidden_size * 4, bias=False)
        self.ffw_layer_2 = Gemma4QuantizableLinear(config, config.hidden_size * 4, config.hidden_size, bias=False)

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, craft_config=config.craft_config)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size, craft_config=config.craft_config)
        self.act_fn = Act2FN(config.hidden_act, craft_config=config.craft_config)

        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

        self.mul = Mul(config.craft_config)
        self.add = Add(config.craft_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # This is needed to avoid any underflow/overflow issues when clipping
        gradient_clipping = min(self.gradient_clipping, torch.finfo(hidden_states.dtype).max)

        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)

        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
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
        batch_size, channels, _ = x.shape
        if state is None:
            conv_padding = torch.zeros((batch_size, channels, padding), dtype=x.dtype, device=x.device)
        else:
            conv_padding = state
            _, _, state_length = conv_padding.shape
            assert state_length == padding, f"State shape {conv_padding.shape} does not match padding {padding}"

        # x is [B, C, T]
        x_cat = torch.cat([conv_padding, x], dim=2)

        out = super().forward(x_cat)

        new_state = x_cat[:, :, x_cat.shape[2] - padding:]

        return out, new_state


class Gemma4AudioLightConv1d(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

        self.linear_start = Gemma4QuantizableLinear(config, config.hidden_size, config.hidden_size * 2, bias=False)
        self.linear_end = Gemma4QuantizableLinear(config, config.hidden_size, config.hidden_size, bias=False)
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
            craft_config=config.craft_config,
        )

        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True, craft_config=config.craft_config)
        self.conv_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True, craft_config=config.craft_config)
        self.act_fn = Act2FN(config.hidden_act, craft_config=config.craft_config)

        self.gradient_clipping = config.gradient_clipping
        self.glu = GLU(config.craft_config)
        self.add = Add(config.craft_config)

    def forward(self, hidden_states: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = self.glu(hidden_states, dim=-1)

        hidden_states, new_state = self.depthwise_conv1d(hidden_states.transpose(1, 2), state=state)
        hidden_states = hidden_states.transpose(1, 2)

        # This is needed to avoid any underflow/overflow issues when clipping
        gradient_clipping = min(self.gradient_clipping, torch.finfo(hidden_states.dtype).max)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        return self.add(residual, hidden_states), new_state


class Gemma4AudioLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.config = config

        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config, layer_idx)
        self.lconv1d = Gemma4AudioLightConv1d(config)

        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size, craft_config=config.craft_config)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size, craft_config=config.craft_config)
        self.norm_out = Gemma4RMSNorm(config.hidden_size, craft_config=config.craft_config)

        self.gradient_clipping = config.gradient_clipping
        self.add = Add(config.craft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None,
        position_embeddings: torch.Tensor,
        state: tuple[dict[str, torch.Tensor], torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, tuple[dict[str, torch.Tensor], torch.Tensor]]:
        # This is needed to avoid any underflow/overflow issues when clipping
        gradient_clipping = min(self.gradient_clipping, torch.finfo(hidden_states.dtype).max)

        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_pre_attn(hidden_states)

        is_streaming = kwargs.get("is_streaming", None)
        hidden_states, _, new_attn_state = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            state=state[0] if state is not None else None,
            is_streaming=is_streaming,
        )

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = self.add(hidden_states, residual)

        hidden_states, new_lconv_state = self.lconv1d(
            hidden_states,
            state=state[1] if state is not None else None,
        )
        hidden_states = self.feed_forward2(hidden_states)

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_out(hidden_states)

        return hidden_states, (new_attn_state, new_lconv_state)


# ---- Vision Encoder Layers ----


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = Linear(3 * self.patch_size**2, self.hidden_size, bias=False, craft_config=config.craft_config)
        self.position_embedding_table = nn.Parameter(torch.ones(2, self.position_embedding_size, self.hidden_size))
        self.add = Add(config.craft_config)
        self.pixel_sfq = CraftModule(config.craft_config)

    def _position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
        """Prepare patch positions map for matmul with positon embedding table."""
        # Expanding and permute patch positions to (batch_size, num_patches, 2, position_embedding_size) for matmul.
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        # Compute positional embeddings and sum across x and y.
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        # Zero out embeddings for any padding patches.
        position_embeddings = torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)
        return position_embeddings

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        # Gemma4 applies no normalization and instead scales in model code
        pixel_values = self.pixel_sfq(2 * (pixel_values - 0.5))
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        position_embeddings = self._position_embeddings(pixel_position_ids, padding_positions)
        return self.add(hidden_states, position_embeddings)


class Gemma4VisionPooler(nn.Module):
    """Scaling and optional spatial pooling for vision encodings"""

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5
        self.mul = Mul(config.craft_config)
        self.matmul = Matmul(config.craft_config)
        self.patch_to_pool_weights_in = CraftModule(config.craft_config)

    def _avg_pool_by_positions(
        self, hidden_states: torch.Tensor, pixel_position_ids: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        2D spatial pooling according to patch positions.
        Pools the input tokens by averaging patches within a `k^2` grid, where `k` is determined by the ratio between
        input and output lengths
        """
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )

        # Clamp padding positions (which are -1) to 0 so they don't break one_hot.
        # Padding patches have zero hidden states so they contribute nothing to the average.
        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        weights = self.patch_to_pool_weights_in(weights)
        output = self.matmul(weights.transpose(1, 2).to(hidden_states.dtype), hidden_states)
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]}). Change the value of `num_soft_tokens` when processing."
            )

        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )

        hidden_states = self.mul(hidden_states, self.root_hidden_size)
        return hidden_states, padding_positions


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Gemma4QuantizableLinear(config, self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Gemma4QuantizableLinear(config, self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Gemma4QuantizableLinear(config, self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = Act2FN(config.hidden_activation, craft_config=config.craft_config)
        self.mul = Mul(config.craft_config)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fused = self.mul(gate, up)
        down_proj = self.down_proj(fused)
        return down_proj


class Gemma4VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma4VisionConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        self.cos_outs = CraftModule(config.craft_config)
        self.sin_outs = CraftModule(config.craft_config)

    @staticmethod
    def compute_default_rope_parameters(
        config: Gemma4VisionConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        # The reference implementation computes RoPE frequencies INDEPENDENTLY
        # for each spatial dimension using the partitioned head_dim (head_dim // ndim),
        # so both x and y dimensions get identical frequency ranges.
        # This is different from splitting the global inv_freq between dimensions.
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / spatial_dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"

        # Multidimensional positions: [batch, num_patches, ndim]. Apply rotations to each spatial dim separately
        all_cos, all_sin = [], []
        for i in range(2):
            dim_position_ids = position_ids[:, :, i]
            dim_position_ids_expanded = dim_position_ids[:, None, :].float()

            with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
                freqs = (inv_freq_expanded.float() @ dim_position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            all_cos.append(cos)
            all_sin.append(sin)

        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return self.cos_outs(cos), self.sin_outs(sin)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1, rope_operator: RotaryEmbeddingOperator | None = None) -> torch.Tensor:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if rope_operator is not None:
        return rope_operator(x, cos, sin)
    else:
        return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: "Gemma4VisionAttention | Gemma4TextAttention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = module.attn_mul(module.attn_matmul_qk(query, key_states.transpose(2, 3)), scaling)

    if softcap is not None:
        attn_weights = module.softcap_div(attn_weights, softcap)
        attn_weights = module.softcap_tanh(attn_weights)
        attn_weights = module.softcap_mul(attn_weights, softcap)
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            attention_mask = torch.isclose(attention_mask, torch.tensor(0.0, device=attention_mask.device), atol=1e-7)
        attention_mask = (~attention_mask) * module.config.attention_invalid_logits_value
        attn_weights = module.attn_add(attn_weights, attention_mask)

    # upcast attention to fp32
    attn_weights = module.attn_softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = module.attn_matmul_av(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
    rope_operators: Sequence[RotaryEmbeddingOperator] | None = None,
) -> torch.Tensor:
    """Applies multidimensional RoPE to inputs.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            If position_ids.ndim + 2 == x.ndim, then this function passes through to `apply_rotary_pos_emb()`.
            Otherwise, position_ids is used to split the inputs, x, into multiple pieces, where each piece is fed to
            `apply_rotary_pos_emb()`, and then concatenated back together.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
      Tensor of shape [B, L, N, H] with RoPE applied.
    """
    ndim = position_ids.shape[-1]
    num_input_channels = x.shape[-1]
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

    if num_rotated_channels_per_dim <= 0:
        raise ValueError(
            "Invalid configuration: num_rotated_channels_per_dim must be > 0, got"
            f" {num_rotated_channels_per_dim} (num_input_channels={num_input_channels},"
            f" ndim={ndim})"
        )

    # Correctly split the input tensor into ndim parts
    split_sizes = [num_rotated_channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        apply_rotary_pos_emb(
            x=x_parts[k],
            cos=cos_parts[k],
            sin=sin_parts[k],
            unsqueeze_dim=unsqueeze_dim,
            rope_operator=rope_operators[k] if rope_operators is not None else None,
        )
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


@use_kernelized_func(apply_rotary_pos_emb)
class Gemma4VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = False
        self.q_proj = Gemma4QuantizableLinear(config, config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = Gemma4QuantizableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = Gemma4QuantizableLinear(config, config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = Gemma4QuantizableLinear(config, config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.k_norm = Gemma4RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False, craft_config=config.craft_config)

        ### CRAFT for attention
        self.attn_matmul_qk = Matmul(config.craft_config)
        self.attn_mul = Mul(config.craft_config)
        self.attn_add = Add(config.craft_config)
        self.attn_softmax = Softmax(config.craft_config)
        self.attn_matmul_av = Matmul(config.craft_config)

        self.softcap_div = Div(config.craft_config)
        self.softcap_tanh = Tanh(config.craft_config)
        self.softcap_mul = Mul(config.craft_config)

        self.q_rope_operator1 = RotaryEmbeddingOperator(craft_config=config.craft_config)
        self.q_rope_operator2 = RotaryEmbeddingOperator(craft_config=config.craft_config)
        self.k_rope_operator1 = RotaryEmbeddingOperator(craft_config=config.craft_config)
        self.k_rope_operator2 = RotaryEmbeddingOperator(craft_config=config.craft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        q_rope_operators = [self.q_rope_operator1, self.q_rope_operator2]
        k_rope_operators = [self.k_rope_operator1, self.k_rope_operator2]

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_multidimensional_rope(query_states, cos, sin, position_ids, rope_operators=q_rope_operators)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        key_states = apply_multidimensional_rope(key_states, cos, sin, position_ids, rope_operators=k_rope_operators)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma4VisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4VisionAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.add1 = Add(config.craft_config)
        self.add2 = Add(config.craft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.add1(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.add2(residual, hidden_states)

        return hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config=config, layer_idx=i) for i in range(self.num_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        pixel_position_ids (torch.Tensor):
            Patch positions as (x, y) coordinates in the image as [batch, num_patches, 2].
        """
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

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

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = Gemma4QuantizableLinear(config, self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Gemma4QuantizableLinear(config, self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Gemma4QuantizableLinear(config, self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = Act2FN(config.hidden_activation, craft_config=config.craft_config)
        self.mul = Mul(config.craft_config)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fused = self.mul(gate, up)
        down_proj = self.down_proj(fused)
        return down_proj


class Gemma4TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma4TextConfig, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            if (rope_type := rope_params["rope_type"]) != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self.compute_default_rope_parameters

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            rope_init_fn_kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Gemma4TextConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        return cos, sin


@use_kernelized_func(apply_rotary_pos_emb)
class Gemma4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    k_cache_scale: torch.Tensor | None
    k_cache_num_bits: torch.Tensor | None
    v_cache_scale: torch.Tensor | None
    v_cache_num_bits: torch.Tensor | None

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_key_value_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # Shared kv cache
        first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # For shared layers, find the last non-shared layer of the same type before sharing starts
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # For non-shared layers, store full-length kv if this is the last non-shared layer of its type
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(
                config.layer_types[layer_idx]
            )

        self.q_proj = Gemma4QuantizableLinear(
            config, config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, craft_config=config.craft_config)

        # Layers sharing kv states don't need any weight matrices
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, craft_config=config.craft_config)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False, craft_config=config.craft_config)

            self.k_proj = Gemma4QuantizableLinear(
                config, config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                None
                if self.use_alternative_attention
                else Gemma4QuantizableLinear(
                    config, config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias
                )
            )

        self.o_proj = Gemma4QuantizableLinear(
            config, config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        ### CRAFT for attention
        self.attn_matmul_qk = Matmul(config.craft_config)
        self.attn_mul = Mul(config.craft_config)
        self.attn_add = Add(config.craft_config)
        self.attn_softmax = Softmax(config.craft_config)
        self.attn_matmul_av = Matmul(config.craft_config)

        self.softcap_div = Div(config.craft_config)
        self.softcap_tanh = Tanh(config.craft_config)
        self.softcap_mul = Mul(config.craft_config)


        self.q_rope_operator = RotaryEmbeddingOperator(craft_config=config.craft_config)
        self.k_rope_operator = RotaryEmbeddingOperator(craft_config=config.craft_config)

        use_quantized_kv_cache = getattr(config, "use_quantized_kv_cache", False)
        if use_quantized_kv_cache:
            self.register_buffer("k_cache_scale", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("v_cache_scale", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("k_cache_num_bits", torch.tensor(0, dtype=torch.int32))
            self.register_buffer("v_cache_num_bits", torch.tensor(0, dtype=torch.int32))
        else:
            self.k_cache_scale = None
            self.v_cache_scale = None
            self.k_cache_num_bits = None
            self.v_cache_num_bits = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2, rope_operator=self.q_rope_operator)
        query_states = query_states.transpose(1, 2)

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
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2, rope_operator=self.k_rope_operator)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

            if self.k_cache_scale is not None and self.k_cache_num_bits is not None:
                key_states = fake_quant_activation(
                    key_states, self.k_cache_scale, int(self.k_cache_num_bits.item())
                )
            if self.v_cache_scale is not None and self.v_cache_num_bits is not None:
                value_states = fake_quant_activation(
                    value_states, self.v_cache_scale, int(self.v_cache_num_bits.item())
                )

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma4MtpCrossAttention(Gemma4TextAttention):
    """Read-only cross-attention for MTP drafter layers."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        # 1. Project Query using Drafter weights
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2, rope_operator=self.q_rope_operator)
        query_states = query_states.transpose(1, 2)

        # 2. Read-only extraction from Backbone Cache
        if past_key_values is not None:
            last_sliding_idx = getattr(self.config, "last_backbone_sliding_idx", None)
            last_full_idx = getattr(self.config, "last_backbone_full_idx", None)

            if self.is_sliding:
                target_layer_idx = last_sliding_idx
            else:
                target_layer_idx = last_full_idx

            if target_layer_idx is None:
                raise ValueError(f"Target backbone layer index not set in config for layer {self.layer_idx}")

            if hasattr(past_key_values, "layers"):
                key_states = past_key_values.layers[target_layer_idx].keys
                value_states = past_key_values.layers[target_layer_idx].values
            else:
                key_states, value_states = past_key_values[target_layer_idx]

            # Device placement
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)

            # Note: We assume the cache already has RoPE applied if needed,
            # and we do not update the cache.

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_experts_implementation
class Gemma4TextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = Act2FN(config.hidden_activation, self.config.craft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Gemma4TextRouter(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5
        self.eps = config.rms_norm_eps

        self.norm = Gemma4RMSNorm(self.hidden_size, eps=self.eps, with_scale=False, craft_config=config.craft_config)
        self.proj = Gemma4QuantizableLinear(config, self.hidden_size, config.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)  # [B*S, E]
        router_probabilities = nn.functional.softmax(expert_scores, dim=-1)

        # topk returns both values (probabilities) and indices directly
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # both [B*S, K]

        # Normalize the top-k weights so they sum to 1 per token
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Apply per-expert scale directly to the weights
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class Gemma4TextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma4TextConfig | Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.add1 = Add(self.config.craft_config)
        self.add2 = Add(self.config.craft_config)
        self.add3 = Add(self.config.craft_config)
        self.mul1 = Mul(self.config.craft_config)
        self.mul2 = Mul(self.config.craft_config)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = Act2FN(config.hidden_activation, self.config.craft_config)
            self.per_layer_input_gate = Gemma4QuantizableLinear(
                config, self.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = Gemma4QuantizableLinear(
                config, self.hidden_size_per_layer_input, self.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config
            )

        self.enable_moe_block = config.enable_moe_block
        if self.enable_moe_block:
            self.router = Gemma4TextRouter(config)
            self.experts = Gemma4TextExperts(config)
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
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
        return hidden_states


class Gemma4AssistantDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4AssistantConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4MtpCrossAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.register_buffer("layer_scalar", torch.ones(1))

        self.add1 = Add(config.craft_config)
        self.add2 = Add(config.craft_config)
        self.mul1 = Mul(config.craft_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
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


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)



class Gemma4CandidateGenerator(AssistedCandidateGenerator):
    """Candidate generator for Gemma4's assistant model.
    Unlike standard assisted generation where the assistant runs
    independently, drafters receive the target model's hidden states
    as input at each drafting round. This enables higher quality
    candidates since the drafter was trained to predict from backbone
    representations.
    The key differences from AssistedCandidateGenerator:
    - Implements `prepare_additional_assistant_inputs()` to capture
      backbone hidden_states between verification rounds
    - Overrides `get_candidates()` to run the drafter loop using
      backbone state instead of calling assistant_model.generate()
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "Gemma4AssistantModel",
        target_model: "Gemma4ForCausalLM",
        generation_config: "GenerationConfig",
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = None,
        logits_processor: LogitsProcessorList | None = None,
    ):
        input_ids = input_ids.to(assistant_model.device)
        self.assistant_model = assistant_model
        self.target_model = target_model
        # Share backbone embeddings with the drafter
        self.assistant_model.set_backbone_embeddings(target_model.get_input_embeddings())
        # --- Generation config ---
        self.generation_config = copy.deepcopy(generation_config)
        self.main_model_min_length = self.generation_config.min_length
        self.main_model_max_length = self.generation_config.max_length
        # Number of draft tokens to propose per iteration
        self.num_assistant_tokens = generation_config.num_assistant_tokens or 4
        self.model_kwargs = model_kwargs
        self.model_kwargs["output_hidden_states"] = True
        # Logits processors
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # Softcapping config from backbone
        self.final_logit_softcapping = getattr(target_model.config, "final_logit_softcapping", None)
        # assistant state: set by prepare_additional_assistant_inputs()
        self._assistant_state: torch.Tensor | None = None
        self._backbone_past_key_values: Cache | None = None
        # --- Attributes expected by parent but unused by us ---
        # Set these to safe defaults so any inherited helper methods
        # that reference them won't crash.
        self.assistant_kwargs = {}
        self.input_ids_key = "input_ids"
        self.assistant_generation_config = copy.deepcopy(generation_config)
        self.assistant_confidence_threshold = None
        # The assistant model needs a generation_config for the parent
        # _assisted_decoding code that checks num_assistant_tokens_schedule
        if not hasattr(assistant_model, "generation_config"):
            assistant_model.generation_config = GenerationConfig()

        # Find last sliding and full layer indices in backbone
        self.last_backbone_sliding_idx = None
        self.last_backbone_full_idx = None
        target_config = getattr(target_model.config, "text_config", target_model.config)
        if hasattr(target_config, "layer_types") and target_config.layer_types is not None:
            layer_types = target_config.layer_types
            first_kv_shared_layer_idx = target_config.num_hidden_layers - getattr(target_config, "num_kv_shared_layers", 0)
            for i in reversed(range(first_kv_shared_layer_idx)):
                if layer_types[i] == "sliding_attention" and self.last_backbone_sliding_idx is None:
                    self.last_backbone_sliding_idx = i
                if layer_types[i] == "full_attention" and self.last_backbone_full_idx is None:
                    self.last_backbone_full_idx = i

        # Fallback if not found
        if self.last_backbone_full_idx is None:
            self.last_backbone_full_idx = getattr(target_model.config, "text_config", target_model.config).num_hidden_layers - 1
        if self.last_backbone_sliding_idx is None:
            self.last_backbone_sliding_idx = self.last_backbone_full_idx

        # Set them on assistant config
        self.assistant_model.config.last_backbone_sliding_idx = self.last_backbone_sliding_idx
        self.assistant_model.config.last_backbone_full_idx = self.last_backbone_full_idx

        self.use_centroid_embedder = getattr(
            self.assistant_model.config, "use_centroid_embedder", False)
        if self.use_centroid_embedder:
            self.num_centroids_per_group = (
                self.assistant_model.config.vocab_size // self.assistant_model.config.num_centroids
            )
            self.canonical_positions_per_cluster = self.assistant_model.token_ordering.view(
                self.assistant_model.config.num_centroids,
                self.num_centroids_per_group
            )
            if self.num_centroids_per_group <= 0:
                raise ValueError(
                    "num_centroids_per_group must be greater than 0, but got "
                    f"{self.num_centroids_per_group}."
                )

    def prepare_additional_assistant_inputs(
        self,
        model_outputs=None,
        **model_kwargs,
    ):
        """Capture the backbone's hidden state and cache for the drafter."""
        if model_outputs is not None:
            if getattr(model_outputs, "hidden_states", None):
                last_hidden = model_outputs.hidden_states[-1]
                if last_hidden is not None:
                    self._assistant_state = last_hidden[:, -1:, :].detach()
            if getattr(model_outputs, "past_key_values", None):
                self._backbone_past_key_values = model_outputs.past_key_values

    def get_candidates(self, input_ids: torch.LongTensor) -> tuple[torch.LongTensor, torch.FloatTensor | None]:
        """Generate draft token candidates using the drafter."""
        device = self.assistant_model.device
        batch_size = input_ids.shape[0]
        max_new_tokens = min(int(self.num_assistant_tokens), self.main_model_max_length - input_ids.shape[1] - 1)
        if max_new_tokens <= 0:
            return input_ids, None

        if self._assistant_state is None:
            return input_ids, None

        # If assistant state is already available (captured from a previous
        # target model forward pass), reuse it and start drafting from the
        # last token in input_ids.
        state = self._assistant_state.to(device)
        first_token = input_ids[:, -1:]
        # Do not add to draft_token_ids as it is already in input_ids
        draft_token_ids = []
        draft_logits_list = []
        draft_top_k_indices_list = []
        num_loop_steps = max_new_tokens

        # --- Drafter autoregressive loop ---
        # The first token comes from input_ids and is in the canonical space.
        embedding = self.assistant_model.get_embedding(first_token)
        cur_pos = input_ids.shape[1]
        b_idx = torch.arange(batch_size, device=device).unsqueeze(-1)
        for _ in range(num_loop_steps):
            # Match Gemax behavior for EEVEE/MTP: position is the same for all drafter steps
            # and points to the last verified token position (cur_pos - 1).
            position_ids = torch.tensor([[cur_pos - 1]], device=device).expand(batch_size, -1)
            with torch.no_grad():
                logits, state, top_k_indices = self.assistant_model(
                    embedding=embedding,
                    state=state,
                    position_ids=position_ids,
                    past_key_values=self._backbone_past_key_values,
                )
            draft_logits_list.append(logits)
            next_token = logits.argmax(dim=-1)
            if self.use_centroid_embedder:
                draft_top_k_indices_list.append(top_k_indices)
                # Map directly to canonical ID using pre-computed 2D view and gather
                selected_canonical_ids = self.canonical_positions_per_cluster[top_k_indices].view(batch_size, 1, -1)
                next_token = torch.gather(selected_canonical_ids, dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1)
            draft_token_ids.append(next_token)
            embedding = self.assistant_model.get_embedding(next_token)
        # --- Assemble output ---
        draft_ids = torch.cat(draft_token_ids, dim=1)
        candidate_ids = torch.cat([input_ids, draft_ids], dim=1)
        candidate_logits = torch.cat(draft_logits_list, dim=1)
        if self.use_centroid_embedder:
            all_top_k_indices = torch.cat(draft_top_k_indices_list, dim=1)
            selected_tokens = self.assistant_model.token_ordering.view(
                self.assistant_model.config.num_centroids,
                self.num_centroids_per_group,
            )[all_top_k_indices].view(batch_size, num_loop_steps, -1)

            mask_value = -torch.finfo(candidate_logits.dtype).max
            output = torch.full(
                (batch_size, num_loop_steps, self.assistant_model.config.vocab_size),
                fill_value=mask_value,
                dtype=candidate_logits.dtype,
                device=candidate_logits.device,
            )
            candidate_logits = output.scatter_(dim=-1, index=selected_tokens, src=candidate_logits)
        return candidate_ids, candidate_logits

    def update_candidate_strategy(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        num_matches: int,
    ):
        """Adjust draft count based on acceptance rate."""
        schedule = getattr(self.generation_config, "num_assistant_tokens_schedule", "constant")
        if schedule in {"heuristic", "heuristic_transient"}:
            if num_matches == len(scores[0]) - 1:
                self.num_assistant_tokens = self.num_assistant_tokens + 2
            else:
                self.num_assistant_tokens = max(1, self.num_assistant_tokens - 1)


# ---- Model Classes ----


class Gemma4PreTrainedModel(PreTrainedModel):
    config: Gemma4Config
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _no_split_modules = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer", "Gemma4AssistantDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "shared_kv_states"]
    input_modalities = ("image", "text", "video", "audio")

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Gemma4VisionPatchEmbedder):
            init.ones_(module.position_embedding_table)
        elif isinstance(module, Gemma4AudioRelPositionalEncoding):
            min_timescale = 1.0
            max_timescale = 10000.0
            num_timescales = module.hidden_size // 2
            log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
            inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
            init.copy_(module.inv_timescales, inv_timescales.unsqueeze(0).unsqueeze(0))
        elif isinstance(module, Gemma4AudioAttention):
            init.constant_(module.softcap, module.attention_logits_soft_cap)
            init.zeros_(module.per_dim_scale)
        elif isinstance(module, Gemma4RMSNorm):
            init.constant_(module.eps, module.eps_val)
        elif isinstance(module, Gemma4TextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_init_fn_kwargs = {"layer_type": layer_type}
                if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
                    rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

                curr_inv_freq, _ = rope_init_fn(module.config, **rope_init_fn_kwargs)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, Gemma4VisionRotaryEmbedding):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)
        elif isinstance(module, Gemma4QuantizableEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, Gemma4TextRouter):
            init.ones_(module.scale)
            init.ones_(module.per_expert_scale)
        elif isinstance(module, Gemma4TextExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, (Gemma4TextDecoderLayer, Gemma4AssistantDecoderLayer)):
            init.ones_(module.layer_scalar)
        elif isinstance(module, Gemma4ClippableLinear) and module.use_clipped_linears:
            init.constant_(module.input_min, -float("inf"))
            init.constant_(module.input_max, float("inf"))
            init.constant_(module.output_min, -float("inf"))
            init.constant_(module.output_max, float("inf"))
        elif isinstance(module, Gemma4VisionModel) and module.config.standardize:
            init.zeros_(module.std_bias)
            init.ones_(module.std_scale)


class Gemma4AssistantModel(Gemma4PreTrainedModel):
    """An assistant model for Gemma4 speculative decoding.

    A lightweight 4-layer transformer that predicts multiple tokens ahead.
    Uses backbone's embed_tokens for input encoding but has its own
    lm_head for decoding (separate embedding table).
    Reads K/V from the backbone's KV cache but never writes to it.
    """

    config_class = Gemma4AssistantConfig
    _can_record_outputs = {
        "hidden_states": Gemma4AssistantDecoderLayer,
        "attentions": Gemma4MtpCrossAttention,
    }

    def __init__(self, config: Gemma4AssistantConfig):
        super().__init__(config)
        self.generation_config = GenerationConfig()
        # Non-owned reference to backbone embeddings — use object.__setattr__
        # to avoid nn.Module registering it as a submodule
        object.__setattr__(self, "_backbone_embed_tokens", None)
        self.hidden_size = config.hidden_size
        self.backbone_hidden_size = config.backbone_hidden_size

        # Pre-projection: concat(embedding, state) → drafter_d
        # concat_state=True means input is 2 × backbone_d
        self.pre_projection = Gemma4QuantizableLinear(
            config,
            2 * self.backbone_hidden_size,
            self.hidden_size,
            bias=False,
        )

        # 4 Transformer layers using specialized MTP decoder layer
        self.layers = nn.ModuleList(
            [Gemma4AssistantDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Final norm applied after all drafter layers
        self.final_norm = Gemma4RMSNorm(
            self.hidden_size,
            eps=config.rms_norm_eps,
            craft_config=config.craft_config,
        )

        # Post-projection: drafter_d → backbone_d (for state propagation)
        self.post_projection = Gemma4QuantizableLinear(
            config,
            self.hidden_size,
            self.backbone_hidden_size,
            bias=False,
        )

        # Separate lm_head for drafter
        self.lm_head = Gemma4QuantizableLinear(
            config,
            self.hidden_size,
            config.vocab_size,
            bias=False,
            skip_act_clip_and_quant=True,
        )
        self.lm_head_out = CraftModule(config.craft_config)

        self.register_buffer(
            "centroids",
            torch.zeros(config.num_centroids, config.hidden_size),
        )

        self.register_buffer(
            "token_ordering",
            torch.zeros(config.vocab_size, dtype=torch.long),
        )
        self.vocab_size_per_centroid = config.vocab_size // config.num_centroids

        self.centroid_intermediate_top_k = config.centroid_intermediate_top_k

        self.matmul_centroid1 = Matmul(config.craft_config)

        self.lm_head_einsum = Gemma4QuantizableEinsum(config, einsum_str="bld,blnd->bln", skip_act_clip_and_quant=True)
        self.lm_head_einsum_out = CraftModule(config.craft_config)

        self.rotary_emb = Gemma4TextRotaryEmbedding(config)

        self.cat = Concat(config.craft_config)
        self.unique_layer_types = set(self.config.layer_types)
        self.cos_in = nn.ModuleDict(
            {layer_type: CraftModule(config.craft_config) for layer_type in self.unique_layer_types}
        )
        self.sin_in = nn.ModuleDict(
            {layer_type: CraftModule(config.craft_config) for layer_type in self.unique_layer_types}
        )

        self.post_init()
        self._reorder_weights_if_necessary()

    def _reorder_weights_if_necessary(self) -> None:
        """Reorder lm_head weights to match centroid order if needed."""
        if self.config.use_centroid_embedder and not self.config.is_head_ordered:
            logger.info("Reordering lm_head weights based on token_ordering...")
            self.lm_head.weight.data = self.lm_head.weight[self.token_ordering]
            if hasattr(self.lm_head, "weight_scale") and self.lm_head.weight_scale is not None:
                self.lm_head.weight_scale.data = self.lm_head.weight_scale[self.token_ordering]

    def _supports_logits_to_keep(self) -> bool:
        return False

    def set_backbone_embeddings(self, embed_tokens: nn.Embedding):
        """Set reference to backbone's embedding table.

        Called by the candidate generator during initialization.
        The drafter needs backbone embeddings to encode draft tokens
        but doesn't own a copy — they're shared by reference.
        """
        object.__setattr__(self, "_backbone_embed_tokens", embed_tokens)

    def get_embedding(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Get token embeddings using the backbone's embedding table."""
        if self._backbone_embed_tokens is None:
            raise ValueError("Backbone embeddings not set. Call set_backbone_embeddings() first.")
        return self._backbone_embed_tokens(token_ids)

    def _apply_centroid_decoding(
        self,
        hidden_states: torch.Tensor,  # [B, L, D]
    ) -> torch.Tensor:
        """Apply centroid decoding by skipping full lm_head computation."""
        # 1. Compute centroid scores: [B, L, D] @ [D, C] -> [B, L, C]
        centroid_logits = self.matmul_centroid1(hidden_states, self.centroids.T)

        # 2. Find top-k centroids
        _, top_k_indices = torch.topk(centroid_logits, k=self.centroid_intermediate_top_k, dim=-1)  # [B, L, top_k]

        # 3. Get token indices for these centroids
        batch_size, seq_len = hidden_states.shape[:2]

        selected_weights = self.lm_head.weight.view(
            self.config.num_centroids,
            self.vocab_size_per_centroid,
            self.hidden_size,
        )[top_k_indices]  # [B, L, top_k, K, D]
        selected_weights = selected_weights.view(
            batch_size, seq_len, -1, self.hidden_size)  # [B, L, top_k * K, D]

        selected_scales = None
        if hasattr(self.lm_head, "weight_scale") and self.lm_head.weight_scale is not None:
            selected_scales = self.lm_head.weight_scale.view(
                self.config.num_centroids,
                self.vocab_size_per_centroid,
                1,
            )[top_k_indices]  # [B, L, top_k, K, 1]
            selected_scales = selected_scales.view(
                batch_size, seq_len, -1)  # [B, L, top_k * K]

        selected_logits = self.lm_head_einsum(
            "bld,blnd->bln",
            hidden_states,
            selected_weights,
            weight_scale=selected_scales,
        )  # [B, L, top_k * K]
        selected_logits = self.lm_head_einsum_out(selected_logits)

        return selected_logits, top_k_indices

    def forward(
        self,
        embedding: torch.Tensor,
        state: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: dict[str, torch.Tensor] | None = None,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
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
        drafter_input = self.cat(embedding, state, dim=-1)
        # drafter_input: [B, L, 2 * backbone_d]

        # Step 2: Pre-projection
        hidden_states = self.pre_projection(drafter_input)
        # hidden_states: [B, L, drafter_d]

        if position_embeddings is None:
            position_embeddings = {}
            for layer_type in set(self.config.layer_types):
                position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)
                position_embeddings[layer_type] = (
                    self.cos_in[layer_type](position_embeddings[layer_type][0]),
                    self.sin_in[layer_type](position_embeddings[layer_type][1]),
                )

        # Step 3: Pass through drafter transformer layers
        for layer in self.layers:
            layer_type = layer.self_attn.layer_type
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask[layer_type] if attention_mask else None,
                position_embeddings=position_embeddings[layer_type] if position_embeddings else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,  # Drafter does NOT write to cache
                cache_position=cache_position,
            )

        # Step 4: Final norm
        hidden_states = self.final_norm(hidden_states)

        # Step 5: Post-projection for state propagation to next step
        projected_state = self.post_projection(hidden_states)

        # Step 6: Logits via drafter's own lm_head OR centroid decoding
        top_k_indices = None
        if self.config.use_centroid_embedder and hasattr(self, "centroids") and self.centroids is not None:
            logits, top_k_indices = self._apply_centroid_decoding(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
            logits = self.lm_head_out(logits)

        return logits, projected_state, top_k_indices


@auto_docstring(custom_intro="The base Gemma 4 language model without a language modeling head.")
class Gemma4TextModel(Gemma4PreTrainedModel):
    config: Gemma4TextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(Gemma4TextRouter, index=0),
        "hidden_states": Gemma4TextDecoderLayer,
        "attentions": Gemma4TextAttention,
    }

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma4 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma4QuantizableEmbedding(
            config, config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, craft_config=config.craft_config)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.unique_layer_types = set(self.config.layer_types)

        self.embed_outs = CraftModule(config.craft_config)
        self.ple_in = CraftModule(config.craft_config)
        # Per-layer embeddings for each transformer layer.
        self.ple_outs = nn.ModuleList(
            [CraftModule(config.craft_config) for _ in range(config.num_hidden_layers)]
        )
        # cos and sin for full_attention, sliding_attention
        self.cos_in = nn.ModuleDict(
            {layer_type: CraftModule(config.craft_config) for layer_type in self.unique_layer_types}
        )
        self.sin_in = nn.ModuleDict(
            {layer_type: CraftModule(config.craft_config) for layer_type in self.unique_layer_types}
        )

        # Per-Layer Embeddings (PLE): auxiliary embedding that feeds a residual signal
        # into each decoder layer. See `get_per_layer_inputs()` and `project_per_layer_inputs()`
        # for the full pipeline. The embedding is packed: total dim = num_layers * per_layer_dim.
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4QuantizableEmbedding(
                config,
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
                num_weight_scales=config.num_hidden_layers,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = Gemma4QuantizableLinear(
                config,
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_model_projection_mul = Mul(config.craft_config)
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps, craft_config=config.craft_config)
            self.add = Add(config.craft_config)
            self.mul = Mul(config.craft_config)

        # Update `_keys_to_ignore_on_load_unexpected` to drop all k/v proj and norms for the shared layers
        self._keys_to_ignore_on_load_unexpected = []
        for i, layer in enumerate(self.layers):
            if layer.self_attn.is_kv_shared_layer:
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
                )

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = self.embed_outs(inputs_embeds)
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)
            position_embeddings[layer_type] = (
                self.cos_in[layer_type](position_embeddings[layer_type][0]),
                self.sin_in[layer_type](position_embeddings[layer_type][1]),
            )

        # Initialize as empty dict - it will be filled in the right layers
        shared_kv_states = {}

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            if per_layer_input is not None:
                per_layer_input = self.ple_outs[i](per_layer_input)

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def get_per_layer_inputs(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> torch.Tensor:
        """Compute the token-identity component of Per-Layer Embeddings (PLE).

        Looks up `input_ids` in `embed_tokens_per_layer` (a scaled embedding that multiplies
        by `sqrt(hidden_size_per_layer_input)`) and reshapes the packed output from
        `[batch, seq, num_hidden_layers * hidden_size_per_layer_input]` to
        `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`.

        If only `inputs_embeds` is provided (no `input_ids`), reverses the main embedding
        to recover `input_ids` for the PLE lookup.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call get_per_layer_inputs() from a model initialized with a config that does not support"
                f" per-layer embeddings. {self.config}"
            )

        # If only inputs_embeds are provided, reverse main embedding to find the input_ids - this allows to `generate`
        # from `inputs_embeds` only as other models (otherwise it would need the value from both embeddings)
        if input_ids is None:
            with torch.no_grad():
                input_ids = (
                    (
                        inputs_embeds[:, :, None, :]
                        == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5
                    )
                    .all(dim=3)
                    .nonzero()[:, 2]
                )
                try:
                    input_ids = input_ids.view(inputs_embeds.shape[:2])
                except RuntimeError:
                    raise RuntimeError(
                        "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and that "
                        "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Gemma4 needs to reverse "
                        "the embedding to compute another embedding, make sure you provide exact `inputs_embeds`"
                    )

        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
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
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        per_layer_inputs = self.ple_in(per_layer_inputs)

        return self.mul(self.add(per_layer_projection, per_layer_inputs), self.per_layer_input_scale)


@auto_docstring(custom_intro="The base Gemma 4 language model with a language modeling head.")
class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma4TextConfig
    base_model_prefix = "model"

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.model = Gemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Gemma4QuantizableLinear(config, config.hidden_size, config.vocab_size, bias=False, skip_act_clip_and_quant=True)
        self.lm_head_outs = CraftModule(config.craft_config)
        self.tanh = Tanh(config.craft_config)
        self.div = Div(config.craft_config)
        self.mul = Mul(config.craft_config)

        self._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        if getattr(config, "use_quantized_model", False):
            self._tied_weights_keys["lm_head.weight_scale"] = "model.embed_tokens.weight_scale"

        # Grab the ones from the child
        self._keys_to_ignore_on_load_unexpected = [
            f"model.{name}" for name in self.model._keys_to_ignore_on_load_unexpected
        ]

        # Initialize weights and apply final processing
        self.post_init()

    def _get_candidate_generator(
        self,
        generation_config,
        input_ids,
        inputs_tensor,
        logits_processor,
        model_kwargs,
        assistant_model=None,
        target_tokenizer=None,
        assistant_tokenizer=None,
    ):
        if isinstance(assistant_model, Gemma4AssistantModel):
            return Gemma4CandidateGenerator(
                input_ids=input_ids,
                assistant_model=assistant_model,
                target_model=self,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
            )
        return super()._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            assistant_model=assistant_model,
            target_tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from ... import AutoTokenizer, Gemma4ForCausalLM

        >>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
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


def sliding_window_mask_function(sliding_window: tuple[int, int]) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        left_window_size, right_window_size = sliding_window

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


class Gemma4AudioModel(Gemma4PreTrainedModel):
    """An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture."""

    config: Gemma4AudioConfig
    main_input_name = "input_features"
    base_model_prefix = "model.audio_tower"  # prefix for Gemma4ForConditionalGeneration saved checkpoints, required for Gemma4AudioModel.from_pretrained()
    _can_record_outputs = {
        "hidden_states": Gemma4AudioLayer,
        "attentions": Gemma4AudioAttention,
    }

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__(config)
        self.config = config
        self.div = Div(config.craft_config)
        self.mul = Mul(config.craft_config)

        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [Gemma4AudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.output_proj = Linear(config.hidden_size, config.output_proj_dims, bias=True, craft_config=config.craft_config)

        self.post_init()

    def _convert_4d_mask_to_blocked_5d(self, mask_4d: torch.Tensor, has_past: bool = False) -> torch.Tensor:
        """Converts a 4D attention mask to a 5D blocked mask.

        This is used to adapt the 4D attention mask to the 5D shape
        `[batch_size, 1, num_blocks, chunk_size, context_size]` expected by the chunked local attention,
        """
        batch_size, _, seq_len, _ = mask_4d.shape
        device = mask_4d.device

        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right

        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)

        past_pad_value = has_past
        mask_5d = F.pad(mask_5d, (max_past_horizon, 0), value=past_pad_value)
        mask_5d = F.pad(mask_5d, (0, max_future_horizon), value=False)

        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(chunk_size + max_past_horizon + max_future_horizon, device=device)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(batch_size, 1, -1, chunk_size, -1)

        return mask_5d.gather(-1, kv_indices)

    def _forward_chunk(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: Sequence | None = None,
        is_last: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4AudioModelOutput:
        subsample_state = state[0] if state is not None else None
        hidden_states, output_mask, new_subsample_state = self.subsample_conv_projection(
            input_features, attention_mask, state=subsample_state, is_last=is_last
        )
        position_embeddings = self.rel_pos_enc(hidden_states)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=output_mask,
            and_mask_function=sliding_window_mask_function(
                (self.config.attention_context_left, self.config.attention_context_right)
            ),
        )
        attention_mask = self._convert_4d_mask_to_blocked_5d(attention_mask, has_past=(state is not None))

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
        if output_mask is not None:
            hidden_states = hidden_states.masked_fill(~output_mask.unsqueeze(-1), 0.0)

        past_key_values = [new_subsample_state] + new_layer_states
        return Gemma4AudioModelOutput(last_hidden_state=hidden_states, attention_mask=output_mask, past_key_values=past_key_values)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring(custom_intro="Encodes audio features to soft tokens.")
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        streaming_chunk_size_tokens: int | None = None,
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
        if streaming_chunk_size_tokens is None:
            # One-shot mode
            return self._forward_chunk(input_features, attention_mask, state=None, is_last=True, **kwargs)

        # Internal chunking mode
        chunk_size = streaming_chunk_size_tokens
        # 4 is the total subsampling factor from the two conv layers with stride 2
        input_chunk_size = chunk_size * 4

        seq_len = input_features.shape[1]
        outputs = []
        masks = []
        current_state = None  # Manage state internally

        if attention_mask is None:
            attention_mask = torch.ones((input_features.shape[0], seq_len), device=input_features.device, dtype=torch.bool)

        for i in range(0, seq_len, input_chunk_size):
            raw_chunk = input_features[:, i:i+input_chunk_size, :]
            raw_mask = attention_mask[:, i:i+input_chunk_size]

            is_last_chunk = (i + input_chunk_size) >= seq_len
            pad_len = input_chunk_size - raw_chunk.shape[1]

            if pad_len > 0:
                chunk = torch.cat([
                    raw_chunk,
                    torch.zeros((raw_chunk.shape[0], pad_len, raw_chunk.shape[2]), device=raw_chunk.device, dtype=raw_chunk.dtype)
                ], dim=1)
                chunk_mask = torch.cat([
                    raw_mask,
                    torch.zeros((raw_mask.shape[0], pad_len), device=raw_mask.device, dtype=torch.bool)
                ], dim=1)
            else:
                chunk = raw_chunk
                chunk_mask = raw_mask

            out = self._forward_chunk(chunk, attention_mask=chunk_mask, state=current_state, is_last=is_last_chunk, is_streaming=True, **kwargs)
            outputs.append(out.last_hidden_state)
            if out.attention_mask is not None:
                masks.append(out.attention_mask)
            current_state = out.past_key_values

        full_output = torch.cat(outputs, dim=1)
        full_mask = torch.cat(masks, dim=1) if masks else None

        # Slice output to the correct length to match one-shot inference
        expected_len = (seq_len + 3) // 4
        full_output = full_output[:, :expected_len, :]
        if full_mask is not None:
            full_mask = full_mask[:, :expected_len]

        return Gemma4AudioModelOutput(last_hidden_state=full_output, attention_mask=full_mask)


class Gemma4VisionModel(Gemma4PreTrainedModel):
    """The Gemma 4 Vision Encoder."""

    config = Gemma4VisionConfig
    _can_record_outputs = {
        "hidden_states": Gemma4VisionEncoderLayer,
        "attentions": Gemma4VisionAttention,
    }

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__(config)
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)

        if self.config.standardize:
            self.register_buffer("std_bias", torch.empty(self.config.hidden_size))
            self.register_buffer("std_scale", torch.empty(self.config.hidden_size))
            self.std_sub = Sub(config.craft_config)
            self.std_mul = Mul(config.craft_config)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring(custom_intro="Encodes image pixels to soft tokens from patches.")
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_position_ids: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` or `list[torch.FloatTensor]`):
            The images to encode. Either a single `[batch, channels, height, width]` tensor
            (all images same size) or a list of `[1, channels, height, width]` tensors (different sizes).
        pixel_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,  # encoder expects True=valid, padding_positions is True=padding
            pixel_position_ids=pixel_position_ids,
            **kwargs,
        )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=output.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        # Strip padding tokens. pooler_mask is True = valid, False = padding.
        hidden_states = hidden_states[pooler_mask]

        if self.config.standardize:
            hidden_states = self.std_sub(hidden_states, self.std_bias)
            hidden_states = self.std_mul(hidden_states, self.std_scale)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Gemma4MultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Gemma4AudioConfig | Gemma4VisionConfig,
        text_config: Gemma4TextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size)
        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size
        self.embedding_projection = Linear(
            self.multimodal_hidden_size,
            self.text_hidden_size,
            bias=False,
            craft_config=multimodal_config.craft_config
        )
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            self.multimodal_hidden_size,
            eps=self.eps,
            with_scale=False,
            craft_config=text_config.craft_config,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.
        Args:
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.
        Returns:
            A torch.Tensor of embeddings with shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


# Identical as Gemma3 but modular can't resolve if we simply import. FIXME: @cyril
def token_type_ids_mask_function(
    token_type_ids: torch.Tensor | None,
    image_group_ids: torch.Tensor | None,
) -> Callable | None:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        seq_length = image_group_ids.shape[-1]

        # clamp indices because with static cache they can go beyond `image_group_ids.shape[-1]`
        q_idx_clamped = q_idx.clamp(max=seq_length - 1)
        kv_idx_clamped = kv_idx.clamp(max=seq_length - 1)

        # Unmask if the q and kv come from same group which is not -1 (i.e. non-text)
        q_group = image_group_ids[batch_idx, q_idx_clamped]
        kv_group = image_group_ids[batch_idx, kv_idx_clamped]
        q_group = torch.where(q_idx < seq_length, q_group, -1)
        kv_group = torch.where(kv_idx < seq_length, kv_group, -1)
        return (q_group == kv_group) & (q_group >= 0)

    return inner_mask


# Similar to Gemma3 but `sliding_mask_kwargs` and `mask_kwargs` are different and `token_type_ids->mm_token_type_ids`
def create_causal_mask_mapping(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    is_training: bool = False,
    is_first_iteration: bool | None = None,
    **kwargs,
) -> dict:
    """
    Overwrites the base `create_masks_for_generate` with `token_type_ids` masking to create the causal mask mapping
    for all kinds of forward passes. Gemma4 uses a bidirectional mask for images.

    Uses `pixel_values` as an optional input to disambiguate edge cases.
    """
    if is_training and mm_token_type_ids is None:
        raise ValueError("`mm_token_type_ids` is required as a model input when training")

    mask_kwargs = {
        "config": config.get_text_config(),
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    sliding_mask_kwargs = mask_kwargs.copy()

    # NOTE: this `may_have_image_input` logic is not flawless, it fails when we're using a cache eagerly initialized
    # (e.g. compiled prefill) AND `pixel_values` are not provided (i.e. the image data is provided through other
    # means). Determining prefill in that case requires checking data values, which is not compile-compatible.
    is_first_iteration = (
        is_first_iteration
        if is_first_iteration is not None
        else (past_key_values is None or not past_key_values.is_initialized or pixel_values is not None)
    )
    if mm_token_type_ids is not None and is_first_iteration:
        # We need to pass an additional mask function to account for token type ids, and it needs to be an `or` (to
        # undo the causal masking)

        # First find where a new vision block starts. Vision tokens cannot attend to
        # future vision tokens, but can attend to all prev tokens and to itself bidirectionally
        is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
        is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
        is_prev_vision[..., 0] = False
        new_vision_starts = is_vision & ~is_prev_vision
        vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
        vision_group_ids = torch.where(is_vision, vision_group_ids, -1)
        sliding_mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
            mm_token_type_ids.to(inputs_embeds.device), vision_group_ids
        )

    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
    }




@auto_docstring(
    custom_intro="""
    The base Gemma 4 model comprising a vision backbone, an audio backbone, and a language model without a
    language modeling head.
    """
)
class Gemma4Model(Gemma4PreTrainedModel):
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModel.from_config(config=config.text_config)
        self.language_model = language_model
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        self.vision_tower = (
            Gemma4VisionModel(config=config.vision_config)
            if config.vision_config is not None
            else None
        )
        self.embed_vision = (
            Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
            if config.vision_config is not None
            else None
        )
        self.audio_tower = (
            Gemma4AudioModel(config.audio_config)
            if config.audio_config is not None
            else None
        )
        self.embed_audio = (
            Gemma4MultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )
        # Grab the ones from the child
        self._keys_to_ignore_on_load_unexpected = [
            f"language_model.{name}" for name in self.language_model._keys_to_ignore_on_load_unexpected
        ]
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the vision model into language model space.")
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            **kwargs,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        vision_outputs.pooler_output = self.embed_vision(inputs_embeds=last_hidden_state)
        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Obtains mask for multimodal placeholders (replaced by soft tokens) and hard text tokens.

        Masks will be obtained from `mm_token_type_ids`, `input_ids`, or `inputs_embeds` as available and in that
        precedence order. If passing `input_ids` or `inputs_embeds`, the image mask will be derived using
        `config.image_token_id`. Same goes for audio and video masks

        Args:
            input_ids: A tensor containing the hard token IDs from the text tokenizer.
            inputs_embeds: A tensor containing the embeddings for all hard text tokens.

        Returns:
            image_mask, video_mask, audio_mask
        """
        if input_ids is not None:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id
        else:
            special_image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
            special_video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)

        return special_image_mask, special_video_mask, special_audio_mask

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
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

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        multimodal_mask = image_mask | video_mask | audio_mask

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if self.config.get_text_config().hidden_size_per_layer_input:
            pad_embedding = self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]
            llm_inputs_embeds = torch.where(multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
            per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
        else:
            per_layer_inputs = None

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
                f" {image_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
            )

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos, video_position_ids, return_dict=True
            ).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_video_tokens = video_mask.sum()
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features:"
                f" {video_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device), video_features.to(inputs_embeds.device)
            )

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_mask_from_encoder = audio_output.attention_mask  # True = valid

            # Strip padding tokens: only keep real (non-padding) audio soft tokens.
            # audio_mask_from_encoder is True for valid positions, False for padding tokens.
            # This mirrors the vision encoder's padding stripping (see Gemma4VisionEncoder.forward).
            audio_features = audio_features[audio_mask_from_encoder]

            n_audio_tokens = audio_mask.sum()
            audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[audio_mask].numel() == audio_features.numel(),
                f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features:"
                f" {audio_features.shape[0] * audio_features.shape[1]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device)
            )

        # It may already have been prepared by, e.g., `generate`
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if self.config.get_text_config().use_bidirectional_attention == "vision":
                # Larger Gemma 4 models use Gemma 3's bidirectional attention mask for vision inputs
                causal_mask_mapping = create_causal_mask_mapping(
                    self.config,
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    position_ids,
                    mm_token_type_ids,
                    pixel_values,
                    is_training=self.training,
                )
            else:
                # Smaller Gemma models use a conventional casual attention mask
                causal_mask_mapping = create_masks_for_generate(
                    self.config,
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    position_ids,
                )

        outputs = self.language_model(
            per_layer_inputs=per_layer_inputs,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        return Gemma4ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the audio encoder into language model space.")
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
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

        chunk_size = getattr(self.config.audio_config, "streaming_chunk_size_tokens", None)
        audio_outputs = self.audio_tower(
            input_features,
            input_features_mask,
            return_dict=True,
            streaming_chunk_size_tokens=chunk_size,
            **kwargs
        )
        audio_outputs.pooler_output = self.embed_audio(inputs_embeds=audio_outputs.last_hidden_state)

        return audio_outputs

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the vision encoder into language model space.")
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        pixel_values_videos = pixel_values_videos.flatten(0, 1)
        video_position_ids = video_position_ids.flatten(0, 1)
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values_videos,
            pixel_position_ids=video_position_ids,
            **kwargs,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        vision_outputs.pooler_output = self.embed_vision(inputs_embeds=last_hidden_state)
        return vision_outputs


@auto_docstring(
    custom_intro="""
    The base Gemma 4 model comprising a vision backbone, an audio backbone, a language model, and a language modeling
    head.
    """
)
class Gemma4ForConditionalGeneration(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    base_model_prefix = "model"

    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.model = Gemma4Model(config)
        self.lm_head = Gemma4QuantizableLinear(config.text_config, config.text_config.hidden_size, config.text_config.vocab_size, bias=False, skip_act_clip_and_quant=True)
        self.lm_head_out = CraftModule(config.craft_config)

        # for softcapping
        if self.config.get_text_config().final_logit_softcapping is not None:
            self.logit_div = Div(config.craft_config)
            self.logit_tanh = Tanh(config.craft_config)
            self.logit_mul = Mul(config.craft_config)

        self._tied_weights_keys = {
            "lm_head.weight": "model.language_model.embed_tokens.weight",
        }
        if getattr(config.text_config, "use_quantized_model", False):
            self._tied_weights_keys["lm_head.weight_scale"] = "model.language_model.embed_tokens.weight_scale"

        if getattr(config, "assistant_config", None) is not None:
            self.assistant_model = Gemma4AssistantModel(config.assistant_config)

        # Grab the ones from the child
        self._keys_to_ignore_on_load_unexpected = [
            f"model.{name}" for name in self.model._keys_to_ignore_on_load_unexpected
        ]
        self.post_init()

    def _get_candidate_generator(
        self,
        generation_config,
        input_ids,
        inputs_tensor,
        logits_processor,
        model_kwargs,
        assistant_model=None,
        target_tokenizer=None,
        assistant_tokenizer=None,
    ):
        if isinstance(assistant_model, Gemma4AssistantModel):
            return Gemma4CandidateGenerator(
                input_ids=input_ids,
                assistant_model=assistant_model,
                target_model=self,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
                logits_processor=logits_processor,
            )
        return super()._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            assistant_model=assistant_model,
            target_tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        return self.model.get_image_features(pixel_values, image_position_ids, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
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
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = self.lm_head_out(logits)
        if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:
            logits = self.logit_div(logits, final_logit_softcapping)
            logits = self.logit_tanh(logits)
            logits = self.logit_mul(logits, final_logit_softcapping)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.get_text_config().vocab_size, **kwargs)

        return Gemma4CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # If we're in cached decoding stage, multimodal inputs are already cached and can be dropped
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ) -> dict:
        if getattr(config.get_text_config(), "use_bidirectional_attention", None) == "vision":
            # Larger Gemma 4 models use Gemma 3's bidirectional attention mask for vision inputs
            return create_causal_mask_mapping(
                config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                position_ids,
                mm_token_type_ids,
                is_first_iteration=is_first_iteration,
                **{k: v for k, v in kwargs.items() if k != "pixel_values"},
            )
        else:
            # Smaller Gemma models use a conventional casual attention mask
            return create_masks_for_generate(
                config, inputs_embeds, attention_mask, past_key_values, position_ids, **kwargs
            )


__all__ = [
    "Gemma4AudioModel",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4Model",
    "Gemma4PreTrainedModel",
    "Gemma4TextModel",
    "Gemma4VisionModel",
]
