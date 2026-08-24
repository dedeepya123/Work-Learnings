# common_typos_disable
# isort: skip_file
# pylint: disable=all
# pytype: skip-file
# refex: disable=pytotw.034
# fmt: off
"""Quantization modules for Gemma4."""

import torch
from torch import nn
import torch.nn.functional as F

from .configuration_gemma4 import Gemma4TextConfig
from .configuration_gemma4 import Gemma4VisionConfig
from .static_fake_quant import fake_quant


def fake_quant_activation(
    x: torch.Tensor,
    scale: torch.Tensor,
    activation_bits: int,
) -> torch.Tensor:
  """Applies activation quantization based on bounds or scale."""
  if activation_bits <= 0:
    raise ValueError("Activation bits must be positive.")
  return fake_quant(
      x,
      int_precision=activation_bits,
      scale=scale + 1e-9,
  )


class Gemma4QuantizableEmbedding(nn.Module):
  """Quantizable Scaled Word Embedding layer for Gemma4.

  Uses config to turn quantization on and off.
  """

  weight_scale: torch.Tensor | None

  def __init__(
      self,
      config: Gemma4TextConfig | Gemma4VisionConfig,
      num_embeddings: int,
      embedding_dim: int,
      padding_idx: int,
      embed_scale: float = 1.0,
      num_weight_scales: int = 1,
      device: torch.device | str | None = None,
      dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}

    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.scalar_embed_scale = embed_scale
    self.register_buffer(
        "embed_scale",
        torch.tensor(embed_scale, **factory_kwargs),
        persistent=False,
    )
    use_quantized_model = getattr(config, "use_quantized_model", False)
    if use_quantized_model:
      self.weight = nn.Parameter(
          torch.empty(
              (num_embeddings, embedding_dim), device=device, dtype=torch.int8
          ),
          requires_grad=False,
      )
      self.weight_scale = nn.Parameter(
          torch.ones((num_embeddings, num_weight_scales), **factory_kwargs),
          requires_grad=False,
      )
    else:
      self.weight = nn.Parameter(
          torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
      )
      self.weight_scale = None

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    if self.weight_scale is not None:
      w_int = F.embedding(input_ids, self.weight, self.padding_idx)
      w_scale = F.embedding(input_ids, self.weight_scale)
      stretch_factor = w_int.shape[-1] // w_scale.shape[-1]
      if stretch_factor > 1:
        w_scale = w_scale.repeat_interleave(stretch_factor, dim=-1)
      embeds = w_int.to(w_scale.dtype) * w_scale
    else:
      embeds = F.embedding(input_ids, self.weight, self.padding_idx)

    return embeds * self.embed_scale.to(embeds.dtype)


class Gemma4QuantizableLinear(nn.Module):
  """Quantizable Linear layer for Gemma4.

  Uses config in the checkpoint to turn quantization and clipping on and off.
  """

  input_min: torch.Tensor | None
  input_max: torch.Tensor | None
  output_min: torch.Tensor | None
  output_max: torch.Tensor | None
  input_scale: torch.Tensor | None
  input_bits: torch.Tensor | None
  weight_scale: torch.Tensor | None
  output_scale: torch.Tensor | None
  output_bits: torch.Tensor | None

  def __init__(
      self,
      config: Gemma4TextConfig | Gemma4VisionConfig,
      in_features: int,
      out_features: int,
      bias: bool = True,
      device: torch.device | str | None = None,
      dtype: torch.dtype | None = None,
      skip_act_clip_and_quant: bool = False,
      weight_bits: int = 8,
  ) -> None:
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}
    self.in_features = in_features
    self.out_features = out_features
    use_quantized_model = getattr(config, "use_quantized_model", False)
    use_clipping = getattr(config, "use_clipped_linears", False)
    if bias:
      self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    else:
      self.register_parameter("bias", None)

    if use_clipping and use_quantized_model:
      raise ValueError(
          "Cannot set both use_clipped_linears and use_quantized_linears to"
          " True."
      )

    if use_clipping and not skip_act_clip_and_quant:
      self.register_buffer(
          "input_min", torch.tensor(-float("inf"), **factory_kwargs)
      )
      self.register_buffer(
          "input_max", torch.tensor(float("inf"), **factory_kwargs)
      )
      self.register_buffer(
          "output_min", torch.tensor(-float("inf"), **factory_kwargs)
      )
      self.register_buffer(
          "output_max", torch.tensor(float("inf"), **factory_kwargs)
      )
    else:
      self.input_min = None
      self.input_max = None
      self.output_min = None
      self.output_max = None

    if use_quantized_model:
      if weight_bits == 8:
        weight_dtype = torch.int8
      elif weight_bits == 16:
        weight_dtype = torch.int16
      else:
        raise ValueError(f"Unsupported weight_bits: {weight_bits}")
      self.weight = nn.Parameter(
          torch.empty(
              (out_features, in_features), device=device, dtype=weight_dtype
          ),
          requires_grad=False,
      )
      self.weight_scale = nn.Parameter(
          torch.ones((out_features, 1), **factory_kwargs),
          requires_grad=False,
      )
      if not skip_act_clip_and_quant:
        self.register_buffer(
            "input_scale", torch.ones((1, 1, 1), **factory_kwargs)
        )
        self.register_buffer(
            "input_bits", torch.tensor(0, **factory_kwargs)
        )
        self.register_buffer(
            "output_scale", torch.ones((1, 1, 1), **factory_kwargs)
        )
        self.register_buffer(
            "output_bits", torch.tensor(0, **factory_kwargs)
        )
      else:
        self.input_scale = None
        self.input_bits = None
        self.output_scale = None
        self.output_bits = None
    else:
      self.input_scale = None
      self.input_bits = None
      self.weight_scale = None
      self.output_scale = None
      self.output_bits = None
      self.weight = nn.Parameter(
          torch.empty((out_features, in_features), **factory_kwargs)
      )

  def forward_clipped(self, x: torch.Tensor) -> torch.Tensor:
    if self.input_min is not None and self.input_max is not None:
      x = torch.clamp(x, self.input_min, self.input_max)

    x = nn.functional.linear(x, self.weight, self.bias)

    if self.output_min is not None and self.output_max is not None:
      x = torch.clamp(x, self.output_min, self.output_max)
    return x

  def forward_quantized(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype

    # 1. Activation Fake-Quantization.
    if self.input_scale is not None and self.input_bits is not None:
      x = fake_quant_activation(
          x, self.input_scale, int(self.input_bits.item())
      )

    # 2. Dense Matrix Multiplication in Floating Point.
    x = nn.functional.linear(x.to(torch.float32), self.weight.to(torch.float32), bias=None)

    # 3. Weight Dequantization.
    x = x * self.weight_scale.to(torch.float32).T
    if self.bias is not None:
      x = x + self.bias.to(torch.float32)

    # 4. Output Activation Fake-Quantization.
    if self.output_scale is not None and self.output_bits is not None:
      x = fake_quant_activation(
          x, self.output_scale, int(self.output_bits.item())
      )

    return x.to(orig_dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.weight_scale is not None:
      return self.forward_quantized(x)
    elif self.input_min is not None or self.output_min is not None:
      return self.forward_clipped(x)
    else:
      return nn.functional.linear(x, self.weight, self.bias)


class Gemma4QuantizableScaledLinear(nn.Module):
  """Quantizable Scaled Linear layer for Gemma4.

  Uses config in the ckeckpoint to turn quantization on and off.
  """

  input_scale: torch.Tensor | None
  input_bits: torch.Tensor | None
  weight_scale: torch.Tensor | None
  output_scale: torch.Tensor | None
  output_bits: torch.Tensor | None

  def __init__(
      self,
      config: Gemma4TextConfig | Gemma4VisionConfig,
      in_features: int,
      out_features: int,
      scalar: float,
      bias: bool = False,
      device: torch.device | str | None = None,
      dtype: torch.dtype | None = None,
      skip_act_clip_and_quant: bool = False,
  ) -> None:
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}
    self.in_features = in_features
    self.out_features = out_features
    use_quantized_model = getattr(config, "use_quantized_model", False)

    self.scalar = scalar
    self.register_buffer(
        "scale", torch.tensor(self.scalar, **factory_kwargs), persistent=False
    )
    if bias:
      self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    else:
      self.register_parameter("bias", None)

    if use_quantized_model:
      self.weight = nn.Parameter(
          torch.empty(
              (out_features, in_features), device=device, dtype=torch.int8
          ),
          requires_grad=False,
      )
      self.weight_scale = nn.Parameter(
          torch.ones((out_features, 1), **factory_kwargs),
          requires_grad=False,
      )
      if not skip_act_clip_and_quant:
        self.register_buffer(
            "input_scale", torch.ones((1, 1, 1), **factory_kwargs)
        )
        self.register_buffer("input_bits", torch.tensor(0, **factory_kwargs))
        self.register_buffer(
            "output_scale", torch.ones((1, 1, 1), **factory_kwargs)
        )
        self.register_buffer("output_bits", torch.tensor(0, **factory_kwargs))
      else:
        self.input_scale = None
        self.input_bits = None
        self.output_scale = None
        self.output_bits = None
    else:
      self.input_scale = None
      self.input_bits = None
      self.weight_scale = None
      self.output_scale = None
      self.output_bits = None
      self.weight = nn.Parameter(
          torch.empty((out_features, in_features), **factory_kwargs)
      )

  def forward_quantized(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype

    # 1. Activation Fake-Quantization.
    if self.input_scale is not None and self.input_bits is not None:
      x = fake_quant_activation(
          x, self.input_scale, int(self.input_bits.item())
      )

    # 2. Dense Matrix Multiplication in Floating Point.
    x = nn.functional.linear(x.to(torch.float32), self.weight.to(torch.float32), bias=None)

    # 3. Weight Dequantization.
    x = x * self.weight_scale.to(torch.float32).T

    if self.bias is not None:
      x = x + self.bias.to(torch.float32)
    x = x * self.scale

    # 4. Output Activation Fake-Quantization.
    if self.output_scale is not None and self.output_bits is not None:
      x = fake_quant_activation(
          x, self.output_scale, int(self.output_bits.item())
      )

    return x.to(orig_dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.weight_scale is not None:
      return self.forward_quantized(x)
    else:
      return nn.functional.linear(x, self.weight, self.bias) * self.scale


class Gemma4QuantizableEinsum(nn.Module):
  """Quantizable Einsum layer for Gemma4."""

  def __init__(
      self,
      config: Gemma4TextConfig | Gemma4VisionConfig,
      einsum_str: str,
      skip_act_clip_and_quant: bool = False,
      dtype: torch.dtype | None = None,
      device: torch.device | str | None = None,
  ) -> None:
    super().__init__()
    self.config = config
    self.einsum_str = einsum_str
    factory_kwargs = {"device": device, "dtype": dtype}

    use_quantized_model = getattr(config, "use_quantized_model", False)
    use_clipping = getattr(config, "use_clipped_linears", False)

    if use_clipping and use_quantized_model:
      raise ValueError(
          "Cannot set both use_clipped_linears and use_quantized_linears to True."
      )

    if use_clipping and not skip_act_clip_and_quant:
      self.register_buffer(
          "input_min", torch.tensor(-float("inf"), **factory_kwargs)
      )
      self.register_buffer(
          "input_max", torch.tensor(float("inf"), **factory_kwargs)
      )
      self.register_buffer(
          "output_min", torch.tensor(-float("inf"), **factory_kwargs)
      )
      self.register_buffer(
          "output_max", torch.tensor(float("inf"), **factory_kwargs)
      )
    else:
      self.input_min = None
      self.input_max = None
      self.output_min = None
      self.output_max = None

    if use_quantized_model and not skip_act_clip_and_quant:
      self.register_buffer(
          "input_scale", torch.ones((1, 1, 1), **factory_kwargs)
      )
      self.register_buffer("input_bits", torch.tensor(0, **factory_kwargs))
      self.register_buffer(
          "output_scale", torch.ones((1, 1, 1), **factory_kwargs)
      )
      self.register_buffer("output_bits", torch.tensor(0, **factory_kwargs))
    else:
      self.input_scale = None
      self.input_bits = None
      self.output_scale = None
      self.output_bits = None

  def _clip_or_quantize_activation(
      self,
      x: torch.Tensor,
      min_val: torch.Tensor | None,
      max_val: torch.Tensor | None,
      scale: torch.Tensor | None,
      bits: torch.Tensor | None,
  ) -> torch.Tensor:
    if min_val is not None and max_val is not None:
      return torch.clamp(x, min_val, max_val)
    elif scale is not None and bits is not None and int(bits.item()) > 0:
      return fake_quant_activation(x, scale, int(bits.item()))
    return x

  def forward(
      self,
      einsum_str: str,
      x: torch.Tensor,
      w: torch.Tensor,
      weight_scale: torch.Tensor | None = None,
  ) -> torch.Tensor:
    assert einsum_str == self.einsum_str, f"Expected einsum_str {self.einsum_str}, but got {einsum_str}"
    orig_dtype = x.dtype

    # 1. Activation Clipping / Quantization
    x = self._clip_or_quantize_activation(
        x, self.input_min, self.input_max, self.input_scale, self.input_bits
    )

    x_f32 = x.to(torch.float32)
    w_f32 = w.to(torch.float32)

    # 2. Einsum
    result = torch.einsum(einsum_str, x_f32, w_f32)

    # 3. Weight Dequantization
    if weight_scale is not None:
      result = result * weight_scale.to(torch.float32)

    # 4. Clipping / Quantization on Output Activation
    result = self._clip_or_quantize_activation(
        result, self.output_min, self.output_max, self.output_scale, self.output_bits
    )

    return result.to(orig_dtype)
