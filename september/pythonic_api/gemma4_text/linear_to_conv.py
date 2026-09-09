
import warnings

import torch

from transformers.models.gemma4.quantization_gemma4 import Gemma4QuantizableLinear

from qairt.experimental.pipeline.torch.common.adaptations.linears_to_conv import rsetattr

from .quant_utils import clip_only_fake_quant


def _get_weight_and_bias(mod: Gemma4QuantizableLinear):
    weight = mod.weight
    if mod.weight_scale is not None:
        weight = weight * mod.weight_scale
    return weight, None


class QcGemma4ConvInplaceLinear(torch.nn.Conv2d):

    def __init__(self, mod):
        if isinstance(mod, Gemma4QuantizableLinear):
            weight, bias = _get_weight_and_bias(mod)
        elif isinstance(mod, torch.nn.Linear):
            weight, bias = mod.weight, mod.bias
        else:
            raise TypeError(
                f"{type(self).__name__} expects a Gemma4QuantizableLinear or Linear module, "
                f"got {type(mod).__name__}"
            )

        self.out_features, self.in_features = weight.shape
        device = weight.data.device
        weight_dtype = weight.dtype

        init_dtype = weight_dtype if weight_dtype.is_floating_point else torch.float32

        super().__init__(
            self.in_features,
            self.out_features,
            1,
            dtype=init_dtype,
            bias=bias is not None,
            device=device,
        )

        if not weight_dtype.is_floating_point:
            del self.weight
            self.register_buffer("weight", weight.data[:, :, None, None].clone())
        else:
            self.weight.data.copy_(weight.data[:, :, None, None])
        if bias is not None:
            assert self.bias is not None
            self.bias.data.copy_(bias.data)


        self.input_bits = None
        self.output_bits = None

        self.register_buffer("input_scale", None)
        self.register_buffer("output_scale", None)
        if getattr(mod, "input_bits", False) and getattr(mod, "output_bits", False):
            assert mod.input_bits > 0, "input_bits must be positive if set"
            assert mod.output_bits > 0, "output_bits must be positive if set"
            assert mod.input_scale > 0, "input_scale must be positive if set"
            assert mod.output_scale > 0, "output_scale must be positive if set"
            self.input_bits = mod.input_bits
            self.output_bits = mod.output_bits
            self.input_scale = mod.input_scale
            self.output_scale = mod.output_scale

    def forward(self, x: torch.Tensor):
        if self.input_bits is not None and self.input_scale is not None:
            x = clip_only_fake_quant(x, self.input_scale, int(self.input_bits.item()))

        ndim = x.ndim
        if ndim == 2:
            x = x.unsqueeze(0).unsqueeze(-1).permute(0, 2, 3, 1)
        elif ndim == 3:
            x = x.unsqueeze(-1).permute(0, 2, 3, 1)
        elif ndim == 4:
            x = x.permute(0, 3, 1, 2)
            warnings.warn(
                f"{type(self).__name__} received an unexpected 4d input, assuming channels-last and proceeding."
            )
        else:
            raise NotImplementedError(f"{type(self).__name__} could not handle input with shape {x.shape}")

        x = super().forward(x)

        if ndim == 2:
            x = x.permute(0, 3, 1, 2).squeeze(-1).squeeze(0)
        elif ndim == 3:
            x = x.permute(0, 3, 1, 2).squeeze(-1)
        elif ndim == 4:
            x = x.permute(0, 2, 3, 1)

        x = x.contiguous()

        if self.output_bits is not None and self.output_scale is not None:
            x = clip_only_fake_quant(x, self.output_scale, int(self.output_bits.item()))

        return x


def replace_linears_with_convs(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, Gemma4QuantizableLinear)):
            conv_layer = QcGemma4ConvInplaceLinear(module)
            rsetattr(model, name, conv_layer)

    return model
