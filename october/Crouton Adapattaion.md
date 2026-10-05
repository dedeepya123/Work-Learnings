Problem Statement:
  We have 2D RopE implementation where we are splitting the intermediate tenor into 4 splits and doing cos, sin some opertaions
  <img width="506" height="284" alt="image" src="https://github.com/user-attachments/assets/1cb69d37-ea13-4911-ac01-c90a2af93126" />
But in the memory layout crouton format that tensor 16 in last dim is not using it fully causing perf siiue so what mahematically can we change so crouton can be fully utilized?
``` python

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


For a 64-dimensional head, 2D RoPE divides channels into two 32-dimensional groups corresponding to X and Y positional encoding. Each 32-dimensional group is further split into 16-dimensional real and imaginary components and rotated independently. This results in four 16-dimensional tensors. However, the HTP crouton size is 32 channels. Therefore the arithmetic kernels operate on 16-channel tensors while the hardware is optimized for 32-channel blocks, leading to approximately 50% crouton utilization. The RoPE computation is mathematically correct, but the tensor partitioning does not align with the hardware's preferred execution granularity.
    result = torch.cat(y_parts, dim=-1)
    return result
