from dataclasses import field
from typing import Tuple, Any

from jax import nn, random
from flax import linen as nn
import jax.numpy as jnp
import math

default_init = nn.initializers.xavier_uniform

class Mish(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * jnp.tanh(nn.softplus(x))

class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.expand_dims(x, axis=-1) * jnp.expand_dims(emb, axis=0)
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(d_model))

    pe = jnp.zeros((d_model, height, width))
    d_model_half = d_model // 2
    div_term = jnp.exp(jnp.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))

    pos_w = jnp.arange(0., width).reshape(-1, 1)
    pos_h = jnp.arange(0., height).reshape(-1, 1)

    pe = pe.at[0:d_model_half:2, :, :].set(jnp.sin(pos_w * div_term).T[:, None, :].repeat(height, axis=1))
    pe = pe.at[1:d_model_half:2, :, :].set(jnp.cos(pos_w * div_term).T[:, None, :].repeat(height, axis=1))
    pe = pe.at[d_model_half::2, :, :].set(jnp.sin(pos_h * div_term).T[:, :, None].repeat(width, axis=2))
    pe = pe.at[d_model_half + 1::2, :, :].set(jnp.cos(pos_h * div_term).T[:, :, None].repeat(width, axis=2))
    
    return pe

class Downsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim, kernel_size=(3,), strides=(2,))(x)

class Upsample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        return nn.ConvTranspose(self.dim, kernel_size=(4,), strides=(2,))(x)


class Conv1dBlock(nn.Module):
    out_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        return nn.Sequential([
            nn.Conv(self.out_channels, kernel_size=(self.kernel_size,), padding=self.kernel_size // 2),
            nn.GroupNorm(self.n_groups),
            Mish(),
        ])(x)

class ConditionalResidualBlock1D(nn.Module):
    out_channels: int
    kernel_size: int
    n_groups: int
    residual_proj: bool

    @nn.compact
    def __call__(self, x, cond):
        residual = x
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)(x) # (B, T, D)     
        cond_channels = self.out_channels * 2
        embed = nn.Sequential([
            Mish(),
            nn.Dense(cond_channels, kernel_init=default_init()),
        ])(cond)
        embed = jnp.expand_dims(embed, axis=1) # (B, 1, 2 * D)
        scale, bias = jnp.split(embed, 2, axis=-1) # (B, D)
        out = scale * out + bias
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)(out)
        
        if self.residual_proj:
            residual_conv = nn.Conv(self.out_channels, kernel_size=(1,), strides=1, padding=0)
            residual = residual_conv(residual)
        return out + residual

class ConditionalUnet1D(nn.Module):
    input_dim: int
    global_cond_dim: int 
    diffusion_step_embed_dim: int = 256
    down_dims: Tuple[int] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    downsample: bool = True

    @nn.compact
    def __call__(self, sample, timestep, global_cond=None, training=True):
        # samples is # (B, T, C)
        timesteps = jnp.array(timestep) if not isinstance(timestep, jnp.ndarray) else timestep
        timesteps = jnp.broadcast_to(timesteps, (sample.shape[0]))
        # (B,)

        diffusion_step_encoder = nn.Sequential([
            SinusoidalPosEmb(self.diffusion_step_embed_dim),
            nn.Dense(self.diffusion_step_embed_dim * 4, kernel_init=default_init()),
            Mish(),
            nn.Dense(self.diffusion_step_embed_dim, kernel_init=default_init()),
        ])

        global_feature = diffusion_step_encoder(timesteps) # (B, dim)

        if global_cond is not None:
            global_feature = jnp.concatenate([global_feature, global_cond], axis=-1)

        x = sample
        h = []

        # down
        for ind, dim_out in enumerate(self.down_dims):
            down_resnet = ConditionalResidualBlock1D(dim_out, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=True)
            down_resnet2 = ConditionalResidualBlock1D(dim_out, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=False)
            x = down_resnet(x, global_feature)
            x = down_resnet2(x, global_feature)
            h.append(x)

            is_last = ind >= (len(self.down_dims) - 1)
            if (self.downsample and not is_last):
                x = Downsample1d(dim_out)(x)

        # mid
        mid_dim = self.down_dims[-1]
        x = ConditionalResidualBlock1D(mid_dim, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=False)(x, global_feature)
        x = ConditionalResidualBlock1D(mid_dim, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=False)(x, global_feature)

        # up
        for ind, dim_in in enumerate(reversed(self.down_dims[:-1])): 
            up_resnet = ConditionalResidualBlock1D(dim_in, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=True)
            up_resnet2 = ConditionalResidualBlock1D(dim_in, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=False)
            x = jnp.concatenate([x, h.pop()], axis=-1) # (B, T, D * 2)
            x = up_resnet(x, global_feature)
            x = up_resnet2(x, global_feature)
            if (self.downsample):
                x = Upsample1d(dim_in)(x)

        final_conv = nn.Sequential([
            Conv1dBlock(self.down_dims[0], kernel_size=self.kernel_size),
            nn.Conv(self.input_dim, kernel_size=(1,)),
        ])

        x = final_conv(x)

        return x
