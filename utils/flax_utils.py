import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
from functools import partial
from omegaconf import OmegaConf
from typing import Any

from jax.scipy.ndimage import map_coordinates
import utils.py_utils as py_utils

nonpytree_field = partial(flax.struct.field, pytree_node=False)

default_init = nn.initializers.xavier_uniform

# https://github.com/NobuoTsukamoto/jax_examples
class TrainStateEMA(train_state.TrainState):
    ema_decay: float = 0.0
    ema_params: Any = None

    def apply_ema(self):
        return jax.tree_util.tree_map(
            lambda ema, param: (ema * self.ema_decay + param * (1.0 - self.ema_decay)),
            self.ema_params,
            self.params,
        )


def calculate_memory_usage(pytree):
    total_memory = 0

    def calculate_array_memory(arr):
        nonlocal total_memory
        if isinstance(arr, jnp.ndarray):
            total_memory += arr.size * arr.dtype.itemsize

    jax.tree_util.tree_map(calculate_array_memory, pytree)
    return total_memory

@jax.jit
def grid_sample_jax(input, grid):
    assert isinstance(input, jax.Array)
    assert isinstance(grid, jax.Array)
    assert len(input.shape) == 4
    assert len(grid.shape) == 4
    assert input.shape[0] == grid.shape[0]
    assert grid.shape[-1] == 2
    # reshape input from BHWC -> BCHW
    input = input.transpose(0, 3, 1, 2)
    B, C, Hi, Wi = input.shape
    _, Ho, Wo, _ = grid.shape

    coordinates = (
        (grid + 1.0) / 2.0 * jnp.array([Hi - 1.0, Wi - 1.0]).reshape(1, 1, 1, 2)
    )
    bilinear_sample_grey = lambda grey, coords: map_coordinates(
        grey, coords.reshape(-1, 2).transpose(), order=1
    )
    bilinear_sample_image = jax.vmap(bilinear_sample_grey, in_axes=[0, None])
    out = jax.vmap(bilinear_sample_image)(input, coordinates).reshape(B, C, Ho, Wo)
     # reshape back from BCHW -> BHWC
    return out.transpose(0, 2, 3, 1)

def random_shift_fn(key, x, pad):
    n, h, w, c = x.shape
    assert h == w
    pad_width = [(0, 0), (pad, pad), (pad, pad), (0, 0)]
    x = jnp.pad(x, pad_width, mode='edge')

    eps = 1.0 / (h + 2 * pad)
    arange = jnp.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad)[:h]
    arange = arange[:, None].repeat(h, axis=1)
    base_grid = jnp.stack([arange, arange.T], axis=-1)
    base_grid = jnp.tile(base_grid[None, ...], (n, 1, 1, 1))

    shift = jax.random.randint(key, (n, 1, 1, 2), 0, 2 * pad + 1)
    shift = shift * (2.0 / (h + 2 * pad))
    grid = base_grid + shift
    x_shifted = grid_sample_jax(x, grid)
    
    return x_shifted

def cfg_to_jnp(obs_normalization):
    if obs_normalization is None:
        return None
    return py_utils.AttrDict(OmegaConf.to_container(obs_normalization, resolve=True)).leaf_apply(lambda x: jnp.array(x) if not isinstance(x, int) else x).as_nested_dict()