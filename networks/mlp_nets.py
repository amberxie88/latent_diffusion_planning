from typing import Any, Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
import flax
import jax

default_init = nn.initializers.xavier_uniform

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

def get_weight_decay_mask(params):
    flattened_params = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.unfreeze(params))

    def decay(k, v):
        if any([(key == 'bias' or 'Input' in key or 'Output' in key)
                for key in k]):
            return False
        else:
            return True

    return flax.core.frozen_dict.freeze(
        flax.traverse_util.unflatten_dict(
            {k: decay(k, v)
             for k, v in flattened_params.items()}))

def torch_init(dtype: Any = jnp.float32):
    """Builds an initializer that returns real uniformly-distributed random arrays.
    https://github.com/pytorch/pytorch/issues/57109

    Args:
    min_range: optional; the lower bound of the uniformly distribution.
    max_range: optional; the upper bound of the uniformly distribution.
    dtype: optional; the initializer's default dtype.

    Returns:
    An initializer that returns arrays whose values are uniformly distributed in
    the range ``[min_range, max_range)``.
    """
    def init(key, shape, dtype=dtype):
        in_feat = shape[0]
        weight_range = 1 / jnp.sqrt(in_feat)
        min_arr = jnp.array(-weight_range, dtype)
        max_arr = jnp.array(weight_range, dtype)
        return jax.random.uniform(key, shape, dtype) * (max_arr - min_arr) + min_arr
    return init

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: str = "relu"
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    weight_init: Optional[str] = 'xavier_uniform'
    bias_init: Optional[str] = None
    use_tanh: Optional[bool] = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.weight_init == 'xavier_uniform':
            kernel_init = default_init()
        elif self.weight_init == 'kaiming_uniform':
            kernel_init = nn.initializers.he_uniform()
        elif self.weight_init == 'torch':
            kernel_init = torch_init()
        else:
            raise NotImplementedError

        if self.bias_init == 'torch':
            bias_init = torch_init()
        elif self.bias_init == 'zeros':
            bias_init = nn.initializers.constant(0.0)
        else:
            bias_init = nn.initializers.constant(0.0)

        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=kernel_init, bias_init=bias_init)(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                # x = self.activations(x)
                if self.activations == 'relu':
                    x = nn.relu(x)
                elif self.activations == 'mish':
                    x = mish(x)
                else:
                    raise NotImplementedError()

        if self.use_tanh:
            x = nn.tanh(x)
        return x