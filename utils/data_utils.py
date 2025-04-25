from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

def postprocess_clip(val, min_val, max_val):
    return jnp.clip(val, min_val, max_val)

def postprocess_normalize_bounds(key, val, min_val, max_val, normalize):
    if normalize:
        val = (val - min_val) / (max_val - min_val) * 2 - 1
    else:
        val = (val + 1) / 2
        val = val * (max_val - min_val) + min_val
        val = jnp.clip(val, min_val, max_val) # I think there are some floating point errs
    return val

def normalize_obs(batch, obs_normalization):
    return normalize_unnormalize_obs(batch, obs_normalization, normalize=True)

def unnormalize_obs(batch, obs_normalization):
    return normalize_unnormalize_obs(batch, obs_normalization, normalize=False)

def normalize_unnormalize_obs(batch, obs_normalization, normalize):
    assert set(batch.keys()).issubset(obs_normalization), f"obs_normalization keys {obs_normalization.keys()} do not match batch keys {batch.keys()}"
    
    new_batch = dict()
    for m in batch:
        if "mean" in obs_normalization[m]:
            raise NotImplementedError
        elif "min" in obs_normalization[m]:
            min_val = obs_normalization[m]["min"]
            max_val = obs_normalization[m]["max"]

            # check shape consistency
            if isinstance(min_val, int):
                pass
            else:
                shape_len_diff = len(batch[m].shape) - len(max_val.shape)
                assert shape_len_diff in [0, 1, 2, 3, 4, 5], "shape length mismatch in normalize_obs"
                assert batch[m].shape[shape_len_diff:] == max_val.shape, f"shape mismatch in normalize obs. {m} {batch[m].shape}, {max_val.shape}"

                # handle case where obs dict is not batched by removing stats batch dimension
                if shape_len_diff == 1:
                    min_val = jnp.expand_dims(min_val, axis=0)
                    max_val = jnp.expand_dims(max_val, axis=0)
                elif shape_len_diff == 2:
                    min_val = jnp.expand_dims(jnp.expand_dims(min_val, axis=0), axis=0)
                    max_val = jnp.expand_dims(jnp.expand_dims(max_val, axis=0), axis=0)
                elif shape_len_diff == 3:
                    min_val = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(min_val, axis=0), axis=0), axis=0)
                    max_val = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(max_val, axis=0), axis=0), axis=0)
                elif shape_len_diff == 4:
                    min_val = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(min_val, axis=0), axis=0), axis=0), axis=0)
                    max_val = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(max_val, axis=0), axis=0), axis=0), axis=0)
                elif shape_len_diff == 5:
                    min_val = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(min_val, axis=0), axis=0), axis=0), axis=0), axis=0)
                    max_val = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(max_val, axis=0), axis=0), axis=0), axis=0), axis=0)

            new_batch[m] = postprocess_normalize_bounds(m, batch[m], min_val, max_val, normalize=normalize)
        elif "clip_min" in obs_normalization[m]:
            min_val = obs_normalization[m]["clip_min"]
            max_val = obs_normalization[m]["clip_max"]

            new_batch[m] = postprocess_clip(batch[m], min_val, max_val)
        else:
            raise NotImplementedError
    return new_batch

def postprocess_batch(batch, obs_normalization):
    new_batch = dict()
    new_batch["obs"] = normalize_obs(batch["obs"], obs_normalization["obs"])
    new_batch["actions"] = normalize_obs(dict(actions=batch["actions"]), obs_normalization)["actions"]
    return new_batch

def postprocess_batch_obs(batch, obs_normalization):
    # THIS IS UGLY! But I'm too lazy to fix.
    new_batch = dict()
    new_batch["obs"] = normalize_obs(batch["obs"], obs_normalization["obs"])
    return new_batch


# batched version of https://github.com/ARISE-Initiative/robosuite/blob/74981fd347660205efb4f8c0005e28ec621f5304/robosuite/utils/transform_utils.py#L490
def quat2axisangle_batch(quats):
    """
    Converts a double batch of quaternions to axis-angle format.
    Returns unit vector directions scaled by their respective angles in radians.

    Args:
        quats (np.array): (N, M, 4) array of quaternions (x, y, z, w)

    Returns:
        np.array: (N, M, 3) array of axis-angle exponential coordinates
    """
    # Clip w component to be within [-1, 1]
    quats[..., 3] = np.clip(quats[..., 3], -1.0, 1.0)

    # Compute denominator
    den = np.sqrt(1.0 - quats[..., 3] ** 2)

    # Find zero rotations (where den is close to 0)
    zero_rotation = np.isclose(den, 0.0)

    # Initialize result
    axis_angles = np.zeros((quats.shape[0], quats.shape[1], 3))

    # Avoid division by zero for zero rotations
    valid = ~zero_rotation
    axis_angles[valid] = (quats[valid, :3] * 2.0 * np.arccos(quats[valid, 3][..., None])) / den[valid][..., None]
    return axis_angles

