from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import hydra
from omegaconf import OmegaConf, open_dict
import optax

from flax.core import FrozenDict
from flax.training import train_state
import utils.flax_utils as flax_utils
from utils.flax_utils import nonpytree_field
from utils.data_utils import postprocess_batch

class StableVAEModel(flax.struct.PyTreeNode):
    vae_state: train_state.TrainState
    obs_normalization: dict[str, Any]
    vae_module: Any = nonpytree_field()
    lr_schedule: Any = nonpytree_field()
    config: dict = nonpytree_field()

    def loss(self, params, batch, rng, use_kl, n_downsample):
        g_rng, rng = jax.random.split(rng)

        img = jnp.concatenate([batch['obs'][rgb_obs][:, 0] for rgb_obs in self.config['rgb_obs']], axis=0).transpose(0, 3, 1, 2)
        z_dist = self.vae_state.apply_fn({"params": params['vae']}, img, method=self.vae_module.encode).latent_dist
        z_rng, rng = jax.random.split(rng)
        hidden_states = z_dist.sample(z_rng)
        pred_img = self.vae_state.apply_fn({"params": params['vae']}, hidden_states, method=self.vae_module.decode).sample

        center_img = img
        mse = jnp.mean((center_img - pred_img) ** 2)
        if use_kl:
            kl = jnp.mean(z_dist.kl()) # (B,) -> ()
        else:
            kl = 0
        loss = mse + self.config['beta'] * kl

        metrics = dict()
        metrics['img_min'] = jnp.min(img)
        metrics['img_max'] = jnp.max(img)
        metrics['img_mean'] = jnp.mean(img)
        metrics['img_std'] = jnp.std(img)
        metrics['loss'] = loss
        metrics['loss_mse'] = mse
        metrics['loss_kl'] = kl
        metrics['z_min'] = jnp.min(hidden_states)
        metrics['z_max'] = jnp.max(hidden_states)
        metrics['z_mean'] = jnp.mean(hidden_states)
        metrics['z_std'] = jnp.std(hidden_states)

        return loss, metrics

    def update(self, batch, rng, step):
        return self.update_step(batch, rng, bool(self.config['use_kl']), self.config['n_downsample'])

    @partial(jax.jit, static_argnames=('use_kl', 'n_downsample'))
    def update_step(self, batch, rng, use_kl, n_downsample):
        batch = postprocess_batch(batch, self.obs_normalization)

        rng, g_rng = jax.random.split(rng)
        params = {"vae": self.vae_state.params}

        grads, metrics = jax.grad(self.loss, has_aux=True)(params, batch, g_rng, use_kl, n_downsample)

        new_vae_state = self.vae_state.apply_gradients(grads=grads['vae'])
        new_vae_state = new_vae_state.replace(ema_params=new_vae_state.apply_ema())
        metrics["vae_lr"] = self.lr_schedule(self.vae_state.step)
        metrics["vae_step"] = self.vae_state.step

        return self.replace(vae_state=new_vae_state), metrics

    def get_metrics(self, batch, rng):
        return self.get_metrics_step(batch, rng, bool(self.config['use_kl']), self.config['n_downsample'])

    @partial(jax.jit, static_argnames=('use_kl', 'n_downsample'))
    def get_metrics_step(self, batch, rng, use_kl, n_downsample):
        batch = postprocess_batch(batch, self.obs_normalization)

        rng, g_rng = jax.random.split(rng)
        params = {"vae": self.vae_state.params}

        _, metrics = self.loss(params, batch, g_rng, use_kl, n_downsample)
        return metrics

    def reconstruct(self, batch, rng, rgb_key):
        batch = postprocess_batch(batch, self.obs_normalization) # maybe jit this?
        batch['obs']['image'] = batch['obs'][rgb_key]
        return self.reconstruct_step(batch, rng)

    @jax.jit
    def reconstruct_step(self, batch, rng):
        img = batch['obs']['image'][:, 0].transpose(0, 3, 1, 2)
        z_dist = self.vae_state.apply_fn({"params": self.vae_state.ema_params}, img, method=self.vae_module.encode).latent_dist
        hidden_states = z_dist.mode()
        pred_img = self.vae_state.apply_fn({"params": self.vae_state.ema_params}, hidden_states, method=self.vae_module.decode).sample
        return pred_img

    def sample(self, rng):
        if self.config['n_downsample'] == 6:
            z_dim = 2
        elif self.config['n_downsample'] == 5:
            z_dim = 4
        elif self.config['n_downsample'] == 4:
            z_dim = 8
        else: 
            raise NotImplementedError
        return self.sample_step(rng, z_dim)

    @partial(jax.jit, static_argnames=('z_dim'))
    def sample_step(self, rng, z_dim):
        noise = jax.random.normal(rng, (4, z_dim, z_dim, self.vae_module.latent_channels))
        pred_img = self.vae_state.apply_fn({"params":  self.vae_state.ema_params}, noise, method=self.vae_module.decode).sample
        return pred_img

    def get_params(self):
        return dict(vae_params=self.vae_state.params, ema_params=self.vae_state.ema_params)

    @classmethod
    def create(
        cls, rng, batch, shape_meta,
        # Hydra Config
        name, vae, rgb_obs, obs_normalization,
        lr, end_lr, warmup_steps, decay_steps, ema_decay,
        use_kl, beta, data_name
    ):
        print(f"Training VAE on RGB observations: {rgb_obs}")
        vae_module = hydra.utils.instantiate(vae)
        init_rng, rng = jax.random.split(rng)
        params = vae_module.init(init_rng, jnp.zeros((2, 3, 64, 64)))['params']
        print(f"vae number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):e}")

        # init vae
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=end_lr,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_lr,
        )
        tx = optax.adam(lr_schedule)
        vae_state = flax_utils.TrainStateEMA.create(
            apply_fn=vae_module.apply,
            params=params,
            tx=tx,
            ema_decay=ema_decay,
            ema_params=params
        )

        # create config with additional variables
        config = flax.core.FrozenDict(dict(
                    rgb_obs=rgb_obs, name=name, use_kl=use_kl,
                    beta=beta, n_downsample=len(vae.down_block_types), data_name=data_name
                    ))
        obs_normalization = flax_utils.cfg_to_jnp(obs_normalization)

        return cls(vae_state, obs_normalization, vae_module, lr_schedule, config)
