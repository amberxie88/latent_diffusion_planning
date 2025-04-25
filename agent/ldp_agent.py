from einops import rearrange
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
import orbax
import orbax.checkpoint as ckpt 
from pathlib import Path
import yaml

from diffusers import FlaxAutoencoderKL
from diffusers.schedulers.scheduling_ddpm_flax import FlaxDDPMScheduler
from flax.core import FrozenDict
from networks.mlp_diffusion_nets import MLPDiffusion
from optax._src import linear_algebra
import utils.flax_utils as flax_utils
from utils.flax_utils import nonpytree_field
from utils.data_utils import postprocess_batch, postprocess_batch_obs, normalize_obs, unnormalize_obs

import numpy as np

class LDPAgent(flax.struct.PyTreeNode):
    planner_state : flax_utils.TrainStateEMA
    idm_state: flax_utils.TrainStateEMA
    vae_module: Any = nonpytree_field()
    vae_params: dict[str, FrozenDict]
    obs_normalization: dict[str, Any]
    use_planner: bool
    use_idm: bool
    alpha_planner: float
    alpha_idm: float
    lr_schedule: Any = nonpytree_field()
    planner_noise_scheduler: Any = nonpytree_field()
    planner_noise_state: Any = nonpytree_field() 
    idm_noise_scheduler: Any = nonpytree_field()
    idm_noise_state: Any = nonpytree_field()
    idm_noise_state: Any = nonpytree_field()
    config: dict = nonpytree_field() # check create function for definition

    @jax.jit
    def vae_encode(self, batch):
        new_batch = dict()
        for key in batch.keys():
            if not f'latent_{key}' in self.config['rgb_obs']:
                new_batch[key] = batch[key]
                continue
            init_obs = batch[key]

            B, H = init_obs.shape[:2]
            init_obs = init_obs.reshape(-1, *init_obs.shape[-3:])
            # init_obs transpose from BHWC to BCHW
            init_obs_resize = jnp.transpose(init_obs, (0, 3, 1, 2))
            z = self.vae_module.apply({"params": self.vae_params}, init_obs_resize, method=self.vae_module.encode)['latent_dist'].mean
            feats = z.reshape(B, H, -1)
            feats = jax.lax.stop_gradient(feats)
            feats = normalize_obs({f'latent_{key}': feats}, self.obs_normalization['obs'])[f'latent_{key}']
            new_batch[f'latent_{key}'] = feats 
        return new_batch

    @jax.jit
    def vae_decode(self, feats):
        B, H = feats.shape[:2]
        if self.config['vae_feature_dim'] == 16:
            feats = feats[:, :, :16]
            z = feats.reshape(B * H, 2, 2, 4)
        elif self.config['vae_feature_dim'] == 32:
            feats = feats[:, :, :32]
            z = feats.reshape(B * H, 2, 2, 8) 
        elif self.config['vae_feature_dim'] == 36:
            feats = feats[:, :, :36]
            z = feats.reshape(B * H, 3, 3, 4)
        elif self.config['vae_feature_dim'] == 64:
            feats = feats[:, :, :64]
            z = feats.reshape(B * H, 4, 4, 4)
        key = self.config['rgb_obs'][0]
        z = unnormalize_obs({key: z}, self.obs_normalization['obs'])[key]
        reconstruct = self.vae_module.apply({"params": self.vae_params}, z, method=self.vae_module.decode).sample
        reconstruct = reconstruct.reshape(B, H, *reconstruct.shape[1:]) # B, H, 3, W, W
        return reconstruct


    @jax.jit
    def get_obs_cond(self, batch):
        lowdim_obs_cond = jnp.concatenate([batch[key] for key in self.config['lowdim_obs']], axis=-1).astype(jnp.float32)
        B, H = lowdim_obs_cond.shape[:2]
        lowdim_obs_cond = lowdim_obs_cond.reshape(-1, *lowdim_obs_cond.shape[2:])
        init_obs = jnp.concatenate([batch[key] for key in self.config['rgb_obs']], axis=1)
        image_features = init_obs.reshape(B, H, -1)
        lowdim_obs_cond = lowdim_obs_cond.reshape(B, H, -1)
        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)
        return obs_cond

    @classmethod
    def _get_obs_cond(cls, batch, rgb_obs, lowdim_obs, obs_horizon, init_enc_rng=None):
        lowdim_obs_cond = jnp.concatenate([batch[key] for key in lowdim_obs], axis=-1).astype(jnp.float32)
        B, H = lowdim_obs_cond.shape[:2]
        lowdim_obs_cond = lowdim_obs_cond.reshape(-1, *lowdim_obs_cond.shape[2:])

        # get image features
        init_obs = jnp.concatenate([batch[key] for key in rgb_obs], axis=1)
        image_features = init_obs.reshape(B, H, -1)
        
        lowdim_obs_cond = lowdim_obs_cond.reshape(B, H, -1)
        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)
        return obs_cond

    def plan_loss(self, params, rng, obs_emb, obs_horizon):
        # noising
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (obs_emb.shape[0],), 0, self.config['planner_n_diffusion_steps'])
        next_obs_emb = obs_emb[:, obs_horizon:]
        noise = jax.random.normal(noise_rng, shape=next_obs_emb.shape)
        noisy_next_obs_emb = self.planner_noise_scheduler.add_noise(self.planner_noise_state, next_obs_emb, noise, t)

        obs_cond = obs_emb[:, :obs_horizon, ...] # tbh I'm not sure this is correct
        obs_cond = obs_cond.reshape(obs_emb.shape[0], -1)
        pred_noise = self.planner_state.apply_fn({"params": params}, noisy_next_obs_emb, t, obs_cond)
        loss = jnp.mean((pred_noise - noise) ** 2)
        metrics = dict()
        return loss, metrics

    def idm_loss(self, params, rng, obs_emb, actions, obs_horizon):
        s_sprime = rearrange(jnp.concatenate((obs_emb[:, obs_horizon-1:-1, :], obs_emb[:, obs_horizon:, :]), axis=-1), 'B H D -> (B H) D')
        actions = jnp.reshape(actions[:, :-1], (-1, *actions.shape[2:]))

        # noising
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (actions.shape[0], 1), 0, self.config['idm_n_diffusion_steps'])
        noise = jax.random.normal(noise_rng, shape=actions.shape)
        noisy_actions = self.idm_noise_scheduler.add_noise(self.idm_noise_state, actions, noise, t)
        pred_noise = self.idm_state.apply_fn({"params": params}, s_sprime, noisy_actions, t)
        loss = jnp.mean((pred_noise - noise) ** 2)
        return loss

    def loss(self, params, batch, rng, use_planner, use_idm, obs_horizon):
        obs_emb = self.get_obs_cond(batch['obs'])
        action = batch['actions']

        if use_planner:
            rng, plan_rng = jax.random.split(rng)
            plan_loss, plan_metrics = self.plan_loss(params['planner'], plan_rng, obs_emb, obs_horizon)
            plan_loss = self.alpha_planner * plan_loss
        else:
            plan_loss = 0
            plan_metrics = dict()

        if use_idm:
            rng, idm_rng = jax.random.split(rng)
            idm_loss = self.idm_loss(params['idm'], idm_rng, obs_emb, action, obs_horizon)
            idm_loss = self.alpha_idm * idm_loss
        else:
            idm_loss = 0

        loss = plan_loss + idm_loss

        metrics = dict(plan_loss=plan_loss, idm_loss=idm_loss, loss=loss)
        metrics.update(plan_metrics)

        metrics['emb_min'] = jnp.min(obs_emb)
        metrics['emb_max'] = jnp.max(obs_emb)
        metrics['emb_mean'] = jnp.mean(obs_emb)
        metrics['emb_std'] = jnp.std(obs_emb)

        metrics['action_min'] = jnp.min(action)
        metrics['action_max'] = jnp.max(action)

        # debugging
        debug_metrics = dict()
        for key in batch['obs']:
            debug_metrics[f"{key}_min"] = jnp.min(batch['obs'][key])#.item()
            debug_metrics[f"{key}_max"] = jnp.max(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_mean"] = jnp.mean(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_std"] = jnp.std(batch['obs'][key])#.item()
        metrics.update(debug_metrics)

        return loss, metrics

    def loss_mixed(self, params, batch, mixed_batch, rng, use_planner, use_idm, obs_horizon):
        obs_emb = self.get_obs_cond(batch['obs'])
        action = batch['actions']
        mixed_obs_emb = self.get_obs_cond(mixed_batch['obs'])
        mixed_action = mixed_batch['actions']

        if use_planner:
            rng, plan_rng = jax.random.split(rng)
            plan_loss, plan_metrics = self.plan_loss(params['planner'], plan_rng, obs_emb, obs_horizon)
            plan_loss = self.alpha_planner * plan_loss
        else:
            plan_loss = 0
            plan_metrics = dict()

        if use_idm:
            rng, idm_rng = jax.random.split(rng)
            idm_loss = self.idm_loss(params['idm'], idm_rng, mixed_obs_emb, mixed_action, obs_horizon)
            idm_loss = self.alpha_idm * idm_loss
        else:
            idm_loss = 0

        loss = plan_loss + idm_loss

        metrics = dict(plan_loss=plan_loss, idm_loss=idm_loss, loss=loss)
        metrics.update(plan_metrics)

        metrics['emb_min'] = jnp.min(obs_emb)
        metrics['emb_max'] = jnp.max(obs_emb)
        metrics['emb_mean'] = jnp.mean(obs_emb)
        metrics['emb_std'] = jnp.std(obs_emb)

        metrics['action_min'] = jnp.min(action)
        metrics['action_max'] = jnp.max(action)

        # debugging
        debug_metrics = dict()
        for key in batch['obs']:
            debug_metrics[f"{key}_min"] = jnp.min(batch['obs'][key])#.item()
            debug_metrics[f"{key}_max"] = jnp.max(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_mean"] = jnp.mean(batch['obs'][key])#.item()
            # debug_metrics[f"{key}_std"] = jnp.std(batch['obs'][key])#.item()
        metrics.update(debug_metrics)

        return loss, metrics

    def update(self, batch, rng, step):
        use_planner = bool(self.use_planner) and step % self.config['update_planner_every'] == 0
        use_idm = bool(self.use_idm) and step % self.config['update_idm_every'] == 0
        use_idm = use_idm and step >= self.config['update_idm_after']
        update_planner = self.config['update_planner_until'] < 0 or step < self.config['update_planner_until']
        update_planner = update_planner and step >= self.config['update_planner_after']
        use_planner = use_planner and update_planner
        return self.update_step(batch, rng, use_planner, use_idm, 
                self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('use_planner', 'use_idm', 'obs_horizon'))
    def update_step(self, batch, rng, use_planner, use_idm, obs_horizon):
        batch = postprocess_batch(batch, self.obs_normalization)
        rng, g_rng = jax.random.split(rng)
        
        # get params
        combined_params = dict()
        if use_planner:
            combined_params['planner'] = self.planner_state.params
        if use_idm:
            combined_params['idm'] = self.idm_state.params
        
        # Compute loss and gradients
        grads, metrics = jax.grad(self.loss, has_aux=True)(combined_params, batch, g_rng, use_planner, use_idm, obs_horizon)
        g_norm = linear_algebra.global_norm(grads)
        metrics['g_norm'] = g_norm
        if use_planner:
            planner_grads = grads['planner']
            new_planner_state = self.planner_state.apply_gradients(grads=planner_grads)
            metrics["planner_lr"] = self.lr_schedule(self.planner_state.step)
            metrics["planner_step"] = self.planner_state.step
        else:
            new_planner_state = self.planner_state
            metrics["planner_lr"] = 0
            metrics["planner_step"] = 0
            metrics["noise_diff"] = 0
        if use_idm:
            idm_grads = grads['idm']
            new_idm_state = self.idm_state.apply_gradients(grads=idm_grads)
            metrics["idm_lr"] = self.lr_schedule(self.idm_state.step)
            metrics["idm_step"] = self.idm_state.step
        else:
            new_idm_state = self.idm_state
            metrics["idm_lr"] = 0
            metrics["idm_step"] = 0

        
        return self.replace(planner_state=new_planner_state, 
                            idm_state=new_idm_state), metrics

    def update_mixed(self, batch, mixed_batch, rng, step):
        use_planner = bool(self.use_planner) and step % self.config['update_planner_every'] == 0
        use_idm = bool(self.use_idm) and step % self.config['update_idm_every'] == 0
        use_idm = use_idm and step >= self.config['update_idm_after']
        update_planner = self.config['update_planner_until'] < 0 or step < self.config['update_planner_until']
        update_planner = update_planner and step >= self.config['update_planner_after']
        use_planner = use_planner and update_planner
        return self.update_mixed_step(batch, mixed_batch, rng, use_planner, use_idm, 
                self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('use_planner', 'use_idm', 'obs_horizon'))
    def update_mixed_step(self, batch, mixed_batch, rng, use_planner, use_idm, obs_horizon):
        batch = postprocess_batch(batch, self.obs_normalization)
        mixed_batch = postprocess_batch(mixed_batch, self.obs_normalization)
        rng, g_rng = jax.random.split(rng)
        
        # get params
        combined_params = dict()
        if use_planner:
            combined_params['planner'] = self.planner_state.params
        if use_idm:
            combined_params['idm'] = self.idm_state.params
        
        # Compute loss and gradients
        grads, metrics = jax.grad(self.loss_mixed, has_aux=True)(combined_params, batch, mixed_batch, g_rng, use_planner, use_idm, obs_horizon)
        g_norm = linear_algebra.global_norm(grads)
        metrics['g_norm'] = g_norm
        if use_planner:
            planner_grads = grads['planner']
            new_planner_state = self.planner_state.apply_gradients(grads=planner_grads)
            metrics["planner_lr"] = self.lr_schedule(self.planner_state.step)
            metrics["planner_step"] = self.planner_state.step
        else:
            new_planner_state = self.planner_state
            metrics["planner_lr"] = 0
            metrics["planner_step"] = 0
            metrics["noise_diff"] = 0
        if use_idm:
            idm_grads = grads['idm']
            new_idm_state = self.idm_state.apply_gradients(grads=idm_grads)
            metrics["idm_lr"] = self.lr_schedule(self.idm_state.step)
            metrics["idm_step"] = self.idm_state.step
        else:
            new_idm_state = self.idm_state
            metrics["idm_lr"] = 0
            metrics["idm_step"] = 0

        return self.replace(planner_state=new_planner_state, 
                            idm_state=new_idm_state), metrics
    def get_metrics(self, batch, rng):

        return self.get_metrics_step(batch, rng, 
                bool(self.use_planner), bool(self.use_idm), 
                self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('use_planner', 'use_idm', 'obs_horizon'))
    def get_metrics_step(self, batch, rng, use_planner, use_idm, obs_horizon):
        batch = postprocess_batch(batch, self.obs_normalization)
        rng, g_rng = jax.random.split(rng)

        # get params
        combined_params = dict()
        if use_planner:
            combined_params['planner'] = self.planner_state.params
        if use_idm:
            combined_params['idm'] = self.idm_state.params
        
        # Compute loss and gradients
        loss, metrics = self.loss(combined_params, batch, g_rng, use_planner, use_idm, obs_horizon)  
        return metrics

    def sample_action_from_plan(self, batch, next_plan, eval_rng):
        if 'actions' in batch.keys():
            batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert len(batch.keys()) == 1
            batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)
        batch['obs'] = self.vae_encode(batch['obs'])
        return self.sample_action_from_plan_step(batch, next_plan, eval_rng, self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('obs_horizon'))
    def sample_action_from_plan_step(self, batch, next_plan, eval_rng, obs_horizon):
        # Planner
        for k, v in batch['obs'].items():
            B = v.shape[0]
            break

        start_plan = self.get_obs_cond(batch['obs'])

        # IDM
        s_sprime = rearrange(jnp.concatenate((start_plan, next_plan), axis=-1), 'B H D -> (B H) D')
        transition_emb = jnp.concatenate((s_sprime[:, :self.config['obs_dim']], s_sprime[:, self.config['obs_dim']:]), axis=1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (transition_emb.shape[0], self.config['action_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['idm_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.idm_state.apply_fn({"params": self.idm_state.params}, transition_emb, noisy_action, k)
            noisy_action = self.idm_noise_scheduler.step(self.idm_noise_state, noise_pred, k, noisy_action, s_rng).prev_sample

            return noisy_action, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))
        action = rearrange(noisy_action, '(B H) D -> B H D', B=B)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action

    def sample_action(self, batch, eval_rng):
        if 'actions' in batch.keys():
            batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert len(batch.keys()) == 1
            batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)
        batch['obs'] = self.vae_encode(batch['obs'])
        return self.sample_action_step(batch, eval_rng, self.config['obs_horizon'])

    @partial(jax.jit, static_argnames=('obs_horizon'))
    def sample_action_step(self, batch, eval_rng, obs_horizon):
        # Planner
        for k, v in batch['obs'].items():
            B = v.shape[0]
            break

        plan = self.get_obs_cond(batch['obs'])

        # IDM
        s_sprime = rearrange(jnp.concatenate((plan[:, :-1, :], plan[:, 1:, :]), axis=-1), 'B H D -> (B H) D')
        transition_emb = jnp.concatenate((s_sprime[:, :self.config['obs_dim']], s_sprime[:, self.config['obs_dim']:]), axis=1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (transition_emb.shape[0], self.config['action_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['idm_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.idm_state.apply_fn({"params": self.idm_state.params}, transition_emb, noisy_action, k)
            noisy_action = self.idm_noise_scheduler.step(self.idm_noise_state, noise_pred, k, noisy_action, s_rng).prev_sample

            return noisy_action, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))
        action = rearrange(noisy_action, '(B H) D -> B H D', B=B)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action

    def sample(self, batch, eval_rng):
        return self.sample_viz(batch, eval_rng)

    def sample_viz(self, batch, eval_rng):
        if 'actions' in batch.keys():
            batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert len(batch.keys()) == 1
            batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)

        batch['obs'] = self.vae_encode(batch['obs'])
        action, metrics = self.sample_viz_step(batch, eval_rng, self.config['obs_horizon'])

        # from training batch. not inference
        if metrics['obs_emb'].shape[1] > self.config['obs_horizon']:
            metrics['plan_mse'] = jnp.mean((metrics['noisy_next_obs'] - metrics['obs_emb'][:, self.config['obs_horizon']:, :]) ** 2)
        metrics.pop('obs_emb')
        metrics.pop('noisy_next_obs')
        return action, metrics

    @partial(jax.jit, static_argnames=('obs_horizon'))
    def sample_viz_step(self, batch, eval_rng, obs_horizon):
        # Planner
        for k, v in batch['obs'].items():
            B = v.shape[0]
            break

        obs_emb = self.get_obs_cond(batch['obs'])
        obs_cond = obs_emb[:, :obs_horizon, ...].reshape(obs_emb.shape[0], -1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_next_obs = jax.random.normal(noise_rng, (B, self.config['pred_horizon'], self.config['obs_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['planner_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_next_obs, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.planner_state.apply_fn({"params": self.planner_state.params}, noisy_next_obs, k, obs_cond)
            noisy_next_obs = self.planner_noise_scheduler.step(self.planner_noise_state, noise_pred, k, noisy_next_obs, s_rng).prev_sample

            return noisy_next_obs, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_next_obs, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_next_obs, s_rng))

        start = 0
        end = start + self.config['action_horizon']
        plan = noisy_next_obs[:, start:end, :] # (B, T, D)
        start_state = obs_emb[:, obs_horizon-1:obs_horizon, :] # during inference, only pass obs_horizon imgs
        plan = jnp.concatenate((start_state, plan), axis=1)
        plan_viz = self.vae_decode(plan)

        # IDM
        s_sprime = rearrange(jnp.concatenate((plan[:, :-1, :], plan[:, 1:, :]), axis=-1), 'B H D -> (B H) D')
        transition_emb = jnp.concatenate((s_sprime[:, :self.config['obs_dim']], s_sprime[:, self.config['obs_dim']:]), axis=1)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (transition_emb.shape[0], self.config['action_dim']), dtype=jnp.float32)

        n_diffusion_steps = self.config['idm_n_diffusion_steps']
        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i

            noise_pred = self.idm_state.apply_fn({"params": self.idm_state.params}, transition_emb, noisy_action, k)
            noisy_action = self.idm_noise_scheduler.step(self.idm_noise_state, noise_pred, k, noisy_action, s_rng).prev_sample

            return noisy_action, eval_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))
        action = rearrange(noisy_action, '(B H) D -> B H D', B=B)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action, dict(plan_viz=plan_viz, noisy_next_obs=noisy_next_obs, obs_emb=obs_emb, plan=plan)

    def get_params(self):
        params = dict()
        if self.use_planner:
            params['planner_params'] = self.planner_state.params
        if self.use_idm:
            params['idm_params'] = self.idm_state.params
        return params

    @classmethod
    def create(
        cls, rng, batch, shape_meta,
        # Hydra Config
        name, planner, idm_net, preprocess_time, cond_encoder, 
        vae_pretrain_path, vae_feature_dim,
        use_planner, use_idm,
        lowdim_obs, rgb_obs, obs_normalization, data_name,
        obs_horizon, pred_horizon, action_horizon, 
        planner_n_diffusion_steps, idm_n_diffusion_steps,
        alpha_planner, alpha_idm,
        lr, end_lr, idm_lr, idm_end_lr, 
        warmup_steps, decay_steps,
        update_planner_every, update_idm_every, update_idm_after,
        update_planner_until, update_planner_after,
        grad_clip,
    ):
        # process data info
        lowdim_obs_dim = 0
        for key in lowdim_obs:
            lowdim_obs_dim += int(np.prod(shape_meta['all_shapes'][key]))
        resnet_feature_dim = vae_feature_dim
        vision_feature_dim = resnet_feature_dim * len(rgb_obs) # ResNet18 has output dim of 512
        obs_dim = lowdim_obs_dim + vision_feature_dim
        action_dim = shape_meta['ac_dim']

        # load_vae
        if "ckpt" in vae_pretrain_path:
            # create encoder
            ckpter = orbax.checkpoint.PyTreeCheckpointer()
            raw_restored = ckpter.restore(vae_pretrain_path)
            model_cfg_path = Path(vae_pretrain_path) / '../../.hydra/config.yaml'
            with open(model_cfg_path, 'r') as f:
                model_cfg_path = OmegaConf.create(yaml.safe_load(f))
            vae_module = hydra.utils.instantiate(model_cfg_path.model.vae)
            vae_params = raw_restored['vae_params']
        else:
            vae_module, vae_params = FlaxAutoencoderKL.from_pretrained(vae_pretrain_path)
            vae_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices('gpu')[0]), vae_params)
        print(f"vae number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(vae_params)):e}")

        # get init batch
        rng, init_enc_rng = jax.random.split(rng)
        init_time = jnp.zeros((1,), dtype=jnp.int32)
        init_action = batch['actions']

        # create encoder
        obs_emb = LDPAgent._get_obs_cond(batch['obs'], rgb_obs, lowdim_obs, obs_horizon, init_enc_rng)

        # create planner
        if use_planner:
            rng, init_rng = jax.random.split(rng)
            with open_dict(planner):
                planner.input_dim = obs_dim # important! model obs not action
                planner.global_cond_dim = obs_dim
                planner._convert_ = 'all'
            planner = hydra.utils.instantiate(planner)
            obs_cond = obs_emb[:, :obs_horizon, ...]
            obs_cond = obs_cond.reshape(obs_emb.shape[0], -1)
            planner_params = planner.init(init_rng, obs_emb[:, obs_horizon:], init_time, obs_cond)["params"]
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(planner_params))
            print(f"planner number of parameters: {param_count:e}")

            # create train state
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=end_lr,
                peak_value=lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
            tx = optax.adam(lr_schedule)
            planner_state = flax_utils.TrainStateEMA.create(
                apply_fn=planner.apply,
                params=planner_params,
                tx=tx,
            )
        else:
            planner_state = None

        # create idm
        if use_idm:
            rng, init_idm_rng = jax.random.split(rng)
            with open_dict(idm_net):
                idm_net.out_dim = action_dim
                idm_net._convert_ = 'all'
            idm_net_cls = lambda: hydra.utils.instantiate(idm_net)
            preprocess_time_cls = lambda: hydra.utils.instantiate(preprocess_time)
            cond_encoder_cls = lambda: hydra.utils.instantiate(cond_encoder)

            idm = MLPDiffusion(cond_encoder_cls, idm_net_cls, preprocess_time_cls)

            # maybe should turn this into a separate jit util function
            # it's possible that with obs_horizon, action / s_sprime could be off??
            s_sprime = rearrange(jnp.concatenate((obs_emb[:, obs_horizon-1:-1, :], obs_emb[:, obs_horizon:, :]), axis=-1), 'B H D -> (B H) D')
            transition_emb = jnp.concatenate((s_sprime[:, :obs_dim], s_sprime[:, obs_dim:]), axis=1)
            init_action = init_action[:, :-1, :]
            init_action = init_action.reshape(-1, *init_action.shape[2:])
            idm_params = idm.init(init_idm_rng, transition_emb, init_action, init_time)["params"]
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(idm_params))
            print(f"IDM number of parameters: {param_count:e}")
            
            # create train state
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=idm_end_lr,
                peak_value=idm_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=idm_end_lr,
            )
            tx = optax.adam(lr_schedule)
            idm_state = flax_utils.TrainStateEMA.create(
                apply_fn=idm.apply,
                params=idm_params,
                tx=tx,
            )
        else:
            idm_state = None

        
        # create noise scheduler
        planner_noise_scheduler = FlaxDDPMScheduler(
                            num_train_timesteps=planner_n_diffusion_steps,
                            beta_schedule='squaredcos_cap_v2',
                            clip_sample=True,
                            prediction_type='epsilon'
                            )
        planner_noise_state = planner_noise_scheduler.create_state()
        idm_noise_scheduler = FlaxDDPMScheduler(
                            num_train_timesteps=idm_n_diffusion_steps,
                            beta_schedule='squaredcos_cap_v2',
                            clip_sample=True,
                            prediction_type='epsilon'
                            )
        idm_noise_state = idm_noise_scheduler.create_state()

        # create config with additional variables
        config = flax.core.FrozenDict(dict(
                    planner_n_diffusion_steps=planner_n_diffusion_steps,
                    idm_n_diffusion_steps=idm_n_diffusion_steps,
                    lowdim_obs=lowdim_obs, rgb_obs=rgb_obs, obs_horizon=obs_horizon,
                    name=name, action_dim=shape_meta['ac_dim'],
                    pred_horizon=pred_horizon, action_horizon=action_horizon,
                    obs_dim=obs_dim, 
                    update_planner_every=update_planner_every, update_idm_every=update_idm_every, 
                    update_planner_until=update_planner_until,
                    update_planner_after=update_planner_after, 
                    update_idm_after=update_idm_after, 
                    vae_feature_dim=vae_feature_dim, data_name=data_name
                    ))
        obs_normalization = flax_utils.cfg_to_jnp(obs_normalization)

        return cls(planner_state, idm_state, vae_module, vae_params,
                 obs_normalization, use_planner, use_idm, 
                 alpha_planner, alpha_idm, lr_schedule,
                 planner_noise_scheduler, planner_noise_state, idm_noise_scheduler, idm_noise_state, 
                 config)
