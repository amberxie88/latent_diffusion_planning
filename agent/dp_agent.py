from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
import chex
import flax
import flax.linen as nn
import hydra
from omegaconf import OmegaConf, open_dict
import optax
import utils.flax_utils as flax_utils
import utils.data_utils as data_utils

from diffusers.schedulers.scheduling_ddpm_flax import FlaxDDPMScheduler
from flax.core import FrozenDict
from flax.training import train_state
from utils.flax_utils import nonpytree_field
from utils.data_utils import postprocess_batch, postprocess_batch_obs, unnormalize_obs

class DPAgent(flax.struct.PyTreeNode):
    planner_state: flax_utils.TrainStateEMA
    encoder_state_dict: dict[str, flax_utils.TrainStateEMA]
    obs_normalization: dict[str, Any]
    lr_schedule: Any = nonpytree_field()
    noise_scheduler: Any = nonpytree_field()
    noise_state: Any = nonpytree_field() 
    config: dict = nonpytree_field() # check create function for definition

    @partial(jax.jit, static_argnames=('shared_encoder'))
    def get_obs_cond(self, encoder_params, batch, shared_encoder):
        lowdim_obs_cond = jnp.concatenate([batch[key][:, :self.config['obs_horizon']] for key in self.config['lowdim_obs']], axis=-1).astype(jnp.float32)
        B = lowdim_obs_cond.shape[0]
        lowdim_obs_cond = lowdim_obs_cond.reshape(lowdim_obs_cond.shape[0], -1)
        if shared_encoder:
            init_obs = jnp.concatenate([batch[key][:, :self.config['obs_horizon']] for key in self.config['rgb_obs']], axis=1)
            init_obs = init_obs.reshape(-1, *init_obs.shape[-3:])
            image_features = self.encoder_state_dict['shared'].apply_fn({"params": encoder_params['shared']}, init_obs)
            image_features = image_features.reshape(B, -1)
        else:
            image_feature_lst = []
            for key in self.config['rgb_obs']:
                init_obs = batch[key][:, :self.config['obs_horizon']]
                init_obs = init_obs.reshape(-1, *init_obs.shape[-3:])
                image_feature = self.encoder_state_dict[key].apply_fn({"params": encoder_params[key]}, init_obs) 
                image_feature = image_feature.reshape(B, -1)
                image_feature_lst.append(image_feature)
            image_features = jnp.concatenate(image_feature_lst, axis=-1)

        # reshape back
        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)
        return obs_cond

    @classmethod
    def _get_obs_cond(cls, encoder_dict, batch, rgb_obs, lowdim_obs, obs_horizon, shared_encoder, init_enc_rng):
        lowdim_obs_cond = jnp.concatenate([batch[key][:, :obs_horizon] for key in lowdim_obs], axis=-1).astype(jnp.float32)
        B = lowdim_obs_cond.shape[0]
        lowdim_obs_cond = lowdim_obs_cond.reshape(lowdim_obs_cond.shape[0], -1)
        
        # initialize encoder
        encoder_params_dict = dict()
        if shared_encoder:
            init_obs = jnp.concatenate([batch[key][:, :obs_horizon] for key in rgb_obs], axis=1)
            init_obs = init_obs.reshape(-1, *init_obs.shape[-3:])
            assert init_obs.shape[-1] == 3
            encoder_params = encoder_dict['shared'].init(init_enc_rng, init_obs)["params"]
            encoder_params_dict['shared'] = encoder_params
            image_features = encoder_dict['shared'].apply({"params": encoder_params}, init_obs)
            image_features = image_features.reshape(B, -1)
        else:
            image_feature_lst = []
            for key in rgb_obs:
                new_rng, init_enc_rng = jax.random.split(init_enc_rng)
                init_obs = batch[key][:, :obs_horizon]
                init_obs = init_obs.reshape(-1, *init_obs.shape[-3:])
                assert init_obs.shape[-1] == 3
                encoder_params = encoder_dict[key].init(new_rng, init_obs)["params"]
                encoder_params_dict[key] = encoder_params
                image_feature = encoder_dict[key].apply({"params": encoder_params}, init_obs)
                image_feature = image_feature.reshape(B, -1)
                image_feature_lst.append(image_feature)
            image_features = jnp.concatenate(image_feature_lst, axis=-1)

        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)
        return obs_cond, encoder_params_dict

    def loss(self, params, batch, rng, shared_encoder):
        debug_metrics = dict()
        action = batch['actions']
        debug_metrics['action_min'] = jnp.min(action)
        debug_metrics['action_max'] = jnp.max(action)
        obs_emb = self.get_obs_cond(params['encoder'], batch['obs'], shared_encoder)

        # noising
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (action.shape[0],), 0, self.config['n_diffusion_steps'])
        noise = jax.random.normal(noise_rng, shape=action.shape)
        noisy_actions = self.noise_scheduler.add_noise(self.noise_state, action, noise, t)

        pred_noise = self.planner_state.apply_fn({"params": params['planner']}, noisy_actions, t, obs_emb)
        loss = jnp.mean((pred_noise - noise) ** 2)

        metrics = dict()
        metrics['obs_min'] = jnp.min(obs_emb)
        metrics['obs_max'] = jnp.max(obs_emb)
        metrics['obs_mean'] = jnp.mean(obs_emb)
        metrics['obs_std'] = jnp.std(obs_emb)
        metrics['loss'] = loss

        return loss, metrics

    def update(self, batch, rng, step):
        return self.update_step(batch, rng, bool(self.config['shared_encoder']))

    @partial(jax.jit, static_argnames=('shared_encoder'))
    def update_step(self, batch, rng, shared_encoder):
        batch = postprocess_batch(batch, self.obs_normalization)

        rng, g_rng = jax.random.split(rng)
        encoder_params_dict = {key: self.encoder_state_dict[key].params for key in self.encoder_state_dict}
        combined_params = {"planner": self.planner_state.params, "encoder": encoder_params_dict}

        grads, metrics = jax.grad(self.loss, has_aux=True)(combined_params, batch, g_rng, shared_encoder)

        new_planner_state = self.planner_state.apply_gradients(grads=grads['planner'])
        new_planner_state = new_planner_state.replace(ema_params=new_planner_state.apply_ema())
        metrics["planner_lr"] = self.lr_schedule(self.planner_state.step)
        metrics["planner_step"] = self.planner_state.step
        new_encoder_state_dict = dict()
        for key in self.encoder_state_dict.keys():
            new_encoder_state = self.encoder_state_dict[key].apply_gradients(grads=grads['encoder'][key])
            new_encoder_state = new_encoder_state.replace(ema_params=new_encoder_state.apply_ema())
            new_encoder_state_dict[key] = new_encoder_state
            metrics[f"enc_{key}_lr"] = self.lr_schedule(self.encoder_state_dict[key].step)
            metrics[f"enc_{key}_step"] = self.encoder_state_dict[key].step
        return self.replace(planner_state=new_planner_state, encoder_state_dict=new_encoder_state_dict), metrics

    def sample_action(self, batch, rng):
        return self.sample(batch, rng)

    def sample(self, batch, eval_rng):
        if 'actions' in batch.keys():
            in_batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert len(batch.keys()) == 1
            in_batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)
        action, metrics = self.sample_step(in_batch, eval_rng, bool(self.config['shared_encoder']))
        batch_metrics = dict()
        for k, v in in_batch['obs'].items():
            batch_metrics[f'{k}_min'] = jnp.min(v)
            batch_metrics[f'{k}_max'] = jnp.max(v)
        metrics.update(batch_metrics)
        return action, metrics

    @partial(jax.jit, static_argnames=('shared_encoder'))
    def sample_step(self, batch, eval_rng, shared_encoder):
        metrics = dict()
        n_diffusion_steps = self.config['n_diffusion_steps']

        for k, v in batch['obs'].items():
            B = v.shape[0]
            break
        encoder_params_dict = {key: self.encoder_state_dict[key].params for key in self.encoder_state_dict}
        planner_params = self.planner_state.params
        obs_emb = self.get_obs_cond(encoder_params_dict, batch['obs'], shared_encoder)
        metrics['obs_min'] = jnp.min(obs_emb)
        metrics['obs_max'] = jnp.max(obs_emb)
        metrics['obs_mean'] = jnp.mean(obs_emb)
        metrics['obs_std'] = jnp.std(obs_emb)
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (B, self.config['pred_horizon'], self.config['action_dim']), dtype=jnp.float32)

        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i
            k_arr = jnp.repeat(jnp.array(k), B)

            noise_pred = self.planner_state.apply_fn({"params": planner_params}, noisy_action, k_arr, obs_emb)
            noisy_action = self.noise_scheduler.step(self.noise_state, noise_pred, k, noisy_action, s_rng).prev_sample
            return noisy_action, s_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))

        start = 0
        end = self.config['action_horizon']
        action = noisy_action[:, start:end, :] # (B, T, D)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action, metrics

    def get_metrics(self, batch, rng):
        batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        metrics = self.get_metrics_step(batch, rng, bool(self.config['shared_encoder']))
        sample_rng, rng = jax.random.split(rng)
        return metrics

    @partial(jax.jit, static_argnames=('shared_encoder'))
    def get_metrics_step(self, batch, rng, shared_encoder):
        rng, g_rng = jax.random.split(rng)
        encoder_params_dict = {key: self.encoder_state_dict[key].params for key in self.encoder_state_dict}
        combined_params = {"planner": self.planner_state.params, "encoder": encoder_params_dict}

        _, metrics = self.loss(combined_params, batch, g_rng, shared_encoder)
        return metrics

    def get_params(self):
        encoder_params_dict = {f'{key}_params': self.encoder_state_dict[key].params for key in self.encoder_state_dict}
        encoder_ema_params_dict = {f'{key}_params': self.encoder_state_dict[key].ema_params for key in self.encoder_state_dict}
        return dict(planner_params=self.planner_state.params, encoder_params=encoder_params_dict,
                    planner_ema_params=self.planner_state.ema_params, encoder_ema_params=encoder_params_dict)

    @classmethod
    def create(
        cls, rng, batch, shape_meta,
        # Hydra Config
        name, planner, encoder, lowdim_obs, rgb_obs, obs_normalization,
        obs_horizon, pred_horizon, action_horizon, n_diffusion_steps,
        lr, end_lr, warmup_steps, decay_steps, shared_encoder,
        planner_ema_decay, encoder_ema_decay,
    ):
        # process data info
        lowdim_obs_dim = 0
        for key in lowdim_obs:
            lowdim_obs_dim += int(np.prod(shape_meta['all_shapes'][key]))
        vision_feature_dim = 512 * len(rgb_obs) # ResNet18 has output dim of 512
        obs_dim = lowdim_obs_dim + vision_feature_dim
        action_dim = shape_meta['ac_dim']

        # create model
        with open_dict(planner):
            planner.input_dim = action_dim
            planner.global_cond_dim = -1 # not used 
            planner._convert_ = 'all'
        planner = hydra.utils.instantiate(planner)

        # create encoder
        encoder_dict = dict()
        if shared_encoder:
            encoder_dict['shared'] = hydra.utils.instantiate(encoder)
        else:
            for key in rgb_obs:
                encoder_dict[key] = hydra.utils.instantiate(encoder)

        # init encoder
        rng, init_plan_rng, init_enc_rng = jax.random.split(rng, 3)
        init_action = batch['actions']
        init_time = jnp.zeros((init_action.shape[0],), dtype=jnp.int32)
        obs_emb, encoder_params_dict = DPAgent._get_obs_cond(encoder_dict, batch['obs'], rgb_obs, lowdim_obs, obs_horizon, shared_encoder, init_enc_rng)
        encoder_state_dict = dict()
        for key in encoder_params_dict:
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(encoder_params_dict[key]))
            print(f"encoder ({key}) number of parameters: {param_count:e}")
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=end_lr,
                peak_value=lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
            tx = optax.adam(lr_schedule)
            encoder_state = flax_utils.TrainStateEMA.create(
                apply_fn=encoder_dict[key].apply,
                params=encoder_params_dict[key],
                tx=tx,
                ema_decay=encoder_ema_decay,
                ema_params=encoder_params_dict[key]
            )
            encoder_state_dict[key] = encoder_state

        # init planner
        planner_params = planner.init(init_plan_rng, init_action, init_time, obs_emb)["params"]
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(planner_params))
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
            ema_decay=planner_ema_decay,
            ema_params=planner_params,
        )
        print(f"planner number of parameters: {param_count:e}")

        # create noise scheduler
        noise_scheduler = FlaxDDPMScheduler(
                            num_train_timesteps=n_diffusion_steps,
                            beta_schedule='squaredcos_cap_v2',
                            clip_sample=True,
                            prediction_type='epsilon'
                            )
        noise_state = noise_scheduler.create_state()

        # create config with additional variables
        config = flax.core.FrozenDict(dict(
                    n_diffusion_steps=n_diffusion_steps, 
                    lowdim_obs=lowdim_obs, rgb_obs=rgb_obs, obs_horizon=obs_horizon,
                    name=name, action_dim=shape_meta['ac_dim'],
                    pred_horizon=pred_horizon, action_horizon=action_horizon,
                    shared_encoder=shared_encoder
                    ))
        obs_normalization = flax_utils.cfg_to_jnp(obs_normalization)

        return cls(planner_state, encoder_state_dict, obs_normalization, lr_schedule, noise_scheduler, noise_state, config)
