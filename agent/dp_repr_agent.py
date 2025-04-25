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

from diffusers.schedulers.scheduling_ddpm_flax import FlaxDDPMScheduler
from flax.core import FrozenDict
from flax.training import train_state
import utils.flax_utils as flax_utils
import utils.data_utils as data_utils
from utils.flax_utils import nonpytree_field, random_shift_fn
from utils.data_utils import postprocess_batch, postprocess_batch_obs, normalize_obs, unnormalize_obs

class DPVAEAgent(flax.struct.PyTreeNode):
    planner_state: flax_utils.TrainStateEMA
    vae_module: Any = nonpytree_field()
    vae_params: dict[str, FrozenDict]
    obs_normalization: dict[str, Any]
    lr_schedule: Any = nonpytree_field()
    noise_scheduler: Any = nonpytree_field()
    noise_state: Any = nonpytree_field() 
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
        lowdim_obs_cond = jnp.concatenate([batch[key][:, :self.config['obs_horizon']] for key in self.config['lowdim_obs']], axis=-1).astype(jnp.float32)
        B = lowdim_obs_cond.shape[0]
        lowdim_obs_cond = lowdim_obs_cond.reshape(lowdim_obs_cond.shape[0], -1)
        init_obs = jnp.concatenate([batch[key][:, :self.config['obs_horizon']] for key in self.config['rgb_obs']], axis=1)
        image_features = init_obs.reshape(B, -1)
        # reshape back
        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)

        return obs_cond

    @classmethod
    def _get_obs_cond(cls, batch, rgb_obs, lowdim_obs, obs_horizon):
        # this is weird but wasn't sure how else to do this (at least conveniently)
        # I should really reformat this if I'm using it for init
        lowdim_obs_cond = jnp.concatenate([batch[key][:, :obs_horizon] for key in lowdim_obs], axis=-1).astype(jnp.float32)
        B = lowdim_obs_cond.shape[0]
        lowdim_obs_cond = lowdim_obs_cond.reshape(lowdim_obs_cond.shape[0], -1)
        
        init_obs = jnp.concatenate([batch[key][:, :obs_horizon] for key in rgb_obs], axis=1)
        image_features = init_obs.reshape(B, -1)

        obs_cond = jnp.concatenate([image_features, lowdim_obs_cond], axis=-1)
        return obs_cond

    def loss(self, params, batch, rng):
        action = batch['actions']
        obs_emb = self.get_obs_cond(batch['obs'])

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

        metrics['action_min'] = jnp.min(action)
        metrics['action_max'] = jnp.max(action)

        # debugging
        debug_metrics = dict()
        for key in batch['obs']:
            debug_metrics[f"{key}_min"] = jnp.min(batch['obs'][key])#.item()
            debug_metrics[f"{key}_max"] = jnp.max(batch['obs'][key])#.item()
            debug_metrics[f"{key}_mean"] = jnp.mean(batch['obs'][key])#.item()
            debug_metrics[f"{key}_std"] = jnp.std(batch['obs'][key])#.item()
        metrics.update(debug_metrics)
        metrics['loss'] = loss

        return loss, metrics

    def update(self, batch, rng, step):
        if self.config['random_shift'] > 0:
            for key in self.config['rgb_obs']:
                shift_rng, rng = jax.random.split(rng)
                B, T, H, W, C = batch['obs'][key].shape
                obs = batch['obs'][key].reshape(-1, H, W, C)
                obs_aug = random_shift_fn(shift_rng, obs, self.config['random_shift'])
                batch['obs'][key] = obs_aug.reshape(B, T, H, W, C)
        return self.update_step(batch, rng)

    @jax.jit
    def update_step(self, batch, rng):
        batch = postprocess_batch(batch, self.obs_normalization)

        rng, g_rng = jax.random.split(rng)
        combined_params = {"planner": self.planner_state.params}

        grads, metrics = jax.grad(self.loss, has_aux=True)(combined_params, batch, g_rng)

        new_planner_state = self.planner_state.apply_gradients(grads=grads['planner'])
        new_planner_state = new_planner_state.replace(ema_params=new_planner_state.apply_ema())
        metrics["planner_lr"] = self.lr_schedule(self.planner_state.step)
        metrics["planner_step"] = self.planner_state.step
        return self.replace(planner_state=new_planner_state), metrics

    def sample(self, batch, eval_rng):
        if 'actions' in batch.keys():
            batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        else:
            assert len(batch.keys()) == 1
            batch = jax.jit(postprocess_batch_obs)(batch, self.obs_normalization)
        batch['obs'] = self.vae_encode(batch['obs'])
        return self.sample_step(batch, eval_rng, bool(self.config['use_ema']))

    @partial(jax.jit, static_argnames=('use_ema'))
    def sample_step(self, batch, eval_rng, use_ema):
        n_diffusion_steps = self.config['n_diffusion_steps']

        for k, v in batch['obs'].items():
            B = v.shape[0]
            break
        if use_ema:
            planner_params = self.planner_state.ema_params
        else:
            planner_params = self.planner_state.params
        obs_cond = self.get_obs_cond(batch['obs'])
        eval_rng, noise_rng = jax.random.split(eval_rng)
        noisy_action = jax.random.normal(noise_rng, (B, self.config['pred_horizon'], self.config['action_dim']), dtype=jnp.float32)

        def sample_loop(i, args):
            noisy_action, eval_rng = args 
            s_rng, eval_rng = jax.random.split(eval_rng)
            k = n_diffusion_steps - 1 - i
            k_arr = jnp.repeat(jnp.array(k), B)

            noise_pred = self.planner_state.apply_fn({"params": planner_params}, noisy_action, k_arr, obs_cond)
            noisy_action = self.noise_scheduler.step(self.noise_state, noise_pred, k, noisy_action, s_rng).prev_sample
            return noisy_action, s_rng

        s_rng, eval_rng = jax.random.split(eval_rng)
        noisy_action, _ = jax.lax.fori_loop(0, n_diffusion_steps, sample_loop, (noisy_action, s_rng))

        start = 0
        end = self.config['action_horizon']
        action = noisy_action[:, start:end, :] # (B, T, D)
        action = unnormalize_obs(dict(actions=action), self.obs_normalization)['actions']
        return action, dict()

    def get_metrics(self, batch, rng):
        batch = jax.jit(postprocess_batch)(batch, self.obs_normalization)
        metrics = self.get_metrics_step(batch, rng, bool(self.config['use_ema']))
        sample_rng, rng = jax.random.split(rng)
        return metrics

    @partial(jax.jit, static_argnames=('use_ema'))
    def get_metrics_step(self, batch, rng, use_ema):
        rng, g_rng = jax.random.split(rng)
        if use_ema:
            combined_params = {"planner": self.planner_state.ema_params}
        else:
            combined_params = {"planner": self.planner_state.params}

        _, metrics = self.loss(combined_params, batch, g_rng)
        return metrics

    def get_params(self):
        return dict(planner_params=self.planner_state.params,
                    planner_ema_params=self.planner_state.ema_params)

    @classmethod
    def create(
        cls, rng, batch, shape_meta,
        # Hydra Config
        name, planner, lowdim_obs, rgb_obs, obs_normalization,
        obs_horizon, pred_horizon, action_horizon, n_diffusion_steps,
        lr, end_lr, warmup_steps, decay_steps,
        random_shift, use_ema, planner_ema_decay,
        vae_pretrain_path, vae_feature_dim
    ):
        # process data info
        lowdim_obs_dim = 0
        for key in lowdim_obs:
            lowdim_obs_dim += int(np.prod(shape_meta['all_shapes'][key]))
        vision_feature_dim = vae_feature_dim * len(rgb_obs) # ResNet18 has output dim of 512
        obs_dim = lowdim_obs_dim + vision_feature_dim
        action_dim = shape_meta['ac_dim']

        # create model
        # deal with obs_horizon here if setting it!
        with open_dict(planner):
            planner.input_dim = action_dim
            planner.global_cond_dim = obs_dim
            planner._convert_ = 'all'
        planner = hydra.utils.instantiate(planner)

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


        # init planner
        rng, init_plan_rng = jax.random.split(rng, 2)
        init_action = batch['actions']
        obs_emb = DPVAEAgent._get_obs_cond(batch['obs'], rgb_obs, lowdim_obs, obs_horizon)
        init_time = jnp.zeros((init_action.shape[0],), dtype=jnp.int32)
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
                    random_shift=random_shift, use_ema=use_ema, vae_feature_dim=vae_feature_dim
                    ))
        obs_normalization = flax_utils.cfg_to_jnp(obs_normalization)

        return cls(planner_state, vae_module, vae_params, obs_normalization, lr_schedule, noise_scheduler, noise_state, config)
