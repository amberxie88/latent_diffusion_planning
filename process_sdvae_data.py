import hydra
import h5py
import numpy as np
import os
from pathlib import Path
import psutil
import time
from tqdm import tqdm
import wandb
import yaml

import jax
import jax.numpy as jnp
import flax
import orbax 
import orbax.checkpoint as ckpt 
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict

from diffusers import FlaxAutoencoderKL
from functools import partial

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        # setup
        self.cfg = cfg
        self.seed = cfg.seed

        # data
        self.data = hydra.utils.instantiate(cfg.data)

    def load_vae(self):
        if self.cfg.pretrain_path is not None:
            vae_module, vae_params = FlaxAutoencoderKL.from_pretrained(self.cfg.pretrain_path)
            return vae_module, vae_params
        if self.cfg.restore_snapshot_path is not None:
            # create encoder
            ckpter = orbax.checkpoint.PyTreeCheckpointer()
            raw_restored = ckpter.restore(self.cfg.restore_snapshot_path)
            model_cfg_path = Path(self.cfg.restore_snapshot_path) / '../../.hydra/config.yaml'
            with open(model_cfg_path, 'r') as f:
                model_cfg_path = OmegaConf.create(yaml.safe_load(f))
            vae_module = hydra.utils.instantiate(model_cfg_path.model.vae)
            return vae_module, raw_restored['vae_params']

    def run(self):
        if "rm" in self.cfg.data.name:
            self.run_rm()
        elif "aloha" in self.cfg.data.name:
            self.run_aloha()

    def run_rm(self):
        # create hdf5 file
        dataset_path = self.work_dir / 'latent.hdf5'
        data_writer = h5py.File(dataset_path, "w")
        data_grp = data_writer.create_group("data")

        vae_module, vae_params = self.load_vae()
        # single-GPU training. For multi-GPU, see flax.jax_utils.replicate
        vae_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices('gpu')[0]), vae_params)

        @jax.jit
        def resize_original(obs):
            obs_resize = jax.image.resize(image=obs, shape=(3,256,256), method="bilinear")
            return obs_resize

        @jax.jit
        def encode(obs):
            z = vae_module.apply({"params": vae_params}, obs, method=vae_module.encode)['latent_dist'].mean
            return z
        resize = jax.vmap(resize_original)

        data = self.data.train_dataset
        min_z, max_z = 0, 0
        for ep in tqdm(data.demos):
            obs_traj = {k: data.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in data.rgb_keys}
            obs_end = {k: data.hdf5_file["data/{}/next_obs/{}".format(ep, k)][-1].astype('float32') for k in data.rgb_keys}
            # concat obs_traj and obs_end
            for k in obs_traj.keys():
                obs_traj[k] = np.concatenate([obs_traj[k], obs_end[k][None]], axis=0)
                
            ep_data_grp = data_grp.create_group(ep)
            for rgb_key, obs_arr in obs_traj.items():
                obs_arr = jnp.array(obs_arr)
                obs_arr = jax.device_put(obs_arr, jax.devices('gpu')[0])
                obs_arr = obs_arr / 255
                obs_arr = (obs_arr - 0.5) / 0.5
                obs_arr = obs_arr.transpose(0,3,1,2)
                if self.cfg.pretrain_path is not None:
                    obs_arr = resize(obs_arr)
                # encode with shards of size self.cfg.shard
                zs = []
                for i in range(0, len(obs_arr), self.cfg.shard):
                    sharded = obs_arr[i:min(i+self.cfg.shard, len(obs_arr))]
                    if sharded.shape[0] == self.cfg.shard:
                        z = encode(sharded)
                        # reconstruct = vae_module.apply({"params": vae_params}, z, method=vae_module.decode).sample
                    else:
                        n_extra = self.cfg.shard - sharded.shape[0]
                        if self.cfg.pretrain_path is not None:
                            pad = jnp.zeros((n_extra, 3, 256, 256))
                        else:
                            pad = jnp.zeros((n_extra, 3, sharded.shape[-1], sharded.shape[-1]))
                        sharded_pad = jnp.concatenate([sharded, pad], axis=0)
                        z = encode(sharded_pad)
                        z = z[:sharded.shape[0]]
                    zs.append(z)
                zs = jnp.concatenate(zs, axis=0)
                min_z = min(min_z, jnp.min(zs))
                max_z = max(max_z, jnp.max(zs))
                ep_data_grp.create_dataset("latent/{}".format(rgb_key), data=np.array(zs))

        data_grp.attrs['total'] = len(data.demos)
        data_grp.attrs['min_z'] = min_z
        data_grp.attrs['max_z'] = max_z
            
        print(f"min_z: {min_z}, max_z: {max_z}")
        print(f"done processing dataset! {dataset_path}")


    def run_aloha(self):
        # create hdf5 file
        dataset_path = self.work_dir / 'latent.hdf5'
        data_writer = h5py.File(dataset_path, "w")
        data_grp = data_writer.create_group("data")

        vae_module, vae_params = self.load_vae()
        # single-GPU training. For multi-GPU, see flax.jax_utils.replicate
        vae_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices('gpu')[0]), vae_params)

        @jax.jit
        def resize_original(obs):
            obs_resize = jax.image.resize(image=obs, shape=(3,256,256), method="bilinear")
            return obs_resize

        @jax.jit
        def encode(obs):
            z = vae_module.apply({"params": vae_params}, obs, method=vae_module.encode)['latent_dist'].mean
            return z
        resize = jax.vmap(resize_original)

        data = self.data.train_dataset
        min_z, max_z = 0, 0
        for ep in tqdm(data.demos):
            obs_traj = {k: data.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in data.rgb_keys}
                
            ep_data_grp = data_grp.create_group(ep)
            for rgb_key, obs_arr in obs_traj.items():
                obs_arr = jnp.array(obs_arr)
                obs_arr = jax.device_put(obs_arr, jax.devices('gpu')[0])
                obs_arr = obs_arr / 255
                obs_arr = (obs_arr - 0.5) / 0.5
                obs_arr = obs_arr.transpose(0,3,1,2)
                if self.cfg.pretrain_path is not None:
                    obs_arr = resize(obs_arr)
                # encode with shards of size self.cfg.shard
                zs = []
                for i in range(0, len(obs_arr), self.cfg.shard):
                    sharded = obs_arr[i:min(i+self.cfg.shard, len(obs_arr))]
                    if sharded.shape[0] == self.cfg.shard:
                        z = encode(sharded)
                        # reconstruct = vae_module.apply({"params": vae_params}, z, method=vae_module.decode).sample
                    else:
                        n_extra = self.cfg.shard - sharded.shape[0]
                        if self.cfg.pretrain_path is not None:
                            pad = jnp.zeros((n_extra, 3, 256, 256))
                        else:
                            pad = jnp.zeros((n_extra, 3, sharded.shape[-1], sharded.shape[-1]))
                        sharded_pad = jnp.concatenate([sharded, pad], axis=0)
                        z = encode(sharded_pad)
                        z = z[:sharded.shape[0]]
                    zs.append(z)
                zs = jnp.concatenate(zs, axis=0)
                min_z = min(min_z, jnp.min(zs))
                max_z = max(max_z, jnp.max(zs))
                ep_data_grp.create_dataset("latent/{}".format(rgb_key), data=np.array(zs))

        data_grp.attrs['total'] = len(data.demos)
        data_grp.attrs['min_z'] = min_z
        data_grp.attrs['max_z'] = max_z
            
        print(f"min_z: {min_z}, max_z: {max_z}")
        print(f"done processing dataset! {dataset_path}")


@hydra.main(config_path='.', config_name='process_sdvae_data')
def main(cfg):
    # create logger
    if cfg.use_wandb:
        import omegaconf
        wandb.init(entity=YOUR_ENTITY, project='latent_diffusion_planning', group=cfg.experiment_folder,
                    name=cfg.experiment_name,tags=[cfg.experiment_folder], sync_tensorboard=True)
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )

    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()