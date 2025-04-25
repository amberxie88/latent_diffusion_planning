import glob
import hydra
from hydra.core.override_parser.overrides_parser import OverridesParser
import numpy as np
import os
from pathlib import Path
import re
import time
import yaml

import jax
import jax.numpy as jnp
import jaxlib
import flax
from flax.training import train_state, orbax_utils
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omegaconf
from omegaconf import OmegaConf, open_dict, dictconfig
import optax
import orbax
import utils.aloha_env_utils as aloha_env_utils 
import utils.rm_env_utils as rm_env_utils
from utils.logger import Logger, MeterDict
import utils.py_utils as py_utils


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.video_dir = self.work_dir / 'video'
        self.video_dir.mkdir(exist_ok=True)
        self.eval_dir = Path(cfg.reload_dir) / 'ckpt'

        # setup
        self.cfg = cfg
        self.seed = cfg.seed

        # data
        self.data = hydra.utils.instantiate(cfg.data)
        self.train_dataloader = self.data.train_dataloader()
        self.eval_dataloader = self.data.eval_dataloader()

        # logging
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb, save_stdout=False)
        self.ckpter = orbax.checkpoint.PyTreeCheckpointer()

        # misc
        self.step = 0
        self.timer = py_utils.Timer()

        devices = jax.local_devices()
        n_devices = len(devices)
        print(f"using {n_devices} devices: {devices}")


    def init_agent(self, rng, init_batch):
        rng, init_rng = jax.random.split(rng)

        rng, agent_rng = jax.random.split(rng)
        agent_class = hydra.utils.get_class(self.cfg.agent._target_)
        OmegaConf.resolve(self.cfg.agent)
        with open_dict(self.cfg.agent):
            self.cfg.agent.pop('_target_')
        agent = agent_class.create(agent_rng, init_batch, self.data.shape_meta,
                            **self.cfg.agent)
        return agent, rng

    def eval_ckpts(self):
        # initialize agent
        train_data_iter = map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.train_dataloader)
        init_batch = next(train_data_iter)
        rng = jax.random.PRNGKey(self.seed)
        self.timer.tick("time/init_agent")
        agent, rng = self.init_agent(rng, init_batch)
        self.timer.tock("time/init_agent")

        files = glob.glob(str(self.eval_dir / '*.ckpt'))
        files.sort(key=lambda file: int(Path(file).name.split('.')[0]))
        pattern = r"^(?P<ts>\d+)\.ckpt"

        for file in files:
            file = Path(file)
            file_name = file.name
            matches = re.search(pattern, file_name) 
            if matches is None:
                print(f"skipping {file_name}")
                continue
            snapshot_step = int(matches.group('ts'))
            if snapshot_step < self.cfg.after:
                continue
            if self.cfg.before > 0 and snapshot_step > self.cfg.before:
                continue

            agent = self.load_snapshot(agent, file)
            if self.cfg.restore_idm_snapshot_path is not None:
                agent = self.load_snapshot(agent, self.cfg.restore_idm_snapshot_path)

            self.epoch = snapshot_step
            self.step = snapshot_step

            rng, eval_rng = jax.random.split(rng)
            self.eval(agent, eval_rng)

    def eval_loss(self, agent, eval_rng):
        if agent.config['name'] == "ldp_hier_agent":
            return dict(), eval_rng

        dataloader = self.train_dataloader

        # device setup
        devices = jax.local_devices()
        n_devices = len(devices)
        print(f"using {n_devices} devices: {devices}")
        assert self.data.batch_size % n_devices == 0
        sharding = jax.sharding.PositionalSharding(devices)
        shard_fn = partial(py_utils.shard_batch, sharding=sharding)

        eval_metrics = dict()
        for data_idx, dataloader in enumerate([self.train_dataloader, self.eval_dataloader]):
            data_iter = map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), dataloader))
            all_metrics = []
            for i in range(10):
                batch = next(data_iter)
                metrics_rng, sample_rng, eval_rng = jax.random.split(eval_rng, 3)
                metrics = agent.get_metrics(batch, metrics_rng)
                if self.cfg.agent.name.startswith("dp"):
                    pred_action_full, _ = agent.sample(batch, sample_rng)                
                    H = pred_action_full.shape[1]
                    metrics['full_action_mse'] = jnp.mean(jnp.square(batch['actions'][:, :H, :] - pred_action_full))
                else:
                    pred_action = agent.sample_action(batch, sample_rng)
                    H = pred_action.shape[1]
                    metrics['action_mse'] = jnp.mean(jnp.square(batch['actions'][:, :H, :] - pred_action[:, :H, :]))
                    metrics['action_mse_0'] = jnp.mean(jnp.square(batch['actions'][:, 0, :] - pred_action[:, 0, :]))
                    try:
                        metrics['action_mse_1'] = jnp.mean(jnp.square(batch['actions'][:, 1, :] - pred_action[:, 1, :]))
                        metrics['action_mse_2'] = jnp.mean(jnp.square(batch['actions'][:, 2, :] - pred_action[:, 2, :]))
                    except:
                        pass
                    if self.cfg.agent.use_planner:
                        pred_action_full, stats = agent.sample(batch, sample_rng)            
                        H = pred_action_full.shape[1]
                        metrics['full_action_mse'] = jnp.mean(jnp.square(batch['actions'][:, :H, :] - pred_action_full))
                        metrics['full_action_mse_0'] = jnp.mean(jnp.square(batch['actions'][:, 0, :] - pred_action_full[:, 0, :]))
                        try:
                            metrics['full_action_mse_1'] = jnp.mean(jnp.square(batch['actions'][:, 1, :] - pred_action_full[:, 1, :]))
                            metrics['full_action_mse_2'] = jnp.mean(jnp.square(batch['actions'][:, 2, :] - pred_action_full[:, 2, :]))
                        except:
                            pass
                        metrics['plan_mse'] = stats['plan_mse']

                all_metrics.append(metrics)
            prefix = "traindata" if data_idx == 0 else "evaldata"
            all_metrics = {f"{prefix}/{k}": float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
            eval_metrics.update(all_metrics)
        return eval_metrics, metrics_rng

    def eval(self, agent, rng):
        self.timer.tick("time/eval")
        eval_rng, rng = jax.random.split(rng)

        metrics = dict()
        if self.data.name.startswith("rm") or self.data.name == "robomimic":
            eval_loss_metrics, eval_metrics = self.eval_loss(agent, eval_rng)
            metrics.update(eval_loss_metrics)
            env_params = self.data.env_params
            env_params['env_kwargs'].update(self.train_dataloader.dataset.env_meta['env_kwargs'])
            env_params['env_kwargs']['env_name'] = self.train_dataloader.dataset.env_meta['env_name']
            assert self.cfg.n_eval_episodes % self.cfg.n_eval_processes == 0
            if (not 'use_planner' in self.cfg.agent) or self.cfg.agent.use_planner:
                env_metrics, videos = rm_env_utils.run_robomimic_eval(env_params, agent, agent.config['name'], self.cfg.n_eval_episodes, self.cfg.n_eval_processes, self.cfg.seed, eval_rng)
                metrics.update(env_metrics)
                self.save_videos(videos)
            else:
                print("failed to run robomimic eval")
            self.timer.tock("time/eval")
        elif "aloha" in self.data.name:
            self.timer.tock("time/eval")
            env_params = self.data.env_params
            env_metrics, videos = aloha_env_utils.run_aloha_eval(env_params, agent, agent.config['name'], self.cfg.n_eval_episodes, self.cfg.n_eval_processes, self.cfg.seed, eval_rng)
            self.save_videos(videos)
            metrics.update(env_metrics)
        else:
            raise NotImplementedError

        for k, v in metrics.items():
            try:
                metrics[k] = float(v)
            except:
                print(f"failed for {k} {type(v)}")
        self.logger.log_metrics(metrics, self.step, ty='eval')
        self.logger.dump(self.step, ty='eval')

    def save_videos(self, videos, tag=""):
        for idx, video in enumerate(videos):
            if idx >= self.cfg.n_videos: 
                return
            save_path = py_utils.save_video(np.array(video), self.video_dir / f"{self.step}_{idx}{tag}.mp4", fps=10)
            print(f"saved video to {save_path}")

    def load_snapshot(self, agent, file):
        print(f"loading checkpoint from {file}")
        restored_prefixes = []
        raw_restored = self.ckpter.restore(file)
        for k in raw_restored.keys():
            if k == "encoder_params":
                if self.cfg.agent.shared_encoder:
                    shared_encoder = agent.encoder_state_dict['shared'].replace(params=raw_restored[k]['shared_params'], ema_params=raw_restored[k]['shared_params'])
                    encoder_state_dict = {"shared": shared_encoder}
                    agent = agent.replace(**{"encoder_state_dict": encoder_state_dict})
                else:
                    encoder_state_dict = dict()
                    for rgb_k in raw_restored[k].keys():
                        rgb_encoder = agent.encoder_state_dict[rgb_k.replace('_params', '')].replace(params=raw_restored[k][rgb_k], ema_params=raw_restored[k][rgb_k])
                        encoder_state_dict[rgb_k.replace('_params', '')] = rgb_encoder

                    agent = agent.replace(**{"encoder_state_dict": encoder_state_dict})
                restored_prefixes.append(k)
            elif "ema" in k:
                continue
            elif k.endswith("_params"):
                prefix = k.replace("_params", "")
                state_name = f"{prefix}_state"
                reload_params = raw_restored[k]
                agent = agent.replace(**{state_name: getattr(agent, state_name).replace(params=reload_params, ema_params=reload_params)})
                restored_prefixes.append(prefix)
        print(f"successfully loaded checkpoint from {file}: {restored_prefixes}")
        return agent

    def save_snapshot(self, agent, batch):
        state = agent.planner_state
        # save checkpoint, forcibly overwriting old ones if it exists
        ckpt = dict(config=dict(self.cfg), data=batch)
        ckpt.update(agent.get_params())
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.ckpter.save(self.ckpt_dir / f"{self.step}.ckpt", ckpt, save_args=save_args, force=True)

def recursively_update(cfg, reloaded_cfg, ignore_keys=[]):
    for k in reloaded_cfg.keys():
        if k in ignore_keys:
            continue
        if OmegaConf.is_interpolation(reloaded_cfg, k): # just changed this, hope it works
            continue 
        if k not in cfg.keys():
            cfg[k] = reloaded_cfg[k]
        elif isinstance(cfg[k], dictconfig.DictConfig):
            recursively_update(cfg[k], reloaded_cfg[k])
        else:
            cfg[k] = reloaded_cfg[k]

def recursively_remove_unnecessary_keys(cfg, reloaded_cfg):
    keys_to_remove = []
    for k in cfg.keys():
        if not k in reloaded_cfg.keys():
            keys_to_remove.append(k)
        elif isinstance(cfg[k], dictconfig.DictConfig):
            recursively_remove_unnecessary_keys(cfg[k], reloaded_cfg[k])
    for key_to_remove in keys_to_remove:
        cfg.pop(key_to_remove)

OmegaConf.register_new_resolver("eval", eval, replace=True)
@hydra.main(config_path='.', config_name='eval_bc')
def main(cfg):
    root_dir = Path.cwd()

    # make sure overrides are still valid for agent 
    # this is used for stuff like guidance scale
    with open(root_dir / '.hydra/overrides.yaml', 'r') as f:
        overrides_file = OmegaConf.create(yaml.safe_load(f))
    overrides = OverridesParser.create().parse_overrides(overrides_file)

    reload_dir = root_dir / f'../../{cfg.experiment_folder}/{cfg.experiment_name}'
    with open(reload_dir / '.hydra/config.yaml', 'r') as f:
        reloaded_cfg = OmegaConf.create(yaml.safe_load(f))
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        # manual overriding of keys
        cfg.obs_horizon = reloaded_cfg.obs_horizon
        cfg.reload_dir = str(reload_dir)
        cfg.horizon = reloaded_cfg.horizon
        cfg.action_horizon = reloaded_cfg.action_horizon

        # Update config for data
        recursively_update(cfg.data, reloaded_cfg.data)
        # Remove unnecessary keys for data config 
        recursively_remove_unnecessary_keys(cfg.data, reloaded_cfg.data)

        # Update config for agent
        agent = reloaded_cfg.agent.name
        curr_agent_cfg = hydra.compose(config_name=f"agent/{agent}")['agent']
        # Add missing keys from current agent cfg (this might error though)
        for k in curr_agent_cfg.keys():
            if not k in cfg.agent.keys():
                if OmegaConf.is_interpolation(curr_agent_cfg, k):
                    if k == 'idm_lr':
                        cfg.agent[k] = cfg.lr
                    elif k == 'idm_end_lr':
                        cfg.agent[k] = cfg.end_lr
                    elif k == 'data_name':
                        cfg.agent[k] = cfg.data.name
                    elif k == 'idm_horizon':
                        cfg.agent[k] = cfg.idm_horizon
                    else:
                        print(f"Need to hard-code {k} for interpolation")
                        raise NotImplementedError
                else:
                    cfg.agent[k] = curr_agent_cfg[k]
            # do a second level of adding missing keys from current agent cfg
            if isinstance(cfg.agent[k], dictconfig.DictConfig):
                if k == "planner":
                    continue 
                if OmegaConf.is_interpolation(cfg.agent, k):
                    continue 
                for k2 in curr_agent_cfg[k].keys():
                    if OmegaConf.is_interpolation(cfg.agent[k], k2):
                        continue
                    if k2 == "out_dim":
                        continue
                    if not k2 in cfg.agent[k].keys():
                        cfg.agent[k][k2] = curr_agent_cfg[k][k2]
        # Update cfg values from reloaded agent config
        agent_override = any([override_i.key_or_group == "agent" for override_i in overrides])
        for k, v in reloaded_cfg.agent.items():
            if agent_override and k in ["name", "_target_"]:
                # if overriding the bc agent, do not update the above keys. A bit hacky.
                # because we explicitly want to load target of the bc agent.
                continue
            if k == "device":
                continue
            if isinstance(cfg.agent[k], dictconfig.DictConfig):
                # second layer of checks-- might want to do deeper checks?
                for k2 in v.keys():
                    try:
                        cfg.agent[k][k2] = v[k2]
                    except omegaconf.errors.MissingMandatoryValue:
                        # deal w/ values that aren't set in the config
                        pass
                    except:
                        print(f"Error in updating {k} {k2}")
                keys_to_remove = []
                for k2 in cfg.agent[k].keys():
                    if k2 in ["input_dim", "global_cond_dim"]:
                        continue
                    # remove keys that are not in the reloaded agent's cfg
                    if not k2 in v.keys():
                        keys_to_remove.append(k2)
                for key_to_remove in keys_to_remove:
                    cfg.agent[k].pop(key_to_remove)
            else:    
                cfg.agent[k] = v
        # Remove unnecessary keys 
        keys_to_remove = []
        for k in cfg.agent.keys():
            if not k in curr_agent_cfg.keys():
                keys_to_remove.append(k)
        for key_to_remove in keys_to_remove:
            cfg.agent.pop(key_to_remove)

    if cfg.restore_idm_snapshot_path is not None:
        # override cfg.agent.idm_net keys since we are separately loading an idm
        reload_idm_dir = Path(cfg.restore_idm_snapshot_path).parent.parent
        with open(reload_idm_dir / '.hydra/config.yaml', 'r') as f:
            reloaded_idm_cfg = OmegaConf.create(yaml.safe_load(f))
        recursively_update(cfg.agent.idm_net, reloaded_idm_cfg.agent.idm_net, ignore_keys=["out_dim"])
        cfg.agent.use_idm = True

    for override_i in overrides:
        override_key = override_i.key_or_group
        override_key_tree = override_key.split(".")
        if override_key_tree[0] == "agent":
            if len(override_key_tree) == 1:
                continue
            if override_key_tree[1] in cfg.agent.keys():
                cfg.agent[override_key_tree[1]] = override_i._value
        elif override_key_tree[0] == "data":
            if len(override_key_tree) == 1:
                continue
            cfg_data = cfg.data
            for override_key in override_key_tree[1:-1]:
                cfg_data = cfg_data[override_key]
            cfg_data[override_key_tree[-1]] = override_i._value
        else:
            cfg[override_key] = override_i._value
            if override_key == 'action_horizon':
                cfg.agent.action_horizon = override_i._value


    with open(root_dir / '.hydra' / 'new_config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    workspace = Workspace(cfg)
    workspace.eval_ckpts()

if __name__ == '__main__':
    main()