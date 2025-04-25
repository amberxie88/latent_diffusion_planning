import hydra
import numpy as np
import os
from pathlib import Path
import time
import wandb

import jax
import jax.numpy as jnp
import jaxlib
import flax
from flax.training import train_state, orbax_utils
from functools import partial
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict
import orbax

import utils.aloha_env_utils as aloha_env_utils 
import utils.data_utils as data_utils
import utils.rm_env_utils as rm_env_utils
from utils.logger import Logger, MeterDict
import utils.py_utils as py_utils

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.ckpt_dir = self.work_dir / 'ckpt'
        self.ckpt_dir.mkdir(exist_ok=True)
        self.video_dir = self.work_dir / 'video'
        self.video_dir.mkdir(exist_ok=True)

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

    def init_agent(self, rng, init_batch):
        rng, init_rng = jax.random.split(rng)

        rng, agent_rng = jax.random.split(rng)
        agent_class = hydra.utils.get_class(self.cfg.agent._target_)
        OmegaConf.resolve(self.cfg.agent)
        with open_dict(self.cfg.agent):
            self.cfg.agent.pop('_target_')
        agent = agent_class.create(agent_rng, init_batch, 
                            self.data.shape_meta, 
                            **self.cfg.agent)

        if self.cfg.restore_snapshot_path is not None:
            print(f"loading checkpoint from {self.cfg.restore_snapshot_path}")
            agent = self.load_snapshot(agent, self.cfg.restore_snapshot_path)
            print(f"successfully loaded checkpoint from {self.cfg.restore_snapshot_path}")
        return agent, rng

    def run(self):
        # device setup
        devices = jax.local_devices()
        n_devices = len(devices)
        print(f"using {n_devices} devices: {devices}")
        assert self.data.batch_size % n_devices == 0
        sharding = jax.sharding.PositionalSharding(devices)
        shard_fn = partial(py_utils.shard_batch, sharding=sharding)

        # init agent and dataset
        train_data_iter = map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.train_dataloader))
        init_batch = next(train_data_iter)
        rng = jax.random.PRNGKey(self.seed)
        self.timer.tick("time/init_agent")
        agent, rng = self.init_agent(rng, init_batch)
        print("no sharding available")
        # agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding.replicate())
        self.timer.tock("time/init_agent")
        print("finished initializing agent")

        # # eval
        # eval_rng, rng = jax.random.split(rng)
        # self.eval(agent, eval_rng)

        eval_every_step = py_utils.Every(self.cfg.eval_every_step)
        save_every_step = py_utils.Every(self.cfg.save_every_step)
        log_every_step = py_utils.Every(self.cfg.log_every_step)
        dump_every_step = py_utils.Every(self.cfg.dump_every_step)
        start_time = time.time()

        while True:
            self.timer.tick("time/update_loop")
            try:
                batch = next(train_data_iter)
            except StopIteration:
                train_data_iter = map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.train_dataloader))
                batch = next(train_data_iter)

            update_rng, rng = jax.random.split(rng)
            agent, metrics = agent.update(batch, update_rng, self.step)
            self.step += 1

            if log_every_step(self.step):
                metrics = jax.tree.map(lambda x: x.item() if isinstance(x, (jnp.ndarray, jaxlib.xla_extension.ArrayImpl)) else x, metrics)
                metrics.update(self.timer.get_average_times())
                metrics['total_time'] = time.time() - start_time
                self.logger.log_metrics(metrics, self.step, ty='train')
            if save_every_step(self.step):
                self.save_snapshot(agent, batch)
            if eval_every_step(self.step):
                eval_rng, rng = jax.random.split(rng)
                self.eval(agent, eval_rng)
            if dump_every_step(self.step):
                metrics['total_time'] = time.time() - start_time
                self.logger.dump(self.step, ty='train')

            try:
                self.timer.tock("time/update_loop")
            except:
                pass # this happens if metrics was just updated

            if self.step >= self.cfg.n_grad_steps:
                break

    def eval(self, agent, rng):
        self.timer.tick("time/eval")
        eval_rng, rng = jax.random.split(rng)

        if not self.eval_dataloader is None:
            sharding = jax.sharding.PositionalSharding(jax.local_devices())
            shard_fn = partial(py_utils.shard_batch, sharding=sharding)
            eval_data_iter = map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.eval_dataloader))
            all_metrics = []
            for idx, batch in enumerate(eval_data_iter):
                metrics_rng, eval_rng, sample_rng = jax.random.split(eval_rng, 3)
                metrics = agent.get_metrics(batch, metrics_rng)
                try:
                    pred_action, _ = agent.sample_action(batch, sample_rng)
                    H = pred_action.shape[1]
                    metrics['action_mse'] = jnp.mean(jnp.square(batch['actions'][:, :H, :] - pred_action[:, :H, :]))
                    metrics['action_l1'] = jnp.mean(jnp.abs(batch['actions'][:, :H, :] - pred_action[:, :H, :]))
                    if self.cfg.agent.name.startswith("dp"):
                        pass
                    elif self.cfg.agent.use_planner:
                        pred_action_full, _ = agent.sample(batch, sample_rng)
                        H = pred_action_full.shape[1]
                        metrics['full_action_mse'] = jnp.mean(jnp.square(batch['actions'][:, :H, :] - pred_action_full))
                        plan_mse = agent.sample_plan_stats(batch, sample_rng)
                        metrics['plan_mse'] = plan_mse
                except:
                    # too lazy to implement everywhere, and these stats aren't imperative
                    pass
                all_metrics.append(metrics)
                if idx >= 10:
                    break

            # take average of metrics
            eval_metrics = {f"evaldata/{k}": float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        else:
            eval_metrics = dict()

        assert self.cfg.n_eval_episodes % self.cfg.n_eval_processes == 0
        if self.data.name.startswith("rm"):
            env_params = self.data.env_params
            env_params['env_kwargs'].update(self.train_dataloader.dataset.env_meta['env_kwargs'])
            env_params['env_kwargs']['env_name'] = self.train_dataloader.dataset.env_meta['env_name']
            if ('use_planner' in self.cfg.agent) and (not self.cfg.agent.use_planner):
                pass
            else:
                env_metrics, videos = rm_env_utils.run_robomimic_eval(env_params, agent, agent.config['name'], self.cfg.n_eval_episodes, self.cfg.n_eval_processes, self.cfg.seed, eval_rng)
                self.save_videos(videos)
                eval_metrics.update(env_metrics)
        elif "aloha" in self.data.name:
            import utils.aloha_env_utils as aloha_env_utils 
            env_params = self.data.env_params
            if ('use_planner' in self.cfg.agent) and (not self.cfg.agent.use_planner):
                pass
            else:
                env_metrics, videos = aloha_env_utils.run_aloha_eval(env_params, agent, agent.config['name'], self.cfg.n_eval_episodes, self.cfg.n_eval_processes, self.cfg.seed, eval_rng)
                eval_metrics.update(env_metrics)
                self.save_videos(videos)
        else:
            raise NotImplementedError

        self.timer.tock("time/eval")
        eval_metrics.update(self.timer.get_average_times())
        self.logger.log_metrics(eval_metrics, self.step, ty='eval')
        self.logger.dump(self.step, ty='eval')

    def save_videos(self, videos, tag=""):
        for idx, video in enumerate(videos):
            if idx >= self.cfg.n_videos: 
                return
            py_utils.save_video(np.array(video), self.video_dir / f"{self.step}_{idx}{tag}.mp4", fps=10)

    def save_snapshot(self, agent, batch):
        # save checkpoint, forcibly overwriting old ones if it exists
        ckpt = dict(data=batch, cfg=OmegaConf.to_container(self.cfg, resolve=True))
        ckpt.update(agent.get_params())
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.ckpter.save(self.ckpt_dir / f"{self.step}.ckpt", ckpt, save_args=save_args, force=True)

    def load_snapshot(self, agent, file):
        print(f"loading checkpoint from {file}")
        restored_prefixes = []
        raw_restored = self.ckpter.restore(file)
        for k in raw_restored.keys():
            if len(self.cfg.restore_keys) > 0 and k not in self.cfg.restore_keys:
                continue
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
                # not loading ema params
                continue
            elif k.endswith("_params"):
                prefix = k.replace("_params", "")
                state_name = f"{prefix}_state"
                reload_params = raw_restored[k]
                agent = agent.replace(**{state_name: getattr(agent, state_name).replace(params=reload_params, ema_params=reload_params)})
                restored_prefixes.append(prefix)
        print(f"successfully loaded checkpoint from {file}: {restored_prefixes}")
        return agent

OmegaConf.register_new_resolver("eval", eval, replace=True)
@hydra.main(config_path='.', config_name='train_bc')
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