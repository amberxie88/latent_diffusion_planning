import hydra
import numpy as np
import os
from pathlib import Path
import psutil
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
import optax
import orbax

import utils.data_utils as data_utils
import utils.html_utils as html_utils
from utils.logger import Logger, MeterDict
import utils.py_utils as py_utils

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.ckpt_dir = self.work_dir / 'ckpt'
        self.ckpt_dir.mkdir(exist_ok=True)
        self.plot_dir = self.work_dir / 'plot'
        self.plot_dir.mkdir(exist_ok=True)

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

    def init_model(self, rng, init_batch):
        rng, init_rng = jax.random.split(rng)

        rng, model_rng = jax.random.split(rng)
        model_class = hydra.utils.get_class(self.cfg.model._target_)
        OmegaConf.resolve(self.cfg.model)
        with open_dict(self.cfg.model):
            self.cfg.model.pop('_target_')
        model = model_class.create(model_rng, init_batch, self.data.shape_meta, 
                            **self.cfg.model)

        if self.cfg.restore_snapshot_path is not None:
            print(f"loading checkpoint from {self.cfg.restore_snapshot_path}")
            raw_restored = self.ckpter.restore(self.cfg.restore_snapshot_path)
            for k in raw_restored.keys():
                if k == "ema_params":
                    state_name = "vae_state" # hardcoded
                    reload_params = raw_restored[k]
                    model = model.replace(**{state_name: getattr(model, state_name).replace(ema_params=reload_params)})
                elif k.endswith("_params"):
                    prefix = k.replace("_params", "")
                    state_name = f"{prefix}_state"
                    reload_params = raw_restored[k]
                    model = model.replace(**{state_name: getattr(model, state_name).replace(params=reload_params)})
            print(f"successfully loaded checkpoint from {self.cfg.restore_snapshot_path}")
        return model, rng

    def run(self):
        # device setup
        devices = jax.local_devices()
        n_devices = len(devices)
        print(f"using {n_devices} devices: {devices}")
        assert self.data.batch_size % n_devices == 0
        sharding = jax.sharding.PositionalSharding(devices)
        shard_fn = partial(py_utils.shard_batch, sharding=sharding)

        # init model and dataset
        train_data_iter = map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.train_dataloader))
        init_batch = next(train_data_iter)
        rng = jax.random.PRNGKey(self.seed)
        self.timer.tick("time/init_model")
        model, rng = self.init_model(rng, init_batch)
        print("no sharding")
        # model = jax.device_put(jax.tree.map(jnp.array, model), sharding.replicate())
        self.timer.tock("time/init_model")
        print("finished initializing model")

        eval_every_step = py_utils.Every(self.cfg.eval_every_step)
        save_every_step = py_utils.Every(self.cfg.save_every_step)
        log_every_step = py_utils.Every(self.cfg.log_every_step)
        dump_every_step = py_utils.Every(self.cfg.dump_every_step)
        start_time = time.time()

        # eval
        eval_rng, rng = jax.random.split(rng)
        self.eval(model, eval_rng)

        while True:
            self.timer.tick("time/data")
            try:
                batch = next(train_data_iter)
            except StopIteration:
                train_data_iter = map(postprocess_fn, map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.train_dataloader)))
                batch = next(train_data_iter)
            self.timer.tock("time/data")

            update_rng, rng = jax.random.split(rng)
            self.timer.tick("time/update")
            model, metrics = model.update(batch, update_rng, self.step)
            self.timer.tock("time/update")
            self.step += 1

            if log_every_step(self.step):
                metrics = jax.tree.map(lambda x: x.item() if isinstance(x, (jnp.ndarray, jaxlib.xla_extension.ArrayImpl)) else x, metrics)
                metrics.update(self.timer.get_average_times())
                metrics['total_time'] = time.time() - start_time
                self.logger.log_metrics(metrics, self.step, ty='train')
            if save_every_step(self.step):
                self.save_snapshot(model, batch)
            if eval_every_step(self.step):
                eval_rng, rng = jax.random.split(rng)
                self.eval(model, eval_rng)
            if dump_every_step(self.step):
                metrics['total_time'] = time.time() - start_time
                self.logger.dump(self.step, ty='train')

            if self.step >= self.cfg.n_grad_steps:
                break

    def eval(self, model, rng):
        eval_rng, rng = jax.random.split(rng)

        sharding = jax.sharding.PositionalSharding(jax.local_devices())
        shard_fn = partial(py_utils.shard_batch, sharding=sharding)
        eval_data_iter = map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.eval_dataloader))
        all_metrics = []
        for idx, batch in enumerate(eval_data_iter):
            metrics_rng, eval_rng = jax.random.split(eval_rng)
            metrics = model.get_metrics(batch, metrics_rng)
            all_metrics.append(metrics)
            if idx >= 10:
                break
        batch = next(eval_data_iter)

        # take average of metrics
        eval_metrics = {f"evaldata/{k}": np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        
        train_data_iter = map(shard_fn, map(lambda batch: jax.tree.map(lambda tensor: tensor.numpy(), batch), self.train_dataloader))
        init_batch = next(train_data_iter)
        eval_batch = batch
        
        t_rng, e_rng, s_rng, rng = jax.random.split(rng, 4)
        train_reconstruct = model.reconstruct(init_batch, t_rng, self.cfg.data.meta.rgb_obs[0])
        eval_reconstruct = model.reconstruct(eval_batch, e_rng, self.cfg.data.meta.rgb_obs[0])
        train_reconstruct = np.clip((np.array(train_reconstruct) + 1) / 2, 0, 1)
        eval_reconstruct = np.clip((np.array(eval_reconstruct) + 1) / 2, 0, 1)
        sample = model.sample(s_rng)
        sample = np.clip((np.array(sample) + 1) / 2, 0, 1)

        webpage = html_utils.HTML(self.plot_dir / f'step{self.step}', f'Checkpoint Step {self.step}')
        webpage.add_header(f'Checkpoint Step {self.step}')

        image_dir = Path(webpage.get_image_dir())
        train_comb_paths = []
        for i in range(3):
            for rgb_obs in self.cfg.data.meta.rgb_obs:
                train_traj = self.train_dataloader.dataset.sample_traj(i)
                train_rng, t_t_rng = jax.random.split(eval_rng)
                train_traj_reconstruct = model.reconstruct(train_traj, t_t_rng, rgb_obs)
                train_traj_reconstruct = np.clip((np.array(train_traj_reconstruct) + 1) / 2 * 255, 0, 255).transpose(0, 2, 3, 1).astype(np.uint8)
                train_traj_img = train_traj['obs'][rgb_obs][:, 0]
                train_comb = np.concatenate([train_traj_img, train_traj_reconstruct, np.clip(train_traj_img - train_traj_reconstruct, 0, 255)], axis=2)
                train_comb_path = py_utils.save_video(train_comb, image_dir / f"step{self.step}_train_traj_comb_{i}_{rgb_obs}.gif")
                train_comb_paths.append(train_comb_path.name)
        webpage.add_header("Train Trajectory Reconstruction")
        webpage.add_images(train_comb_paths, ['gt (left), recon (middle), gt - recon (right)'] * len(train_comb_paths))

        eval_comb_paths = []
        for i in range(3):
            for rgb_obs in self.cfg.data.meta.rgb_obs:
                eval_traj = self.eval_dataloader.dataset.sample_traj(i)
                eval_rng, e_t_rng = jax.random.split(eval_rng)
                eval_traj_reconstruct = model.reconstruct(eval_traj, e_t_rng, rgb_obs)
                eval_traj_reconstruct = np.clip((np.array(eval_traj_reconstruct) + 1) / 2 * 255, 0, 255).transpose(0, 2, 3, 1).astype(np.uint8)
                eval_traj_img = eval_traj['obs'][rgb_obs][:, 0]
                eval_comb = np.concatenate([eval_traj_img, eval_traj_reconstruct, np.clip(eval_traj_img - eval_traj_reconstruct, 0, 255)], axis=2)
                eval_comb_path = py_utils.save_video(eval_comb, image_dir / f"step{self.step}_eval_traj_comb_{i}_{rgb_obs}.gif")
                eval_comb_paths.append(eval_comb_path.name)
        webpage.add_header("Eval Trajectory Reconstruction")
        webpage.add_images(eval_comb_paths, ['gt (left), recon (middle), gt - recon (right)'] * len(eval_comb_paths))

        webpage.add_header("Train Reconstruction")
        for i in range(3):
            # single image reconstruction
            train_gt_path = py_utils.save_image(np.array(init_batch['obs'][self.cfg.data.meta.rgb_obs[0]][i][0]), image_dir / f"step{self.step}_train_{i}.png")
            train_recon_path = py_utils.save_image(train_reconstruct[i], image_dir / f"step{self.step}_train_reconstruct_{i}.png")
            webpage.add_images([train_gt_path.name, train_recon_path.name], ['train_gt', 'train_recon'])
            
        webpage.add_header("Eval Reconstruction")
        for i in range(3):
            eval_gt_path = py_utils.save_image(np.array(eval_batch['obs'][self.cfg.data.meta.rgb_obs[0]][i][0]), image_dir / f"step{self.step}_eval_{i}.png")
            eval_recon_path = py_utils.save_image(eval_reconstruct[i], image_dir / f"step{self.step}_eval_reconstruct_{i}.png")
            webpage.add_images([eval_gt_path.name, eval_recon_path.name], ['eval_gt', 'eval_recon'])

        sample_names = []
        for i in range(3):
            sample_path = py_utils.save_image(sample[i], image_dir / f"step{self.step}_sample_{i}.png")
            sample_names.append(sample_path.name)
        webpage.add_header("Samples")
        webpage.add_images(sample_names, ['sample'] * len(sample_names))

        for k, v in eval_metrics.items():
            try:
                eval_metrics[k] = float(v)
                webpage.add_text(f'{k}: {eval_metrics[k]}')
            except:
                print(f"failed for {k} {type(v)}")
        webpage.save()
        self.logger.log_metrics(eval_metrics, self.step, ty='eval')
        self.logger.dump(self.step, ty='eval')


    def save_videos(self, videos, tag=""):
        for idx, video in enumerate(videos):
            if idx >= self.cfg.n_videos: 
                return
            py_utils.save_video(np.array(video), self.video_dir / f"{self.step}_{idx}{tag}.mp4", fps=10)

    def save_snapshot(self, model, batch):
        # save checkpoint, forcibly overwriting old ones if it exists
        ckpt = dict(data=batch)
        ckpt.update(model.get_params())
        save_args = orbax_utils.save_args_from_target(ckpt)
        self.ckpter.save(self.ckpt_dir / f"{self.step}.ckpt", ckpt, save_args=save_args, force=True)


@hydra.main(config_path='.', config_name='train_vae')
def main(cfg):
    # create logger
    if cfg.use_wandb:
        import omegaconf
        wandb.init(entity=YOUR_ENTITY, project='ldp', group=cfg.experiment_folder,
                    name=cfg.experiment_name,tags=[cfg.experiment_folder], sync_tensorboard=True)
        wandb.config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )

    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()