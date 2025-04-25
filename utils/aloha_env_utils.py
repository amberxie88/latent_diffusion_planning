import time
from tqdm.auto import tqdm
import h5py
import json
import psutil
import os
import copy
from collections import defaultdict, deque
from pathlib import Path
from PIL import Image
import traceback

import numpy as np
import torch.multiprocessing as mp
# if mp.get_start_method(allow_none=True) != "spawn":
#     mp.set_start_method("spawn")
import jax
from utils.py_utils import save_image, save_video

from envs.alohasim_env import make_sim_env
from envs.alohasim_ee_env import sample_insertion_pose, sample_box_pose

os.environ['MUJOCO_GL'] = 'egl'

import torch
from torchvision import transforms

def process_aloha_obs(ob_dict, env_params):
    new_ob_dict = dict()
    for key in env_params['lowdim_obs']:
        if key == 'optimal': 
            sample_obs = ob_dict[env_params['lowdim_obs'][0]] 
            new_ob_dict['optimal'] = np.ones((1), dtype=sample_obs.dtype)
        else:
            new_ob_dict[key] = ob_dict[key] # (D,)
    for key in list(env_params['rgb_obs']) + [env_params['rgb_viz']]:
        rgb_img = ob_dict['images'][key.replace('_image', '').replace('latent_', '')]
        if rgb_img.min() > -0.01 and rgb_img.max() < 1.1:
            rgb_img = rgb_img * 255
        assert rgb_img.min() >= 0 and rgb_img.max() > 50 and rgb_img.max() < 257, f"min: {rgb_img.min()} max: {rgb_img.max()}"
        if rgb_img.shape[-1] != 3:
            # reshape BTCHW to BTHWC
            rgb_img = rgb_img.transpose(0, 1, 3, 4, 2)
        assert rgb_img.shape[-1] == 3
        new_ob_dict[key.replace('latent_', '')] = rgb_img 
    return new_ob_dict

def run_aloha_eval(env_params, policy, policy_name, n_rollout, n_proc, seed, eval_rng, verbose=True):
    return run_aloha_eval_single(env_params, policy, policy_name, n_rollout, n_proc, seed, eval_rng, verbose)

def run_aloha_eval_single(env_params, policy, policy_name, n_rollout, n_proc, seed, eval_rng, verbose=True):
    from envs.alohasim_env import make_sim_env, BOX_POSE
    env = make_sim_env(**env_params['env_kwargs'])
    env_max_reward = env.task.max_reward
    episode_len = 400 # in aloha_constants

    t = time.time()
    results = dict()
    ob_dict_stats = dict()

    for i in range(n_rollout):
        np.random.seed(seed + 100 + i) # + 100 since train data collected from seeds [0, 49]
        if 'insertion' in env_params['env_kwargs']['task_name']:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())
        elif 'transfer' in env_params['env_kwargs']['task_name']:  
            BOX_POSE[0] = sample_box_pose()
        else:
            raise NotImplementedError
        ts = env.reset()
        ob_dict = process_aloha_obs(ts.observation, env_params)
        debug_obs = []
        total_reward = 0

        obs_horizon = env_params["obs_horizon"]
        obs_deque = deque(
                    [ob_dict] * obs_horizon, maxlen=obs_horizon)

        env_steps = 0
        while True:
            # NOTE: obs["obs"] should be a cpu tensor because it
            # is more complicated to move cuda tensors around.
            obs_dict = dict()
            for k in obs_deque[0].keys():
                v = np.stack([x[k] for x in obs_deque])
                if k in obs_dict.keys():
                    obs_dict[k].append(v)
                else:
                    obs_dict[k] = [v]
            for k, v in obs_dict.items():
                obs_dict[k] = np.array(v, dtype=np.float32)

            s_rng, eval_rng = jax.random.split(eval_rng)
            visualize_plan = policy.config['name'] == 'ldp_agent'
            if visualize_plan:
                action, plan_dict = policy.sample_viz(dict(obs=obs_dict), s_rng)
                plan_viz = ((plan_dict['plan_viz'] + 1)/2 * 255).astype(np.uint8).transpose(0, 1, 3, 4, 2)
            else:
                action, metrics = policy.sample(dict(obs=obs_dict), s_rng)
                for k, v in metrics.items():
                    if k in ob_dict_stats:
                        if 'min' in k:
                            ob_dict_stats[k] = min(ob_dict_stats[k], v)
                        else:
                            ob_dict_stats[k] = max(ob_dict_stats[k], v)
                    else:
                        ob_dict_stats[k] = v

            action = jax.device_get(action)
            r = 0

            for idx, ac in enumerate(action[0]):
                try:
                    ts = env.step(ac)
                except:
                    # broke
                    done = True
                ob_dict = process_aloha_obs(ts.observation, env_params)
                obs_deque.append(ob_dict)
                if visualize_plan:
                    debug_obs.append(np.concatenate([ob_dict[env_params['rgb_viz']], plan_viz[0, idx]], axis=1))
                else:
                    scene_vis = np.concatenate([ts.observation['images'][k] for k in ['top', 'angle', 'vis']], axis=1)
                    debug_obs.append(scene_vis)
                total_reward += ts.reward
                env_steps += 1

                done = ts.reward == env_max_reward
                if env_steps > episode_len:
                    done = True
                    break
                if done: 
                    break

            if done or ts.reward == env_max_reward:
                break

        results[i] = dict(success=ts.reward == env_max_reward, reward=total_reward, horizon=env_steps,
                            avg_reward=total_reward/env_steps, debug_obs=debug_obs)
        print(f"{i}: {results[i]['success']} ({total_reward/env_steps}/{env_max_reward})")

    rollout_logs = dict()
    videos = []
    for result in results.values():
        for k, v in result.items():
            if k.startswith('debug'):
                videos.append(v)
                continue
            elif k in rollout_logs:
                rollout_logs[k].append(v)
            else:
                rollout_logs[k] = [v]
    total_time = time.time() - t
    rollout_logs = dict((k, np.mean(v)) for k, v in rollout_logs.items())
    rollout_logs['total_time'] = total_time
    process = psutil.Process(os.getpid())
    rollout_logs['RAM_MB'] = int(process.memory_info().rss / 1000000)
    rollout_logs['RAM_GB'] = float(rollout_logs['RAM_MB'] / 1000)
    for k, v in ob_dict_stats.items():
        rollout_logs[k] = np.array(v)
    if verbose:
        print(f"total time {time.time() - t:.2f}")
        print(rollout_logs)
    return rollout_logs, videos

def run_aloha_data_collection(save_path, unsuccessful_only, successful_only, env_params, policy, policy_name, n_rollout, n_proc, seed, eval_rng, verbose=True):
    # write to data
    save_path = Path(save_path)
    # create dir if does not exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data_writer = h5py.File(save_path, "w")
    data_grp = data_writer.create_group("data")

    def add_to_traj_dict(traj_dict, obs_dict):
        for k in obs_dict.keys():
            if k in traj_dict:
                traj_dict[k].append(obs_dict[k])
            else:
                traj_dict[k] = [obs_dict[k]]

    from envs.alohasim_env import make_sim_env, BOX_POSE
    env = make_sim_env(**env_params['env_kwargs'])
    env_max_reward = env.task.max_reward
    episode_len = 400 # in aloha_constants

    t = time.time()
    results = dict()
    ob_dict_stats = dict()

    n_samples = 0
    n_env_attempt, n_demo = 0, 0
    while n_demo < n_rollout:
        np.random.seed(seed + n_env_attempt)
        if 'insertion' in env_params['env_kwargs']['task_name']:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())
        elif 'transfer' in env_params['env_kwargs']['task_name']:  
            BOX_POSE[0] = sample_box_pose()
        else:
            raise NotImplementedError
        ts = env.reset()
        ob_dict = process_aloha_obs(ts.observation, env_params)
        traj_dict = dict()
        add_to_traj_dict(traj_dict, ob_dict)

        debug_obs = []
        total_reward = 0

        obs_horizon = env_params["obs_horizon"]
        obs_deque = deque(
                    [ob_dict] * obs_horizon, maxlen=obs_horizon)

        env_steps = 0
        while True:
            # NOTE: obs["obs"] should be a cpu tensor because it
            # is more complicated to move cuda tensors around.
            obs_dict = dict()
            for k in obs_deque[0].keys():
                v = np.stack([x[k] for x in obs_deque])
                if k in obs_dict.keys():
                    obs_dict[k].append(v)
                else:
                    obs_dict[k] = [v]
            for k, v in obs_dict.items():
                obs_dict[k] = np.array(v, dtype=np.float32)

            s_rng, eval_rng = jax.random.split(eval_rng)
            visualize_plan = policy.config['name'] == 'ldp_agent'
            if visualize_plan:
                action, plan_dict = policy.sample_viz(dict(obs=obs_dict), s_rng)
                plan_viz = ((plan_dict['plan_viz'] + 1)/2 * 255).astype(np.uint8).transpose(0, 1, 3, 4, 2)
            else:
                action, metrics = policy.sample(dict(obs=obs_dict), s_rng)
                for k, v in metrics.items():
                    if k in ob_dict_stats:
                        if 'min' in k:
                            ob_dict_stats[k] = min(ob_dict_stats[k], v)
                        else:
                            ob_dict_stats[k] = max(ob_dict_stats[k], v)
                    else:
                        ob_dict_stats[k] = v

            action = jax.device_get(action)
            r = 0

            for idx, ac in enumerate(action[0]):
                try:
                    ts = env.step(ac)
                except:
                    # broke
                    done = True

                done = ts.reward == env_max_reward
                if env_steps >= episode_len - 1:
                    done = True

                ob_dict = process_aloha_obs(ts.observation, env_params)
                add_to_traj_dict(traj_dict, ob_dict)
                add_to_traj_dict(traj_dict, dict(actions=ac, rewards=ts.reward, dones=done))
                obs_deque.append(ob_dict)
                if visualize_plan:
                    debug_obs.append(np.concatenate([ob_dict[env_params['rgb_viz']], plan_viz[0, idx]], axis=1))
                else:
                    scene_vis = np.concatenate([ts.observation['images'][k] for k in ['top', 'angle', 'vis']], axis=1)
                    debug_obs.append(scene_vis)
                total_reward += ts.reward
                env_steps += 1

                if done: 
                    break

            if done or ts.reward == env_max_reward:
                break

        n_env_attempt += 1
        if unsuccessful_only and ts.reward == env_max_reward:
            print("successful trajectory; retrying")
            continue
        if successful_only and ts.reward != env_max_reward:
            print("unsuccessful trajectory; retrying")
            continue

        for k in traj_dict.keys():
            traj_dict[k] = np.array(traj_dict[k])
        traj_dict['wrist64_image'] = (traj_dict['wrist64_image']).astype(np.uint8)

        # save info
        ep_data_grp = data_grp.create_group(f"demo_{n_demo}")
        for key in ['actions', 'rewards', 'dones']:
            ep_data_grp.create_dataset(key, data=traj_dict[key])
        for key in ['wrist64_image', 'qpos', 'qvel']:
            ep_data_grp.create_dataset(f"obs/{key}", data=traj_dict[key][:-1])
            ep_data_grp.create_dataset(f"next_obs/{key}", data=traj_dict[key][1:])
        ep_data_grp.attrs['num_samples'] = len(traj_dict['actions'])
        print(f"saved {'successful' if ts.reward == env_max_reward else 'unsuccessful'} demo {n_demo}")

        n_samples += len(traj_dict['actions'])
        n_demo += 1

    data_writer.close()
    print(f"saved to {save_path}")
    print(f"total number of samples: {n_samples}")
    print(f"num_env_attempt: {n_env_attempt} num_demo: {n_demo}")
    return
