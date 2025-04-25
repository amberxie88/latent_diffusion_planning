import time
from tqdm.auto import tqdm
import h5py
import json
import psutil
import os
import copy
from collections import defaultdict, deque
from pathlib import Path
import traceback

import numpy as np
import torch.multiprocessing as mp
import jax
from utils.py_utils import save_image, save_video

# adapted from Hengyuan Hu
class EvalProc:
    def __init__(self, policy_name, seeds, process_id, env_params, terminal_queue: mp.Queue):
        self.policy_name = policy_name
        self.seeds = seeds
        self.process_id = process_id
        self.env_params = env_params
        self.terminal_queue = terminal_queue
        self.send_queue = mp.Queue()
        self.recv_queue = mp.Queue()

    def start(self):
        try: 
            t = time.time()
            from envs.robosuite_env import RobosuiteEnv
            env = RobosuiteEnv(**self.env_params['env_kwargs'])

            results = {}
            for seed in self.seeds:
                debug_obs = []
                np.random.seed(seed)
                ob_dict = env.reset()
                state_dict = env.get_state()
                # hack that is necessary for robosuite tasks for deterministic action playback
                ob_dict = env.reset_to(state_dict)
                success = { k: False for k in env.is_success() }
                total_reward = 0

                obs_horizon = self.env_params["obs_horizon"]
                obs_deque = deque(
                            [ob_dict] * obs_horizon, maxlen=obs_horizon)

                env_steps = 0
                while True:
                    # NOTE: obs["obs"] should be a cpu tensor because it
                    # is more complicated to move cuda tensors around.
                    self.send_queue.put((self.process_id, obs_deque))
                    out = self.recv_queue.get()
                    if len(out) == 1:
                        action = out[0]
                        visualize_plan = False
                    elif len(out) == 2:
                        action, plan_viz = out
                        visualize_plan = True
                    r = 0
                    for idx, ac in enumerate(action):
                        ob_dict, r_tmp, done, _ = env.step(ac)
                        env_steps += 1
                        obs_deque.append(ob_dict)
                        if visualize_plan:
                            gt_img = ob_dict[self.env_params['rgb_viz']].transpose((2,0,1))
                            debug_obs.append(np.concatenate([gt_img, plan_viz[idx]], axis=-1))
                        else:
                            debug_obs.append(ob_dict[self.env_params['rgb_viz']])
                        r += r_tmp

                        if done: 
                            break

                    # compute reward
                    total_reward += r

                    cur_success_metrics = env.is_success()
                    for k in success:
                        success[k] = success[k] or cur_success_metrics[k]

                    if done or success["task"]:
                        self.send_queue.put((self.process_id, [dict(reset=True)]))
                        break

                results[seed] = dict(success=float(success['task']), reward=total_reward, horizon=env_steps)
                results[seed]['debug_obs'] = debug_obs

            self.terminal_queue.put((self.process_id, results))
            env.env.close()
            return
        except Exception as e:
            self.terminal_queue.put((self.process_id, 'error', traceback.format_exc()))

def run_robomimic_eval(env_params, policy, policy_name, n_rollout, n_proc, seed, eval_rng, verbose=True):
    return run_robomimic_eval_multi(env_params, policy, policy_name, n_rollout, n_proc, seed, eval_rng, verbose)
    
def run_robomimic_eval_multi(env_params, policy, policy_name, n_rollout, n_proc, seed, eval_rng, verbose=True):
    assert n_rollout % n_proc == 0

    rollouts_per_proc = n_rollout // n_proc
    terminal_queue = mp.Queue()

    eval_procs = []
    for i in range(n_proc):
        seeds = list(range(seed + i * rollouts_per_proc, seed + (i + 1) * rollouts_per_proc))
        eval_procs.append(EvalProc(policy_name, seeds, i, env_params, terminal_queue))

    put_queues = {i: proc.recv_queue for i, proc in enumerate(eval_procs)}
    get_queues = {i: proc.send_queue for i, proc in enumerate(eval_procs)}

    processes = {i: mp.Process(target=proc.start) for i, proc in enumerate(eval_procs)}
    for _, p in processes.items():
        p.start()

    t = time.time()
    results = dict()
    while len(processes) > 0:
        while not terminal_queue.empty():
            out = terminal_queue.get()
            if len(out) == 3:
                print(f"process {out[0]} failed with error {out[2]}")
                p_alive = [p.is_alive() for p in processes.values()]
                print(f"Some process died. Processes alive: {p_alive}") 
                raise NotImplementedError
            elif len(out) == 2:
                term_idx, proc_results = out
            results.update(proc_results)
            processes[term_idx].join()
            processes.pop(term_idx)
            get_queues.pop(term_idx)
            put_queues.pop(term_idx)
        p_alive = [p.is_alive() for p in processes.values()]
        if len(p_alive) > 0 and not all(p_alive):
            print(f"Some process died. Processes alive: {p_alive}")
            remove_ps = []
            exit_codes = {i: p.exitcode for i, p in processes.items()}
            print(f"Exit Codes: {exit_codes}")
            for k, p in processes.items():
                p.terminate()
                remove_ps.append(k)
            for k in remove_ps:
                processes.pop(k)
                get_queues.pop(k)
                put_queues.pop(k)
            if len(p_alive) > 0 and not all(p_alive):
                raise NotImplementedError

        idxs = []
        obs_dict = dict()
        for _, get_queue in get_queues.items():
            if get_queue.empty():
                continue
            data = get_queue.get()
            obs_deque = data[1]
            if 'reset' in obs_deque[0] and obs_deque[0]['reset'] is True:
                continue

            for k in obs_deque[0].keys():
                v = np.stack([x[k] for x in obs_deque])
                if k in obs_dict.keys():
                    obs_dict[k].append(v)
                else:
                    obs_dict[k] = [v]
            idxs.append(data[0])

        for k, v in obs_dict.items():
            obs_dict[k] = np.array(v, dtype=np.float32)
        
        if len(idxs) == 0:
            continue

        # process obs
        rgb_obs = [key.replace('latent_', '') if key.startswith('latent_') else key for key in env_params['env_kwargs']['rgb_obs']]
        agent_obs_dims = rgb_obs + env_params['env_kwargs']['lowdim_obs']
        if 'optimal' in agent_obs_dims:
            sample_obs = obs_dict[env_params['env_kwargs']['lowdim_obs'][0]]
            obs_dict['optimal'] = np.ones((sample_obs.shape[0], 1, 1), dtype=sample_obs.dtype)
        obs_dict = {k: obs_dict[k] for k in agent_obs_dims}

        s_rng, eval_rng = jax.random.split(eval_rng)
        visualize_plan = policy.config['name'] in ['ldp_agent', 'ldp_hier_agent']
        if visualize_plan:
            batch_action, plan_dict = policy.sample_viz(dict(obs=obs_dict), s_rng)
            plan_viz = plan_dict['plan_viz']
            plan_viz = (np.clip((np.array(plan_viz) + 1)/2, 0, 1) * 255).astype(np.uint8)
            batch_action = jax.device_get(batch_action)
            batch_action = np.array(batch_action)

            for idx_enum, idx_queue in enumerate(idxs):
                put_queues[idx_queue].put((batch_action[idx_enum], plan_viz[idx_enum]))
        else:
            batch_action, _ = policy.sample(dict(obs=obs_dict), s_rng)
            batch_action = jax.device_get(batch_action)
            batch_action = np.array(batch_action)

            for idx_enum, idx_queue in enumerate(idxs):
                put_queues[idx_queue].put((batch_action[idx_enum],))

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
    if verbose:
        print(f"total time {time.time() - t:.2f}")
        print(rollout_logs)
    return rollout_logs, videos

def run_robomimic_data_collection(save_path, reference_hdf5, unsuccessful_only, successful_only, env_params, policy, policy_name, n_rollout, seed, noise, eval_rng):
    from envs.robosuite_env import RobosuiteEnv
    env = RobosuiteEnv(**env_params['env_kwargs'])

    # write to data
    save_path = Path(save_path)
    # create dir if does not exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data_writer = h5py.File(save_path, "w")
    data_grp = data_writer.create_group("data")
    data_grp.attrs['env_args'] = json.dumps(env.serialize(), indent=4)

    def add_to_traj_dict(traj_dict, obs_dict):
        obs_dict = copy.deepcopy(obs_dict)
        for k in obs_dict.keys():
            if k in ['traj_id']:
                # don't care about this key
                pass
            elif k in traj_dict:
                traj_dict[k].append(obs_dict[k])
            else:
                traj_dict[k] = [obs_dict[k]]

    n_samples = 0
    n_env_attempt, n_demo = 0, 0
    while n_demo < n_rollout:
        debug_obs = []
        np.random.seed(seed + n_env_attempt)
        ob_dict = env.reset()
        state_dict = env.get_state()
        ob_dict = env.reset_to(state_dict)

        traj_dict = dict()
        add_to_traj_dict(traj_dict, ob_dict)
        total_reward = 0

        obs_horizon = env_params["obs_horizon"]
        obs_deque = deque(
                    [ob_dict] * obs_horizon, maxlen=obs_horizon)

        env_steps = 0
        while True:
            # NOTE: obs["obs"] should be a cpu tensor because it
            # is more complicated to move cuda tensors around.
            batch = dict()
            for k in obs_deque[0].keys():
                v = np.stack([x[k] for x in obs_deque])
                if k in batch.keys():
                    batch[k].append(v)
                else:
                    batch[k] = [v]
            for k, v in batch.items():
                batch[k] = np.array(v, dtype=np.float32)
            rgb_obs = [key.replace('latent_', '') if key.startswith('latent_') else key for key in env_params['env_kwargs']['rgb_obs']]
            agent_obs_dims = rgb_obs + env_params['env_kwargs']['lowdim_obs']
            obs = {k: batch[k] for k in agent_obs_dims}

            s_rng, eval_rng = jax.random.split(eval_rng)
            action, _ = policy.sample(dict(obs=obs), s_rng)
            action = jax.device_get(action)
            r = 0
            for ac in action[0]:
                if noise > 0:
                    ac = np.clip(ac + np.random.normal(0, noise, size=ac.shape), -1, 1)
                ob_dict, r, done, _ = env.step(ac)
                obs_deque.append(ob_dict)
                add_to_traj_dict(traj_dict, ob_dict)
                add_to_traj_dict(traj_dict, dict(actions=ac, rewards=r, dones=done))
                total_reward += r
                env_steps += 1

                img = env.render(width=512, height=512)
                if done: 
                    break

            if done:
                break
            if env.is_success()['task']:
                break

        n_env_attempt += 1
        if unsuccessful_only and env.is_success()['task']:
            print("successful trajectory; retrying")
            continue
        if successful_only and not env.is_success()['task']:
            print("unsuccessful trajectory; retrying")
            continue

        for k in traj_dict.keys():
            traj_dict[k] = np.array(traj_dict[k])
        traj_dict['agentview_image'] = traj_dict['agentview_image'].astype(np.uint8)
        traj_dict['robot0_eye_in_hand_image'] = traj_dict['robot0_eye_in_hand_image'].astype(np.uint8)

        # save info
        ep_data_grp = data_grp.create_group(f"demo_{n_demo}")
        for key in ['actions', 'rewards', 'dones']:
            ep_data_grp.create_dataset(key, data=traj_dict[key])
        for key in ['agentview_image', 'robot0_eye_in_hand_image', 'object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
            ep_data_grp.create_dataset(f"obs/{key}", data=traj_dict[key][:-1])
            ep_data_grp.create_dataset(f"next_obs/{key}", data=traj_dict[key][1:])
        ep_data_grp.attrs['num_samples'] = len(traj_dict['actions'])
        print(f"saved {'successful' if env.is_success()['task'] else 'unsuccessful'} demo {n_demo}")

        n_samples += len(traj_dict['actions'])
        n_demo += 1


    data_grp.attrs['total'] = n_samples
    data_writer.close()
    env.env.close()
    return
