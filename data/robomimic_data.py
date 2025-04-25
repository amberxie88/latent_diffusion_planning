from contextlib import contextmanager
import h5py
import json
from omegaconf import OmegaConf
import os
import numpy as np

import torch
import utils.py_utils as py_utils

class RobomimicDataset(torch.utils.data.IterableDataset):
    def __init__(self, hdf5_path, obs_keys, rgb_keys, dataset_keys,
        frame_stack, seq_length, hdf5_use_swmr, n_overfit, optimal
    ):
        super(RobomimicDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self._hdf5_file = None

        # get all keys that needs to be fetched
        self.optimal = optimal
        self.obs_keys = tuple(obs_keys)
        self.rgb_keys = tuple(rgb_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_overfit = n_overfit # the maximum number of samples to load
        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1
        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.load_demo_info()
        self.data = self.weld_demos()
        self.env_meta = json.loads(self.hdf5_file["data"].attrs["env_args"])

        # obs_normalization_stats = self.normalize_obs() # use for new datasets to get stats
        # np.set_printoptions(suppress=True,precision=5)
        self.close_and_delete_hdf5_handle()

    def load_demo_info(self):
        self.demos = list(self.hdf5_file["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        print(f"dataset will load {f'all ({len(self.demos)})' if self.n_overfit is None else self.n_overfit} trajectories")
        if self.n_overfit is not None:
            assert self.n_overfit <= len(self.demos), "The lower bound of sample size must be at most the number of available samples!"
            self.demos = self.demos[0 : self.n_overfit] # chopping off the rest
        self.n_demos = len(self.demos)

        # determine index mapping
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()
        self.total_n_sequences = 0

        for ep in self.demos:
            demo_length = self.hdf5_file[f"data/{ep}"].attrs["num_samples"]
            demo_length += 1 # deal with next_obs
            self._demo_id_to_start_indices[ep] = self.total_n_sequences #keeping track of where things start
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length

            assert demo_length >= 1  # sequence needs to have at least one sample
            num_sequences = max(num_sequences, 1)
    
            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_n_sequences] = ep
                self.total_n_sequences += 1

    def weld_demos(self):
        # this will return a flattened representation of the dataset, for use in weighting purposes
        dict_list = {}
        dict_list["actions"] = list()
        for key in self.obs_keys:
            dict_list[key] = list()
        for demo in self.demos:
            for key in self.obs_keys: 
                if key == 'optimal':
                    obs = self.optimal * np.ones((self._demo_id_to_demo_length[demo], 1))
                else:
                    obs = self.hdf5_file['data'][demo]['obs'][key][:]
                    last_obs = np.expand_dims(self.hdf5_file['data'][demo]['next_obs'][key][-1], 0)
                    obs = np.concatenate([obs, last_obs], axis=0)
                dict_list[key].append(obs)
            actions = self.hdf5_file['data'][demo]['actions'][:]
            dummy_action = np.expand_dims(actions[-1], 0)
            actions = np.concatenate([actions, dummy_action], axis=0)
            dict_list["actions"].append(actions)
        final_dict = {}
        for modality in dict_list:
            final_dict[modality] = np.concatenate(dict_list[modality], axis=0)
        return final_dict

    def _sample(self):
        index = np.random.randint(self.total_n_sequences)
        return self.get_item(index)

    def get_item(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_end_index = demo_start_index + self._demo_id_to_demo_length[demo_id]

        seq_start_index = max(index - self.n_frame_stack + 1, demo_start_index)
        seq_end_index = min(index + self.seq_length, demo_end_index)
        n_pad_start = max(self.n_frame_stack - (index - seq_start_index + 1), 0)
        n_pad_end = max(self.seq_length - (seq_end_index - index), 0)
        return self._get_batch(seq_start_index, seq_end_index, n_pad_start, n_pad_end)

    def _get_batch(self, seq_start_index, seq_end_index, n_pad_start, n_pad_end):
        batch = dict()
        for key in self.dataset_keys:
            seq = self.data[key][seq_start_index:seq_end_index]
            if n_pad_start > 0:
                seq = np.concatenate([np.expand_dims(seq[0], axis=0)] * n_pad_start + [seq], axis=0)
            if n_pad_end > 0:
                seq = np.concatenate([seq] + [np.expand_dims(seq[-1], axis=0)] * n_pad_end, axis=0)
            # no frame stacking
            seq = seq[self.n_frame_stack - 1:]
            batch[key] = seq

        batch['obs'] = dict()
        for key in self.obs_keys:
            seq = self.data[key][seq_start_index:seq_end_index]
            if n_pad_start > 0:
                seq = np.concatenate([np.expand_dims(seq[0], axis=0)] * n_pad_start + [seq], axis=0)
            if n_pad_end > 0:
                seq = np.concatenate([seq] + [np.expand_dims(seq[-1], axis=0)] * n_pad_end, axis=0)
            batch['obs'][key] = seq
        return batch

    def sample_traj(self, ep_id):
        # get the start/end indices for this datapoint
        demo_id = self.demos[ep_id]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_end_index = demo_start_index + self._demo_id_to_demo_length[demo_id]

        batch = self._get_batch(demo_start_index, demo_end_index, 0, 0)
        for key in batch['obs'].keys():
            batch['obs'][key] = np.expand_dims(batch['obs'][key], axis=1)
        return batch

    def normalize_obs(self):
        def _compute_traj_stats(traj_obs_dict):
            traj_stats = { k : {} for k in traj_obs_dict }
            for k in traj_obs_dict:
                traj_stats[k]['min'] = traj_obs_dict[k].min(axis=0, keepdims=True)
                traj_stats[k]['max'] = traj_obs_dict[k].max(axis=0, keepdims=True)
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            merged_stats = {}
            for k in traj_stats_a:
                merged_stats[k] = dict(min=np.minimum(traj_stats_a[k]["min"], traj_stats_b[k]["min"]), max=np.maximum(traj_stats_a[k]["max"], traj_stats_b[k]["max"]))
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file[f"data/{ep}/obs/{k}"][()].astype('float32') for k in self.obs_keys}
        merged_stats = _compute_traj_stats(obs_traj)
        print("ModernDataset: normalizing observations...")
        for ep in self.demos[1:]:
            obs_traj = {k: self.hdf5_file[f"data/{ep}/obs/{k}"][()].astype('float32') for k in self.obs_keys}
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = { k : {} for k in merged_stats }
        for k in merged_stats:
            obs_normalization_stats[k]["min"] = merged_stats[k]["min"]
            obs_normalization_stats[k]["max"] = merged_stats[k]["max"]
            obs_normalization_stats[k]["adj_min"] = np.where(obs_normalization_stats[k]["min"] < 0, obs_normalization_stats[k]["min"] * 1.1, obs_normalization_stats[k]["min"] * 0.9)
            obs_normalization_stats[k]["adj_max"] = np.where(obs_normalization_stats[k]["max"] < 0, obs_normalization_stats[k]["max"] * 0.9, obs_normalization_stats[k]["max"] * 1.1)
        return obs_normalization_stats

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __iter__(self):
        while True:
            yield self._sample()

class RobomimicData:
    """
    Load offline robomimic data.
    """

    def __init__(self, name, train_path, eval_path,
                train_n_episode_overfit, eval_n_episode_overfit, 
                batch_size, n_workers, prefetch_factor, 
                obs_horizon, seq_length, 
                env_params, meta):
        self.name = name
        self.train_path = train_path
        self.eval_path = eval_path
        self.train_n_episode_overfit = train_n_episode_overfit
        self.eval_n_episode_overfit = eval_n_episode_overfit
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.prefetch_factor = prefetch_factor
        self.obs_horizon = obs_horizon
        self.seq_length = seq_length
        
        self.env_params = OmegaConf.to_container(env_params, resolve=True)
        self.meta = meta
        self.shape_meta = meta.shape_meta
        self.setup_robomimic()
        self._train_dataset = None
        self._val_dataset = None

    def setup_robomimic(self):
        all_obs_keys = list(self.meta.lowdim_obs) + list(self.meta.rgb_obs)
        # setup dataset kwargs
        self.ds_kwargs = dict(
            obs_keys=all_obs_keys,
            dataset_keys=['actions'],
            frame_stack=self.obs_horizon,
            seq_length=self.seq_length,
            hdf5_use_swmr=True,
            n_overfit=self.train_n_episode_overfit,
            rgb_keys=list(self.meta.rgb_obs),
            optimal=1,
        )

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self.ds_kwargs['hdf5_path'] = self.train_path
            self.ds_kwargs['n_overfit'] = self.train_n_episode_overfit
            self._train_dataset = RobomimicDataset(**self.ds_kwargs)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self.ds_kwargs['hdf5_path'] = self.eval_path
            self.ds_kwargs['n_overfit'] = self.eval_n_episode_overfit
            self._val_dataset = RobomimicDataset(**self.ds_kwargs)
        return self._val_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.n_workers,
                    pin_memory=False,
                    shuffle=False,
                    # don't kill worker process after each epoch
                    persistent_workers=True,
                    prefetch_factor=self.prefetch_factor,
                    )

    def eval_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.n_workers,
                    pin_memory=False,
                    shuffle=False,
                    # don't kill worker process after each epoch
                    persistent_workers=True,
                    prefetch_factor=self.prefetch_factor,
                    )