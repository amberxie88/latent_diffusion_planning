from collections import defaultdict
from contextlib import contextmanager
import h5py
import omegaconf
from omegaconf import OmegaConf
import os
import random
import numpy as np
from tqdm import tqdm

from data.alohasim_latent_data import ALOHASimLatentDataset as ALOHASimLatentDataset
import torch
import utils.py_utils as py_utils

class ALOHASimMixedLatentData(torch.utils.data.IterableDataset):
	"""
	Load offline ALOHA sim data.
	"""

	def __init__(self, name, train_paths, eval_paths,
				train_latent_paths, eval_latent_paths,
				batch_size, n_workers,
				prefetch_factor, train_n_episode_overfit, eval_n_episode_overfit,
				train_split, eval_split, obs_horizon, seq_length,
				env_params, meta):
		self.name = name
		self.train_paths = train_paths
		self.eval_paths = eval_paths
		self.train_latent_paths = train_latent_paths
		self.eval_latent_paths = eval_latent_paths
		self.batch_size = batch_size
		self.n_workers = n_workers
		self.prefetch_factor = prefetch_factor
		self.train_n_episode_overfit = train_n_episode_overfit
		self.eval_n_episode_overfit = eval_n_episode_overfit
		self.train_split = train_split
		self.eval_split = eval_split
		self.obs_horizon = obs_horizon
		self.seq_length = seq_length

		# Assert valid inputs
		assert len(self.train_n_episode_overfit) == len(self.train_paths)
		# assert len(self.eval_n_episode_overfit) == len(self.eval_paths)
		if isinstance(self.train_split, omegaconf.listconfig.ListConfig):
			assert sum(self.train_split) == 1
		else:
			self.train_split = [self.train_split, 1-self.train_split]
		if isinstance(self.eval_split, omegaconf.listconfig.ListConfig):
			assert sum(self.eval_split) == 1
		else:
			self.eval_split = [self.eval_split, 1-self.eval_split]
		
		self.meta = meta
		self.env_params = OmegaConf.to_container(env_params, resolve=True)
		self.shape_meta = meta.shape_meta
		self.setup_alohasim()

		self._val_dataset = None
		self.train_datasets = []
		for idx, (dataset_path, n_overfit) in enumerate(zip(self.train_paths, self.train_n_episode_overfit)):
			self.ds_kwargs['hdf5_path'] = dataset_path
			self.ds_kwargs['n_overfit'] = n_overfit
			self.ds_kwargs['optimal'] = idx == 0 
			self.ds_kwargs['latent_path'] = self.train_latent_paths[idx]
			self.train_datasets.append(ALOHASimLatentDataset(**self.ds_kwargs))

	def setup_alohasim(self):
		all_obs_keys = list(self.meta.lowdim_obs) + list(self.meta.rgb_obs)
		# setup dataset kwargs
		self.ds_kwargs = dict(
		    obs_keys=all_obs_keys,
		    dataset_keys=['actions'],
		    frame_stack=self.obs_horizon,
		    seq_length=self.seq_length,
		    hdf5_use_swmr=True,
		    n_overfit=None, # TBD per each sub-dataset!
		    rgb_keys=list(self.meta.rgb_obs),
		)

	def __iter__(self):
		while True:
			yield self._sample()

	def _sample(self):
		# randomly pick a dataset based on train_split
		dataset_idx = random.choices(range(len(self.train_datasets)), self.train_split)[0]
		out = self.train_datasets[dataset_idx]._sample()
		return out

	@property
	def val_dataset(self):
		if self._val_dataset is None:
			# a lot of ugly legacy code here from when I was reloading agents w old configs...
			if isinstance(self.eval_paths, omegaconf.listconfig.ListConfig): 
				if len(self.eval_paths) > 0:
					eval_paths = self.eval_paths[0]
				else:
					eval_paths = self.train_paths[0]
			else:
				eval_paths = self.eval_paths
			if isinstance(self.eval_n_episode_overfit, omegaconf.listconfig.ListConfig):
				if len(self.eval_n_episode_overfit) > 0:
					eval_n_episode_overfit = self.eval_n_episode_overfit[0]
				else:
					eval_n_episode_overfit = 10
			else:
				eval_n_episode_overfit = 10
			self.ds_kwargs['hdf5_path'] = eval_paths
			self.ds_kwargs['n_overfit'] = eval_n_episode_overfit
			self.ds_kwargs['optimal'] = 0
			self.ds_kwargs['latent_path'] = self.eval_latent_paths
			self._val_dataset = ALOHASimLatentDataset(**self.ds_kwargs)
		return self._val_dataset

	def train_dataloader(self):
		self.train_dataset = self
		return torch.utils.data.DataLoader(
				self.train_dataset,
				batch_size=self.batch_size,
				num_workers=self.n_workers,
				pin_memory=False,
				shuffle=False,
				# don't kill worker process after each epoch
				persistent_workers=True
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
