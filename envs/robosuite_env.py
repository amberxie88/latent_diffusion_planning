"""
This file contains the robosuite environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""
import json
import numpy as np
from copy import deepcopy

import robosuite

class RobosuiteEnv:
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        lowdim_obs,
        rgb_obs,
        render=True, 
        render_offscreen=True, 
        **kwargs,
    ):
        self.lowdim_obs = lowdim_obs
        self.rgb_obs = [key.replace('latent_', '') if key.startswith('latent_') else key for key in rgb_obs]
        self._env_name = kwargs['env_name']

        # robosuite version check
        is_v1 = (robosuite.__version__.split(".")[0] == "1")
        assert is_v1

        # update kwargs based on passed arguments
        kwargs = deepcopy(kwargs)
        update_kwargs = dict(
            has_renderer=render,
            has_offscreen_renderer=True,
            ignore_done=False,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
        )
        kwargs.update(update_kwargs)

        if kwargs["has_offscreen_renderer"]:
            # ensure that we select the correct GPU device for rendering by testing for EGL rendering
            # NOTE: this package should be installed from this link (https://github.com/StanfordVL/egl_probe)
            import egl_probe
            valid_gpu_devices = egl_probe.get_available_devices()
            if len(valid_gpu_devices) > 0:
                kwargs["render_gpu_device_id"] = valid_gpu_devices[0]

        self._init_kwargs = deepcopy(kwargs)
        self.env = robosuite.make(**kwargs)

        # Make sure joint position observations and eef vel observations are active
        for ob_name in self.env.observation_names:
            if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

        print(f"Created environment with name {self.name}")
        print(f"Action size is {self.action_dimension}")

    def step(self, action):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        obs = self.get_observation(obs)
        return obs, r, done, info

    def reset(self):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        di = self.env.reset()
        return self.get_observation(di)

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            return self.get_observation()
        return None

    def render(self, mode="rgb_array", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            im = self.env.sim.render(height=height, width=width, camera_name=camera_name)
            return im[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True)
        ret = {}
        for k in di:
            if k in self.rgb_obs:
                # by default images from mujoco are flipped in height
                ret[k] = di[k][::-1] # (H, W, C), [0, 255], float32

        # # "object" key contains object information
        if 'object' in self.lowdim_obs:
            ret["object"] = np.array(di["object-state"])

        for robot in self.env.robots:
            # add all robot-arm-specific observations. Note the (k not in ret) check
            # ensures that we don't accidentally add robot wrist images a second time
            pf = robot.robot_model.naming_prefix
            for k in di:
                if k.startswith(pf) and (k not in ret) and \
                        (not k.endswith("proprio-state")):
                    ret[k] = np.array(di[k])
        return ret

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        xml = self.env.sim.model.get_xml() # model xml file
        state = np.array(self.env.sim.get_state().flatten()) # simulator state
        return dict(model=xml, states=state)

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        succ = self.env._check_success()
        if isinstance(succ, dict):
            assert "task" in succ
            return succ
        return { "task" : succ }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return self.env.action_spec[0].shape[0]

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def version(self):
        """
        Returns version of robosuite used for this environment, eg. 1.2.0
        """
        return robosuite.__version__

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(
            env_name=self.name,
            env_version=self.version,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.env

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
