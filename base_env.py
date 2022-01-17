"""
This module is adapted from Islam Elnabarawy
https://github.com/islamelnabarawy/sc2gym
"""

import gym
import sys
import numpy as np
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions, features
from absl import flags

_NO_OP = actions.FUNCTIONS.no_op.id

class SC2Env(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    default_settings = {'agent_interface_format': sc2_env.parse_agent_interface_format(
        feature_screen=64,
        feature_minimap=64,
    )}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.environment = None
        if not flags.FLAGS.is_parsed():
            flags.FLAGS([""])

        self._episode = 0
        self.step_count = 0
        self.ep_reward = 0
        self._total_reward = 0

    def get_info(self):
        return {
            "episode": {
                "r": self.ep_reward,
                "l": self.step_count
            }
        }

    def step(self, action):
        return self.safe_step(action)

    def safe_step(self, action):
        self.step_count = self.step_count + 1
        if action[0] not in self.available_actions:
            action = [_NO_OP]
        obs = self.environment.step([actions.FunctionCall(action[0], action[1:])])[0]
        self.available_actions = obs.observation['available_actions']
        reward = obs.reward
        self.ep_reward = self.ep_reward + reward
        self._total_reward = self._total_reward + reward
        return self.preprocess_obs(obs), reward, obs.step_type == StepType.LAST, {}

    def reset(self):
        if self.environment is None:
            self.create_environment()
        self.observation += 1
        self.step_count = 0
        self.ep_reward = 0
        obs = self.environment.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return self.preprocess_obs(obs)

    def save_replay(self, replay_dir):
        self.environment.save_replay(replay_dir)

    def create_environment(self):
        args = {**self.default_settings, **self._kwargs}
        self.environment = sc2_env.SC2Env(**args)

    def close(self):
        if self.environment is not None:
            self.environment.close()
        super().close()

    def preprocess_obs(self, obs):
        screen = obs.observation.feature_screen
        minimap = obs.observation.feature_minimap
        info = obs.observation.player
        screen = self.spatial_preprocess(screen, features.SCREEN_FEATURES)
        minimap = self.spatial_preprocess(minimap, features.MINIMAP_FEATURES)
        info = np.log(info + 1)
        return screen, minimap, info

    def spatial_preprocess(self, feature_map, feature_info):
        
        frames = []

        def _one_hot(input, limit):
            arange = np.arange(limit).reshape(-1, 1, 1)
            return (arange == input).astype(np.float32)
        for i, (_, _, _, _, scale, feature_type, _, _) in enumerate(feature_info):
            if feature_type == features.FeatureType.CATEGORICAL:
                frames.append(_one_hot(np.expand_dims(feature_map[i], 0), scale))
            elif feature_type == features.FeatureType.SCALAR:
                frames.append(np.log(np.expand_dims(feature_map[i], 0) + 1))

        return frames

    def __del__(self):
        self.close()

    @property
    def settings(self):
        return self._kwargs

    @property
    def action_spec(self):
        if self.environment is None:
            self.create_environment()
        return self.environment.action_spec()

    @property
    def observation_spec(self):
        if self.environment is None:
            self.create_environment()
        return self.environment.observation_spec()

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self.step_count

    @property
    def episode_reward(self):
        return self.ep_reward

    @property
    def total_reward(self):
        return self._total_reward


if __name__ == "__main__":
    FLAGS(sys.argv)
    _MAP_NAME = 'MoveToBeacon'
    env = SC2Env(map_name='MoveToBeacon')
    obs = env.reset()
    print([[frame.shape for frame in ob] if isinstance(ob, list) else ob.shape for ob in obs])
