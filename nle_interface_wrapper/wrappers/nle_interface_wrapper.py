from functools import partial
from string import ascii_lowercase, ascii_uppercase

import gymnasium as gym
from nle import nle_language_obsv
from nle.language_wrapper.wrappers import nle_language_wrapper
from nle.nethack import actions as A

from nle_interface_wrapper.spaces import Strings


class NLEInterfaceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        self.nle_language = nle_language_obsv.NLELanguageObsv()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        return obs, reward, terminated, truncated, info
