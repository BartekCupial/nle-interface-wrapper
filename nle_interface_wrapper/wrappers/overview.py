import gymnasium as gym
from nle.nethack import actions as A

from nle_interface_wrapper.wrappers.properties.blstats import BLStats


class AddTextOverview(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.overview = {}
        self.cache_overview()

        return self.populate_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        blstats: BLStats = self.env.get_wrapper_attr("blstats")
        # update the overview when we go to a new level
        if (blstats.dungeon_number, blstats.depth) != (self.overview["dungeon_number"], self.overview["depth"]):
            self.cache_overview()

        return self.populate_obs(obs), reward, terminated, truncated, info

    def populate_obs(self, obs):
        return {**obs, "text_overview": self.get_cached_overview()}

    def cache_overview(self):
        obs, *_ = self.env.step(self.env.actions.index(A.Command.OVERVIEW))
        blstats: BLStats = self.get_blstats(obs)
        message: str = self.get_message(obs)

        self.overview = {
            "message": message,
            "dungeon_number": blstats.dungeon_number,
            "depth": blstats.depth,
        }

    def get_cached_overview(self):
        return self.overview["message"] if self.overview else ""
