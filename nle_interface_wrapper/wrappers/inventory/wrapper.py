import gymnasium as gym

from nle_interface_wrapper.wrappers.inventory.inventory import Inventory
from nle_interface_wrapper.wrappers.inventory.item_database import ItemDatabase


class AddTextInventory(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.inventory = Inventory()
        self.item_database = ItemDatabase()

        self.update(obs)

        return self.populate_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.update(obs)

        return self.populate_obs(obs), reward, terminated, truncated, info

    def update(self, obs):
        if not self.hallu and not self.blind:
            self.inventory.update(
                obs["inv_strs"],
                obs["inv_letters"],
                obs["inv_oclasses"],
                obs["inv_glyphs"],
                self.item_database,
            )

    def populate_obs(self, obs):
        print(str(self.inventory))

        return {
            **obs,
            "text_inventory": str(self.inventory),
        }
