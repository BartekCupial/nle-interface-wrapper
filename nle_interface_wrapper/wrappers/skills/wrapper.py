import re

import gymnasium as gym
from nle.nethack import actions as A

from nle_interface_wrapper.wrappers.skills.properties import Alignment, Gender, Race, Role
from nle_interface_wrapper.wrappers.skills.skill import CharacterSkills


class AddTextSkills(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.update()

        return self.populate_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # TODO: we should update the attributes if skill is enchanted

        return self.populate_obs(obs), reward, terminated, truncated, info

    def populate_obs(self, obs):
        return {**obs, "text_known_spells": str(self)}

    def update(self):
        obs, *_ = self.env.step(self.env.actions.index(A.Command.ATTRIBUTES))
        message = "\n".join([bytes(line).decode("latin-1") for line in obs["tty_chars"]])
        self.parse_welcome(message)
        obs, *_ = self.env.step(self.env.actions.index(A.Command.ESC))

    def parse_welcome(self, message):
        """
        parse agent gender, race, role, alignment form starting message
        """
        self.race = Race.parse(message)
        self.gender = Gender.parse(message)
        self.role = Role.parse(message)
        self.alignment = Alignment.parse(message)
        self.skill = CharacterSkills.from_role(self.role)

    def __str__(self):
        return ", ".join(self.skill.get_skill_str_list())

    def __repr__(self):
        return str(self)
