import re

import gymnasium as gym
from nle.nethack import actions as A

from nle_interface_wrapper.wrappers.spells.spell import Spell


class AddTextSpells(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.update()

        return self.populate_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        success_regex = (
            r"(?:learn .+\."  # matches: learn <spell name>.
            r"|add .+ to your repertoire\."  # matches: add <spell name> to your repertoire.
            r"|Your knowledge of .+ is keener\."
            r"|Your knowledge of .+ is restored\."
            r"|You know .+ quite well already\.)"
        )

        # we should only update if we know that we have learned a new spell
        if re.search(success_regex, self.message):
            self.update(obs)

        return self.populate_obs(obs), reward, terminated, truncated, info

    def populate_obs(self, obs):
        return {**obs, "text_known_spells": str(self)}

    def update(self):
        self.known_spells = {}
        obs, *_ = self.env.step(self.env.actions.index(A.Command.CAST))
        text = obs["text_message"]
        self.parse_spellcast_view(text)

    def parse_spellcast_view(self, text):
        # Pattern for spells
        spell_pattern = r"^([a-z])\s*-\s*([a-z ]+?)\s+(\d+)\s+([a-z]+)\s+(\d+)%\s+(\d+)%$"

        # Pattern for end markers
        end_pattern = r"\(end\)"
        page_pattern = r"\((\d+) of \1\)"  # Same number (e.g., "2 of 2")
        diff_page_pattern = r"\((\d+) of (\d+)\)"  # Different numbers (e.g., "1 of 2")

        spells = re.finditer(spell_pattern, text, re.MULTILINE)
        for match in spells:
            spell = Spell(*match.groups())
            self.known_spells[spell.name] = spell

        # Check for (end)
        if re.search(end_pattern, text):
            self.env.step(self.env.actions.index(A.MiscAction.MORE))

        # Check for (n of n) - same number
        elif re.search(page_pattern, text):
            self.env.step(self.env.actions.index(A.MiscAction.MORE))

        # Check for (n of m) - different numbers
        elif re.search(diff_page_pattern, text):
            obs, *_ = self.env.step(self.env.actions.index(A.MiscAction.SPACE))
            text = obs["text_message"]
            self.parse_spellcast_view(text)

    def __str__(self):
        return ", ".join([spell for spell in self.known_spells.keys()])

    def __repr__(self):
        return ", ".join([str(spell) for spell in self.known_spells.values()])
