from typing import Any, List, Tuple, Union

import gymnasium as gym
import numpy as np
from nle import nethack

from nle_interface_wrapper.wrappers.properties.blstats import BLStats
from nle_interface_wrapper.wrappers.properties.entity import Entity
from nle_interface_wrapper.wrappers.properties.glyph import G
from nle_interface_wrapper.wrappers.properties.utils import isin


class Properties(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.is_lycanthrope = False

        self.obs, self.info = self.env.reset(**kwargs)
        self.last_obs = self.obs

        self.update()
        self.last_obs = self.obs
        self.start_glyph = self.entity.glyph

        return self.obs, self.info

    def step(self, action):
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.last_obs = self.obs

        self.update()
        self.last_obs = self.obs

        return self.obs, reward, terminated, truncated, self.info

    def update(self):
        internal = self.env.unwrapped.last_observation[self.env.unwrapped._internal_index]
        self.in_yn_function = internal[1]
        self.in_getlin = internal[2]
        self.xwaitingforspace = internal[3]

        self.blstats = self.get_blstats(self.obs)
        self.glyphs = self.get_glyphs(self.obs)
        self.message = self.get_message(self.obs)
        self.tty_chars = self.get_tty_chars(self.obs)
        self.tty_colors = self.get_tty_colors(self.obs)
        self.cursor = self.get_cursor(self.obs)
        self.entity = self.get_entity(self.obs)
        self.entities = self.get_entities(self.obs)

        if "You feel feverish." in self.message:
            self.is_lycanthrope = True
        if "You feel purified." in self.message:
            self.is_lycanthrope = False

    def add_message(self, message):
        self.obs["text_message"] += "\n" + message

    def get_blstats(self, last_obs) -> BLStats:
        return BLStats(*last_obs["blstats"])

    def get_glyphs(self, last_obs):
        return last_obs["glyphs"]

    def get_message(self, last_obs):
        return last_obs["text_message"]

    def get_tty_chars(self, last_obs):
        return last_obs["tty_chars"]

    def get_tty_colors(self, last_obs):
        return last_obs["tty_colors"]

    def get_cursor(self, last_obs):
        return tuple(last_obs["tty_cursor"])

    def get_entity(self, last_obs) -> Entity:
        """
        Returns:
            Entity object with the player
        """
        blstats = self.get_blstats(last_obs)
        position = (blstats.y, blstats.x)
        return Entity(position, self.get_glyphs(last_obs)[position])

    def get_entities(self, last_obs) -> List[Union[Any, Entity]]:
        """
        Returns:
            List of Entity objects with the monsters
        """
        glyphs = self.get_glyphs(last_obs)
        blstats = self.get_blstats(last_obs)
        monster_mask = isin(glyphs, G.MONS, G.INVISIBLE_MON)
        monster_mask[blstats.y, blstats.x] = 0

        return [Entity(position, glyphs[position]) for position in list(zip(*np.where(monster_mask)))]

    @property
    def lycantropy(self):
        return self.is_lycanthrope

    @property
    def engulfed(self):
        return isin(self.glyphs, G.SWALLOW).any()

    @property
    def stone(self):
        """Stoned"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_STONE else False

    @property
    def slime(self):
        """Slimed"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_SLIME else False

    @property
    def strngl(self):
        """Strangled"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_STRNGL else False

    @property
    def foodpois(self):
        """Food Poisoning"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_FOODPOIS else False

    @property
    def termill(self):
        """Terminally Ill"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_TERMILL else False

    @property
    def blind(self):
        """Blind"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_BLIND else False

    @property
    def deaf(self):
        """Deaf"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_DEAF else False

    @property
    def stun(self):
        """Stunned"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_STUN else False

    @property
    def conf(self):
        """Confused"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_CONF else False

    @property
    def hallu(self):
        """Hallucinating"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_HALLU else False

    @property
    def lev(self):
        """Levitating"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_LEV else False

    @property
    def fly(self):
        """Flying"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_FLY else False

    @property
    def ride(self):
        """Riding"""
        return True if self.blstats.prop_mask & nethack.BL_MASK_RIDE else False

    @property
    def poly(self):
        """Polymorphed"""
        return self.start_glyph != self.entity.glyph
