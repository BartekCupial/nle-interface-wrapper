import gymnasium as gym
import nle  # NOQA: F401
from gymnasium import registry
from nle import nethack

from nle_interface_wrapper.wrappers import (
    AddTextInventory,
    AddTextMap,
    AddTextOverview,
    AddTextSkills,
    AddTextSpells,
    AutoMore,
    AutoRender,
    AutoSeed,
    NoProgressAbort,
    PlayLanguage,
    PlayNLE,
    Properties,
)

NETHACK_ENVS = [env_spec.id for env_spec in registry.values() if "NetHack" in env_spec.id]


def create_env(env_name, cfg, env_config, render_mode=None):
    observation_keys = (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        "glyphs",
        "inv_glyphs",
        "inv_strs",
        "inv_letters",
        "inv_oclasses",
    )

    # NetHack options
    options = []
    for option in nethack.NETHACKOPTIONS:
        if option == "autopickup" and not cfg.autopickup:
            options.append("!autopickup")
            continue
        options.append(option)

    if not cfg.pet:
        options += ("pettype:none",)

    kwargs = dict(
        observation_keys=observation_keys,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
        penalty_mode=cfg.fn_penalty_step,
        savedir=cfg.savedir,
        save_ttyrec_every=cfg.save_ttyrec_every,
        actions=nethack.ACTIONS,
        options=options,
    )

    param_mapping = {
        "max_episode_steps": cfg.max_episode_steps,
        "character": cfg.character,
        "allow_all_yn_questions": cfg.allow_all_yn_questions,
        "allow_all_modes": cfg.allow_all_modes,
    }

    for param_name, param_value in param_mapping.items():
        if param_value is not None:
            kwargs[param_name] = param_value

    env = gym.make(env_name, render_mode=render_mode, **kwargs)
    env = AutoRender(env)
    env = AutoSeed(env)
    env = NoProgressAbort(env)
    env = AutoMore(env)

    env = Properties(env)
    env = AddTextOverview(env)
    env = AddTextMap(env)
    env = AddTextInventory(env)
    env = AddTextSpells(env)
    env = AddTextSkills(env)

    if cfg.play_mode == "language":
        env = PlayLanguage(env)
    elif cfg.play_mode == "nle":
        env = PlayNLE(env)

    return env
