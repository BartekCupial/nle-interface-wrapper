import random
import timeit
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import tyro

from nle_interface_wrapper.envs.nle_env import create_env


@dataclass
class Args:
    env: str = "NetHackScore-v0"
    character: str = "@"
    autopickup: bool = False
    pet: bool = True
    penalty_step: float = 0.0
    penalty_time: float = 0.0
    reward_shaping_coefficient: float = 0.1
    fn_penalty_step: Literal["constant", "exp", "square", "linear", "always"] = "constant"
    max_episode_steps: Optional[int] = 100_000
    allow_all_yn_questions: bool = True
    allow_all_modes: bool = True
    savedir: Optional[str] = None
    save_ttyrec_every: Optional[int] = 0
    seed: Optional[int] = None
    render_mode: Optional[Literal["human", "rgb_array"]] = "human"


def play(cfg):
    env = create_env(
        cfg.env,
        cfg=cfg,
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        render_mode=cfg.render_mode,
    )

    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
    obs, info = env.reset(seed=cfg.seed)

    steps = 0
    reward = 0.0
    total_reward = 0.0
    action = None

    total_start_time = timeit.default_timer()
    start_time = total_start_time

    while True:
        action = env.get_wrapper_attr("get_action")(obs)
        if action is None:
            break

        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += reward

        if not (terminated or truncated):
            continue

        time_delta = timeit.default_timer() - start_time

        print("Final reward:", reward)
        print("End status:", info.get("end_status"), "")
        print(f"Total reward: {total_reward}, Steps: {steps}, SPS: {steps / time_delta}", total_reward)

        break
    env.close()

    return info


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    play(cfg)
