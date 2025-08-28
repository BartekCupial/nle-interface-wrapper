import gymnasium as gym


class AutoSeed(gym.Wrapper):
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.env.seed(seed, seed, reseed=False)

        result = self.env.reset(seed=seed, **kwargs)

        return result
