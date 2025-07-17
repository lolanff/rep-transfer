from typing import Any

import numpy as np
from forager.config import ForagerConfig
from forager.Env import ForagerEnv
from forager.objects import LargeDeathCap, LargeMorel, LargeOyster
from rlglue.environment import BaseEnvironment


class ForagerTwoBiomeLarge(BaseEnvironment):
    def __init__(self, seed: int, aperture: int):
        config = ForagerConfig(
            size=(15, 15),
            object_types={
                "deathcap": LargeDeathCap,
                "morel": LargeMorel,
                "oyster": LargeOyster,
            },
            aperture=aperture,
            seed=seed,
        )
        self.env = ForagerEnv(config)
        # 012345678901234
        # ___AA_____BB___
        # 30 * (0.5 * 30) / 300 = 1.5
        self.env.generate_objects(0.5, "morel", (3, 0), (5, 15))

        # 30 * (0.5 * 1 + 0.25 * -1) / 10 = 0.75
        self.env.generate_objects(0.5, "oyster", (10, 0), (12, 15))
        self.env.generate_objects(0.25, "deathcap", (10, 0), (12, 15))

    def start(self) -> Any:
        obs = self.env.start()
        return obs.astype(np.float32)

    def step(self, a: int):
        obs, r = self.env.step(a)
        return (obs.astype(np.float32), r, False, False, {})

    def render(self):
        return self.env.render(mode="world")
