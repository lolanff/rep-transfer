import numpy as np
from RlGlue.environment import BaseEnvironment
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

class Memory(BaseEnvironment):
    def __init__(self, max_steps=100, seed=np.random.randint(int(1e5))):
        # A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.
        env = gym.make("MiniGrid-MemoryS11-v0", render_mode="rgb_array", max_steps=max_steps)
        self.rng = np.random.RandomState(seed)
        self.env = ImgObsWrapper(env)
        self.gamma = 1 # The environment itself give reward based on time
    
    def start(self):
        observation, _ = self.env.reset(seed=self.rng.randint(0, 2**32))
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return reward, observation, terminated or truncated, self.get_info()
        
    def get_info(self):
        return {"gamma": self.gamma}
