import numpy as np
from rlglue.environment import BaseEnvironment
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

class Memory(BaseEnvironment):
    def __init__(self, max_steps=100, seed=np.random.randint(int(1e5))):
        # The reward is changed to 1 if success, 0 otherwise
        env = gym.make("MiniGrid-MemoryS7-v0", render_mode="rgb_array", max_steps=max_steps)
        self.rng = np.random.RandomState(seed)
        self.env = ImgObsWrapper(env)
        self.gamma = 0.99
    
    def start(self):
        observation, _ = self.env.reset(seed=self.rng.randint(0, 2**32))
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        return observation, float(reward != 0), terminated, truncated, self.get_info()
        
    def get_info(self):
        return {"gamma": self.gamma}
