import numpy as np
from rlglue.environment import BaseEnvironment

class SimpleTestEnv(BaseEnvironment):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.size = 5
        self.state = 0
        
    def start(self):
        self.state = 0
        return np.array([self.state], dtype=np.float32)
    
    def step(self, action):
        # 0: left, 1: right
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(self.size, self.state + 1)
            
        reward = 0.0
        terminal = False
        success = False
        
        if self.state == self.size:
            reward = 1.0
            terminal = True
            success = True
            
        return np.array([self.state], dtype=np.float32), reward, terminal, False, {'success': success}

    def observation_shape(self):
        return (1,)
    
    def num_actions(self):
        return 2
