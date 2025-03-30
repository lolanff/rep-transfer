# Implement T-maze from https://papers.nips.cc/paper/2001/hash/a38b16173474ba8b1a95bcbc30d3b8a5-Abstract.html
import numpy as np
from rlglue.environment import BaseEnvironment
from collections import Counter

class TMaze(BaseEnvironment):
    def __init__(self, corridor_length=10, seed=np.random.randint(int(1e5))):
        # Disambiguity: corridor_length is excluding the junction.
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.99
        self.corridor_length = corridor_length
        self.x = 0
        self.y = 0
        self.right_goal_reward = 4
        self.wrong_goal_reward = -1
        self.other_state_reward = -0.1
        self.freq = Counter()
        
    def generate_goal(self):
        # True goal is up, False goal is down
        self.goal_is_up = self.rng.rand() < 0.5
        
    def get_sign_state(self):
        if self.goal_is_up:
            return np.array([[[1, 1, 0]]])
        else:
            return np.array([[[0, 1, 1]]])
        
    def get_corridor_state(self):
        return np.array([[[1, 0, 1]]])
    
    def get_junction_state(self):
        return np.array([[[0, 1, 0]]])
    
    def get_goal_state(self):
        return self.get_corridor_state()
    
    def is_at_sign(self):
        return self.x == 0
    
    def is_at_junction(self):
        return self.x == self.corridor_length and self.y == 0
    
    def is_at_goal(self):
        return self.x == self.corridor_length and self.y != 0
    
    def get_state(self):
        if self.is_at_junction():
            return self.get_junction_state()
        elif self.is_at_goal():
            return self.get_goal_state()
        elif self.is_at_sign():
            return self.get_sign_state()
        else:
            return self.get_corridor_state()
        
    def get_reward(self):
        if self.is_at_goal():
            if (self.goal_is_up and self.y == 1) or ((not self.goal_is_up) and self.y == -1):
                return self.right_goal_reward
            else:
                return self.wrong_goal_reward
        else:
            return self.other_state_reward
    
    def start(self):
        print(self.freq)
        self.freq.clear()
        self.x = 0
        self.y = 0
        self.generate_goal()
        return self.get_state()

    def step(self, action):
        self.freq[action] += 1
        # actions: (0) up, (1) right, (2) down, (3) left
        match action:
            case 0:
                if self.is_at_junction():
                    self.y += 1
                    return self.get_state(), self.get_reward(), True, False, self.get_info()
                else:
                    return self.get_state(), self.get_reward(), False, False, self.get_info()
                
            case 1:
                if self.is_at_junction():
                    return self.get_state(), self.get_reward(), False, False, self.get_info()
                else:
                    self.x += 1
                    return self.get_state(), self.get_reward(), False, False, self.get_info()
                    
            case 2:
                if self.is_at_junction():
                    self.y -= 1
                    return self.get_state(), self.get_reward(), True, False, self.get_info()
                else:
                    return self.get_state(), self.get_reward(), False, False, self.get_info()
                
            case 3:
                if self.is_at_sign():
                    return self.get_state(), self.get_reward(), False, False, self.get_info()
                else:
                    self.x -= 1
                    return self.get_state(), self.get_reward(), False, False, self.get_info()
                
            case _:
                raise NotImplementedError("Illegal action")
        
    def get_info(self):
        return {"gamma": self.gamma}
