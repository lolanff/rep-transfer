# Implement T-maze from https://papers.nips.cc/paper/2001/hash/a38b16173474ba8b1a95bcbc30d3b8a5-Abstract.html
# Note the following changes
#   wrong_goal_reward is -1 instead of -0.1
#   gamma is 0.95 instead of 0.98
#   observation is repeated along HW for (2, 2) unlike the original of (1, 1)
import numpy as np
from rlglue.environment import BaseEnvironment
import gymnasium as gym
from gymnasium import spaces

class TMaze(BaseEnvironment):
    def __init__(self, corridor_length=10, seed=np.random.randint(int(1e5))):
        # Disambiguity: corridor_length is excluding the junction.
        self.rng = np.random.RandomState(seed)
        self.gamma = 0.95
        self.corridor_length = corridor_length
        self.x = 0
        self.y = 0
        self.right_goal_reward = 4
        self.wrong_goal_reward = -1
        self.other_state_reward = -0.1
        
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
            state = self.get_junction_state()
        elif self.is_at_goal():
            state = self.get_goal_state()
        elif self.is_at_sign():
            state = self.get_sign_state()
        else:
            state = self.get_corridor_state()
        
        return np.tile(state, (2,2,1))
        
    def get_reward(self):
        if self.is_at_goal():
            if (self.goal_is_up and self.y == 1) or ((not self.goal_is_up) and self.y == -1):
                return self.right_goal_reward
            else:
                return self.wrong_goal_reward
        else:
            return self.other_state_reward
        
    def get_state_value(self, x = None):
        if x is None: x = self.x
        if self.is_at_goal():
            return 0
        ret = self.right_goal_reward
        for _ in range(self.corridor_length, x, -1):
            ret = self.other_state_reward + self.gamma * ret
        return ret
    
    def get_action_value(self, action, x = None):
        if x is None: x = self.x
        if self.is_at_goal():
            return 0
        
        match action:
            case 0:
                if x == self.corridor_length:
                    return self.right_goal_reward if self.goal_is_up else self.wrong_goal_reward
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                
            case 1:
                if x == self.corridor_length:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x + 1)
                
            case 2:
                if x == self.corridor_length:
                    return self.right_goal_reward if not self.goal_is_up else self.wrong_goal_reward
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                
            case 3:
                if x == 0:
                    return self.other_state_reward + self.gamma * self.get_state_value(x)
                else:
                    return self.other_state_reward + self.gamma * self.get_state_value(x - 1)
                
            case _:
                return 0

    def get_action_values(self, x = None):
        return [self.get_action_value(a) for a in range(4)]
        
    def start(self):
        self.x = 0
        self.y = 0
        self.generate_goal()
        return self.get_state()

    def step(self, action):
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

class TMazeGymWrapper(gym.Env):
    def __init__(self, corridor_length=10):
        super().__init__()
        self.env = TMaze(corridor_length)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, 1, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        state = self.env.start()
        return state.astype(np.float32), {}

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return state.astype(np.float32), reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Agent is at x={self.env.x}, y={self.env.y}, goal_is_up={self.env.goal_is_up}")

    def close(self):
        pass
