from environments.GridworldGoal import GridHardRGBGoal
from environments.TMaze import TMaze
import numpy as np
import copy

def test_gridworld_goals():
    env = GridHardRGBGoal("-1")
    assert env.goal_x == 9
    assert env.goal_y == 9

    env = GridHardRGBGoal("0")
    assert env.goal_x == 9
    assert env.goal_y == 10

    env = GridHardRGBGoal("171")
    assert env.goal_x == 0
    assert env.goal_y == 0

def test_tmaze():
    env = TMaze(corridor_length=3, seed=0)
    obs = env.start()
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[0, 1, 1]]])
    assert env.x == 0
    assert env.y == 0
    
    obs, reward, terminated, truncated, others = env.step(0)
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[0, 1, 1]]])
    assert reward == -0.1
    assert terminated == False
    assert truncated == False
    assert env.x == 0
    assert env.y == 0
    
    obs, reward, terminated, truncated, others = env.step(1)
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[1, 0, 1]]])
    assert reward == -0.1
    assert terminated == False
    assert truncated == False
    assert env.x == 1
    assert env.y == 0
    
    obs, reward, terminated, truncated, others = env.step(2)
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[1, 0, 1]]])
    assert reward == -0.1
    assert terminated == False
    assert truncated == False
    assert env.x == 1
    assert env.y == 0
    
    obs, reward, terminated, truncated, others = env.step(3)
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[0, 1, 1]]])
    assert reward == -0.1
    assert terminated == False
    assert truncated == False
    assert env.x == 0
    assert env.y == 0
    
    obs, reward, terminated, truncated, others = env.step(1)
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[1, 0, 1]]])
    assert reward == -0.1
    assert terminated == False
    assert truncated == False
    assert env.x == 1
    assert env.y == 0
    
    obs, reward, terminated, truncated, others = env.step(1)
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[1, 0, 1]]])
    assert reward == -0.1
    assert terminated == False
    assert truncated == False
    assert env.x == 2
    assert env.y == 0
    
    obs, reward, terminated, truncated, others = env.step(1)
    assert env.goal_is_up == False
    assert np.allclose(obs, [[[0, 1, 0]]])
    assert reward == -0.1
    assert terminated == False
    assert truncated == False
    assert env.x == 3
    assert env.y == 0
    
    env2 =  copy.deepcopy(env)
    
    obs, reward, terminated, truncated, others = env.step(0)
    assert env.goal_is_up == False
    assert reward == -1
    assert terminated == True
    assert truncated == False
    assert env.x == 3
    assert env.y == 1
    
    obs, reward, terminated, truncated, others = env2.step(2)
    assert env2.goal_is_up == False
    assert reward == 4
    assert terminated == True
    assert truncated == False
    assert env2.x == 3
    assert env2.y == -1