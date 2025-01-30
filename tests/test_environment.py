from environments.GridworldGoal import GridHardRGBGoal

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