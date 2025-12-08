from environments.GridworldGoal import GridHardRGBGoal
import numpy as np
import jax.numpy as jnp

class GridHardRGBGoalPartial(GridHardRGBGoal):
    def __init__(self, goal_id, fov, repeat=(1,1), goal_original=(12,9), seed=np.random.randint(int(1e5))):
        super().__init__(goal_id, repeat=repeat, goal_original=goal_original, seed=seed)
        self.fov = fov
        self.goal_x, self.goal_y = goal_original

    def get_inner_obstacles_map(self):
        _map = np.zeros([20, 14])
        _map[0, [5, 11]] = 1.0
        _map[1, [5, 11]] = 1.0
        _map[2, [5, 11]] = 1.0
        _map[2, [2, 8]] = 1.0
        _map[3, [2, 8]] = 1.0
        _map[4, [2, 8]] = 1.0
        _map[5, :6] = 1.0
        _map[5, 8:] = 1.0
        _map[6, 5] = 1.0
        _map[6, 11] = 1.0
        _map[7, 5] = 1.0
        _map[7, 11] = 1.0
        _map[8, 2:7] = 1.0
        _map[8, 9:12] = 1.0
        _map[11, [2, 5]] = 1.0
        _map[11, 8:] = 1.0
        _map[12, [2, 5, 8]] = 1.0
        _map[13, [2, 5, 8]] = 1.0
        _map[14, 2:6] = 1.0
        _map[14, 8:12] = 1.0
        _map[15, [5]] = 1.0
        _map[16, [5]] = 1.0
        _map[17, 0:3] = 1.0
        _map[17, 5] = 1.0
        _map[17, 8:] = 1.0
        _map[18, 5] = 1.0
        _map[19, 5] = 1.0

        return _map
        
    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color off
        state[x][y][2] = 255.0  # turning the blue color on
        
        pad = self.fov // 2
        H, W, _ = self.state_dim
        wall = jnp.array([255.0, 0.0, 0.0])
        new_state = jnp.full((H + 2*pad, W + 2*pad, 3), wall)
        new_state = new_state.at[pad:pad+H, pad:pad+W, :].set(state)
        state = new_state[x : x + self.fov, y : y + self.fov, :]
        return state