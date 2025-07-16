from environments.GridworldGoal import GridHardRGBGoal
import numpy as np
import jax.numpy as jnp

class GridHardRGBGoalPartial(GridHardRGBGoal):
    def __init__(self, goal_id, fov, seed=np.random.randint(int(1e5))):
        super().__init__(goal_id, seed)
        self.fov = fov
        
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