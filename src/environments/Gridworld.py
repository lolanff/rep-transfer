# Modified from https://github.com/erfanMhi/LTA-Representation-Properties/blob/main/core/environment/gridworlds.py
import numpy as np
import jax.numpy as jnp
from rlglue.environment import BaseEnvironment
from tqdm import tqdm

class GridHardXY(BaseEnvironment):
    def __init__(self, repeat=(1,1), goal=(9,9), seed=np.random.randint(int(1e5))):
        self.rng = np.random.RandomState(seed)
        self.state_dim = (2,)
        self.action_dim = 4
        self.obstacles_map = self.get_obstacles_map(repeat)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.max_x, self.max_y = self.obstacles_map.shape[0] - 1, self.obstacles_map.shape[1] - 1
        self.min_x, self.min_y = 0, 0
        self.goal_x, self.goal_y = goal
        self.current_state = None
        self.gamma = 0.99

    def generate_state(self, coords):
        return np.array(coords)

    def info(self, key):
        return

    def start(self):
        while True:
            rx = self.rng.randint(low=self.min_x, high=self.max_x)
            ry = self.rng.randint(low=self.min_y, high=self.max_y)
            if not int(self.obstacles_map[rx][ry]) and not (rx == self.goal_x and ry == self.goal_y):
                self.current_state = rx, ry
                return self.generate_state(self.current_state)

    def step(self, action):
        dx, dy = self.actions[action]
        assert self.current_state is not None, "Call start() before step()"
        x, y = self.current_state

        nx = x + dx
        ny = y + dy

        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)

        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny

        self.current_state = x, y
        if x == self.goal_x and y == self.goal_y:
            return self.generate_state([x, y]), 1.0, True, False, {**self.get_info(), "success": True}
        else:
            return self.generate_state([x, y]), 0.0, False, False, {**self.get_info(), "success": False}

    def get_info(self):
        return {"gamma": self.gamma, "pos": np.array(self.current_state)}

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(self.max_x + 1)
                       for y in range(self.max_y + 1) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        # goal_coords = [[9, 9], [0, 0], [14, 0], [7, 14]]
        # goal_coords = [[9, 9], [9, 12], [13, 11], [8, 6], [13, 2]] # 0 5 25 50 100
        # goal_coords = [[9, 9], [13, 11], [8, 6], [9, 1], [13, 2], [4, 13], [1, 3]] # 0 25 50 75 100 125 150
        goal_coords = [[9, 9], [3, 4], [1, 0], [0, 14], [3, 14], [7, 14], [10, 3], [14, 4]] #
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)
    
    def get_inner_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[2, 0:6] = 1.0
        _map[2, 8:] = 1.0
        _map[3, 5] = 1.0
        _map[4, 5] = 1.0
        _map[5, 2:7] = 1.0
        _map[5, 9:] = 1.0
        _map[8, 2] = 1.0
        _map[8, 5] = 1.0
        _map[8, 8:] = 1.0
        _map[9, 2] = 1.0
        _map[9, 5] = 1.0
        _map[9, 8] = 1.0
        _map[10, 2] = 1.0
        _map[10, 5] = 1.0
        _map[10, 8] = 1.0
        _map[11, 2:6] = 1.0
        _map[11, 8:12] = 1.0
        _map[12, 5] = 1.0
        _map[13, 5] = 1.0
        _map[14, 5] = 1.0

        return _map

    def get_obstacles_map(self, repeat=(1,1)):
        return np.tile(self.get_inner_obstacles_map(), repeat)

    def get_useful(self, state=None):
        if state:
            return state
        else:
            return self.current_state
        
    def get_num_states(self):
        rows, cols = self.obstacles_map.shape
        num_states = 0
        for x in range(rows):
            for y in range(cols):
                if not self.obstacles_map[x][y]:
                    num_states += 1
        return num_states

    def get_optimal_policy(self, tolerance=0.00):
        rows, cols = self.obstacles_map.shape
        num_states = rows * cols
        V = np.zeros(num_states)
        policy = np.full(num_states, -1, dtype=int)

        def get_idx(x, y):
            return x * cols + y

        def step_model(x, y, action):
            dx, dy = self.actions[action]
            nx, ny = x + dx, y + dy
            nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)
            if not self.obstacles_map[nx][ny]:
                x, y = nx, ny
            
            reward = 0.0
            is_terminal = False
            if x == self.goal_x and y == self.goal_y:
                reward = 1.0
                is_terminal = True
            
            return x, y, reward, is_terminal

        while True:
            delta = 0
            for x in range(rows):
                for y in range(cols):
                    if self.obstacles_map[x][y]:
                        continue
                    
                    s_idx = get_idx(x, y)
                    if x == self.goal_x and y == self.goal_y:
                        continue

                    v = V[s_idx]
                    action_values = []
                    for a in range(self.action_dim):
                        nx, ny, r, term = step_model(x, y, a)
                        ns_idx = get_idx(nx, ny)
                        val = r + (0.0 if term else self.gamma * V[ns_idx])
                        action_values.append(val)
                    
                    best_val = max(action_values)
                    V[s_idx] = best_val
                    delta = max(delta, abs(v - best_val))
            
            if delta <= tolerance:
                break

        for x in range(rows):
            for y in range(cols):
                if self.obstacles_map[x][y]:
                    continue
                if x == self.goal_x and y == self.goal_y:
                    continue
                    
                s_idx = get_idx(x, y)
                action_values = []
                for a in range(self.action_dim):
                    nx, ny, r, term = step_model(x, y, a)
                    ns_idx = get_idx(nx, ny)
                    val = r + (0.0 if term else self.gamma * V[ns_idx])
                    action_values.append(val)
                
                policy[s_idx] = np.argmax(action_values)
                
        return policy

    def get_successor_representation(self, policy):
        rows, cols = self.obstacles_map.shape
        
        # Map valid coordinates to indices
        valid_coords = []
        coord_to_idx = {}
        idx = 0
        for x in range(rows):
            for y in range(cols):
                if not self.obstacles_map[x][y]:
                    valid_coords.append((x, y))
                    coord_to_idx[(x, y)] = idx
                    idx += 1
        
        num_valid = len(valid_coords)
        P = np.zeros((num_valid, num_valid))
        
        def get_original_idx(x, y):
            return x * cols + y
            
        def step_model(x, y, action):
            dx, dy = self.actions[action]
            nx, ny = x + dx, y + dy
            nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)
            if not self.obstacles_map[nx][ny]:
                x, y = nx, ny
            return x, y

        for i, (x, y) in enumerate(valid_coords):
            if x == self.goal_x and y == self.goal_y:
                continue
            
            s_idx_orig = get_original_idx(x, y)
            action = policy[s_idx_orig]
            nx, ny = step_model(x, y, action)
            
            # nx, ny is guaranteed to be valid
            ns_idx = coord_to_idx[(nx, ny)]
            P[i, ns_idx] = 1.0
        
        Phi = np.eye(num_valid)
            
        Psi = np.linalg.inv(np.eye(num_valid) - self.gamma * P) @ Phi
            
        return Psi

    def get_similarity_ranks(self):
        rows, cols = self.obstacles_map.shape
        original_goal = (self.goal_x, self.goal_y)
        
        # Get representation of the true goal
        pi_true = self.get_optimal_policy()
        sr_true = self.get_successor_representation(pi_true)
        vec_true = sr_true.flatten()
        
        similarities = np.full((rows, cols), -np.inf)
        
        for x in tqdm(range(rows)):
            for y in range(cols):
                if self.obstacles_map[x][y]:
                    continue
                
                self.goal_x, self.goal_y = x, y
                
                pi = self.get_optimal_policy()
                sr = self.get_successor_representation(pi)
                vec = sr.flatten()
                
                similarities[x, y] = np.dot(vec, vec_true)
        
        self.goal_x, self.goal_y = original_goal

        # Rank
        flat_sim = similarities.flatten()
        # Sort descending
        sorted_indices = np.argsort(flat_sim)[::-1]
        
        ranks = np.full_like(flat_sim, -1, dtype=int)
        
        current_rank = 0
        for idx in sorted_indices:
            if flat_sim[idx] > -np.inf:
                ranks[idx] = current_rank
                current_rank += 1
            
        return ranks.reshape(rows, cols)

class GridHardGS(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 1)

        """
        # Gray-scale image
            walls are black: 0.0
            agent is gray:   128.0
            open space is white: 255.0
        """
        self.gray_template = np.ones(self.state_dim) * 255.0
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.gray_template[x][y] = 0.0

    def generate_state(self, coords):
        state = np.copy(self.gray_template)
        x, y = coords
        state[x][y] = 128.0
        return state

    def get_features(self, state):
        raise NotImplementedError

class GridHardRGB(GridHardXY):
    def __init__(self, repeat=(1,1), goal=(9,9), seed=np.random.randint(int(1e5))):
        super().__init__(repeat=repeat, goal=goal, seed=seed)

        xd, yd = self.obstacles_map.shape
        self.state_dim = (xd, yd, 3)

        """
        # Gray-scale image
            Walls are Red
            Open spaces are Green
            Agent is Blue
        """
        self.rgb_template = np.zeros(self.state_dim)

        for x in range(xd):
            for y in range(yd):
                if self.obstacles_map[x][y]:
                    self.rgb_template[x][y][0] = 255.0
                else:
                    self.rgb_template[x][y][1] = 255.0

    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color off
        state[x][y][2] = 255.0  # turning the blue color on
        return state

    def get_features(self, state):
        raise NotImplementedError

    def get_useful(self, state=None):
        blue = np.array([0., 0., 255.])
        if state is None:
            state = self.generate_state(self.current_state)
        idx = np.where(np.all(state==blue, axis=2) == True)
        coord = np.array([idx[0][0], idx[1][0]])
        return coord


class GridTwoRoomXY(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goal_x, self.goal_y = 8, 14

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        goal_coords = [[8, 14], [14, 0], [7, 7]]
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[7, :7] = 1.0
        _map[7, 9:] = 1.0
        return _map


class GridTwoRoomRGB(GridTwoRoomXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)

        self.rgb_template = np.zeros(self.state_dim)
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.rgb_template[x][y][0] = 255.0
                else:
                    self.rgb_template[x][y][1] = 255.0

    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        return state

    def get_useful(self, state=None):
        blue = np.array([0., 0., 255.])
        if state is None:
            state = self.generate_state(self.current_state)
        idx = np.where(np.all(state==blue, axis=2) == True)
        coord = np.array([idx[0][0], idx[1][0]])
        return coord


class GridOneRoomXY(GridHardXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        self.goal_x, self.goal_y = 14, 14

    def get_visualization_segment(self):
        state_coords = [[x, y] for x in range(15)
                       for y in range(15) if not int(self.obstacles_map[x][y])]
        states = [self.generate_state(coord) for coord in state_coords]
        goal_coords = [[8, 14], [14, 0], [7, 7]]
        goal_states = [self.generate_state(coord) for coord in goal_coords]
        return np.array(states), np.array(state_coords), np.array(goal_states), np.array(goal_coords)

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        return _map


class GridOneRoomRGB(GridOneRoomXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)

        self.rgb_template = np.zeros(self.state_dim)
        for x in range(d):
            for y in range(d):
                self.rgb_template[x][y][1] = 255.0

    def generate_state(self, coords):
        state = np.copy(self.rgb_template)
        x, y = coords
        assert state[x][y][1] == 255.0 and state[x][y][2] == 0.0

        state[x][y][1] = 0.0    # setting the green color on
        state[x][y][2] = 255.0  # turning the blue color on
        return state

    def get_useful(self, state=None):
        blue = np.array([0., 0., 255.])
        if state is None:
            state = self.generate_state(self.current_state)
        idx = np.where(np.all(state==blue, axis=2) == True)
        coord = np.array([idx[0][0], idx[1][0]])
        return coord
