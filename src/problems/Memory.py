from ml_instrumentation.Collector import Collector
from environments.Memory import Memory as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class Memory(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        # Underlying environment is MiniGrid-MemoryS7-v0, changed reward to be binary and limit action space to 3
        super().__init__(exp, idx, collector)

        self.env = Env(max_steps=self.env_params.get('max_steps', 200), 
                       binary=self.env_params.get('binary_reward', False), 
                       seed=self.seed)
        self.actions = 3
        self.observations = (7, 7, 3)
        self.gamma = 0.95
