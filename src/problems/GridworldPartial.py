from ml_instrumentation.Collector import Collector
from environments.GridworldPartial import GridHardRGBGoalPartial as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class GridworldPartial(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        fov = self.env_params.get('fov',5)
        self.env = Env(self.env_params.get('goal_id',0), fov, self.seed)
        self.actions = 4
        self.observations = (fov, fov, 3)
        self.gamma = 0.99
        