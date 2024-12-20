from PyExpUtils.collection.Collector import Collector
from environments.GridworldGoal import GridHardRGBGoal as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class Gridworld(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(self.env_params.get('goal_id',0), self.seed)
        self.actions = 4
        self.observations = (15, 15, 3)
        self.gamma = 0.99
