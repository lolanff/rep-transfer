from copy import deepcopy

import chex
import pytest
from PyExpUtils.collection.Collector import Collector
from RlGlue import RlGlue

from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint


class TestCheckpoint:
    def test_load_from_checkpoint_agent(self):
        exp = ExperimentModel.load("experiments/Gridworld/train/DQN-ReLU-A.json")

        idx = 0
        Problem = getProblem(exp.problem)
        problem = Problem(exp, idx, None)
        chk_before = Checkpoint(exp, idx)
        chk_before.build('a', problem.getAgent)
        
        idx = 1
        Problem = getProblem(exp.problem)
        collector = Collector()
        problem = Problem(exp, idx, collector)
        chk_load = Checkpoint(exp, idx)
        agent = chk_load.build('a', problem.getAgent)
        env = problem.getEnvironment()
        glue = RlGlue(agent, env)
        glue.start()
        for _ in range(100):
            glue.step()

        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(chk_before["a"].state.target_params["phi"], chk_load["a"].state.target_params["phi"])
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(chk_before["a"].state.optim, chk_load["a"].state.optim)

        chk_after = deepcopy(chk_before)  # agent init occurs
        chk_after.load_from_checkpoint(
            chk_load,
            {
                "a": {
                    "buffer": False,
                    "state": {
                        "optim": False,
                        "params": {"phi": True, "q": False},
                        "target_params": {"phi": True, "q": False},
                    },
                }
            },
        )
        
        chex.assert_trees_all_close(chk_after["a"].state.params["phi"], chk_load["a"].state.params["phi"]) 
        chex.assert_trees_all_close(chk_after["a"].state.params["q"], chk_before["a"].state.params["q"])
        chex.assert_trees_all_close(chk_after["a"].state.target_params["phi"], chk_load["a"].state.target_params["phi"])
        chex.assert_trees_all_close(chk_after["a"].state.target_params["q"], chk_before["a"].state.target_params["q"])

        #chk_after = deepcopy(chk_before)
        #assert chk_before["a"].buffer.size() != chk_load["a"].buffer.size()
        #chk_after.load_from_checkpoint(chk_load, None)
        # TypeError when comparing buffer, so we just check the size
        #chex.assert_trees_all_close(chk_after["a"].buffer, chk_load["a"].buffer)
        #assert chk_after["a"].buffer.size() == chk_load["a"].buffer.size()
        #chex.assert_trees_all_close(chk_after["a"].state, chk_load["a"].state)
        #chex.assert_trees_all_close(chk_after["a"].state.params["phi"], chk_load["a"].state.params["phi"])  # somehow, the behavior network is not loaded at all
        