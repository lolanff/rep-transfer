import Box2D     # we need to import this first because cedar is stupid
import os
import sys

sys.path.append(os.getcwd())

import time
import socket
import logging
import argparse
import numpy as np
import jax.numpy as jnp
from rlglue import RlGlue
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler
from utils.policies import egreedy_probabilities, sample
from problems.registry import getProblem
from ml_instrumentation.Collector import Collector
from ml_instrumentation.Sampler import Identity, Ignore, MovingAverage, Subsample
from ml_instrumentation.utils import Pipe
from ml_instrumentation.metadata import attach_metadata
from PyExpUtils.results.tools import getParamsAsDict
import jax

from tqdm import tqdm
from environments.GridworldGoal import GridHardRGBGoal as Env

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------

device = 'gpu' if args.gpu else 'cpu'
jax.config.update('jax_platform_name', device)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname() or args.silent
if args.debug or not prod:
    logger.setLevel(logging.DEBUG)


# ----------------------
# -- Experiment Def'n --
# ----------------------
timeout_handler = TimeoutHandler()

exp = ExperimentModel.load(args.exp)
indices = args.idxs

Problem = getProblem(exp.problem)
for idx in indices:
    chk = Checkpoint(exp, idx, base_path=args.checkpoint_path)
    chk.load_if_exists()
    timeout_handler.before_cancel(chk.save)

    collector = chk.build('collector', lambda: Collector(
        # specify which keys to actually store and ultimately save
        # Options are:
        #  - Identity() (save everything)
        #  - Window(n)  take a window average of size n
        #  - Subsample(n) save one of every n elements
        config={
            'return': Identity(),
            'episode': Identity(),
            'steps': Identity(),
            'success': Identity(),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Ignore(),
    ))
    collector.set_experiment_id(idx)
    run = exp.getRun(idx)

    # set random seeds. add an offset for transfer tasks
    params = exp.get_hypers(idx)
    seed = run + params.get("experiment", {}).get("seed_offset", 0)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # build stateful things and attach to checkpoint
    problem = chk.build('p', lambda: Problem(exp, idx, collector))
    problem.seed = seed
    agent = chk.build('a', problem.getAgent)
    env = chk.build('e', problem.getEnvironment)

    glue = chk.build('glue', lambda: RlGlue(agent, env))
    chk.initial_value('episode', 0)

    # Load nn parameters from checkpoint
    load_params = problem.exp_params.get("load", {})
    if isinstance(load_params, dict):
        loaded_chk = Checkpoint(exp, run, base_path=args.checkpoint_path, load_path=load_params['path'])
        loaded_chk.load()
        chk.load_from_checkpoint(loaded_chk, load_params.get("config"))

    # Run the experiment
    start_time = time.time()

    # if we haven't started yet, then make the first interaction
    if glue.total_steps == 0:
        glue.start()
        
    # Number of consecutive completion of experiments
    consecutive_completion_counter = 0

    env = Env("0")

    for step in tqdm(range(glue.total_steps, exp.total_steps)):
        collector.next_frame()
        chk.maybe_save()
        interaction = glue.step()

        if interaction.term or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()

            # collect some data
            collector.collect('return', glue.total_reward)
            collector.collect('episode', chk['episode'])
            collector.collect('steps', glue.num_steps)
            collector.collect('success', agent.is_successful)

            # track how many episodes are completed (cutoff is counted as termination for this count)
            chk['episode'] += 1

            # compute the average time-per-step in ms
            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            fps = step / (time.time() - start_time)

            episode = chk['episode']
            
            if False:
                q_diff = []
                for _ in range(5):
                    obs = env.start().astype(jnp.float32)
                    pi, _, carry = agent.ext_policy(obs)
                    obs = env.step(sample(pi, rng=rng))[0].astype(jnp.float32)
                    val = agent.values(obs, carry=carry)[0]
                    val2 = agent.values(obs)[0]
                    q_diff.append(np.linalg.norm(val-val2)/np.abs(np.max((val, val2))))
                mean_q_diff = np.mean(q_diff).item()
                collector.collect('q_diff', mean_q_diff)
            
            logger.debug(f'{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}')

            # stop the experiment if condition met
            if not (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
                consecutive_completion_counter += 1
                if exp.early_saving > -1 and consecutive_completion_counter >= exp.early_saving:
                    break
            else:
                consecutive_completion_counter = 0
                
            glue.start()

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    context = exp.buildSaveContext(idx, base=args.save_path)
    save_path = context.resolve('results.db')
    meta = getParamsAsDict(exp, idx)
    meta |= {'seed': exp.getRun(idx)}
    attach_metadata(save_path, idx, meta)
    collector.merge(context.resolve('results.db'))
    if problem.exp_params.get("save", {}): 
        chk.save()
    else: 
        chk.delete()
    collector.close()
