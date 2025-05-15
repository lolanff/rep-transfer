import os
import sys
sys.path.append(os.getcwd())

import gc
import json
import time
import socket
import logging
import argparse
import numpy as np
import jax
from rlglue import RlGlue
from experiment import ExperimentModel
from utils.checkpoint import Checkpoint
from utils.preempt import TimeoutHandler
from problems.registry import getProblem
from PyExpUtils.results.sqlite import saveCollector
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore, MovingAverage, Subsample
from PyExpUtils.collection.utils import Pipe
from tqdm import tqdm
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
device = 'gpu' if args.gpu else 'cpu'
jax.config.update('jax_platform_name', device)

logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname() or args.silent
if not prod:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)


# ----------------------
# -- Experiment Def'n --
# ----------------------
timeout_handler = TimeoutHandler()

exp = ExperimentModel.load(args.exp)
indices = args.idxs

Problem = getProblem(exp.problem)
for idx in indices:
    chk = Checkpoint(exp, idx, base_path=args.checkpoint_path, save_every=120)
    chk.load_if_exists()

    # Test checkpointing (fail early if it doesn't work)
    chk.save()
    chk.delete()

    timeout_handler.before_cancel(chk.save)

    collector = chk.build('collector', lambda: Collector(
        # specify which keys to actually store and ultimately save
        # Options are:
        #  - Identity() (save everything)
        #  - Window(n)  take a window average of size n
        #  - Subsample(n) save one of every n elements
        config={
            'reward': Pipe(
                MovingAverage(0.999),
                Subsample(100),
            ),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Ignore(),
    ))
    collector.setIdx(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly
    hypers = exp.get_hypers(idx)
    seed = run + hypers.get("experiment", {}).get("seed_offset", 0)
    np.random.seed(seed)

    # build stateful things and attach to checkpoint
    problem = chk.build('p', lambda: Problem(exp, idx, collector))
    agent = chk.build('a', problem.getAgent)
    env = chk.build('e', problem.getEnvironment)

    glue = chk.build('glue', lambda: RlGlue(agent, env))

    # Run the experiment
    start_time = time.time()

    context = exp.buildSaveContext(0, base=args.save_path)
    path = context.ensureExists()
    path += f'/{idx}'
    os.makedirs(path, exist_ok=True)

    # if we haven't started yet, then make the first interaction
    if glue.total_steps == 0:
        glue.start()

    recorded_frames = []
    video_frequency = int(0.1 * exp.total_steps)
    video_length = 1_000

    with open(path + '/hypers.json', 'w') as f:
        hypers["run"] = run
        json.dump(hypers, f, indent=2)

    rgb_array = env.render()
    image = Image.fromarray(rgb_array)
    image = image.resize((rgb_array.shape[1] * 10, rgb_array.shape[0] * 10), Image.Resampling.NEAREST)
    image.save(path + f"/env.png")

    for step in tqdm(range(glue.total_steps, exp.total_steps)):
        collector.next_frame()
        chk.maybe_save()
        interaction = glue.step()

        collector.collect('reward', interaction.reward)

        if step % 500 == 0 and step > 0:
            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            fps = step / (time.time() - start_time)

            avg_reward = collector.get_last('reward')
            logger.debug(f'{step} {avg_reward} {avg_time:.4}ms {int(fps)}')

        if step > 899999 and step % video_frequency < video_length or (exp.total_steps - 1) - step < video_length:
            rgb_array = env.render()
            image = Image.fromarray(rgb_array)
            image = image.resize((rgb_array.shape[1] // 2, rgb_array.shape[0] // 2), Image.Resampling.NEAREST)
            frame = np.array(image)
            recorded_frames.append(frame)
        elif step % video_frequency == video_length and len(recorded_frames) > 0:
            clip = ImageSequenceClip(recorded_frames, fps=8)
            clip.write_videofile(path + f"/{step - video_length}-{step - 1}.mp4")
            recorded_frames = []

    if len(recorded_frames) > 0:
        clip = ImageSequenceClip(recorded_frames, fps=8)
        clip.write_videofile(path + f"/{exp.total_steps - video_length}-{exp.total_steps - 1}.mp4")


    collector.reset()
    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
    chk.delete()