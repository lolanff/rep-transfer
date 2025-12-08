#!/bin/bash
set -e

export JAX_PLATFORMS=cpu
python experiments/Gridworld/A4/P7/collect_samples_alt.py
