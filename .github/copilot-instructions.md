# GitHub Copilot Instructions

This document helps AI coding agents quickly become productive in the `rep-transfer` codebase by outlining project structure, key workflows, conventions, and integration points.

## 1. Project Overview
- Implements reinforcement learning experiments with transferable neural network representations.
- Defines agents, problems, and experiments as JSON specifications under `experiments/`.
- Core training loops live in `src/main.py` (single-task) and `src/continuing_main.py` (continuing tasks).

## 2. High-Level Architecture
```
config.json           # base paths & IO settings
experiments/          # JSON files describe experiments
src/                  # source code
  algorithms/         # agent implementations + registry
    nn/               # JAX/Haiku agents (DQN, DRQN, Aux variants)
  representations/    # Haiku network builders
  environments/       # Gym-like environment dynamics
  experiment/         # ExperimentModel + tools to parse JSON
  utils/              # checkpointing, plotting, policies, iterators
scripts/              # SLURM job scripts & helpers
```
- `algorithms.registry.getAgent(name)` maps JSON `agent` strings to classes.
- Parameter sweeps driven by `metaParameters` field in JSON.
- Data flows: experiment JSON → `ExperimentModel` → `Agent` + `Problem` → loop `{state, buffer, collector}`.

## 3. Developer Workflows
### Local iteration
```bash
env setup:
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
run a quick experiment:
  python src/main.py -e experiments/Gridworld/E1/P0/DQN-Relu.json -i 0
```
- Use `-i` to set random seed index.
- Metrics logged with `DEBUG:exp` prefix; collectors save to `results/`.

### Cluster (SLURM)
- Prepare `clusters/cedar.json` and ensure `pyproject.sif` exists.
- Example submission:
  ```bash
  apptainer exec -C -B .:$HOME pyproject.sif python scripts/slurm.py \
    --cluster clusters/cedar.json --runs 5 \
    -e experiments/Gridworld/E1/P0/DQN-Relu.json
  ```
- Generated SLURM scripts live in `slurm_scripts/`.

## 4. Conventions & Patterns
- **One agent per file** in `src/algorithms` and `src/algorithms/nn`.
- Neural nets built via `representations.networks.NetworkBuilder` and Haiku modules.
- Replay buffer interface: `ReplayTables.ReplayBuffer.Batch` exposes fields `x, a, r, xp, gamma, trans_id`.
- Loss functions in `utils.jax` (e.g., `huber_loss`, `mse_loss`).
- Checkpoint states in `utils.checkpoint.py`; loaded via `--resume` flags.

## 5. Testing & Quality
- Unit tests under `tests/` using `pytest`:
  ```bash
  pytest -q
  ```
- Notebooks in `analysis/` and `test_fta.ipynb` illustrate data pipelines.

## 6. External Integrations
- **JAX + Haiku + Optax** for neural network training.
- **ML-Instrumentation** for metrics collection (`Collector`).
- **SLURM** via `scripts/slurm.py` and `run.sh`, `submit.sh` wrappers.

---
*Let me know if any section needs more detail or clarification!*