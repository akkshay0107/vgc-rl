# vgc-rl

Setup and run guide for the training pipeline:

`replay_gen -> behaviour cloning seed -> PPO train loop`

## Prereqs

If you cloned the repo without submodules, initialize them first:

```bash
git submodule update --init --recursive
```

## Python setup with uv

Install Python 3.13 into `uv` if you do not already have it:

```bash
uv python install 3.13
```

Create the virtualenv and install Python dependencies from `pyproject.toml` / `uv.lock`:

```bash
uv sync
```

From here on, run Python commands through `uv`:

```bash
uv run python <script>
```

## Pokemon Showdown setup

The project battles against a local Pokemon Showdown server.

Install the server dependencies once:

```bash
cd pokemon-showdown
npm install
cd ..
```

For replay generation, start one local server in a separate terminal:

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

That uses port `8000` by default, which matches the replay generation scripts.

## Pipeline

### Generate replay data

Run this from the repo root while the local Showdown server is running:

```bash
uv run python src/replay_gen.py -n <number of replays>
```

This writes replay shards under:

- `replays/fuzzy_heuristic/`
- `replays/simple_heuristic/`
- `replays/max_base_power/`

Increase `-n` if you want more expert trajectories.

### Seed the opponent pool with behaviour cloning

Train behaviour-cloned policies from those replay folders:

```bash
uv run python src/seed_pool.py
```

This creates seed checkpoints in:

- `checkpoints/pool/seed_max_base_power.pt`
- `checkpoints/pool/seed_simple_heuristic.pt`
- `checkpoints/pool/seed_fuzzy_heuristic.pt`

`src/train_loop.py` will automatically bootstrap from `seed_fuzzy_heuristic.pt` if there is no PPO checkpoint yet.

### Run the PPO training loop

Start training from the repo root:

```bash
uv run python src/train_loop.py
```

`train_loop.py` starts its own local Showdown processes, so you do not need to manually start the server for this step.

Training outputs:

- main checkpoint: `checkpoints/ppo_checkpoint.pt`
- opponent snapshots: `checkpoints/pool/`
- TensorBoard logs: `runs/ppo_training/`
- text log: `training.log`

## PPO config

Training reads optional overrides from `.ppoconfig` in the repo root.

Current local example (for my CPU):

```text
num_episodes=4
num_envs=1
n_jobs=2
batch_size=4
```

If `.ppoconfig` is missing, defaults from `src/ppo_utils.py` are used.

## Typical run order

```bash
uv sync
cd pokemon-showdown && npm install && cd ..

# terminal 1
cd pokemon-showdown && node pokemon-showdown start --no-security

# terminal 2
uv run python src/replay_gen.py -n <number of replays>
uv run python src/seed_pool.py
uv run python src/train_loop.py
```
