## vgc-rl: Reinforcement Learning for Gen 9 Pokémon VGC

This repository implements a multi-stage reinforcement learning framework for playing Pokémon Gen 9 VGC (Regulation H) for a fixed metagame. The training procedure combines trajectory bootstrapping via Behavioural Cloning with Proximal Policy Optimization (PPO) in a league based self play setup (partially inspired by AlphaStar and OpenAI Five).

---

## Setup and Installation

### Prerequisites

The project relies on a local instance of the Pokémon Showdown server. Initialize the submodules to fetch the source:

```bash
git submodule update --init --recursive
```

### Environment Configuration

`uv` is used for Python 3.13 environment and dependency management:

```bash
uv python install 3.13
uv sync
```

Install the Node.js dependencies for the Pokémon Showdown server:

```bash
cd pokemon-showdown && npm install && cd ..
```

### Execution Workflow

A full training run follows a sequential pipeline:

1. Start a local Showdown server.

   ```bash
   cd pokemon-showdown && node pokemon-showdown start --no-security
   ```

2. Generate replays from heuristics.

   ```bash
   uv run python src/replay_gen.py -n 500
   ```

   After this step, you can stop the local showdown from step 1.

3. Create the initial pool of policies through behaviour cloning across the heuristic data.

   ```bash
   uv run python src/seed_pool.py
   ```

4. Launch the PPO training loop. This script manages its own pokemon-showdown server processes.
   ```bash
   uv run python src/train_loop.py
   ```

---

## Training Procedure

### 1. Replay Generation

The pipeline begins by recording battles between various heuristic agents (Fuzzy Heuristics, Simple Heuristics, and Max Base Power). MaxBasePower and SimpleHeuristic come from the implementation in poke-env and FuzzyHeuristic can be found in `src/heuristic.py`. They aren't particularly good (an average human player should be able to beat them comfortably) but they provide a base for weeding out poor moves (such as KOing your own teammate unnecessarily).

### 2. Behavioural Cloning Bootstrapping

The replays from the previous step are used in supervised learning to "seed" the neural network. By training the model to predict the actions of these heuristics, we get a semi-decent starting pool of policies to start out the league with and don't waste compute trying to explore extremely poor moves.

### 3. League-Based PPO Training

The agent is optimized via Proximal Policy Optimization (PPO) within a league-based environment. The objective of the pool of opponents is to search for more general strategies rather than a chain of policies that learn to exploit the previous model. As of right now, the latest policy faces the following opponents:

- **Latest Self**: The current version of the training policy (50% probability).
- **Opponent Pool**: Historical snapshots of the agent and initial seeds. FPSP used for sampling, where the agent is more likely to face opponents that currently have a good win rate against the previous few policies. Once the pool is large enough, the policy with the lowest win rate is evicted.

---

## Architecture Summary

### Observation structure

The observation to the model is a static embedding produced by passing text inputs from observed features in the ongoing battle into **TinyBERT** and using the concatenation of the CLS token and the mean. It consists of:

- H1, H2, H3: Text summary embeddings of the outcomes of the last 3 turns
- Field State: Text embedding of the global and local effects on the field (Weather, Terrain, Trick Room, Tailwind, etc).
- Global Info: Text embedding containing any other global info. Contains only the terastallization info for now, but has the space to potentially encode more info.
- Player and Opponents pokemon: 2 text embeddings per pokemon for both player and opponent. Contains info about ability, type, moveset, etc.

The remaining features are purely numerical, but also extracted from the poke-env Battle object.

- Field Nums: Turns of each type of global / local field effect.
- Player and Opponents pokemon nums: Base stats, boosts, status condition (one hot), etc.

### Hybrid State Representation

The model can be thought of as a spatio-temporal model, using an Encoder only Transformer to attend to the spatial features (pokemon + field info) from the static encoding + one time dependent hidden state token (HG).

Over the time axis, the model acts as an RNN, updating the hidden state (HG) with the new information from the latest turn.

The update logic follows:

```math
\text{HG}_t = g(\text{HG}_{t-1}, \text{CLS}_{t-1}, \text{H1}_t)
```

```math
\text{CLS}_t = f(\text{HG}_t, \text{O}_t)
```

The CLS token becomes the internal state representation for the turn, and also the shared backbone, from which the policy and value network split.

### Autoregressive Doubles Policy

Doubles coordination is modeled as a sequential decision process:

1. **First Action Prediction**: The policy predicts the action for the first Pokémon slot, $P(a_1 \mid z)$
2. **Conditional Action Prediction**: The first action is embedded and used to condition the prediction for the second slot autoregressively, $P(a_2 \mid z, a_1)$.
3. **Sequential Masking**: Legal move filtering is applied at each step, preventing coordination errors such as double-terastallization or invalid switch-ins.

The value head is pretty standard, a small MLP with GELU as the activation function with a single scalar output at the end of it. In the latest policy, it also has a custom grad scaler that scales down the value losses before the enter the shared backbone to reduce value interference with the shared backbone.

**NOTE:** In the old policy, the model was not autoregressive, instead it generated $P(a_1 \mid z)$ and $P(a_2 \mid z)$ independently and applied sequential masking to prevent illegal moves from happening. It lead to a decent model, but ran into the coordination issues. The old policy also did not do the grad scaling from the value loss.

---

### Configuration

Hyperparameters (learning rates, entropy coefficients, batch sizes) can be modified via a `.ppoconfig` file in the root directory. If absent, the system defaults to the parameters defined in `src/ppo_utils.py`.

### Analytics

- **TensorBoard**: Training metrics (Win Rate, KL Divergence, Explained Variance) are logged to `runs/ppo_training/`.
- **Logging**: A text output of the same is saved to `training.log`.
- **Checkpoints**: The primary PPO checkpoint is stored at `checkpoints/ppo_checkpoint.pt`, with opponent snapshots archived in `checkpoints/pool/`. `checkpoints/pool/pool_state.json` gives you more information about how the latest checkpoint is doing against the other policies in the pool.

---

### Results

Will be added soon. Unfortunately, due to the nature of the closed metagame, I will not be able to find a stable elo rating for this on the ladder + the fact that Reg H no longer is a format on the official Pokemon Showdown server.
