# VGC-RL

`WIKI.md` for all resources

Find a good name for the project as well

### Installation procedure

Clone the repo and initialize the pokemon-showdown submodule

```bash
git clone https://github.com/akkshay0107/vgc-rl.git
git submodule init
git submodule update --init --recursive
```

#### uv setup

- Install uv (see https://docs.astral.sh/uv/getting-started/installation/)

- Install all dependencies

```bash
uv sync
uv pip install .
```

- Run code through the venv (Alternatively, activate venv manually and use regular python commands)

```bash
uv run <command> [args..]
```

#### Local showdown server setup

From the root of the project,

```bash
cd pokemon-showdown
npm i # first time install
node pokemon-showdown start --no-security
```

It should default to starting the server at port 8000. Turn off any browser shields / blocker and retry if server fails to load properly.
