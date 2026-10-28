# my-mjlab-project

Robot manipulation tasks built on [mjlab](https://github.com/mujocolab/mjlab).
The project includes reaching and vision-based cube-lifting tasks for several
robot arms.

## Installation

Install [uv](https://docs.astral.sh/uv/) first, then run:

```bash
uv sync
```

This project requires Python 3.14 or newer. To check that the custom tasks are
registered correctly:

```bash
uv run train --help
```

## Getting started

Tasks are selected by their task ID. Common task IDs include:

- `Reach-Droid`, `Reach-xArm`, `Reach-Gen3`, `Reach-Kuka`, `Reach-UR5e`
- `Lift-Droid`, `Lift-xArm`, `Lift-Gen3`, `Lift-Kuka`, `Lift-UR5e`
- `Lift-Droid-RGB`, `Lift-xArm-RGB`, `Lift-Gen3-RGB`, `Lift-Kuka-RGB`,
  `Lift-UR5e-RGB`

Use the task's help output to see all available options:

```bash
uv run train Lift-Gen3-RGB --help
uv run play Lift-Gen3-RGB --help
```

## Training

For example, train the Gen3 RGB lifting policy with 256 parallel environments:

```bash
uv run train Lift-Gen3-RGB --env.scene.num-envs 256
```

Training outputs are written under:

```text
logs/rsl_rl/<experiment>/<timestamp>/
```

The directory contains checkpoints such as `model_1000.pt`.

## Visualize the policy

Play a trained checkpoint with the mjlab viewer:

```bash
uv run play Lift-Gen3-RGB \
  --checkpoint-file logs/rsl_rl/lift_gen3_rgb/<timestamp>/model_1000.pt \
  --viewer viser
```

Replace `<timestamp>` and the checkpoint name with the files in your training
run directory.

## Sim2sim

The `sim2sim` package runs exported policies in ManiSkill 3 / SAPIEN. It uses a
separate Python environment because ManiSkill 3 supports Python 3.12, while the
main mjlab project uses Python 3.14.

### 1. Install the sim2sim environment

```bash
cd sim2sim
uv sync
cd ..
```

### 2. Export a trained run

Run this from the repository root. The command creates a bundle containing the
compiled scene, policy, observation/action metadata, and a reference rollout:

```bash
uv run python sim2sim/tools/export_mjlab.py \
  logs/rsl_rl/lift_gen3_rgb/<timestamp>
```

To select a checkpoint explicitly:

```bash
uv run python sim2sim/tools/export_mjlab.py \
  logs/rsl_rl/lift_gen3_rgb/<timestamp> \
  --checkpoint model_1000.pt \
  --task Lift-Gen3-RGB
```

Bundles are written to:

```text
sim2sim/bundles/<experiment>/<timestamp>/
```

### 3. Inspect and run the bundle

Run these commands from `sim2sim/`:

```bash
cd sim2sim

uv run sim2sim-inspect bundles/lift_gen3_rgb/<timestamp>
uv run sim2sim-compare bundles/lift_gen3_rgb/<timestamp>
uv run sim2sim-play bundles/lift_gen3_rgb/<timestamp> --render human
```

`sim2sim-inspect` checks the exported bundle and observation shapes.
`sim2sim-compare` measures the difference between mjlab and SAPIEN.
`sim2sim-play --render human` opens the SAPIEN viewer.

For a saved video instead of an interactive viewer:

```bash
uv run sim2sim-play \
  bundles/lift_gen3_rgb/<timestamp> \
  --render video \
  --save-wrist
```
