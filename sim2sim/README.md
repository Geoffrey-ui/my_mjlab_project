# mjlab-sim2sim

Run policies trained with **mjlab** (MuJoCo-Warp) inside **ManiSkill3** (SAPIEN / PhysX), and measure the gap.

The package is robot-agnostic. It converts the *compiled* MuJoCo scene that mjlab simulated (robot, attached gripper, table, objects), so any robot in `src/mjlab_manipulation/robots` works without per-robot code.

```
 mjlab env (Python 3.14)                          ManiSkill3 env (Python 3.12)
 ───────────────────────                          ───────────────────────────
 logs/rsl_rl/<exp>/<run>/model_N.pt
          │  tools/export_mjlab.py
          ▼
 bundles/<exp>/<run>/
   spec.json      obs terms, action terms, timing, camera, command, baseline ─┐
   scene.mjb      compiled MuJoCo model                                       ├─► MjlabSim2Sim-v0
   policy.onnx    actor + obs normalizer (checked against torch)              │   sim2sim-play / -compare
   reference.npz  one mjlab rollout (obs, actions, qpos, targets)            ─┘
```

There are two Python environments because ManiSkill 3.0.1 / SAPIEN 3.0.3 only supports Python ≤ 3.12, while this project runs mjlab on 3.14. Bundles are plain files, so the two sides never import each other.

## Install

```bash
cd sim2sim
uv sync            # creates sim2sim/.venv (Python 3.12, mani_skill, mujoco==3.11.0, onnxruntime)
```

`mujoco` is pinned to the same version as the mjlab project because `scene.mjb` is version-specific. If you upgrade mjlab, bump the pin in `pyproject.toml` and re-export. If you see `VIRTUAL_ENV ... does not match` warnings, the root `.venv` is still active; you can ignore the warning or run `deactivate`.

## Workflow

**1. Export a run** (from the repository root, using the mjlab environment):

```bash
uv run python sim2sim/tools/export_mjlab.py logs/rsl_rl/lift_kuka_rgb/2026-10-20_17-12-35
#   --checkpoint model_1000.pt   (default: highest model_N.pt)
#   --task Lift-Kuka-RGB         (default: inferred from the experiment folder)
#   --eval-episodes 20           (mjlab baseline metrics stored in spec.json)
#   --ref-steps 50               (reference rollout length, 0 disables)
```

**2. Inspect / compare / play** (from `sim2sim/`):

```bash
cd sim2sim
uv run sim2sim-inspect bundles/lift_kuka_rgb/2026-10-20_17-12-35      # what was exported, shape checks
uv run sim2sim-compare bundles/lift_kuka_rgb/2026-10-20_17-12-35      # quantify the sim2sim gap
uv run sim2sim-play    bundles/lift_kuka_rgb/2026-10-20_17-12-35 --episodes 20 --camera-gamma 1.45
uv run sim2sim-play    bundles/reach_ur5e/2026-10-22_16-34-42 --render human        # SAPIEN viewer
uv run sim2sim-play    bundles/reach_ur5e/2026-10-22_16-34-42 --render video --save-wrist
uv run sim2sim-play    bundles/reach_ur5e/2026-10-22_16-34-42 --num-envs 16 --sim-backend gpu
```

`sim2sim-compare` starts ManiSkill from the exact initial state of the mjlab reference rollout and reports:

1. **Observations at t=0**: a per-term diff. Joint, EE and goal terms should be ~1e-7. It also saves an mjlab-vs-ManiSkill camera image and prints the best-fit `--camera-gamma`.
2. **Open-loop replay** of the mjlab actions: per-term drift, which isolates the *dynamics* gap.
3. **Closed loop**: the policy's action difference and the final task error.

`sim2sim-play` prints ManiSkill metrics next to the mjlab baseline, using the same metric definitions in both sims:

- success rate
- final and best position error
- lift rate (the object rose 2 cm or more above the table)

Use from Python:

```python
import gymnasium as gym
import mjlab_sim2sim  # registers MjlabSim2Sim-v0
from mjlab_sim2sim import Bundle, OnnxPolicy

bundle = Bundle.load("bundles/reach_ur5e/2026-10-22_16-34-42")
env = gym.make("MjlabSim2Sim-v0", bundle=bundle, num_envs=1, max_episode_steps=bundle.episode_steps)
policy = OnnxPolicy(bundle)
obs, _ = env.reset(seed=0)            # obs = {"actor": (N, D), "camera": (N, 3, H, W), ...}
obs, rew, term, trunc, info = env.step(policy(obs))   # info: success, position_error, ...
```

## Package layout

| module | role |
| --- | --- |
| `tools/export_mjlab.py` | mjlab side: spec, compiled scene, ONNX, reference rollout, baseline |
| `bundle.py` | loads a bundle (`spec.json` + `scene.mjb` + `policy.onnx`) |
| `mj_to_sapien.py` | compiled `MjModel` → SAPIEN articulation and actors, loop closures, mimic joints, lights |
| `agent.py` | a ManiSkill agent generated per bundle: PD gains, passive joints, wrist cameras |
| `kinematics.py` | MuJoCo used as the kinematics oracle (FK, site poses, Jacobians for IK) |
| `actions.py` | mjlab action terms: joint position, relative joint position, tendon length, differential IK |
| `observations.py` | mjlab observation functions, keyed by function name |
| `commands.py` | task commands: goal sampling, object resets, success metrics (reach, lift) |
| `env.py` | `MjlabSim2SimEnv`: per-substep action application, passive forces, observation groups |
| `policy.py` | ONNX runner that maps observation groups onto ONNX inputs |
| `cli.py`, `compare.py` | `sim2sim-play`, `sim2sim-inspect`, `sim2sim-compare` |

### Extending

New mjlab terms only need to be written once and then work for every robot. Use the registry keys printed by `sim2sim-inspect` / stored in `spec.json`:

```python
from mjlab_sim2sim.observations import register_obs
from mjlab_sim2sim.actions import register_action, ActionTerm
from mjlab_sim2sim.commands import register_command, Command

@register_obs("joint_pos")                # mjlab observation function name
def joint_pos(env, params):
    return env.ctx.qpos(params["asset_cfg"]["joints"])

@register_action("joint_velocity")        # action "type" written by the exporter
class JointVelocityTerm(ActionTerm): ...

@register_command("MyCommandCfg")         # mjlab command cfg class name
class MyCommand(Command): ...
```

The exporter raises `NotImplementedError` for action types it does not know. The runtime does the same for observation functions and commands without an implementation, so missing pieces fail loudly instead of silently producing wrong observations.

## What is transferred, and how

| MuJoCo / mjlab | ManiSkill3 |
| --- | --- |
| bodies, hinge/slide joints, inertia, armature | articulation links / joints (fixed base at the mocap root pose) |
| meshes (compiled, re-centered) | `.obj` cache in `bundle/.mesh_cache`; visual mesh + **convex hull** collision (as in MuJoCo) |
| `contype` / `conaffinity` | SAPIEN collision groups 0/1; robot self-collision disabled |
| `connect` equality (gripper 4-bars) | locked 3-DoF drive between the two links |
| joint equality `q1 = c0 + c1 q2` | PhysX fixed tendon (ManiSkill mimic joint) |
| position actuators (kp, kv, forcerange, ctrlrange) | PD joint drives, targets clamped to ctrlrange |
| tendon position actuator (gripper `split`) | equal PD targets on the tendon joints, gains distributed by tendon coefficients |
| joint damping / frictionloss / springs | explicit joint forces every substep (PhysX drives clamp damping with the actuator force) |
| `decimation`, `apply_actions` every substep | same: e.g. 500 Hz physics, 50 Hz policy, targets recomputed each substep |
| `reset_joints_by_offset` | default pose clamped into soft joint limits |
| camera (pose, fovy, resolution) | ManiSkill camera mounted on the same link |
| lights / headlight | ambient + directional lights |

### Known approximations (sources of sim2sim gap)

- **Contacts:** MuJoCo's soft contacts and its soft `connect` constraints let the gripper linkage deform when it grasps. PhysX is rigid. Grasps can behave differently, especially for thin pads.
- **frictionloss:** MuJoCo's frictionloss is a soft constraint. Here it is `tanh(q̇ / v0)` with `--frictionloss-v0 0.1`; 0 disables it, which matched mjlab's reach accuracy best for xArm.
- **Rendering:** textures are approximated by their mean color, and the MuJoCo headlight becomes a static light. SAPIEN output is display-encoded, so use the gamma printed by `sim2sim-compare`. The remaining pixel error is typically about 0.04–0.06.
- **Friction combination:** PhysX averages the friction of the two surfaces, while MuJoCo takes the maximum. Torsional and rolling friction (`condim 6`) are not modelled.
- **Not supported:** observation history, weld equalities, non-linear joint equalities, robots with floating bases.

## Results on this repository's runs (20 episodes, 1 s episodes as in mjlab)

| bundle | metric | ManiSkill3 | mjlab |
| --- | --- | --- | --- |
| reach_ur5e | success / final error | 100% / 3.9 mm | 100% / 1.2 mm |
| reach_xarm  | success / final error | 100% / ~6 mm (`v0=0`: ~2 mm) | 100% / 1.5 mm |
| lift_kuka_rgb (2F85) | success / lift rate | 90% / 90% | 80% / 80% |
| lift_gen3_rgb (2F85) | success / lift rate | 0% / 70% | 0% / 45% (weak policy in both) |
| lift_xarm_rgb (xArm hand) | success / lift rate | 0% / 70% | 70% / 100% |

The xArm-hand lift is the one real transfer failure. The fingers close on the cube but it slips out while lifting, and the policy keeps re-grasping. This is a contact-model gap: thin pads plus a 4-bar linkage that is compliant in MuJoCo and rigid in PhysX. It is not a kinematics, observation, or actuation mismatch: `sim2sim-compare` shows exact observations and millimetre-level open-loop arm tracking. Training with more friction or contact randomization in mjlab is the usual remedy.
