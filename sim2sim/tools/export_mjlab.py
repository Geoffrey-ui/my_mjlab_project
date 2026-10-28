"""Export an mjlab training run into a self-contained sim2sim bundle.

Run this with the *mjlab* environment (the root project), e.g.:

  uv run python sim2sim/tools/export_mjlab.py logs/rsl_rl/lift_xarm_rgb/2026-10-21_11-49-25

The bundle directory contains:
  spec.json    -- observation / action / sim / camera / command description
  scene.mjb    -- compiled MuJoCo scene (robot + table + objects)
  policy.onnx  -- actor exported from the chosen checkpoint (normalizer baked in)
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import tyro

import mjlab_manipulation  # noqa: F401  (registers the custom tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import DifferentialIKAction
from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  RelativeJointPositionAction,
  TendonLengthAction,
)
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

SPEC_VERSION = 1


@dataclasses.dataclass
class ExportArgs:
  run_dir: tyro.conf.Positional[Path]
  """Training run directory, e.g. logs/rsl_rl/reach_xarm/2026-10-23_11-37-12."""
  checkpoint: str | None = None
  """Checkpoint file name inside run_dir (default: highest model_<N>.pt)."""
  task: str | None = None
  """mjlab task id (default: inferred from the experiment folder name)."""
  out: Path | None = None
  """Output bundle dir (default: sim2sim/bundles/<experiment>/<run>)."""
  device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
  ref_steps: int | None = None
  """Steps of reference rollout to record (default: one episode, 0 disables)."""
  eval_episodes: int = 20
  """Episodes used to measure the mjlab baseline (0 disables)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_checkpoint(run_dir: Path) -> Path:
  ckpts = sorted(
    run_dir.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])
  )
  if not ckpts:
    raise FileNotFoundError(f"No model_*.pt found in {run_dir}")
  return ckpts[-1]


def _infer_task(experiment: str) -> str:
  matches = [t for t in list_tasks() if load_rl_cfg(t).experiment_name == experiment]
  if not matches:
    raise ValueError(f"No registered task has experiment_name={experiment!r}; pass --task.")
  # Prefer the shortest id (e.g. Lift-xArm-RGB over Lift-xArm-RGB-Play variants).
  return sorted(matches, key=len)[0]


def _to_jsonable(x):
  if isinstance(x, torch.Tensor):
    return x.detach().cpu().tolist()
  if isinstance(x, np.ndarray):
    return x.tolist()
  if isinstance(x, (np.floating, np.integer)):
    return x.item()
  if dataclasses.is_dataclass(x) and not isinstance(x, type):
    return {f.name: _to_jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
  if isinstance(x, dict):
    return {str(k): _to_jsonable(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_to_jsonable(v) for v in x]
  if isinstance(x, (str, int, float, bool)) or x is None:
    return x
  return repr(x)


def _ids_to_list(ids, n: int) -> list[int]:
  if isinstance(ids, slice):
    return list(range(n))[ids]
  if isinstance(ids, torch.Tensor):
    return ids.cpu().tolist()
  return list(ids)


def _full_names(env: ManagerBasedRlEnv, entity_name: str, kind: str, ids) -> list[str]:
  """Resolve entity-local ids to full MuJoCo names (e.g. 'robot/joint_1')."""
  entity = env.scene[entity_name]
  global_ids = getattr(entity.indexing, f"{kind}_ids").cpu().tolist()
  lookup = getattr(env.sim.mj_model, kind)
  return [lookup(global_ids[i]).name for i in _ids_to_list(ids, len(global_ids))]


# ---------------------------------------------------------------------------
# Spec extraction
# ---------------------------------------------------------------------------


def _export_obs_term(env: ManagerBasedRlEnv, group: str, name: str, dim) -> dict:
  cfg = env.observation_manager.get_term_cfg(group, name)
  params: dict = {}
  for key, val in cfg.params.items():
    if hasattr(val, "joint_ids") and hasattr(val, "name"):  # SceneEntityCfg
      entry = {"entity": val.name}
      if val.joint_names is not None:
        entry["joints"] = _full_names(env, val.name, "joint", val.joint_ids)
      if val.site_names is not None:
        entry["sites"] = _full_names(env, val.name, "site", val.site_ids)
      if val.body_names is not None:
        entry["bodies"] = _full_names(env, val.name, "body", val.body_ids)
      params[key] = entry
    else:
      params[key] = _to_jsonable(val)
  scale = cfg.scale
  return {
    "name": name,
    "func": getattr(cfg.func, "__name__", type(cfg.func).__name__),
    "shape": list(dim),
    "params": params,
    "scale": _to_jsonable(scale) if scale is not None else None,
    "clip": list(cfg.clip) if cfg.clip is not None else None,
    "history_length": cfg.history_length,
  }


def _export_action_term(env: ManagerBasedRlEnv, name: str) -> dict:
  term = env.action_manager.get_term(name)
  ent_name = term.cfg.entity_name
  m = env.sim.mj_model
  out: dict = {"name": name, "dim": term.action_dim, "entity": ent_name}

  if isinstance(term, DifferentialIKAction):
    c = term.cfg
    jids = term._joint_ids.cpu().tolist()
    out.update(
      type="differential_ik",
      joints=_full_names(env, ent_name, "joint", jids),
      frame_type=c.frame_type,
      frame=m.site(term._frame_id).name if c.frame_type == "site"
      else m.body(term._frame_id).name if c.frame_type == "body"
      else m.geom(term._frame_id).name,
      use_relative_mode=c.use_relative_mode,
      delta_pos_scale=c.delta_pos_scale,
      delta_ori_scale=c.delta_ori_scale,
      damping=c.damping,
      max_dq=c.max_dq,
      position_weight=c.position_weight,
      orientation_weight=c.orientation_weight,
      joint_limit_weight=c.joint_limit_weight,
      posture_weight=c.posture_weight,
      posture_target=_to_jsonable(term._posture_target[0]),
      joint_lower=_to_jsonable(term._joint_lower[0]),
      joint_upper=_to_jsonable(term._joint_upper[0]),
    )
  elif isinstance(term, (JointPositionAction, RelativeJointPositionAction)):
    # Joint targets: map actuator targets back to joint names.
    joint_names = _full_names(env, ent_name, "joint", term._target_ids)
    out.update(
      type="relative_joint_position"
      if isinstance(term, RelativeJointPositionAction)
      else "joint_position",
      joints=joint_names,
    )
  elif isinstance(term, TendonLengthAction):
    names = _full_names(env, ent_name, "tendon", term._target_ids)
    tendons = []
    for tname in names:
      tid = m.tendon(tname).id
      act = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON
             and m.actuator_trnid[a, 0] == tid]
      if not act:
        raise ValueError(f"No actuator drives tendon {tname}")
      tendons.append(_tendon_actuator(m, act[0]))
    out.update(type="tendon_length", tendons=tendons)
  else:
    raise NotImplementedError(f"Action term {name!r} of type {type(term).__name__}")

  if out["type"] != "differential_ik":
    out["scale"] = _to_jsonable(term._scale[0] if isinstance(term._scale, torch.Tensor) else term._scale)
    out["offset"] = _to_jsonable(term._offset[0] if isinstance(term._offset, torch.Tensor) else term._offset)
    out["clip"] = _to_jsonable(term._clip[0]) if getattr(term, "_clip", None) is not None else None
  return out


def _tendon_actuator(m: mujoco.MjModel, a: int) -> dict:
  """Fixed-tendon position actuator: tendon joints, coefficients and gains."""
  tid = m.actuator_trnid[a, 0]
  adr, num = m.tendon_adr[tid], m.tendon_num[tid]
  return {
    "name": m.actuator(a).name,
    "tendon": m.tendon(tid).name,
    "joints": [m.joint(m.wrap_objid[k]).name for k in range(adr, adr + num)],
    "coefs": [float(m.wrap_prm[k]) for k in range(adr, adr + num)],
    "gain": float(m.actuator_gainprm[a, 0]),
    "bias": m.actuator_biasprm[a, :3].tolist(),
    "ctrlrange": m.actuator_ctrlrange[a].tolist() if m.actuator_ctrllimited[a] else None,
    "forcerange": m.actuator_forcerange[a].tolist() if m.actuator_forcelimited[a] else None,
  }


def _export_tendon_actuators(env: ManagerBasedRlEnv) -> list[dict]:
  """All tendon actuators of the robot -- also those no action term commands
  (e.g. the gripper in reach tasks, which MuJoCo still holds at ctrl=0)."""
  m = env.sim.mj_model
  return [_tendon_actuator(m, a) for a in range(m.nu)
          if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON
          and m.actuator(a).name.startswith("robot/")]


def _export_actuators(env: ManagerBasedRlEnv) -> list[dict]:
  """Joint-transmission position actuators (kp, kv, force limits)."""
  m = env.sim.mj_model
  out = []
  for a in range(m.nu):
    if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
      continue
    out.append({
      "name": m.actuator(a).name,
      "joint": m.joint(m.actuator_trnid[a, 0]).name,
      "kp": float(m.actuator_gainprm[a, 0]),
      "kv": float(-m.actuator_biasprm[a, 2]),
      "ctrlrange": m.actuator_ctrlrange[a].tolist() if m.actuator_ctrllimited[a] else None,
      "forcerange": m.actuator_forcerange[a].tolist() if m.actuator_forcelimited[a] else None,
    })
  return out


def _export_cameras(env: ManagerBasedRlEnv) -> list[dict]:
  from mjlab.sensor import CameraSensorCfg

  m = env.sim.mj_model
  out = []
  for s in env.cfg.scene.sensors or ():
    if not isinstance(s, CameraSensorCfg):
      continue
    cam = m.camera(s.camera_name)
    out.append({
      "sensor_name": s.name,
      "camera": cam.name,
      "body": m.body(m.cam_bodyid[cam.id]).name,
      "pos": m.cam_pos[cam.id].tolist(),
      "quat": m.cam_quat[cam.id].tolist(),
      "fovy": float(s.fovy if s.fovy is not None else m.cam_fovy[cam.id]),
      "width": s.width,
      "height": s.height,
      "data_types": list(s.data_types),
      "znear": float(m.vis.map.znear * m.stat.extent),
      "zfar": float(m.vis.map.zfar * m.stat.extent),
    })
  return out


def _export_commands(env: ManagerBasedRlEnv) -> list[dict]:
  out = []
  for name in env.command_manager.active_terms:
    cfg = env.command_manager.get_term(name).cfg
    d = _to_jsonable(cfg)
    d.pop("viz", None)
    out.append({"name": name, "type": type(cfg).__name__, "cfg": d})
  return out


def _export_policy(env, runner, out_dir: Path, obs_groups: dict) -> dict:
  runner.export_policy_to_onnx(str(out_dir), "policy.onnx")
  sess = ort.InferenceSession(str(out_dir / "policy.onnx"), providers=["CPUExecutionProvider"])

  # Map each ONNX input onto the mjlab observation groups it consumes.
  actor_groups = list(obs_groups.get("actor", ["actor"]))
  obs = env.get_observations()
  flat = [g for g in actor_groups if obs[g].ndim == 2]
  images = [g for g in actor_groups if obs[g].ndim > 2]
  inputs = {}
  for inp in sess.get_inputs():
    if len(inp.shape) == 2:
      inputs[inp.name] = flat
    else:
      inputs[inp.name] = [images.pop(0)]

  # Sanity check: ONNX vs torch inference policy on a real observation.
  policy = runner.get_inference_policy(device=env.device)
  with torch.no_grad():
    ref = policy(obs).cpu().numpy()
  feed = {
    k: torch.cat([obs[g] for g in gs], dim=-1).cpu().numpy().astype(np.float32)
    for k, gs in inputs.items()
  }
  got = sess.run(None, feed)[0]
  err = float(np.abs(ref - got).max())
  print(f"[export] ONNX vs torch max abs diff: {err:.2e}")
  if err > 1e-3:
    print("[export] WARNING: ONNX output differs from the torch policy!")
  return {
    "file": "policy.onnx",
    "inputs": inputs,
    "output": sess.get_outputs()[0].name,
    "clip_actions": runner.cfg.get("clip_actions"),
    "onnx_vs_torch_max_err": err,
  }


def _record_reference(env: ManagerBasedRlEnv, vec_env, runner, steps: int, out: Path) -> None:
  """Roll out the policy in mjlab and save what ManiSkill should reproduce.

  Saved per step t (state *before* action t is applied):
    obs/<group>, action, qpos (full MjData.qpos), command/<name>
  """
  policy = runner.get_inference_policy(device=env.device)
  obs = vec_env.get_observations()
  rec: dict[str, list] = {}

  def push(key, val):
    rec.setdefault(key, []).append(torch.as_tensor(val).detach().cpu().numpy()[0].copy())

  for _ in range(steps):
    for g in obs.keys():
      push(f"obs/{g}", obs[g])
    push("qpos", env.sim.data.qpos)
    for name in env.command_manager.active_terms:
      push(f"command/{name}", env.command_manager.get_command(name))
    with torch.no_grad():
      action = policy(obs)
    push("action", action)
    obs, _, dones, extras = vec_env.step(action)
    if bool(dones[0]):
      break
  np.savez_compressed(out / "reference.npz", **{k: np.stack(v) for k, v in rec.items()})
  print(f"[export] recorded {len(rec['action'])}-step reference rollout")


def summarize_episodes(episodes: list[dict], lift_height: float | None) -> dict:
  """Same definitions as mjlab_sim2sim.cli.summarize_episodes."""
  out = {
    "episodes": len(episodes),
    "success_rate": float(np.mean([e["episode_success"][-1] for e in episodes])),
    "final_error": float(np.mean([e["position_error"][-1] for e in episodes])),
    "min_error": float(np.mean([np.min(e["position_error"]) for e in episodes])),
  }
  if lift_height is not None and all("object_height" in e for e in episodes):
    out["lift_rate"] = float(np.mean([np.max(e["object_height"]) > lift_height for e in episodes]))
  return out


def _evaluate_baseline(env: ManagerBasedRlEnv, vec_env, runner, episodes: int) -> dict:
  policy = runner.get_inference_policy(device=env.device)
  name = env.command_manager.active_terms[0]
  term = env.command_manager.get_term(name)
  lift_height = getattr(term.cfg, "table_height", None)
  lift_height = None if lift_height is None else lift_height + 0.02
  results = []
  for _ in range(episodes):
    obs, _ = vec_env.reset()
    ep: dict[str, list] = {}
    while True:
      with torch.no_grad():
        obs, _, dones, _ = vec_env.step(policy(obs))
      for k in ("position_error", "episode_success", "object_height"):
        if k in term.metrics:
          ep.setdefault(k, []).append(float(term.metrics[k][0]))
      if bool(dones[0]):
        break
    results.append(ep)
  summary = summarize_episodes(results, lift_height)
  print(f"[export] mjlab baseline over {episodes} episodes: {summary}")
  return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: ExportArgs) -> Path:
  run_dir = args.run_dir.resolve()
  experiment = run_dir.parent.name
  task = args.task or _infer_task(experiment)
  ckpt = run_dir / args.checkpoint if args.checkpoint else _latest_checkpoint(run_dir)
  out = (args.out or Path(__file__).resolve().parents[1] / "bundles" / experiment / run_dir.name)
  out.mkdir(parents=True, exist_ok=True)
  print(f"[export] task={task}  checkpoint={ckpt}  ->  {out}")

  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.auto_reset = False  # keep terminal states readable for the baseline
  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(env_cfg, device=args.device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=args.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=args.device)

  m = env.sim.mj_model
  robot = env.scene["robot"]
  om = env.observation_manager

  spec = {
    "version": SPEC_VERSION,
    "task": task,
    "experiment": experiment,
    "run": run_dir.name,
    "checkpoint": ckpt.name,
    "exported_at": datetime.now().isoformat(timespec="seconds"),
    "mujoco_version": mujoco.__version__,
    "scene_model": "scene.mjb",
    "sim": {
      "timestep": float(m.opt.timestep),
      "decimation": int(env_cfg.decimation),
      "episode_length_s": float(env_cfg.episode_length_s),
      "gravity": m.opt.gravity.tolist(),
    },
    "robot": {
      "entity": "robot",
      "root_body": m.body(robot.indexing.root_body_id).name,
      "joints": _full_names(env, "robot", "joint", slice(None)),
      "default_joint_pos": _to_jsonable(robot.data.default_joint_pos[0]),
      "soft_joint_pos_limits": _to_jsonable(robot.data.soft_joint_pos_limits[0]),
      "actuators": _export_actuators(env),
      "tendon_actuators": _export_tendon_actuators(env),
    },
    "observations": {
      g: [_export_obs_term(env, g, n, d) for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g])]
      for g in om.active_terms
    },
    "actions": [_export_action_term(env, n) for n in env.action_manager.active_terms],
    "cameras": _export_cameras(env),
    "commands": _export_commands(env),
  }
  obs_groups = agent_cfg.obs_groups or {"actor": ["actor"]}
  spec["policy"] = _export_policy(vec_env, runner, out, obs_groups)

  steps = args.ref_steps if args.ref_steps is not None else round(
    env_cfg.episode_length_s / (m.opt.timestep * env_cfg.decimation)
  )
  if steps > 0:
    vec_env.reset()
    _record_reference(env, vec_env, runner, steps, out)
  if args.eval_episodes > 0:
    spec["baseline"] = _evaluate_baseline(env, vec_env, runner, args.eval_episodes)

  mujoco.mj_saveModel(m, str(out / "scene.mjb"))
  (out / "spec.json").write_text(json.dumps(spec, indent=2))
  params_dir = run_dir / "params"
  if params_dir.exists():
    shutil.copytree(params_dir, out / "params", dirs_exist_ok=True)
  print(f"[export] wrote {out}")
  env.close()
  return out


if __name__ == "__main__":
  main(tyro.cli(ExportArgs))
