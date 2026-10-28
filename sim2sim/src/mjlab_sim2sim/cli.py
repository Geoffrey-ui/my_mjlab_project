"""Command line entry points: ``sim2sim-play`` and ``sim2sim-inspect``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import tyro

import mjlab_sim2sim  # noqa: F401  (registers the env)
from mjlab_sim2sim.bundle import Bundle
from mjlab_sim2sim.env import ENV_ID
from mjlab_sim2sim.policy import OnnxPolicy


@dataclass
class PlayArgs:
  bundle: tyro.conf.Positional[Path]
  """Bundle directory produced by tools/export_mjlab.py."""
  episodes: int = 5
  num_envs: int = 1
  sim_backend: Literal["auto", "cpu", "gpu"] = "auto"
  episode_steps: int | None = None
  """Steps per episode (default: mjlab episode_length_s / control dt)."""
  render: Literal["none", "human", "video"] = "none"
  """'human' opens the SAPIEN viewer, 'video' writes mp4s into --out."""
  out: Path = Path("outputs")
  save_wrist: bool = False
  """Also save the policy's camera images as a video (camera policies only)."""
  camera_gamma: float = 1.0
  """Tone correction for camera policies (see sim2sim-compare)."""
  frictionloss_v0: float = 0.1
  """Smoothing of MuJoCo joint frictionloss (0 disables)."""
  agent: Literal["trained", "zero", "random"] = "trained"
  seed: int = 0


def make_env(bundle: Bundle, args: PlayArgs):
  steps = args.episode_steps or bundle.episode_steps
  env = gym.make(
    ENV_ID,
    bundle=bundle,
    num_envs=args.num_envs,
    sim_backend=args.sim_backend,
    render_mode={"none": None, "human": "human", "video": "rgb_array"}[args.render],
    max_episode_steps=steps,
    camera_gamma=args.camera_gamma,
    frictionloss_v0=args.frictionloss_v0,
  )
  return env, steps


def play(args: PlayArgs) -> dict:
  bundle = Bundle.load(args.bundle)
  print(bundle.summary())
  env, steps = make_env(bundle, args)
  base = env.unwrapped
  policy = OnnxPolicy(bundle)
  run_out = args.out / bundle.spec["experiment"] / bundle.spec["run"]
  frames, wrist = [], []

  lift = next((c["cfg"] for c in bundle.commands if "table_height" in c["cfg"]), None)
  lift_height = lift["table_height"] + 0.02 if lift else None
  episodes = []
  obs, _ = env.reset(seed=args.seed)
  for ep in range(args.episodes):
    if ep > 0:
      obs, _ = env.reset()
    log: dict[str, list] = {}
    for _ in range(steps):
      if args.agent == "trained":
        action = policy(obs)
      elif args.agent == "zero":
        action = torch.zeros(base.num_envs, bundle.action_dim, device=base.device)
      else:
        action = 2 * torch.rand(base.num_envs, bundle.action_dim, device=base.device) - 1
      obs, _, terminated, truncated, info = env.step(action)
      for k in ("position_error", "episode_success", "object_height"):
        if k in info:
          log.setdefault(k, []).append(info[k].float().cpu().numpy())
      if args.render == "human":
        env.render()
      elif args.render == "video":
        frames.append(common_frame(env.render()))
      if args.save_wrist and base.sensor_data:
        wrist.append(next(iter(base.sensor_data.values()))["rgb"][0].cpu().numpy())
      if (terminated | truncated).all():
        break
    # One record per parallel env so num_envs > 1 counts as more episodes.
    for i in range(base.num_envs):
      episodes.append({k: np.array(v)[:, i] for k, v in log.items()})
    print(f"[ep {ep}] success={np.mean(log['episode_success'][-1]):.2f}  "
          f"final_pos_error={np.mean(log['position_error'][-1]):.4f} m")

  summary = summarize_episodes(episodes, lift_height)
  baseline = bundle.spec.get("baseline")
  print(f"\n== {bundle.name} ==")
  print(f"{'metric':<14}{'maniskill':>12}{'mjlab':>12}")
  for k, v in summary.items():
    ref = baseline.get(k, float("nan")) if baseline else float("nan")
    print(f"{k:<14}{v:>12.4f}{ref:>12.4f}")
  if frames or wrist:
    import imageio

    run_out.mkdir(parents=True, exist_ok=True)
    fps = round(1.0 / bundle.control_dt)
    if frames:
      imageio.mimsave(run_out / "render.mp4", frames, fps=fps)
      print(f"video  -> {run_out / 'render.mp4'}")
    if wrist:
      imageio.mimsave(run_out / "wrist.mp4", [np.repeat(np.repeat(w, 4, 0), 4, 1) for w in wrist], fps=fps)
      print(f"wrist  -> {run_out / 'wrist.mp4'}")
  env.close()
  return summary


def summarize_episodes(episodes: list[dict], lift_height: float | None) -> dict:
  """Metric definitions shared with tools/export_mjlab.py (keep in sync)."""
  out = {
    "episodes": len(episodes),
    "success_rate": float(np.mean([e["episode_success"][-1] for e in episodes])),
    "final_error": float(np.mean([e["position_error"][-1] for e in episodes])),
    "min_error": float(np.mean([np.min(e["position_error"]) for e in episodes])),
  }
  if lift_height is not None and all("object_height" in e for e in episodes):
    out["lift_rate"] = float(np.mean([np.max(e["object_height"]) > lift_height for e in episodes]))
  return out


def common_frame(frame) -> np.ndarray:
  if isinstance(frame, torch.Tensor):
    frame = frame.cpu().numpy()
  return frame[0] if frame.ndim == 4 else frame


def main() -> None:
  play(tyro.cli(PlayArgs))


@dataclass
class InspectArgs:
  bundle: tyro.conf.Positional[Path]
  build: bool = True
  """Also build the ManiSkill env and check observation shapes against the policy."""


def inspect_main() -> None:
  args = tyro.cli(InspectArgs)
  bundle = Bundle.load(args.bundle)
  print(bundle.summary())
  if not args.build:
    return
  env = gym.make(ENV_ID, bundle=bundle, num_envs=1)
  obs, _ = env.reset(seed=0)
  base = env.unwrapped
  print("\nManiSkill articulation joints:", [j.name for j in base.agent.robot.get_active_joints()])
  print("driven:", base.agent.driven_joints)
  print("passive:", base.agent.passive_joints)
  for g, v in obs.items():
    print(f"obs[{g}] shape={tuple(v.shape)}")
  sess_inputs = {i.name: i.shape for i in OnnxPolicy(bundle).session.get_inputs()}
  print("policy inputs:", sess_inputs)
  print("first action:", OnnxPolicy(bundle)(obs).cpu().numpy().round(3))
  env.close()
