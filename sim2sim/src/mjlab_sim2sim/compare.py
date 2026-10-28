"""``sim2sim-compare``: quantify the mjlab -> ManiSkill gap using reference.npz.

Three checks, all starting from the exact initial state of the mjlab rollout:
  1. observation  -- per-term difference at t=0 (+ side-by-side camera image)
  2. open-loop    -- replay mjlab's actions, track per-term drift (dynamics gap)
  3. closed-loop  -- run the policy in ManiSkill, compare actions / task error
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro

from mjlab_sim2sim.bundle import Bundle
from mjlab_sim2sim.env import ENV_ID
from mjlab_sim2sim.policy import OnnxPolicy


@dataclass
class CompareArgs:
  bundle: tyro.conf.Positional[Path]
  steps: int | None = None
  """Number of steps to compare (default: full reference length)."""
  out: Path = Path("outputs")


def _term_slices(bundle: Bundle, group: str) -> list[tuple[str, slice]]:
  out, i = [], 0
  for t in bundle.observations[group]:
    n = int(np.prod(t["shape"]))
    out.append((t["name"], slice(i, i + n)))
    i += n
  return out


def _fmt(x: np.ndarray, n: int = 4) -> str:
  vals = " ".join(f"{v:+.3f}" for v in x[:n])
  return f"[{vals}{' ...' if len(x) > n else ''}]"


def _table(rows: list[tuple], header: tuple) -> str:
  widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
  fmt = "  ".join(f"{{:<{w}}}" for w in widths)
  return "\n".join([fmt.format(*header), fmt.format(*["-" * w for w in widths])] + [fmt.format(*r) for r in rows])


def _reset_to_reference(env, ref, t: int = 0):
  base = env.unwrapped
  targets = {k.split("/", 1)[1]: ref[k][t] for k in ref.files if k.startswith("command/")}
  base.set_mujoco_state(ref["qpos"][t], targets)
  return base.get_obs()


def main() -> None:
  args = tyro.cli(CompareArgs)
  bundle = Bundle.load(args.bundle)
  ref_file = bundle.path / "reference.npz"
  if not ref_file.exists():
    raise SystemExit(f"{ref_file} missing: re-export with --ref-steps > 0")
  ref = np.load(ref_file)
  steps = min(args.steps or len(ref["action"]), len(ref["action"]))
  flat_groups = [g for g in bundle.policy_groups() if len(bundle.observations[g][0]["shape"]) == 1]
  img_groups = [g for g in bundle.policy_groups() if g not in flat_groups]

  env = gym.make(ENV_ID, bundle=bundle, num_envs=1, sim_backend="cpu")
  env.reset(seed=0)
  policy = OnnxPolicy(bundle)
  out_dir = args.out / bundle.spec["experiment"] / bundle.spec["run"]
  out_dir.mkdir(parents=True, exist_ok=True)

  # 1. Observation check ----------------------------------------------------
  obs = _reset_to_reference(env, ref)
  print(f"\n### 1. Observations at t=0 (identical state)  [{bundle.name}]")
  for g in flat_groups:
    ours = obs[g][0].cpu().numpy()
    theirs = ref[f"obs/{g}"][0]
    rows = [(n, f"{np.abs(ours[s] - theirs[s]).max():.2e}", _fmt(theirs[s]), _fmt(ours[s]))
            for n, s in _term_slices(bundle, g)]
    print(_table(rows, ("term", "max|diff|", "mjlab", "maniskill")))
  for g in img_groups:
    ours = obs[g][0].cpu().numpy()
    theirs = ref[f"obs/{g}"][0]
    print(f"{g}: mean |diff| = {np.abs(ours - theirs).mean():.3f} (pixel range 0-1)")
    if env.unwrapped.camera_gamma == 1.0:
      gammas = np.arange(0.5, 2.55, 0.05)
      fit = [np.abs(ours ** gm - theirs).mean() for gm in gammas]
      best = gammas[int(np.argmin(fit))]
      print(f"  best-fit --camera-gamma {best:.2f} -> mean |diff| = {min(fit):.3f}")
    import imageio

    pair = np.concatenate([theirs, ours], axis=-1).transpose(1, 2, 0)
    imageio.imwrite(out_dir / f"{g}_mjlab_vs_maniskill.png",
                    (np.repeat(np.repeat(pair, 4, 0), 4, 1) * 255).astype(np.uint8))
    print(f"  saved {out_dir / f'{g}_mjlab_vs_maniskill.png'} (left: mjlab, right: ManiSkill)")
  a0 = policy(obs)[0].cpu().numpy()
  print(f"policy action t=0  mjlab={np.round(ref['action'][0], 3)}  maniskill={np.round(a0, 3)}")

  # 2. Open-loop replay -----------------------------------------------------
  _reset_to_reference(env, ref)
  errs: dict[str, list] = {}
  for t in range(steps - 1):
    obs, *_ = env.step(torch.as_tensor(ref["action"][t])[None])
    for g in flat_groups:
      ours, theirs = obs[g][0].cpu().numpy(), ref[f"obs/{g}"][t + 1]
      for n, s in _term_slices(bundle, g):
        errs.setdefault(n, []).append(np.abs(ours[s] - theirs[s]).mean())
  print(f"\n### 2. Open-loop replay of {steps - 1} mjlab actions (mean |diff| per term)")
  rows = [(n, f"{np.mean(e):.4f}", f"{e[-1]:.4f}") for n, e in errs.items()]
  print(_table(rows, ("term", "avg over traj", "final")))

  # 3. Closed-loop -------------------------------------------------------------
  obs = _reset_to_reference(env, ref)
  act_err, info = [], {}
  for t in range(steps):
    a = policy(obs)
    act_err.append(np.abs(a[0].cpu().numpy() - ref["action"][t]).mean())
    obs, _, _, _, info = env.step(a)
  print(f"\n### 3. Closed-loop policy for {steps} steps")
  print(f"mean |action diff| vs mjlab: first={act_err[0]:.4f}  avg={np.mean(act_err):.4f}")
  print(f"ManiSkill final position error {info['position_error'].item():.4f} m, "
        f"episode success={bool(info['episode_success'].item())}")
  env.close()
