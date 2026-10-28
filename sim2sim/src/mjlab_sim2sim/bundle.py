"""Loading of sim2sim bundles produced by ``tools/export_mjlab.py``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import mujoco


@dataclass
class Bundle:
  """A self-contained description of a trained mjlab policy and its task."""

  path: Path
  spec: dict = field(repr=False)

  @classmethod
  def load(cls, path: str | Path) -> "Bundle":
    path = Path(path).expanduser().resolve()
    spec_file = path / "spec.json"
    if not spec_file.exists():
      raise FileNotFoundError(
        f"{spec_file} not found. Export a run first with tools/export_mjlab.py."
      )
    spec = json.loads(spec_file.read_text())
    if spec.get("mujoco_version") != mujoco.__version__:
      print(
        f"[sim2sim] WARNING: bundle exported with mujoco {spec.get('mujoco_version')}, "
        f"running {mujoco.__version__}. scene.mjb may fail to load."
      )
    return cls(path=path, spec=spec)

  # -- Convenience accessors ------------------------------------------------

  @cached_property
  def model(self) -> mujoco.MjModel:
    return mujoco.MjModel.from_binary_path(str(self.path / self.spec["scene_model"]))

  @property
  def name(self) -> str:
    return f"{self.spec['experiment']}/{self.spec['run']}"

  @property
  def policy_path(self) -> Path:
    return self.path / self.spec["policy"]["file"]

  @property
  def robot(self) -> dict:
    return self.spec["robot"]

  @property
  def actions(self) -> list[dict]:
    return self.spec["actions"]

  @property
  def observations(self) -> dict[str, list[dict]]:
    return self.spec["observations"]

  @property
  def cameras(self) -> list[dict]:
    return self.spec["cameras"]

  @property
  def commands(self) -> list[dict]:
    return self.spec["commands"]

  @property
  def physics_dt(self) -> float:
    return self.spec["sim"]["timestep"]

  @property
  def decimation(self) -> int:
    return self.spec["sim"]["decimation"]

  @property
  def control_dt(self) -> float:
    return self.physics_dt * self.decimation

  @property
  def action_dim(self) -> int:
    return sum(a["dim"] for a in self.actions)

  @property
  def episode_steps(self) -> int:
    return max(1, round(self.spec["sim"]["episode_length_s"] / self.control_dt))

  def policy_groups(self) -> set[str]:
    """Observation groups consumed by the policy."""
    return {g for gs in self.spec["policy"]["inputs"].values() for g in gs}

  def summary(self) -> str:
    lines = [
      f"bundle     : {self.path}",
      f"task       : {self.spec['task']}  (checkpoint {self.spec['checkpoint']})",
      f"timing     : dt={self.physics_dt}s x {self.decimation} -> {1 / self.control_dt:.0f} Hz control",
      f"robot root : {self.robot['root_body']}  ({len(self.robot['joints'])} joints)",
      "actions    :",
    ]
    for a in self.actions:
      lines.append(f"  - {a['name']:<14} {a['type']:<24} dim={a['dim']}")
    for g in sorted(self.policy_groups()):
      lines.append(f"obs[{g}]:")
      for t in self.observations[g]:
        lines.append(f"  - {t['name']:<20} {t['func']:<40} shape={t['shape']}")
    for c in self.cameras:
      lines.append(f"camera     : {c['camera']} {c['width']}x{c['height']} fovy={c['fovy']}")
    for c in self.commands:
      lines.append(f"command    : {c['name']} ({c['type']})")
    return "\n".join(lines)
