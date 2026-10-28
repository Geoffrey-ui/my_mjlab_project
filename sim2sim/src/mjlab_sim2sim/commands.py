"""Task commands (goal sampling, object resets, success metrics).

Commands are looked up by the mjlab command config class name stored in
spec.json. Add a new task with ``@register_command("MyCommandCfg")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import sapien
import torch
from mani_skill.utils.building import actors

from mjlab_sim2sim import math_utils as mu

if TYPE_CHECKING:
  from mjlab_sim2sim.env import MjlabSim2SimEnv

COMMANDS: dict[str, type["Command"]] = {}


def register_command(cfg_type: str) -> Callable:
  def deco(cls):
    COMMANDS[cfg_type] = cls
    return cls
  return deco


def _uniform(lo, hi, n, device) -> torch.Tensor:
  lo = torch.as_tensor(lo, device=device, dtype=torch.float32)
  hi = torch.as_tensor(hi, device=device, dtype=torch.float32)
  return lo + (hi - lo) * torch.rand(n, *lo.shape, device=device)


class Command:
  def __init__(self, env: "MjlabSim2SimEnv", name: str, cfg: dict):
    self.env, self.name, self.cfg = env, name, cfg
    n, dev = env.num_envs, env.device
    self.target_pos = torch.zeros(n, 3, device=dev)
    self.time_left = torch.zeros(n, device=dev)
    self.episode_success = torch.zeros(n, dtype=torch.bool, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def load(self, scene) -> None:
    """Create debug markers (hidden from robot cameras)."""
    self.marker = actors.build_sphere(
      scene, radius=0.03, color=[0, 1, 0, 0.5], name=f"{self.name}_goal",
      body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(p=[0, 0, -1]),
    )
    self.env._hidden_objects.append(self.marker)

  def _sample_target(self, env_idx):
    r = self.cfg["target_position_range"]
    lo, hi = [r["x"][0], r["y"][0], r["z"][0]], [r["x"][1], r["y"][1], r["z"][1]]
    self.target_pos[env_idx] = _uniform(lo, hi, len(env_idx), self.env.device)

  def resample(self, env_idx: torch.Tensor) -> None:
    self.episode_success[env_idx] = False
    self._sample_target(env_idx)
    lo, hi = self.cfg["resampling_time_range"]
    self.time_left[env_idx] = _uniform(lo, hi, len(env_idx), self.env.device)
    raw = self.marker.pose.raw_pose.clone()
    raw[env_idx, :3] = self.target_pos[env_idx]
    raw[env_idx, 3:] = torch.tensor([1.0, 0, 0, 0], device=raw.device)
    self._set_pose(self.marker, raw)

  def _set_pose(self, actor, raw_pose: torch.Tensor, zero_velocity: bool = False) -> None:
    """ManiSkill only writes the envs selected by the scene reset mask (all of
    them outside of reset), and expects exactly that many rows."""
    mask = self.env.scene._reset_mask
    actor.set_pose(raw_pose[mask])
    if zero_velocity:
      zeros = torch.zeros(int(mask.sum()), 3, device=raw_pose.device)
      actor.set_linear_velocity(zeros)
      actor.set_angular_velocity(zeros)

  def step(self, dt: float) -> None:
    self.time_left -= dt
    expired = torch.nonzero(self.time_left <= 0).flatten()
    if len(expired):
      self.resample(expired)

  def error(self) -> torch.Tensor:
    raise NotImplementedError

  def evaluate(self) -> dict[str, torch.Tensor]:
    err = self.error()
    success = err < self.cfg["success_threshold"]
    self.episode_success |= success
    return {"success": success, "position_error": err, "episode_success": self.episode_success.clone()}


@register_command("ReachCommandCfg")
class ReachCommand(Command):
  def load(self, scene):
    super().load(scene)
    m = self.env.ctx.kin.m
    prefix, suffix = f"{self.cfg['robot_name']}/", f"/{self.cfg['ee_site_name']}"
    names = [m.site(i).name for i in range(m.nsite)]
    self.site_id = next(i for i, n in enumerate(names) if n.startswith(prefix) and n.endswith(suffix))

  def error(self):
    self.env.ctx.sync_kinematics()
    p, _ = self.env.ctx.kin.frame_pose("site", self.site_id)
    return torch.linalg.norm(self.target_pos - self.env.ctx.tensor(p), dim=-1)


@register_command("LiftingCommandCfg")
class LiftingCommand(Command):
  def load(self, scene):
    super().load(scene)
    prefix = f"{self.cfg['entity_name']}/"
    matches = [a for n, a in self.env.world_actors.items() if n.startswith(prefix)]
    if not matches:
      raise KeyError(f"No actor for entity {self.cfg['entity_name']!r} in {list(self.env.world_actors)}")
    self.object = matches[0]

  def resample(self, env_idx):
    super().resample(env_idx)
    n, dev = len(env_idx), self.env.device
    r = self.cfg["object_pose_range"]
    xy = _uniform([r["x"][0], r["y"][0]], [r["x"][1], r["y"][1]], n, dev)
    z = torch.full((n, 1), self.cfg["table_height"] + self.cfg["object_half_height"], device=dev)
    yaw = _uniform(r["yaw"][0], r["yaw"][1], n, dev)
    raw = self.object.pose.raw_pose.clone()
    raw[env_idx] = torch.cat([xy, z, mu.quat_from_yaw(yaw)], dim=-1)
    self._set_pose(self.object, raw, zero_velocity=True)

  def error(self):
    return torch.linalg.norm(self.target_pos - self.object.pose.p, dim=-1)

  def evaluate(self):
    info = super().evaluate()
    info["object_height"] = self.object.pose.p[:, 2] - self.cfg["object_half_height"]
    return info
