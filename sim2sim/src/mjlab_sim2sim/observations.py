"""Re-implementations of mjlab observation functions on ManiSkill state.

Terms are looked up by the *mjlab function name* stored in spec.json, so a
term only has to be written once to work for every robot. Add a new one with:

  @register_obs("my_mjlab_func")
  def my_term(env, params) -> torch.Tensor:  # (num_envs, ...)
    ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from mjlab_sim2sim import math_utils as mu

if TYPE_CHECKING:
  from mjlab_sim2sim.env import MjlabSim2SimEnv

OBS_TERMS: dict[str, Callable] = {}


def register_obs(func_name: str) -> Callable:
  def deco(fn):
    OBS_TERMS[func_name] = fn
    return fn
  return deco


def _to_root(env: "MjlabSim2SimEnv", pos_w: torch.Tensor) -> torch.Tensor:
  root_p, root_q = env.ctx.root_pose()
  return mu.quat_apply_inverse(root_q, pos_w - root_p)


@register_obs("joint_pos_rel")
def joint_pos_rel(env, params):
  joints = params["asset_cfg"]["joints"]
  return env.ctx.qpos(joints) - env.ctx.default_qpos(joints)


@register_obs("joint_vel_rel")
def joint_vel_rel(env, params):
  return env.ctx.qvel(params["asset_cfg"]["joints"])


@register_obs("last_action")
def last_action(env, params):
  name = params.get("action_name")
  if name is None:
    return env.action_manager.action
  term = next(t for t in env.action_manager.terms if t.spec["name"] == name)
  return term.raw


@register_obs("generated_commands")
def generated_commands(env, params):
  return env.commands[params["command_name"]].command


@register_obs("ee_position_in_robot_base_frame")
def ee_position_in_robot_base_frame(env, params):
  site = params["asset_cfg"]["sites"][0]
  env.ctx.sync_kinematics()
  pos, _ = env.ctx.kin.frame_pose("site", env.ctx.kin.m.site(site).id)
  return _to_root(env, env.ctx.tensor(pos))


@register_obs("target_position_in_robot_base_frame")
def target_position_in_robot_base_frame(env, params):
  return _to_root(env, env.commands[params["command_name"]].command)


@register_obs("object_position_in_robot_root_frame")
def object_position_in_robot_root_frame(env, params):
  return _to_root(env, env.commands[params["command_name"]].object.pose.p)


@register_obs("object_orientation_in_robot_root_frame")
def object_orientation_in_robot_root_frame(env, params):
  _, root_q = env.ctx.root_pose()
  return mu.quat_mul(mu.quat_conjugate(root_q), env.commands[params["command_name"]].object.pose.q)


@register_obs("camera_rgb")
def camera_rgb(env, params):
  rgb = env.sensor_data[params["sensor_name"]]["rgb"]  # (N, H, W, 3) uint8
  rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
  # SAPIEN output is display-encoded, mjlab's renderer is closer to linear.
  return rgb if env.camera_gamma == 1.0 else rgb.pow(env.camera_gamma)


@register_obs("camera_depth")
def camera_depth(env, params):
  depth = env.sensor_data[params["sensor_name"]]["depth"].float() / 1000.0  # mm -> m
  depth = depth.permute(0, 3, 1, 2)
  return torch.clamp(depth, min=params.get("min_depth", 0.01), max=params["cutoff_distance"]) / params["cutoff_distance"]


class ObservationManager:
  def __init__(self, groups: dict[str, list[dict]], used_groups: set[str]):
    self.groups = {g: t for g, t in groups.items() if g in used_groups}
    missing = sorted({t["func"] for ts in self.groups.values() for t in ts if t["func"] not in OBS_TERMS})
    if missing:
      raise NotImplementedError(
        f"Observation functions {missing} have no sim2sim implementation; add them with @register_obs."
      )
    for ts in self.groups.values():
      for t in ts:
        if t.get("history_length", 0):
          raise NotImplementedError(f"Observation history for term {t['name']!r} is not supported yet.")

  @property
  def needs_camera(self) -> bool:
    return any(t["func"].startswith("camera_") for ts in self.groups.values() for t in ts)

  def compute(self, env) -> dict[str, torch.Tensor]:
    out = {}
    for g, terms in self.groups.items():
      vals = []
      for t in terms:
        v = OBS_TERMS[t["func"]](env, t["params"]).float()
        if t["clip"] is not None:
          v = torch.clamp(v, *t["clip"])
        if t["scale"] is not None:
          v = v * torch.as_tensor(t["scale"], device=v.device)
        vals.append(v)
      out[g] = torch.cat(vals, dim=-1) if vals[0].ndim == 2 else torch.cat(vals, dim=1)
    return out
