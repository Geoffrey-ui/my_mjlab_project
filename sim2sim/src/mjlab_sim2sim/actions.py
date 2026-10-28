"""Re-implementations of mjlab action terms on top of ManiSkill joint drives.

Like mjlab, every term has two phases:
  process(raw)  -- once per policy step
  apply()       -- once per physics substep, writes joint position targets

Add support for a new mjlab action type with ``@register_action("<type>")``
(the type string is written by tools/export_mjlab.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from mjlab_sim2sim import math_utils as mu

if TYPE_CHECKING:
  from mjlab_sim2sim.env import RobotContext

ACTION_TERMS: dict[str, type["ActionTerm"]] = {}


def register_action(type_name: str) -> Callable:
  def deco(cls):
    ACTION_TERMS[type_name] = cls
    return cls
  return deco


class ActionTerm:
  def __init__(self, spec: dict, ctx: "RobotContext"):
    self.spec = spec
    self.ctx = ctx
    self.dim = spec["dim"]
    self.raw = torch.zeros(ctx.num_envs, self.dim, device=ctx.device)
    self.processed = torch.zeros_like(self.raw)

  def _affine(self, a: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(self.spec.get("scale", 1.0), device=a.device, dtype=a.dtype)
    offset = torch.as_tensor(self.spec.get("offset", 0.0), device=a.device, dtype=a.dtype)
    out = a * scale + offset
    clip = self.spec.get("clip")
    if clip is not None:
      c = torch.as_tensor(clip, device=a.device, dtype=a.dtype)
      out = torch.clamp(out, c[..., 0], c[..., 1])
    return out

  def process(self, a: torch.Tensor) -> None:
    self.raw[:] = a
    self.processed = self._affine(a)

  def apply(self) -> tuple[list[str], torch.Tensor]:
    """Return (joint_names, targets[N, k])."""
    raise NotImplementedError

  def reset(self, env_idx: torch.Tensor) -> None:
    self.raw[env_idx] = 0.0


@register_action("joint_position")
class JointPositionTerm(ActionTerm):
  def apply(self):
    return self.spec["joints"], self.processed


@register_action("relative_joint_position")
class RelativeJointPositionTerm(ActionTerm):
  """target = current joint pos + scale * action (re-evaluated every substep)."""

  def apply(self):
    joints = self.spec["joints"]
    return joints, self.ctx.qpos(joints) + self.processed


@register_action("tendon_length")
class TendonLengthTerm(ActionTerm):
  """Tendon position actuator -> equal targets on the tendon's joints.

  MuJoCo affine actuator: force = gain * ctrl + b1 * length + b2 * velocity, so
  the steady-state tendon length is gain * ctrl / -b1.
  """

  def apply(self):
    joints, targets = [], []
    for i, t in enumerate(self.spec["tendons"]):
      ctrl = self.processed[:, i]
      if t["ctrlrange"] is not None:
        ctrl = torch.clamp(ctrl, *t["ctrlrange"])
      length = t["gain"] * ctrl / -t["bias"][1]
      per_joint = length / sum(t["coefs"])
      for jn in t["joints"]:
        joints.append(jn)
        targets.append(per_joint)
    return joints, torch.stack(targets, dim=-1)


@register_action("differential_ik")
class DifferentialIKTerm(ActionTerm):
  """Damped least-squares IK, mirroring mjlab.envs.mdp.actions.DifferentialIKAction."""

  def __init__(self, spec, ctx):
    super().__init__(spec, ctx)
    m = ctx.kin.m
    ft, frame = spec["frame_type"], spec["frame"]
    self.frame_type = ft
    self.frame_id = {"site": m.site, "body": m.body, "geom": m.geom}[ft](frame).id
    self.body_id = {
      "site": lambda: int(m.site_bodyid[self.frame_id]),
      "body": lambda: self.frame_id,
      "geom": lambda: int(m.geom_bodyid[self.frame_id]),
    }[ft]()
    self.dofs = [ctx.kin.dofadr[j] for j in spec["joints"]]
    dev = ctx.device
    self.lower = torch.tensor(spec["joint_lower"], device=dev)
    self.upper = torch.tensor(spec["joint_upper"], device=dev)
    self.posture = torch.tensor(spec["posture_target"], device=dev)
    self.des_pos = torch.zeros(ctx.num_envs, 3, device=dev)
    self.des_quat = torch.zeros(ctx.num_envs, 4, device=dev)
    self.des_quat[:, 0] = 1.0

  def _frame(self) -> tuple[torch.Tensor, torch.Tensor]:
    self.ctx.sync_kinematics()
    p, q = self.ctx.kin.frame_pose(self.frame_type, self.frame_id)
    return self.ctx.tensor(p), self.ctx.tensor(q)

  def process(self, a):
    s = self.spec
    self.raw[:] = a
    pos, quat = self._frame()
    if self.dim == 3:
      self.des_pos[:] = pos + a * s["delta_pos_scale"] if s["use_relative_mode"] else a
      self.des_quat[:] = quat
    elif self.dim == 6:
      delta = a.clone()
      delta[:, :3] *= s["delta_pos_scale"]
      delta[:, 3:] *= s["delta_ori_scale"]
      self.des_pos[:], self.des_quat[:] = mu.apply_delta_pose(pos, quat, delta)
    else:
      self.des_pos[:], self.des_quat[:] = a[:, :3], a[:, 3:7]

  def apply(self):
    s = self.spec
    pos, quat = self._frame()
    pos_err, rot_err = mu.compute_pose_error(pos, quat, self.des_pos, self.des_quat)
    jp, jr = self.ctx.kin.jacobian(self.body_id, pos.cpu().numpy(), self.dofs)
    jacp, jacr = self.ctx.tensor(jp), self.ctx.tensor(jr)

    wp2, wo2 = s["position_weight"] ** 2, s["orientation_weight"] ** 2
    JTJ = wp2 * torch.einsum("bti,btj->bij", jacp, jacp) + wo2 * torch.einsum("bti,btj->bij", jacr, jacr)
    JTdx = wp2 * torch.einsum("bti,bt->bi", jacp, pos_err) + wo2 * torch.einsum("bti,bt->bi", jacr, rot_err)

    q = self.ctx.qpos(s["joints"])
    r_limit = (self.upper - q).clamp(max=0) + (self.lower - q).clamp(min=0)
    violated = (r_limit != 0).float()
    wl2 = s["joint_limit_weight"] ** 2
    JTJ.diagonal(dim1=-2, dim2=-1).add_(wl2 * violated)
    JTdx.add_(wl2 * violated * r_limit)

    wpost2 = s["posture_weight"] ** 2
    JTJ.diagonal(dim1=-2, dim2=-1).add_(wpost2)
    JTdx.add_(wpost2 * (self.posture - q))

    lam = max(s["damping"], 1e-6)
    JTJ.diagonal(dim1=-2, dim2=-1).add_(lam * lam)
    dq = torch.linalg.solve(JTJ, JTdx).clamp(-s["max_dq"], s["max_dq"])
    return s["joints"], q + dq

  def reset(self, env_idx):
    super().reset(env_idx)
    self.des_pos[env_idx] = 0.0
    self.des_quat[env_idx] = torch.tensor([1.0, 0, 0, 0], device=self.des_quat.device)


class ActionManager:
  """Concatenated policy action -> per-substep targets for the driven joints."""

  def __init__(self, specs: list[dict], ctx: "RobotContext", driven_joints: list[str],
               default_targets: torch.Tensor, ctrl_ranges: list):
    missing = [s["type"] for s in specs if s["type"] not in ACTION_TERMS]
    if missing:
      raise NotImplementedError(f"Unsupported action types {missing}; add them with @register_action.")
    self.terms = [ACTION_TERMS[s["type"]](s, ctx) for s in specs]
    self.index = {j: i for i, j in enumerate(driven_joints)}
    self.default_targets = default_targets
    self.targets = default_targets.clone()
    lo = [r[0] if r else -np.inf for r in ctrl_ranges]
    hi = [r[1] if r else np.inf for r in ctrl_ranges]
    self.lo = torch.tensor(lo, device=ctx.device, dtype=torch.float32)
    self.hi = torch.tensor(hi, device=ctx.device, dtype=torch.float32)
    self.dim = sum(t.dim for t in self.terms)

  @property
  def action(self) -> torch.Tensor:
    return torch.cat([t.raw for t in self.terms], dim=-1)

  def process(self, action: torch.Tensor) -> None:
    i = 0
    for t in self.terms:
      t.process(action[:, i : i + t.dim])
      i += t.dim

  def apply(self) -> torch.Tensor:
    for t in self.terms:
      joints, tgt = t.apply()
      cols = [self.index[j] for j in joints]
      self.targets[:, cols] = tgt.to(self.targets.dtype)
    return torch.clamp(self.targets, self.lo, self.hi)

  def reset(self, env_idx: torch.Tensor) -> None:
    for t in self.terms:
      t.reset(env_idx)
    self.targets[env_idx] = self.default_targets[env_idx]
