"""Quaternion helpers (wxyz), matching mjlab.utils.lab_api.math."""

from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  w1, x1, y1, z1 = a.unbind(-1)
  w2, x2, y2, z2 = b.unbind(-1)
  return torch.stack([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  xyz = q[..., 1:]
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_apply(quat_conjugate(q), v)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
  half = 0.5 * angle.unsqueeze(-1)
  return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
  z = torch.zeros_like(yaw)
  return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], dim=-1)


def axis_angle_from_quat(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
  q = q * (1.0 - 2.0 * (q[..., 0:1] < 0.0))
  mag = torch.linalg.norm(q[..., 1:], dim=-1)
  half = torch.atan2(mag, q[..., 0])
  angle = 2.0 * half
  k = torch.where(angle.abs() > eps, torch.sin(half) / angle, 0.5 - angle * angle / 48)
  return q[..., 1:4] / k.unsqueeze(-1)


def compute_pose_error(p0, q0, p1, q1) -> tuple[torch.Tensor, torch.Tensor]:
  q_err = quat_mul(q1, quat_conjugate(q0))
  return p1 - p0, axis_angle_from_quat(q_err)


def apply_delta_pose(pos, quat, delta, eps: float = 1.0e-6):
  target_pos = pos + delta[:, :3]
  rot = delta[:, 3:6]
  angle = torch.linalg.vector_norm(rot, dim=1)
  axis = rot / angle.clamp_min(eps).unsqueeze(-1)
  identity = torch.tensor([1.0, 0, 0, 0], device=pos.device).expand_as(quat)
  dq = torch.where(angle.unsqueeze(-1) > eps, quat_from_angle_axis(angle, axis), identity)
  return target_pos, quat_mul(dq, quat)
