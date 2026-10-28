"""MuJoCo as a kinematics oracle.

Joint positions are read from ManiSkill and written into per-env MjData
copies; forward kinematics and frame Jacobians are then computed exactly as
mjlab computes them (sites, mesh frames and attached grippers included).
"""

from __future__ import annotations

import mujoco
import numpy as np


class MjKinematics:
  def __init__(self, model: mujoco.MjModel, joint_names: list[str], num_envs: int):
    self.m = model
    self.datas = [mujoco.MjData(model) for _ in range(num_envs)]
    self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
    self.dofadr = {n: int(model.jnt_dofadr[model.joint(n).id]) for n in joint_names}
    for d in self.datas:
      mujoco.mj_kinematics(model, d)

  def update(self, qpos: np.ndarray) -> None:
    """qpos: (num_envs, len(joint_names)) in the order given at construction."""
    for d, q in zip(self.datas, qpos):
      d.qpos[self.qadr] = q
      mujoco.mj_kinematics(self.m, d)
      mujoco.mj_comPos(self.m, d)

  def frame_pose(self, frame_type: str, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    """World position (N, 3) and quaternion wxyz (N, 4) of a site / body / geom."""
    pos, quat = [], []
    for d in self.datas:
      if frame_type == "body":
        pos.append(d.xpos[frame_id].copy())
        quat.append(d.xquat[frame_id].copy())
        continue
      xpos, xmat = (d.site_xpos, d.site_xmat) if frame_type == "site" else (d.geom_xpos, d.geom_xmat)
      q = np.empty(4)
      mujoco.mju_mat2Quat(q, xmat[frame_id])
      pos.append(xpos[frame_id].copy())
      quat.append(q)
    return np.stack(pos), np.stack(quat)

  def jacobian(self, body_id: int, points: np.ndarray, dofs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Translational / rotational Jacobians (N, 3, len(dofs)) of a point on a body."""
    nv = self.m.nv
    jp, jr = [], []
    for d, p in zip(self.datas, points):
      jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
      mujoco.mj_jac(self.m, d, jacp, jacr, p, body_id)
      jp.append(jacp[:, dofs])
      jr.append(jacr[:, dofs])
    return np.stack(jp), np.stack(jr)
