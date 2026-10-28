"""A ManiSkill agent generated from a bundle's compiled MuJoCo robot."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PassiveControllerConfig, PDJointPosControllerConfig
from mani_skill.sensors.camera import CameraConfig

from mjlab_sim2sim.bundle import Bundle
from mjlab_sim2sim.mj_to_sapien import GeomOptions, MeshCache, build_articulation, close_loops, pose

# MuJoCo cameras look along -z with +y up; SAPIEN cameras look along +x with +z up.
_MJ_CAM_TO_SAPIEN = sapien.Pose(q=[0.5, -0.5, 0.5, 0.5])


def driven_joint_gains(bundle: Bundle) -> dict[str, dict]:
  """PD gains for every joint that receives a position target.

  Joint actuators map 1:1. Tendon position actuators (e.g. the "split" tendon of
  parallel grippers) are distributed over the tendon's joints assuming the
  joints move together, which is what the joint equality enforces in MuJoCo.
  """
  gains: dict[str, dict] = {}
  for a in bundle.robot["actuators"]:
    gains[a["joint"]] = {
      "kp": a["kp"],
      "kv": a["kv"],
      "force_limit": max(abs(x) for x in a["forcerange"]) if a["forcerange"] else 1e10,
      "ctrlrange": a["ctrlrange"],
      "idle_target": None,
    }
  tendons = bundle.robot.get("tendon_actuators") or [
    t for act in bundle.actions for t in act.get("tendons", [])
  ]
  for t in tendons:
    total = sum(t["coefs"])
    # Joint target while no action term commands the tendon (MuJoCo ctrl = 0).
    ctrl0 = float(np.clip(0.0, *t["ctrlrange"])) if t["ctrlrange"] else 0.0
    idle = t["gain"] * ctrl0 / -t["bias"][1] / total
    for jn, c in zip(t["joints"], t["coefs"]):
      gains[jn] = {
        "kp": -t["bias"][1] * c * total,
        "kv": -t["bias"][2] * c * total,
        "force_limit": max(abs(x) for x in t["forcerange"]) * abs(c) if t["forcerange"] else 1e10,
        "ctrlrange": None,
        "idle_target": idle,
      }
  return gains


def make_agent_class(bundle: Bundle, cache_dir: Path, visual_groups=(0, 1, 2, 3)) -> type[BaseAgent]:
  m = bundle.model
  root = m.body(bundle.robot["root_body"]).id
  gains = driven_joint_gains(bundle)
  driven = [j for j in bundle.robot["joints"] if j in gains]
  passive = [j for j in bundle.robot["joints"] if j not in gains]
  meshes = MeshCache(m, cache_dir)
  opts = GeomOptions(visual_groups=visual_groups)

  class MjlabAgent(BaseAgent):
    uid = f"mjlab-{bundle.spec['experiment']}"
    fix_root_link = True
    driven_joints = driven
    passive_joints = passive
    joint_gains = gains

    def _load_articulation(self, initial_pose=None):
      self.robot = build_articulation(self.scene, m, root, self.uid, meshes, opts, initial_pose)
      self.robot_link_names = [link.name for link in self.robot.get_links()]

    def _after_loading_articulation(self):
      self.loop_drives = close_loops(self.scene, self.robot, m)

    @property
    def _controller_configs(self):
      g = [gains[j] for j in driven]
      cfg = {
        "driven": PDJointPosControllerConfig(
          driven, lower=None, upper=None,
          stiffness=[x["kp"] for x in g], damping=[x["kv"] for x in g],
          force_limit=[x["force_limit"] for x in g],
          normalize_action=False,
        )
      }
      if passive:
        # Passive dynamics (damping, frictionloss, springs) are applied as joint
        # forces by the env, see passive_joint_params().
        cfg["passive"] = PassiveControllerConfig(passive, damping=0.0, friction=0.0)
      return {"pd_joint_pos": cfg}

    @property
    def _sensor_configs(self):
      cams = []
      for c in bundle.cameras:
        if "rgb" not in c["data_types"] and "depth" not in c["data_types"]:
          continue
        cams.append(CameraConfig(
          uid=c["sensor_name"],
          pose=pose(c["pos"], c["quat"]) * _MJ_CAM_TO_SAPIEN,
          width=c["width"],
          height=c["height"],
          fov=float(np.deg2rad(c["fovy"])),
          near=c["znear"],
          far=c["zfar"],
          mount=self.robot.links_map[c["body"]],
        ))
      return cams

  return MjlabAgent


def passive_joint_params(m: mujoco.MjModel, joint_names: list[str]) -> dict[str, np.ndarray]:
  """MuJoCo passive joint parameters, ordered like ``joint_names``.

  PhysX drives clamp damping together with the actuator force and PhysX joint
  friction is a coefficient rather than a torque, so these are applied
  explicitly every substep to keep MuJoCo semantics:
    qf = -damping * qd - frictionloss * tanh(qd / v0) - stiffness * (q - springref)
  """
  ids = [m.joint(n).id for n in joint_names]
  dofs = [m.jnt_dofadr[j] for j in ids]
  return {
    "damping": m.dof_damping[dofs].astype(np.float32),
    "frictionloss": m.dof_frictionloss[dofs].astype(np.float32),
    "stiffness": m.jnt_stiffness[ids].astype(np.float32),
    "springref": m.qpos_spring[[m.jnt_qposadr[j] for j in ids]].astype(np.float32),
  }


def robot_world_pose(bundle: Bundle) -> sapien.Pose:
  m = bundle.model
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  b = m.body(bundle.robot["root_body"]).id
  return pose(d.xpos[b], d.xquat[b])
