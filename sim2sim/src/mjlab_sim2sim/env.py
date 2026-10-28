"""Generic ManiSkill3 environment that replays an mjlab task from a bundle."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

from mjlab_sim2sim.actions import ActionManager
from mjlab_sim2sim.agent import make_agent_class, passive_joint_params, robot_world_pose
from mjlab_sim2sim.bundle import Bundle
from mjlab_sim2sim.commands import COMMANDS
from mjlab_sim2sim.kinematics import MjKinematics
from mjlab_sim2sim.mj_to_sapien import GeomOptions, MeshCache, build_world, load_lighting
from mjlab_sim2sim.observations import ObservationManager

ENV_ID = "MjlabSim2Sim-v0"


class RobotContext:
  """State accessors shared by action / observation / command terms."""

  def __init__(self, env: "MjlabSim2SimEnv"):
    self.env = env
    self.robot = env.agent.robot
    self.num_envs = env.num_envs
    self.device = env.device
    self.joint_index = {j.name: i for i, j in enumerate(self.robot.get_active_joints())}
    self.joint_names = list(self.joint_index)
    spec = env.bundle.robot
    self._default = dict(zip(spec["joints"], spec["default_joint_pos"]))
    # mjlab's reset_joints_by_offset clamps the default pose into the soft limits.
    self._reset = {
      j: float(np.clip(q, lo, hi))
      for j, q, (lo, hi) in zip(spec["joints"], spec["default_joint_pos"], spec["soft_joint_pos_limits"])
    }
    self.kin = MjKinematics(env.bundle.model, self.joint_names, self.num_envs)
    self._kin_dirty = True
    self.passive = {k: self.tensor(v) for k, v in passive_joint_params(env.bundle.model, self.joint_names).items()}

  def passive_forces(self, frictionloss_v0: float) -> torch.Tensor:
    """MuJoCo passive joint forces. frictionloss is smoothed with tanh(qd / v0);
    MuJoCo's soft friction constraint behaves more viscous than a hard static
    friction, so a fairly large v0 matches best (v0 <= 0 disables it)."""
    q, qd, p = self.robot.get_qpos(), self.robot.get_qvel(), self.passive
    qf = -p["damping"] * qd - p["stiffness"] * (q - p["springref"])
    if frictionloss_v0 > 0:
      qf = qf - p["frictionloss"] * torch.tanh(qd / frictionloss_v0)
    return qf

  def tensor(self, x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

  def _idx(self, joints: list[str]) -> list[int]:
    return [self.joint_index[j] for j in joints]

  def qpos(self, joints: list[str]) -> torch.Tensor:
    return self.robot.get_qpos()[:, self._idx(joints)]

  def qvel(self, joints: list[str]) -> torch.Tensor:
    return self.robot.get_qvel()[:, self._idx(joints)]

  def default_qpos(self, joints: list[str]) -> torch.Tensor:
    return self.tensor([self._default.get(j, 0.0) for j in joints]).expand(self.num_envs, -1)

  def reset_qpos(self, joints: list[str]) -> torch.Tensor:
    return self.tensor([self._reset.get(j, 0.0) for j in joints]).expand(self.num_envs, -1)

  def root_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    p = self.robot.pose
    return p.p, p.q

  def invalidate(self) -> None:
    self._kin_dirty = True

  def sync_kinematics(self) -> None:
    if self._kin_dirty:
      self.kin.update(self.robot.get_qpos().cpu().numpy())
      self._kin_dirty = False


@register_env(ENV_ID, max_episode_steps=None)
class MjlabSim2SimEnv(BaseEnv):
  """Args (in addition to ManiSkill's BaseEnv args):
    bundle: path to an exported bundle directory (or a Bundle).
    visual_groups: MuJoCo geom groups rendered in SAPIEN (mjlab cameras use 0-3).
    camera_gamma: exponent applied to camera RGB in [0, 1] to match mjlab's tone
      (``sim2sim-compare`` reports a fitted value).
    terminate_on_success: ManiSkill's default; mjlab tasks only end on time-out.
    frictionloss_v0: velocity scale of the smoothed joint frictionloss (0 disables).
  """

  SUPPORTED_REWARD_MODES = ["sparse", "none"]
  SUPPORTED_ROBOTS = []

  def __init__(self, *args, bundle: str | Path | Bundle, visual_groups=(0, 1, 2, 3),
               camera_gamma: float = 1.0, terminate_on_success: bool = False,
               frictionloss_v0: float = 0.1, **kwargs):
    self.frictionloss_v0 = frictionloss_v0
    self.bundle = bundle if isinstance(bundle, Bundle) else Bundle.load(bundle)
    self.visual_groups = tuple(visual_groups)
    self.camera_gamma = camera_gamma
    self.terminate_on_success = terminate_on_success
    cache = self.bundle.path / ".mesh_cache"
    self._meshes = MeshCache(self.bundle.model, cache)
    agent_cls = make_agent_class(self.bundle, cache, self.visual_groups)
    self._obs_manager = ObservationManager(self.bundle.observations, self.bundle.policy_groups())
    kwargs.setdefault("obs_mode", "rgb" if self._obs_manager.needs_camera else "state_dict")
    kwargs.setdefault("reward_mode", "sparse")
    self.sensor_data: dict = {}
    super().__init__(*args, robot_uids=agent_cls, **kwargs)

    dim = self.action_manager.dim
    self.single_action_space = gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
    self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
    self._orig_single_action_space = self.single_action_space

  # -- Configuration --------------------------------------------------------

  @property
  def _default_sim_config(self):
    b = self.bundle
    sim_freq = round(1.0 / b.physics_dt)
    return SimConfig(
      sim_freq=sim_freq,
      control_freq=sim_freq // b.decimation,
      scene_config=SceneConfig(
        gravity=np.array(b.spec["sim"]["gravity"]),
        solver_position_iterations=20,
        solver_velocity_iterations=1,
      ),
    )

  @property
  def _default_human_render_camera_configs(self):
    pose = sapien_utils.look_at(eye=[1.5, -1.2, 1.3], target=[0.35, 0.0, 0.55])
    return CameraConfig("render_camera", pose, 640, 480, fov=1.0, near=0.01, far=100)

  # -- Scene ----------------------------------------------------------------

  def _load_agent(self, options: dict):
    super()._load_agent(options, robot_world_pose(self.bundle))

  def _load_lighting(self, options: dict):
    load_lighting(self.scene, self.bundle.model, shadow=self.enable_shadow)

  def _load_scene(self, options: dict):
    m = self.bundle.model
    robot_root = m.body(self.bundle.robot["root_body"]).id
    self.world_actors = build_world(
      self.scene, m, {robot_root}, self._meshes, GeomOptions(self.visual_groups)
    )
    self.ctx = RobotContext(self)
    driven = self.agent.driven_joints
    defaults = self.ctx.reset_qpos(driven).clone()
    for i, j in enumerate(driven):  # tendon actuators hold their ctrl=0 target
      if self.agent.joint_gains[j]["idle_target"] is not None:
        defaults[:, i] = self.agent.joint_gains[j]["idle_target"]
    ranges = [self.agent.joint_gains[j]["ctrlrange"] for j in driven]
    self.action_manager = ActionManager(self.bundle.actions, self.ctx, driven, defaults, ranges)
    self.commands = {}
    for c in self.bundle.commands:
      if c["type"] not in COMMANDS:
        raise NotImplementedError(f"No sim2sim command for {c['type']}; add one with @register_command.")
      self.commands[c["name"]] = COMMANDS[c["type"]](self, c["name"], c["cfg"])
      self.commands[c["name"]].load(self.scene)

  def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    n = len(env_idx)
    self.agent.reset(self.ctx.reset_qpos(self.ctx.joint_names)[:n])
    self.action_manager.reset(env_idx)
    self.agent.controller.controllers["driven"].set_drive_targets(self.action_manager.targets[env_idx])
    for c in self.commands.values():
      c.resample(env_idx)
    self.ctx.invalidate()

  def set_mujoco_state(self, qpos: np.ndarray, command_targets: dict[str, np.ndarray] | None = None):
    """Teleport robot + free bodies to a MuJoCo ``qpos`` (same in every env).

    Used by ``sim2sim-compare`` to start from the exact state of the mjlab
    reference rollout.
    """
    m = self.bundle.model
    q = self.ctx.tensor([qpos[m.jnt_qposadr[m.joint(j).id]] for j in self.ctx.joint_names])
    q = q.expand(self.num_envs, -1)
    self.agent.robot.set_qpos(q)
    self.agent.robot.set_qvel(torch.zeros_like(q))
    self.action_manager.reset(torch.arange(self.num_envs, device=self.device))
    self.agent.controller.controllers["driven"].set_drive_targets(self.action_manager.targets)
    for name, actor in self.world_actors.items():
      b = m.body(name).id
      if m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == 0:  # free joint
        adr = m.jnt_qposadr[m.body_jntadr[b]]
        actor.set_pose(self.ctx.tensor(qpos[adr : adr + 7]).expand(self.num_envs, -1))
        actor.set_linear_velocity(torch.zeros(self.num_envs, 3, device=self.device))
        actor.set_angular_velocity(torch.zeros(self.num_envs, 3, device=self.device))
    for name, tgt in (command_targets or {}).items():
      cmd = self.commands[name]
      cmd.target_pos[:] = self.ctx.tensor(tgt)
      raw = cmd.marker.pose.raw_pose.clone()
      raw[:, :3] = cmd.target_pos
      cmd.marker.set_pose(raw)
    if self.gpu_sim_enabled:
      self.scene._gpu_apply_all()
      self.scene.px.gpu_update_articulation_kinematics()
      self.scene._gpu_fetch_all()
    self.ctx.invalidate()

  # -- Stepping -------------------------------------------------------------

  def _step_action(self, action):
    a = common.to_tensor(action, device=self.device).float()
    if a.ndim == 1:
      a = a.unsqueeze(0)
    clip = self.bundle.spec["policy"].get("clip_actions")
    if clip is not None:
      a = torch.clamp(a, -clip, clip)
    self.action_manager.process(a)
    driven = self.agent.controller.controllers["driven"]
    for _ in range(self._sim_steps_per_control):
      driven.set_drive_targets(self.action_manager.apply())
      self.agent.robot.set_qf(self.ctx.passive_forces(self.frictionloss_v0))
      if self.gpu_sim_enabled:
        self.scene.px.gpu_apply_articulation_target_position()
        self.scene.px.gpu_apply_articulation_qf()
      self.scene.step()
      if self.gpu_sim_enabled:
        self.scene._gpu_fetch_all()
      self.ctx.invalidate()
    for c in self.commands.values():
      c.step(self.control_timestep)
    return a

  def step(self, action):
    obs, reward, terminated, truncated, info = super().step(action)
    if not self.terminate_on_success:
      terminated = torch.zeros_like(terminated)
    return obs, reward, terminated, truncated, info

  # -- Observations / evaluation ---------------------------------------------

  def get_obs(self, info=None, unflattened=False):
    if self._obs_manager.needs_camera:
      self.sensor_data = self._get_obs_sensor_data()
    return self._obs_manager.compute(self)

  def evaluate(self):
    info = {}
    for c in self.commands.values():
      info.update(c.evaluate())
    return info

  def compute_sparse_reward(self, obs, action, info):
    return info.get("success", torch.zeros(self.num_envs, device=self.device)).float()
