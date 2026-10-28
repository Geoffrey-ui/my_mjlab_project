"""Build SAPIEN / ManiSkill articulations and actors from a compiled MjModel.

Working from the *compiled* model (rather than parsing MJCF) means every
frame, mesh re-centering, attached gripper and mjlab spec edit is already
resolved, so any robot that mjlab can simulate can be converted.

Mapping summary:
  body            -> link (articulation) or actor
  hinge / slide   -> revolute / prismatic joint (axis aligned to joint x)
  geom contype/conaffinity != 0 -> collision shape (mesh -> convex hull)
  geom group in visual_groups   -> visual shape
  connect equality              -> 3-DoF locked drive (closes linkages)
  contype / conaffinity bitmasks -> SAPIEN collision groups 0 / 1
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import sapien
import trimesh
from sapien.wrapper.articulation_builder import MimicJointRecord

_GEOM = mujoco.mjtGeom
_X_TO_Z = sapien.Pose(q=[np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])  # rotates +x onto +z
_SELF_COLLISION_BIT = 1 << 29  # ManiSkill convention for "same articulation"


def pose(pos, quat) -> sapien.Pose:
  return sapien.Pose(p=np.asarray(pos, dtype=np.float32), q=np.asarray(quat, dtype=np.float32))


def _quat_from_x_to(axis: np.ndarray) -> np.ndarray:
  """Quaternion (wxyz) rotating +x onto ``axis``."""
  axis = axis / np.linalg.norm(axis)
  x = np.array([1.0, 0.0, 0.0])
  d = float(np.dot(x, axis))
  if d > 1 - 1e-9:
    return np.array([1.0, 0.0, 0.0, 0.0])
  if d < -1 + 1e-9:
    return np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
  c = np.cross(x, axis)
  q = np.array([1.0 + d, *c])
  return q / np.linalg.norm(q)


def body_name(m: mujoco.MjModel, b: int) -> str:
  return m.body(b).name


def subtree(m: mujoco.MjModel, root: int) -> list[int]:
  """Bodies of the subtree rooted at ``root`` in depth-first (parent-first) order."""
  children: dict[int, list[int]] = {}
  for b in range(1, m.nbody):
    children.setdefault(int(m.body_parentid[b]), []).append(b)
  order, stack = [], [root]
  while stack:
    b = stack.pop()
    order.append(b)
    stack.extend(reversed(children.get(b, [])))
  return order


def body_world_pose(m: mujoco.MjModel, b: int) -> sapien.Pose:
  """World pose of a body at qpos0 (bodies are static in their parents' frames)."""
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  return pose(d.xpos[b], d.xquat[b])


class MeshCache:
  """Writes compiled MuJoCo meshes to .obj files so SAPIEN can load them."""

  def __init__(self, m: mujoco.MjModel, cache_dir: Path):
    self.m = m
    self.dir = Path(cache_dir)
    self.dir.mkdir(parents=True, exist_ok=True)

  def path(self, mesh_id: int) -> str:
    f = self.dir / f"mesh_{mesh_id:04d}.obj"
    if not f.exists():
      m = self.m
      va, vn = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
      fa, fn = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
      tm = trimesh.Trimesh(m.mesh_vert[va : va + vn], m.mesh_face[fa : fa + fn], process=False)
      tm.export(f)
    return str(f)


class GeomOptions:
  def __init__(self, visual_groups=(0, 1, 2, 3), collisions: bool = True, visuals: bool = True):
    self.visual_groups = set(visual_groups)
    self.collisions = collisions
    self.visuals = visuals


def _texture_mean(m: mujoco.MjModel, tex: int) -> np.ndarray:
  adr, n = m.tex_adr[tex], m.tex_width[tex] * m.tex_height[tex] * m.tex_nchannel[tex]
  data = m.tex_data[adr : adr + n].reshape(-1, m.tex_nchannel[tex])[:, :3]
  return data.mean(axis=0) / 255.0


def _geom_rgba(m: mujoco.MjModel, g: int) -> np.ndarray:
  """Material color wins over geom rgba, matching mjlab's camera renderer.
  Textures are approximated by their mean color (meshes carry no UVs here)."""
  mat = m.geom_matid[g]
  if mat < 0:
    return m.geom_rgba[g].astype(np.float32)
  rgba = m.mat_rgba[mat].astype(np.float32).copy()
  tex = int(m.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB])
  if tex >= 0:
    rgba[:3] *= _texture_mean(m, tex)
  return rgba


def load_lighting(scene, m: mujoco.MjModel, shadow: bool = False) -> None:
  """Approximate MuJoCo lighting: model lights + the camera headlight.

  The headlight follows the camera in MuJoCo; here it becomes ambient light
  plus a top-down directional light, a reasonable match for wrist cameras.
  """
  hl = m.vis.headlight
  ambient = np.asarray(hl.ambient) * bool(hl.active) + m.light_ambient.sum(axis=0)
  scene.set_ambient_light(np.clip(ambient, 0, 1).tolist())
  if hl.active:
    scene.add_directional_light([0, 0, -1], np.asarray(hl.diffuse).tolist())
  for i in range(m.nlight):
    color = m.light_diffuse[i].tolist()
    if m.light_type[i] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
      scene.add_directional_light(m.light_dir[i].tolist(), color, shadow=shadow and bool(m.light_castshadow[i]))
    else:
      scene.add_point_light(m.light_pos[i].tolist(), color, shadow=shadow and bool(m.light_castshadow[i]))


def add_body_geoms(
  builder,
  m: mujoco.MjModel,
  bodies: list[int],
  meshes: MeshCache,
  opts: GeomOptions,
  frame: sapien.Pose = sapien.Pose(),
) -> tuple[int, int]:
  """Add geoms of ``bodies`` to ``builder``. ``frame`` maps body-local poses into
  the builder frame (used when welding a subtree into a single actor).

  Returns the OR'ed (contype, conaffinity) bitmasks of the added collision shapes.
  """
  contype = conaffinity = 0
  for b in bodies:
    rel = frame
    if b != bodies[0]:
      # Pose of b relative to bodies[0] at qpos0.
      root_w = body_world_pose(m, bodies[0])
      rel = frame * root_w.inv() * body_world_pose(m, b)
    for g in np.nonzero(m.geom_bodyid == b)[0]:
      g = int(g)
      gp = rel * pose(m.geom_pos[g], m.geom_quat[g])
      gt, size = m.geom_type[g], m.geom_size[g]
      rgba = _geom_rgba(m, g)
      collide = bool(m.geom_contype[g] or m.geom_conaffinity[g])
      visible = opts.visuals and int(m.geom_group[g]) in opts.visual_groups and rgba[3] > 0

      if collide and opts.collisions:
        contype |= int(m.geom_contype[g])
        conaffinity |= int(m.geom_conaffinity[g])
        mu = float(m.geom_friction[g, 0])
        material = sapien.physx.PhysxMaterial(mu, mu, 0.0)
        if gt == _GEOM.mjGEOM_BOX:
          builder.add_box_collision(gp, half_size=size, material=material)
        elif gt == _GEOM.mjGEOM_SPHERE:
          builder.add_sphere_collision(gp, radius=size[0], material=material)
        elif gt == _GEOM.mjGEOM_CAPSULE:
          builder.add_capsule_collision(gp * _X_TO_Z, radius=size[0], half_length=size[1], material=material)
        elif gt == _GEOM.mjGEOM_CYLINDER:
          builder.add_cylinder_collision(gp * _X_TO_Z, radius=size[0], half_length=size[1], material=material)
        elif gt == _GEOM.mjGEOM_ELLIPSOID:
          builder.add_sphere_collision(gp, radius=float(np.mean(size)), material=material)
        elif gt == _GEOM.mjGEOM_PLANE:
          builder.add_plane_collision(gp * _X_TO_Z, material=material)  # SAPIEN plane normal is +x
        elif gt == _GEOM.mjGEOM_MESH:
          builder.add_convex_collision_from_file(meshes.path(int(m.geom_dataid[g])), gp, material=material)

      if visible:
        mat = sapien.render.RenderMaterial(base_color=rgba)
        if gt == _GEOM.mjGEOM_BOX:
          builder.add_box_visual(gp, half_size=size, material=mat)
        elif gt == _GEOM.mjGEOM_SPHERE:
          builder.add_sphere_visual(gp, radius=size[0], material=mat)
        elif gt == _GEOM.mjGEOM_CAPSULE:
          builder.add_capsule_visual(gp * _X_TO_Z, radius=size[0], half_length=size[1], material=mat)
        elif gt == _GEOM.mjGEOM_CYLINDER:
          builder.add_cylinder_visual(gp * _X_TO_Z, radius=size[0], half_length=size[1], material=mat)
        elif gt == _GEOM.mjGEOM_ELLIPSOID:
          builder.add_sphere_visual(gp, radius=float(np.mean(size)), material=mat)
        elif gt == _GEOM.mjGEOM_PLANE:
          half = size[:2] if size[0] > 0 else (10.0, 10.0)
          builder.add_box_visual(gp * sapien.Pose(p=[0, 0, -0.005]), half_size=(*half, 0.005), material=mat)
        elif gt == _GEOM.mjGEOM_MESH:
          builder.add_visual_from_file(meshes.path(int(m.geom_dataid[g])), gp, material=mat)
  return contype, conaffinity


def _set_inertia(builder, m: mujoco.MjModel, b: int) -> None:
  mass = max(float(m.body_mass[b]), 1e-4)
  inertia = np.maximum(m.body_inertia[b], 1e-7)
  builder.set_mass_and_inertia(mass, pose(m.body_ipos[b], m.body_iquat[b]), inertia)


def build_articulation(scene, m: mujoco.MjModel, root: int, name: str, meshes: MeshCache,
                       opts: GeomOptions, initial_pose: sapien.Pose | None = None,
                       disable_self_collisions: bool = True):
  """Build a fixed-base ManiSkill articulation from the subtree at ``root``.

  Link and joint names are the full MuJoCo names (e.g. ``robot/joint_1``).
  """
  builder = scene.create_articulation_builder()
  builder.initial_pose = initial_pose or body_world_pose(m, root)
  links = {}
  for b in subtree(m, root):
    parent = None if b == root else links[int(m.body_parentid[b])]
    lb = builder.create_link_builder(parent)
    lb.set_name(body_name(m, b))
    _set_inertia(lb, m, b)
    contype, conaffinity = add_body_geoms(lb, m, [b], meshes, opts)
    lb.collision_groups = [contype, conaffinity, _SELF_COLLISION_BIT if disable_self_collisions else 0, 0]

    jnts = [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])] if m.body_jntnum[b] > 0 else []
    body_pose = pose(m.body_pos[b], m.body_quat[b])
    if parent is None:
      lb.set_joint_properties("fixed", [], sapien.Pose(), sapien.Pose())
      lb.set_joint_name(f"{body_name(m, b)}_root")
    elif not jnts:
      lb.set_joint_properties("fixed", [], body_pose, sapien.Pose())
      lb.set_joint_name(f"{body_name(m, b)}_fixed")
    else:
      if len(jnts) > 1:
        raise NotImplementedError(f"Body {body_name(m, b)} has {len(jnts)} joints; only 1 supported.")
      j = jnts[0]
      jt = m.jnt_type[j]
      if jt not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        raise NotImplementedError(f"Joint {m.joint(j).name}: type {jt} unsupported in articulations.")
      in_child = pose(m.jnt_pos[j], _quat_from_x_to(m.jnt_axis[j]))
      limits = [m.jnt_range[j].tolist()] if m.jnt_limited[j] else [[-np.inf, np.inf]]
      lb.set_joint_properties(
        "revolute" if jt == mujoco.mjtJoint.mjJNT_HINGE else "prismatic",
        limits,
        body_pose * in_child,
        in_child,
      )
      lb.set_joint_name(m.joint(j).name)
    links[b] = lb
  builder.mimic_joint_records = _mimic_records(m, set(links))
  art = builder.build(name=name, fix_root_link=True)

  # Joint armature is not part of the builder API.
  for joint in art.get_active_joints():
    j = m.joint(joint.name).id
    for obj in joint._objs:
      obj.set_armature([float(m.dof_armature[m.jnt_dofadr[j]])])
  return art


def close_loops(scene, art, m: mujoco.MjModel) -> list:
  """Replace MuJoCo ``connect`` equalities inside ``art`` with locked drives."""
  links = {l.name: l for l in art.get_links()}
  drives = []
  for e in range(m.neq):
    if not m.eq_active0[e]:
      continue
    if m.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT and m.eq_objtype[e] == mujoco.mjtObj.mjOBJ_BODY:
      b1, b2 = body_name(m, m.eq_obj1id[e]), body_name(m, m.eq_obj2id[e])
      if b1 not in links or b2 not in links:
        continue
      a1, a2 = m.eq_data[e, 0:3], m.eq_data[e, 3:6]
      drive = scene.create_drive(links[b1], sapien.Pose(a1), links[b2], sapien.Pose(a2))
      drive.set_limit_x(0, 0)
      drive.set_limit_y(0, 0)
      drive.set_limit_z(0, 0)
      drives.append(drive)
    elif m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
      pass  # Built as a PhysX fixed tendon in build_articulation().
    else:
      print(f"[sim2sim] WARNING: equality {e} (type {m.eq_type[e]}) is not transferred.")
  return drives


def _mimic_records(m: mujoco.MjModel, bodies: set[int]) -> list[MimicJointRecord]:
  """MuJoCo joint equalities q1 = c0 + c1 * q2 -> ManiSkill mimic joints, which
  become PhysX fixed tendons (they must exist before the articulation is added
  to the scene, otherwise GPU init fails)."""
  records = []
  for e in range(m.neq):
    if not m.eq_active0[e] or m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
      continue
    j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
    if m.jnt_bodyid[j1] not in bodies or j2 < 0:
      continue
    c0, c1 = float(m.eq_data[e, 0]), float(m.eq_data[e, 1])
    b1, b2 = m.jnt_bodyid[j1], m.jnt_bodyid[j2]
    siblings = m.body_parentid[b1] == m.body_parentid[b2]
    parent_child = m.body_parentid[b1] == b2 and m.body_parentid[b2] != 0
    if np.any(np.abs(m.eq_data[e, 2:5]) > 1e-9) or abs(c1) < 1e-9 or not (siblings or parent_child):
      print(f"[sim2sim] WARNING: joint equality {m.joint(j1).name} = f({m.joint(j2).name}) not transferred.")
      continue
    records.append(MimicJointRecord(joint=m.joint(j2).name, mimic=m.joint(j1).name, multiplier=c1, offset=c0))
  return records


def build_world(scene, m: mujoco.MjModel, skip_roots: set[int], meshes: MeshCache,
                opts: GeomOptions) -> dict[str, object]:
  """Build every top-level body that is not in ``skip_roots`` as a ManiSkill actor.

  Bodies with a free joint become dynamic actors; everything else (terrain,
  mocap tables, ...) is welded into static actors.
  """
  actors = {}
  world_geoms = [g for g in np.nonzero(m.geom_bodyid == 0)[0]]
  roots = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and b not in skip_roots]
  if world_geoms:
    roots = [0] + roots
  for b in roots:
    bodies = [0] if b == 0 else subtree(m, b)
    free = [x for x in bodies if m.body_jntnum[x] > 0
            and m.jnt_type[m.body_jntadr[x]] == mujoco.mjtJoint.mjJNT_FREE]
    if free and len(bodies) > 1:
      raise NotImplementedError(f"Free body {body_name(m, free[0])} with children is unsupported.")

    builder = scene.create_actor_builder()
    builder.initial_pose = body_world_pose(m, b)
    contype, conaffinity = add_body_geoms(builder, m, bodies, meshes, opts)
    builder.collision_groups = [contype, conaffinity, 0, 0]
    if not builder.collision_records and not builder.visual_records:
      continue
    name = body_name(m, b) or "world"
    if free:
      _set_inertia(builder, m, b)
      actors[name] = builder.build(name=name)
    else:
      actors[name] = builder.build_static(name=name)
  return actors
