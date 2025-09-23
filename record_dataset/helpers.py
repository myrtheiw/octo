#!/usr/bin/env python3
"""
Minimal helpers for null-space collision avoidance with MuJoCo + Panda.

This module provides small utilities that you can import into `oracle_test.py`:

- `damped_pinv(J, lam=0.05)`: damped least-squares pseudoinverse (3x7 -> 7x3)
- `body_pos(model, data, body_name)`: world position of a body and its id
- `approx_body_radius(model, body_id, default=0.04)`: quick radius estimate from geoms
- `jacobian_body_point(model, data, body_name, arm_dof_idx)`: 3x7 Jacobian of the body CoM

You can run this file directly to perform a few sanity checks against your
current Panda model and targets from `oracle_test.py`.

Usage:
  python avoidance_helpers.py

Then import in your IK code, e.g.:
  from avoidance_helpers import (
      damped_pinv as _damped_pinv,
      body_pos as _body_pos,
      approx_body_radius as _approx_body_radius,
      jacobian_body_point as _jacobian_body_point,
  )
"""

from __future__ import annotations
import numpy as np
import mujoco
import os
import sys
sys.path.insert(1, '/home/myrtheiw/octo_ws/octo')
from record_dataset.generate_tomato_plant import generate_tomato_plant_xml
from record_dataset.getlocation import get_side_stem_origins_and_quats
from record_dataset.getlocation import get_side_stem_grasp_points

# ---- Split policy (edit ratios as you like) ----
SPLIT_RATIOS = dict(train=0.90, val=0.10, test=0.0)
# ------------------------------- Helpers --------------------------------------
import json, re, glob, os, shutil

def expected_ds_dir(root_dir, ds_config):
    return os.path.join(root_dir, ds_config.name, str(ds_config.version))

def move_stray_shards_into_version_dir(root_dir, ds_config):
    """Recursively find any *.tfrecord-* shards under root_dir and move them into
    <root_dir>/<name>/<version>/, preserving the shard suffix."""
    expected = expected_ds_dir(root_dir, ds_config)
    os.makedirs(expected, exist_ok=True)

    # Pick up shards even if EnvLogger created nested subdirs
    pattern = os.path.join(root_dir, "**", f"{ds_config.name}-*.tfrecord-*")
    stray = sorted(glob.glob(pattern, recursive=True))
    moved = 0
    for src in stray:
        # Skip ones already in the expected dir
        if os.path.commonpath([os.path.abspath(src), os.path.abspath(expected)]) == os.path.abspath(expected):
            continue
        dest = os.path.join(expected, os.path.basename(src))
        try:
            shutil.move(src, dest)
            moved += 1
        except Exception:
            # If a same-named shard exists, make a unique name
            base, ext = os.path.splitext(os.path.basename(src))
            i = 1
            while os.path.exists(dest):
                dest = os.path.join(expected, f"{base}.{i}{ext}")
                i += 1
            shutil.move(src, dest)
            moved += 1
    if moved:
        print(f"[TFDS] Moved {moved} stray shard(s) into {expected}")
    return expected

def repair_tfds_splits_at_dir(version_dir, name):
    """Rebuild dataset_info.json 'splits' to match shards present in version_dir."""
    info_p = os.path.join(version_dir, "dataset_info.json")
    if not os.path.exists(info_p):
        print(f"[TFDS] No dataset_info.json at {info_p}, skipping repair.")
        return

    pat = re.compile(rf"{re.escape(name)}-(?P<split>[^.]+)\.tfrecord-")
    shards_by_split = {}
    for f in glob.glob(os.path.join(version_dir, f"{name}-*.tfrecord-*")):
        m = pat.search(os.path.basename(f))
        if m:
            shards_by_split.setdefault(m.group("split"), []).append(f)

    if not shards_by_split:
        print("[TFDS] No shards found, skipping split repair.")
        return

    with open(info_p, "r") as fh:
        info = json.load(fh)

    splits = []
    for split, files in sorted(shards_by_split.items()):
        files_sorted = sorted(files)
        sizes = []
        total_bytes = 0
        for f in files_sorted:
            sizes.append("1")              # shardLengths entries
            total_bytes += os.path.getsize(f)
        splits.append({
            "name": split,
            "shardLengths": sizes,
            "numBytes": total_bytes,
        })

    info["splits"] = splits
    with open(info_p, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"[TFDS] Rewrote split table: {', '.join(sorted(shards_by_split.keys()))}")

# --- Back-compat alias used by older call sites ---
def repair_tfds_splits(version_dir, name):
    return repair_tfds_splits_at_dir(version_dir, name)


def choose_split_for_episode(ep_idx: int) -> str:
    """Deterministic split assignment by episode index."""
    # 90/10/0 pattern by modulo; avoids needing dataset sizes known up-front
    r_train = int(round(SPLIT_RATIOS["train"] * 100))
    r_val   = int(round(SPLIT_RATIOS["val"] * 100))
    m = (ep_idx % 100)
    if m < r_train:
        return "train"
    elif m < r_train + r_val:
        return "val"
    else:
        return "test"

def choose_split_for_plant(plant_idx: int, p_train=0.85, p_val=0.075) -> str:
    """Deterministically map a plant index to 'train' / 'val' / 'test'.
    Keeps *all* episodes from a given plant in the same split.
    """
    h = (1103515245 * (plant_idx + 1) + 12345) & 0xFFFFFFFF  # LCG-ish hash
    frac = h / 4294967296.0
    if frac < p_train:
        return "train"
    elif frac < p_train + p_val:
        return "val"
    else:
        return "test"

def collect_plant_obstacles(model):
    """Return names of all plant bodies to use as obstacles."""
    prefixes = ("tomato_plant", "side_stem", "truss", "leaf")
    obs = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        if any(name == p or name.startswith(p) for p in prefixes):
            obs.append(name)
    return obs

def weighted_dls(J, e, lam=0.05, wdiag=None):
    """
    Weighted damped least squares:
        argmin_dq ||W dq||  s.t.  J dq ≈ e
    dq = W^{-1} J^T (J W^{-1} J^T + lam I)^{-1} e
    """
    import numpy as np
    if wdiag is None:
        # fall back to ordinary damped pinv
        return _damped_pinv(J, lam=lam) @ e
    w = np.asarray(wdiag, dtype=float)
    Winv = np.diag(1.0 / np.maximum(1e-6, w))
    A = J @ Winv @ J.T + lam * np.eye(J.shape[0])
    return Winv @ J.T @ np.linalg.solve(A, e)


def damped_pinv(J: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """Damped least-squares pseudoinverse.

    Args:
      J: shape (m, 7)
      lam: damping factor (>=0)
    Returns:
      J^+ of shape (7, m)
    """
    J = np.asarray(J, dtype=float)
    m, n = J.shape
    if n != 7:
        raise ValueError(f"Expected J with 7 columns, got {n}")
    Id = np.eye(m)
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * Id)


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str):
    """Return (world position, body_id) for a named body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"Body '{body_name}' not found in model")
    return data.xpos[bid].copy(), bid


def approx_body_radius(model: mujoco.MjModel, body_id: int, default: float = 0.04) -> float:
    """Approximate a body's radius from its first sphere/capsule/cylinder geom.
    Falls back to `default` if nothing suitable is found.
    """
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == int(body_id):
            gtype = int(model.geom_type[gid])
            if gtype in (
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                return float(model.geom_size[gid][0])  # radius in size[0]
    return float(default)


def jacobian_body_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    arm_dof_idx: np.ndarray,
) -> np.ndarray:
    """3x7 geometric Jacobian of a body's CoM for the arm DOFs.

    Returns a position Jacobian (no orientation rows) sliced to the 7 DOFs
    in `arm_dof_idx`.
    """
    Jp = np.zeros((3, model.nv), dtype=float)
    Jr = np.zeros((3, model.nv), dtype=float)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"Body '{body_name}' not found in model")
    mujoco.mj_jacBody(model, data, Jp, Jr, bid)
    return Jp[:, arm_dof_idx]

def approach_normal_lateral(model, data, goal_pos, plant_body="tomato_plant"):
    """Unit vector from plant center to goal, projected to the horizontal plane."""
    c_plant, _ = _body_pos(model, data, plant_body)
    n = goal_pos - c_plant
    n[2] = 0.0  # prefer lateral approach, not vertical
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)  # fallback
    return n / n_norm

EPISODES_TOTAL = 100
EPISODES_PER_PLANT = 2  # grasp top two, then regenerate

SHOW_DEBUG_VIZ = True  # turn off to hide frames

def place_frame_mocap(model, data, body_name, origin, x_axis_hint, z_axis_hint=np.array([0,0,1.0])):
    """
    Set a mocap body's pose so its local +x ~ x_axis_hint and +z ~ z_axis_hint.
    Both axes are re-orthogonalized to make a proper rotation matrix.
    """
    import numpy as np
    def _n(v):
        n = float(np.linalg.norm(v))
        return v / (n if n > 1e-9 else 1.0)
    x = _n(np.asarray(x_axis_hint, float))
    z = _n(np.asarray(z_axis_hint, float))
    # make y = z x x, then recompute z = x x y to ensure orthonormal basis
    y = _n(np.cross(z, x))
    z = _n(np.cross(x, y))
    R = np.stack([x, y, z], axis=1)  # columns are basis vectors

    # convert to quaternion (w, x, y, z) expected by mj
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = np.sign(R[2,1] - R[1,2]) * np.sqrt(max(0.0, 1.0 + R[0,0] - R[1,1] - R[2,2])) / 2.0
    qy = np.sign(R[0,2] - R[2,0]) * np.sqrt(max(0.0, 1.0 - R[0,0] + R[1,1] - R[2,2])) / 2.0
    qz = np.sign(R[1,0] - R[0,1]) * np.sqrt(max(0.0, 1.0 - R[0,0] - R[1,1] + R[2,2])) / 2.0
    quat = np.array([qw, qx, qy, qz], dtype=float)

    mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if mocap_id < 0:
        return
    # mocap bodies appear after the regular bodies in mocap_* arrays
    idx = mocap_id - model.nbody + model.nmocap
    data.mocap_pos[idx]  = np.asarray(origin, float)
    data.mocap_quat[idx] = quat

def scene_dir_from_model_path(model_path: str) -> str:
    """Prefer directory that contains model_path; fall back to cwd."""
    base = os.path.dirname(model_path) if model_path else ""
    return base if os.path.isdir(base) else os.getcwd()

def write_scene_dynamic(scene_path: str):
    """Write a scene xml that includes panda.xml + tomato_plant.xml and
    preserves floor/camera/visual/asset from your original scene."""
    scene_txt = """<mujoco model="panda scene">
  <include file="panda.xml"/>
  <include file="tomato_plant.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <compiler angle="degree"/>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="third_person_cam" pos="0.25 -1.5 2" euler="50 0 0"/>

    <!-- ===== DEBUG VIZ (mocap bodies you can pose from Python) ===== -->
    <body name="goal_viz" mocap="true">
      <!-- axis frame (x:red, y:green, z:blue) -->
      <geom type="capsule" fromto="0 0 0 0.06 0 0" size="0.003" rgba="1 0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0.06 0" size="0.003" rgba="0 1 0 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.003" rgba="0 0 1 1"/>
      <site name="goal_site" size="0.01" rgba="1 0 0 0.25"/>
    </body>

    <body name="pregrasp_viz" mocap="true">
      <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.0025" rgba="1 0.4 0.4 1"/>
      <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.0025" rgba="0.4 1 0.4 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.0025" rgba="0.4 0.4 1 1"/>
      <site name="pregrasp_site" size="0.01" rgba="1 0.4 0.4 0.25"/>
    </body>

    <body name="retreat_viz" mocap="true">
      <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.0025" rgba="0.8 0.5 0.1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0.04 0" size="0.0025" rgba="0.5 0.8 0.1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.0025" rgba="0.1 0.5 0.8 1"/>
      <site name="retreat_site" size="0.01" rgba="0.8 0.5 0.1 0.25"/>
    </body>
    <!-- ============================================================= -->
  </worldbody>

</mujoco>"""
    with open(scene_path, "w") as f:
        f.write(scene_txt)

def build_and_load_scene(model_path: str, plant_base=(0.5, 0.0, 0.15)):
    """Generate a new plant, write scene_dynamic.xml, and load model+data."""
    basedir = scene_dir_from_model_path(model_path)
    plant_path = os.path.join(basedir, "tomato_plant.xml")
    scene_path = os.path.join(basedir, "scene_dynamic.xml")

    # 1) Generate a fresh plant file (randomized)
    generate_tomato_plant_xml(output_file=plant_path, base_pos=plant_base)

    # 2) Write scene that includes panda + plant
    write_scene_dynamic(scene_path)

    # 3) Load it
    model = mujoco.MjModel.from_xml_path(scene_path)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def find_side_stem_targets(model, data, k=2, s=0.66):
    """
    Returns a list of (stem_name, grasp_pos, approach_dir_xy, tip_body) for the top-k
    stems sorted by the grasp point Z (highest first).
    """
    info = get_side_stem_grasp_points(model, data, s=s)
    items = []
    for name, d in info.items():
        items.append((name, d["grasp_pos"], d["approach_dir_xy"], d["tip_body"]))
    ordered = sorted(items, key=lambda t: float(t[1][2]), reverse=True)
    return ordered[:k]

def harvest_any_tfrecords(root_dir: str, version_dir: str, ds_name: str, split_name: str) -> int:
    """Recursively find any *.tfrecord* under root_dir and move/rename them to:
       {version_dir}/{ds_name}-{split_name}.tfrecord-<suffix>.
       Returns number of files moved."""
    import glob, os, shutil
    os.makedirs(version_dir, exist_ok=True)
    moved = 0
    for src in sorted(glob.glob(os.path.join(root_dir, "**", "*.tfrecord*"), recursive=True)):
        # skip already-in-place files
        if os.path.commonpath([os.path.abspath(src), os.path.abspath(version_dir)]) == os.path.abspath(version_dir):
            continue
        base = os.path.basename(src)
        # keep shard suffix if present, else synthesize one
        if ".tfrecord-" in base:
            suffix = base.split(".tfrecord-", 1)[1]
        else:
            suffix = "00000-of-00001"
        dest = os.path.join(version_dir, f"{ds_name}-{split_name}.tfrecord-{suffix}")
        if os.path.exists(dest):
            i = 1
            while True:
                alt = os.path.join(version_dir, f"{ds_name}-{split_name}.tfrecord-{i:05d}-of-00100")
                if not os.path.exists(alt):
                    dest = alt
                    break
                i += 1
        shutil.move(src, dest)
        moved += 1
    if moved:
        print(f"[TFDS] Harvested {moved} shard(s) into {version_dir}")
    return moved

def repair_tfds_splits(version_dir, name):
    return repair_tfds_splits_at_dir(version_dir, name)

# Backward-compatible aliases (so you can import with underscores if you like)
_damped_pinv = damped_pinv
_body_pos = body_pos
_approx_body_radius = approx_body_radius
_jacobian_body_point = jacobian_body_point
_collect_plant_obstacles = collect_plant_obstacles
_approach_normal_lateral = approach_normal_lateral
_build_and_load_scene = build_and_load_scene
_find_top_targets = find_side_stem_targets
_write_scene_dynamic = write_scene_dynamic
_scene_dir_from_model_path = scene_dir_from_model_path
_weighted_dls = weighted_dls
_place_frame_mocap = place_frame_mocap
_choose_split_for_plant = choose_split_for_plant
_choose_split_for_episode = choose_split_for_episode
_expected_ds_dir = expected_ds_dir
_move_stray_shards_into_version_dir = move_stray_shards_into_version_dir
_repair_tfds_splits_at_dir = repair_tfds_splits_at_dir
_harvest_any_tfrecords = harvest_any_tfrecords
_repair_tfds_splits = repair_tfds_splits
# ------------------------------- Self-test ------------------------------------

def approx_body_radius_max(model, body_id, pad=0.01, default=0.03):
    """Max radius across all sphere/capsule/cylinder geoms on a body (+pad)."""
    radii = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == int(body_id):
            gtype = int(model.geom_type[gid])
            if gtype in (
                mujoco.mjtGeom.mjGEOM_SPHERE,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                radii.append(float(model.geom_size[gid][0]))
    return (max(radii) + pad) if radii else float(default)




def _self_test():
    """Run a few basic checks against your Panda model and plant obstacles."""
    import oracle as ot
    from getlocation import get_side_stem_origins_and_quats

    # 1) Load model/data FIRST
    print("[TEST] Loading model:", ot.MODEL_PATH)
    model = mujoco.MjModel.from_xml_path(ot.MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 2) Collect whole-plant obstacles
    obstacles = collect_plant_obstacles(model)
    print(f"[AVOID] using plant obstacles ({len(obstacles)} bodies):",
          obstacles[:8], "..." if len(obstacles) > 8 else "")

    # 3) Arm mapping
    arm_act_ids, arm_qpos_addr = ot.build_arm_mapping_from_model(model, prefer_position=True)
    arm_dof_idx = ot.build_arm_dof_indices(model, arm_act_ids)
    print("[TEST] arm_dof_idx:", arm_dof_idx)

    # 4) EE Jacobian sanity
    ee_kind, ee_name = getattr(ot, "EE_REF", ("body", "hand"))
    if ee_kind != "body":
        ee_name = "hand"
    J = jacobian_body_point(model, data, ee_name, arm_dof_idx)
    print("[TEST] J shape (should be 3x7):", J.shape, "  ||J||_F:", np.linalg.norm(J))
    Jp = damped_pinv(J, lam=0.1)
    ident_err = np.linalg.norm(J @ Jp - np.eye(3))
    print("[TEST] ||J J^+ - I||:", ident_err)

    # 5) Quick obstacle radius sample
    if obstacles:
        name = obstacles[0]
        c_obs, bid = body_pos(model, data, name)
        R1 = approx_body_radius(model, bid, default=0.04)
        R2 = approx_body_radius_max(model, bid, pad=0.01, default=0.03)
        print(f"[TEST] obstacle='{name}'  center={c_obs}")
        print(f"[TEST] approx_body_radius={R1:.3f}   approx_body_radius_max={R2:.3f}")

    # 6) Locator sanity (optional)
    stems = get_side_stem_origins_and_quats(model, data, prefix="side_stem")
    if stems:
        any_name = next(iter(stems))
        pos, _ = stems[any_name]
        print(f"[TEST] locator goal_pos for '{any_name}' =", pos)
    else:
        print("[TEST] No 'side_stem*' bodies found by locator.")

    # 7) Random J pinv check
    rng = np.random.default_rng(0)
    Jrand = rng.normal(size=(3, 7))
    err_rand = np.linalg.norm(Jrand @ damped_pinv(Jrand, lam=0.1) - np.eye(3))
    print("[TEST] random ||J J^+ - I||:", err_rand)

    print("[TEST] Done.")


if __name__ == "__main__":
    _self_test()
