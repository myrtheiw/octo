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

from typing import Optional

from record_dataset.generate_tomato_plant import generate_tomato_plant_xml
from record_dataset.getlocation import get_side_stem_origins_and_quats
from record_dataset.getlocation import get_side_stem_grasp_points

# ---- Split policy (edit ratios as you like) ----
SPLIT_RATIOS = dict(train=0.90, val=0.10, test=0.0)
# ------------------------------- Helpers --------------------------------------
import json, re, glob, os, shutil


def _ensure_worldbody_camera(scene_text: str, cam_xml: str) -> str:
    """Insert a camera XML snippet just after <worldbody> if it's missing."""
    # If worldbody doesn't exist, just return unchanged (very unusual for MuJoCo scenes).
    if "<worldbody" not in scene_text:
        return scene_text
    # Insert right after the opening <worldbody ...>
    return re.sub(r"(<worldbody[^>]*>)", r"\1\n    " + cam_xml.strip() + "\n", scene_text, count=1)

def _ensure_front_cam(scene_text: str) -> str:
    """Make sure a <camera name='front_cam' .../> exists."""
    if re.search(r'<camera\s+name\s*=\s*"front_cam"\b', scene_text):
        return scene_text  # already present
    # Reasonable default that matches your scene.xml (pos/axes/fovy).
    front_cam_xml = """
    <camera name="front_cam"
            pos="0.65 -1.0 1.60"
            xyaxes="0.988936 0.148340 -0.000000   -0.113436 0.756243 0.644382"
            fovy="57"/>
    """.strip()
    return _ensure_worldbody_camera(scene_text, front_cam_xml)

def expected_ds_dir(root_dir, ds_config):
    return os.path.join(root_dir, ds_config.name, str(ds_config.version))

def move_stray_shards_into_version_dir(root_dir, ds_config):
    """Recursively find any *.tfrecord-* shards under root_dir and move them into
    <root_dir>/<name>/<version>/, preserving the shard suffix."""
    expected = expected_ds_dir(root_dir, ds_config)
    os.makedirs(expected, exist_ok=True)

    # Envlogger writes shards directly under `data_directory`; only scoop those.
    pattern = os.path.join(root_dir, f"{ds_config.name}-*.tfrecord-*")
    stray = sorted(glob.glob(pattern, recursive=False))
    moved = 0
    for src in stray:
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


def compute_joint_delta_action(
    q_now,
    q_target,
    include_gripper: Optional[float] = None,
) -> np.ndarray:
    """Return the joint delta needed to reach q_target from q_now."""
    q_now_arr = np.asarray(q_now, dtype=np.float32)
    q_next_arr = np.asarray(q_target, dtype=np.float32)
    delta = (q_next_arr[:7] - q_now_arr[:7]).astype(np.float32, copy=False)
    if include_gripper is None:
        return delta
    grip = np.array([float(include_gripper)], dtype=np.float32)
    return np.concatenate([delta, grip], axis=0)

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

# --- add near the top of helpers.py ---
import re
import os
import mujoco
from typing import Optional

def _swap_plant_include(scene_text: str, new_include: str = "tomato_plant.xml") -> str:
    """
    Replace any tomato-plant include line with `tomato_plant.xml`.
    Handles tomato_plant_v10.xml or older names. If none found, we insert
    right after the panda include.
    """
    # 1) Try to replace any tomato_plant*.xml include
    pat = r'<include\s+file\s*=\s*"tomato_plant[^"]*\.xml"\s*/?>'
    if re.search(pat, scene_text):
        return re.sub(pat, f'<include file="{new_include}"/>', scene_text, count=99)

    # 2) Otherwise, insert after panda include (best-effort)
    panda_pat = r'(<include\s+file\s*=\s*"panda\.xml"\s*/?>)'
    if re.search(panda_pat, scene_text):
        return re.sub(panda_pat,
                      r'\1\n  <include file="{}"/>'.format(new_include),
                      scene_text, count=1)
    # 3) Fallback: just prepend one at top-level after <mujoco ...>
    head_pat = r'(<mujoco[^>]*>)'
    if re.search(head_pat, scene_text):
        return re.sub(head_pat,
                      r'\1\n  <include file="{}"/>'.format(new_include),
                      scene_text, count=1)
    return scene_text  # last resort, unchanged


def _dedupe_named_tags(scene_text: str, tag: str, name: str) -> str:
    """
    Keep only the first <tag name="name" ...> ... </tag> (or self-closing) block.
    Removes subsequent duplicates to avoid MuJoCo 'repeated name' errors.
    """
    # Matches both self-closing <camera .../> and block <camera ...>...</camera>
    pat = re.compile(
        rf'<{tag}\s+[^>]*name\s*=\s*"{re.escape(name)}"[^>]*\/>|'
        rf'<{tag}\s+[^>]*name\s*=\s*"{re.escape(name)}"[^>]*>.*?<\/{tag}>',
        flags=re.DOTALL | re.IGNORECASE
    )
    matches = list(pat.finditer(scene_text))
    if len(matches) <= 1:
        return scene_text  # nothing to do
    # Keep the first, remove the rest
    keep_start, keep_end = matches[0].span()
    pieces = [scene_text[:keep_end]]
    last = keep_end
    for m in matches[1:]:
        s, e = m.span()
        pieces.append(scene_text[last:s])  # skip the duplicate block
        last = e
    pieces.append(scene_text[last:])
    return "".join(pieces)


def write_scene_dynamic(scene_path: str,
                        template_scene_path: Optional[str] = None):
    if template_scene_path and os.path.isfile(template_scene_path):
        with open(template_scene_path, "r") as f:
            txt = f.read()
        # swap tomato include so dynamic plant comes from tomato_plant.xml
        txt2 = _swap_plant_include(txt, "tomato_plant.xml")

        # NEW: ensure we don’t duplicate cameras that may also be defined in included files
        txt2 = _dedupe_named_tags(txt2, tag="camera", name="front_cam")
        txt2 = _dedupe_named_tags(txt2, tag="camera", name="third_person_cam")

        with open(scene_path, "w") as f:
            f.write(txt2)
        print(f"[scene] wrote dynamic scene from template: {template_scene_path}")
        return

    # (fallback scene text unchanged)

    # ---- minimal fallback (old behavior) ----
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
  </worldbody>
</mujoco>"""
    # ensure front_cam for the fallback file too
    scene_txt = _ensure_front_cam(scene_txt)
    with open(scene_path, "w") as f:
        f.write(scene_txt)
    print("[scene] wrote minimal dynamic scene (fallback).")


_REUSE_DYNAMIC_SCENE = bool(int(os.environ.get("OCTO_REUSE_SCENE", "0")))


def build_and_load_scene(model_path: str, plant_base=(0.5, 0.0, 0.15)):
    """Generate a new plant, write scene_dynamic.xml based on template scene, and load."""
    basedir = scene_dir_from_model_path(model_path)
    plant_path = os.path.join(basedir, "tomato_plant.xml")
    scene_path = os.path.join(basedir, "scene_dynamic.xml")

    if _REUSE_DYNAMIC_SCENE and os.path.exists(scene_path):
        print("[scene] Reusing existing dynamic scene XML (OCTO_REUSE_SCENE=1).")
    else:
        # 1) Fresh randomized plant
        generate_tomato_plant_xml(output_file=plant_path, base_pos=plant_base)

        # 2) Clone your scene.xml look & swap plant include
        write_scene_dynamic(scene_path, template_scene_path=model_path)

    # 3) Load dynamic scene
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
    """Recursively find *.tfrecord* under root_dir and move/rename them into:
       {version_dir}/{ds_name}-{split_name}.tfrecord-<suffix>.
       Returns the number of files moved."""
    import glob, os, shutil

    os.makedirs(version_dir, exist_ok=True)
    moved = 0

    # Accept both .tfrecord and .tfrecord-<shard> names
    cand_patterns = ["**/*.tfrecord", "**/*.tfrecord-*"]
    candidates = []
    for pat in cand_patterns:
        candidates.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))

    # Dedup and sort for stability
    seen = set()
    for src in sorted(set(candidates)):
        src_abs = os.path.abspath(src)
        # skip anything already inside the version_dir
        try:
            if os.path.commonpath([src_abs, os.path.abspath(version_dir)]) == os.path.abspath(version_dir):
                continue
        except Exception:
            # If commonpath fails on different drives, ignore and continue
            pass

        base = os.path.basename(src)
        # keep shard suffix if present, else synthesize one
        if ".tfrecord-" in base:
            suffix = base.split(".tfrecord-", 1)[1]
        else:
            # Envlogger sometimes writes plain ".tfrecord" → normalize to a single shard
            suffix = "00000-of-00001"

        dest = os.path.join(version_dir, f"{ds_name}-{split_name}.tfrecord-{suffix}")
        # If collision, synthesize sequential shard ids (kept consistent width)
        i = 0
        while os.path.exists(dest):
            i += 1
            dest = os.path.join(version_dir, f"{ds_name}-{split_name}.tfrecord-{i:05d}-of-00100")

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
        moved += 1

    if moved:
        print(f"[TFDS] Harvested {moved} shard(s) into {version_dir}")
    return moved




def repair_tfds_splits(version_dir, name):
    return repair_tfds_splits_at_dir(version_dir, name)


def ee_pose(model, data, ee_ref):
    kind, name = ee_ref
    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid].copy()
        mat = data.site_xmat[sid].reshape(3,3).copy()
    else:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[bid].copy()
        mat = data.xmat[bid].reshape(-1,3,3)[bid].copy()
    # convert rotation matrix to axis-angle (magnitude=angle)
    # MuJoCo has mj_mat2Quat; we’ll do a small utility here:
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return pos, quat

def quat_ang_dist(q1, q2):
    # shortest-angle distance between orientations
    dq = np.array([q2[0]*q1[0] + q2[1]*q1[1] + q2[2]*q1[2] + q2[3]*q1[3]], dtype=float)  # dot
    dq = np.clip(dq, -1.0, 1.0)
    return 2.0 * np.arccos(np.abs(dq))[0]

def trapezoid_times(total, vmax, amax):
    """Return cumulative times for a 1D path of length `total` under trapezoid limits."""
    total = float(max(total, 0.0))
    if total <= 1e-9:
        return [0.0]
    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc**2
    if 2*d_acc >= total:  # triangle
        t_acc = np.sqrt(total / amax)
        t_peak = t_acc
        t_total = 2*t_acc
    else:
        d_cruise = total - 2*d_acc
        t_cruise = d_cruise / vmax
        t_peak = t_acc + t_cruise
        t_total = 2*t_acc + t_cruise
    return [0.0, t_peak, t_total]

def time_parameterize_by_ee_limits(model, data, arm_qpos_addr, ee_ref,
                                    traj_q: np.ndarray,
                                    rate_hz: float,
                                    vmax_trans: float, amax_trans: float,
                                    vmax_rot: float,   amax_rot: float) -> np.ndarray:
    """Resample joint waypoints so that EE motion obeys trans/rot trapezoidal limits, then sample at `rate_hz`."""
    if len(traj_q) < 2:
        return traj_q.astype(np.float32, copy=False)

    # FK: EE positions and orientation deltas
    pos_list, quat_list = [], []
    _bak = data.qpos.copy()
    for q in traj_q:
        data.qpos[arm_qpos_addr] = q[:7]
        mujoco.mj_forward(model, data)
        p, qh = _ee_pose(model, data, ee_ref)
        pos_list.append(p); quat_list.append(qh.copy())
    data.qpos[:] = _bak; mujoco.mj_forward(model, data)

    pos = np.stack(pos_list, axis=0)
    # translational arc-length along the polyline
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s_trans = np.concatenate([[0.0], np.cumsum(dp)])
    total_trans = float(s_trans[-1])

    # rotational arc (angle distance)
    dang = [0.0]
    for i in range(1, len(quat_list)):
        dang.append(_quat_ang_dist(quat_list[i-1], quat_list[i]))
    s_rot = np.cumsum(dang)
    s_rot = np.concatenate([[0.0], s_rot])
    total_rot = float(s_rot[-1])

    # times to complete each modality under trapezoid, then take the max
    t_marks_trans = _trapezoid_times(total_trans, vmax_trans, amax_trans)
    t_marks_rot   = _trapezoid_times(total_rot,   vmax_rot,   amax_rot)
    t_total = max(t_marks_trans[-1], t_marks_rot[-1])

    # Desired sampling at fixed rate_hz
    if rate_hz <= 0:
        rate_hz = 10.0
    dt = 1.0 / float(rate_hz)
    times = np.arange(0.0, t_total + 1e-9, dt)

    # Map desired time -> path fraction for each modality, then choose the limiting fraction
    def frac_from_trap(t, t_marks, total):
        t = float(np.clip(t, 0.0, t_marks[-1]))
        if len(t_marks) == 1 or total <= 1e-12:
            return 0.0
        # compute trapezoid params
        if len(t_marks) == 3:
            t_acc, t_peak, t_total_loc = t_marks[0], t_marks[1], t_marks[2]
        else:
            t_acc, t_peak, t_total_loc = 0.0, t_marks[0], t_marks[-1]
        vmax = vmax_trans if total == total_trans else vmax_rot
        amax = amax_trans if total == total_trans else amax_rot

        if 2*(0.5*amax*(vmax/amax)**2) >= total:  # triangle case recompute
            t_acc = np.sqrt(total/amax)
            t_peak = t_acc
            t_total_loc = 2*t_acc

        if t <= t_acc:
            s = 0.5*amax*t**2
        elif t <= t_peak:
            s = 0.5*amax*t_acc**2 + vmax*(t - t_acc)
        else:
            t_dec = t - t_peak
            s = total - 0.5*amax*(t_total_loc - t)**2
        return float(np.clip(s / max(total, 1e-12), 0.0, 1.0))

    fracs = []
    for t in times:
        f_trans = frac_from_trap(t, t_marks_trans, total_trans)
        f_rot   = frac_from_trap(t, t_marks_rot,   total_rot)
        fracs.append(max(f_trans, f_rot))
    fracs = np.asarray(fracs)

    # Convert path fractions to indices over the original joint path (piecewise linear)
    idx_float = fracs * (len(traj_q) - 1)
    idx0 = np.floor(idx_float).astype(int)
    idx1 = np.clip(idx0 + 1, 0, len(traj_q) - 1)
    alpha = (idx_float - idx0).reshape(-1, 1).astype(np.float32)

    traj_resampled = (1.0 - alpha) * traj_q[idx0, :7] + alpha * traj_q[idx1, :7]
    if traj_q.shape[1] > 7:
        g = (1.0 - alpha[:,0]) * traj_q[idx0, 7] + alpha[:,0] * traj_q[idx1, 7]
        traj_out = np.concatenate([traj_resampled, g[:, None]], axis=1)
    else:
        traj_out = traj_resampled
    return traj_out.astype(np.float32, copy=False)


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
_compute_joint_delta_action = compute_joint_delta_action
_choose_split_for_plant = choose_split_for_plant
_choose_split_for_episode = choose_split_for_episode
_expected_ds_dir = expected_ds_dir
_move_stray_shards_into_version_dir = move_stray_shards_into_version_dir
_repair_tfds_splits_at_dir = repair_tfds_splits_at_dir
_harvest_any_tfrecords = harvest_any_tfrecords
_repair_tfds_splits = repair_tfds_splits
_ee_pose = ee_pose
_quat_ang_dist = quat_ang_dist
_trapezoid_times = trapezoid_times
_time_parameterize_by_ee_limits = time_parameterize_by_ee_limits 




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
