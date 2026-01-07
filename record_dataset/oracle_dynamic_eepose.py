
#!/usr/bin/env python3
"""
oracle_dynamic.py
-----------------
Panda oracle for tomato picking in MuJoCo with:
- LM/DLS Jacobian IK (+ null-space collision avoidance with a goal-aware corridor)
- Pregrasp → goal → retreat staging
- PD torque control for smooth execution
- Dynamic-plant regeneration and multi-episode RLDS logging
- Wrist-cam + third-person images in the observation
- Conservative, robust defaults (pregrasp/retreat, repeats, holds)

This file consolidates the "working old" logic with your newer pieces so it
works end-to-end with dynamic plants and randomized geometry.

HOW TO RUN
----------
$ python oracle_dynamic.py

Key toggles are at the top (e.g., USE_DYNAMIC_PLANT, EPISODES_TOTAL, etc.).
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")   # try EGL first
import time
import random
from pathlib import Path
import numpy as np
import mujoco
import dm_env
from dm_env import specs, TimeStep
import json, re, glob, os, shutil, hashlib

try:
    import imageio.v2 as _imageio
except ModuleNotFoundError:
    _imageio = None
    try:
        from PIL import Image as _PILImage
    except ModuleNotFoundError:
        _PILImage = None
else:
    _PILImage = None

# Rendering
from mujoco import viewer

# RLDS / TFDS
import tensorflow as tf
import tensorflow_datasets as tfds
import envlogger
from envlogger.backends import tfds_backend_writer

# Target detection utils (you already have these)
from getlocation import get_side_stem_origins_and_quats, get_side_stem_grasp_points

# Collision avoidance + dynamic scene helpers (provided in your helpers.py)
from helpers import (
    damped_pinv as _damped_pinv,
    body_pos as _body_pos,
    approx_body_radius_max as _approx_body_radius,
    jacobian_body_point as _jacobian_body_point,
    collect_plant_obstacles as _collect_plant_obstacles,
    approach_normal_lateral as _approach_normal_lateral,
    weighted_dls as _weighted_dls,
    build_and_load_scene as _build_and_load_scene,     # dynamic plant writer/loader
    find_side_stem_targets as _find_top_targets,        # compute grasp + approach
    place_frame_mocap as _place_frame_mocap,            # viz helper (optional)
    scene_dir_from_model_path as _scene_dir,
    _choose_split_for_plant as choose_split_for_plant,
    _repair_tfds_splits_at_dir as repair_tfds_splits,
    _expected_ds_dir as expected_ds_dir,
    _move_stray_shards_into_version_dir as move_stray_shards_into_version_dir,
    _harvest_any_tfrecords as harvest_any_tfrecords,
    _repair_tfds_splits as repair_tfds_splits,
    ee_pose as ee_pose,
    quat_ang_dist as _quat_ang_dist,
    trapezoid_times as _trapezoid_times,
    time_parameterize_by_ee_limits as _time_parameterize_by_ee_limits,
)

import helpers, inspect  # noqa: E402,N812
print(f"[helpers] using: {inspect.getfile(helpers)}")

# ------------------------------- Config ---------------------------------------

# Base model/scene: we will swap to a dynamic scene if USE_DYNAMIC_PLANT=True
MODEL_PATH = os.environ.get("MODEL_PATH", 
    "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"
)

# End-effector reference: prefer TCP site if present; else hand body
EE_REF = ("site", "tcp")   # will auto-fallback to ("body","hand") at runtime if tcp is missing

# Cameras
PRIMARY_CAM_NAME = os.environ.get("PRIMARY_CAM_NAME", "front_cam")
WRIST_CAM_NAME = os.environ.get("WRIST_CAM_NAME", "gripper_cam")

IMG_H, IMG_W     = 256, 256

# Motion staging & pacing
START_JOINTS        = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=float)
PREGRASP_OFFSET     = 0.1   # meters along lateral normal (bigger is safer near foliage)
RETREAT_OFFSET      = 0.0     # meters opposite lateral normal (back out)
PREGRASP_DWELL_SEC = 0.3  # shorter hold to reduce stagnant frames

N_CART_WAYPOINTS    = 60       # resampled joint waypoints total (smoothness vs length)
WAYPOINT_REPEAT     = 1        # repeat each waypoint to slow motion (stability)
WAYPOINT_DOWNSAMPLE = 1        # keep every waypoint (dedupe handles smoothing)
PAUSE_PREGRASP_STEPS= 40       # dwell at pregrasp while open
EXTRA_HOLD_STEPS    = 10       # hold final pose after retreat for logging stability
HAND_ROLL_AT_GRASP_DEG = 30.0   # wrist roll amount before closing
ROLL_RAMP_STEPS        = 20     # spread the roll across N dwell steps

# ===== Realistic pacing targets =====
# Action/logging rate target (10 Hz ⇒ ~0.1s per outer step)
TARGET_ACTION_DT = 0.10  # seconds per outer step

# Pauses ("beats") in seconds — converted to steps at runtime
PREGRASP_DWELL_SEC     = 0.5
GRIPPER_CLOSE_SEC      = 0.3
POST_GRASP_SETTLE_SEC  = 0.5
POST_PLACE_SETTLE_SEC  = 0.5
FINAL_HOLD_SEC         = 0.5  # replaces/augments DATASET_FINAL_HOLD_STEPS

# End-effector limits (for time-parameterization)
V_MAX_TRANS = 0.15        # m/s
A_MAX_TRANS = 0.5         # m/s^2
V_MAX_ROT   = np.deg2rad(60.0)   # rad/s
A_MAX_ROT   = np.deg2rad(180.0)  # rad/s^2


# IK tuning (LM/DLS)
IK_MAX_ITERS = 200
IK_POS_TOL   = 1e-3
LM_STEP      = 0.4
LM_DAMP      = 0.12

# Null-space collision avoidance
USE_NULLSPACE_AVOID = True
AVOID_LINKS = ("link3", "link4", "link5", "link6", "hand")
AVOID_D0    = 0.10
AVOID_GAIN  = 0.6
AVOID_LAM   = 0.05

HAND_ROLL_AT_GRASP_DEG = 30.0   # how much to roll the last joint before closing
ROLL_RAMP_STEPS        = 20     # how many trajectory steps to spread that roll over


# PD controller gains (simple diagonal)
PD_KP = 100.0
PD_KD = 2.0 * np.sqrt(PD_KP)

# Optional: widen joint limits for the 7 arm joints (helps if model has tiny ranges)
AUTO_WIDEN_LIMITS = True
TYPICAL_FRANKA_LIMITS = [
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
]


# Plant & dataset settings
USE_DYNAMIC_PLANT  = True     # regenerate plant geometry every couple episodes
EPISODES_TOTAL     = int(os.environ.get("EPISODES_TOTAL", 100))
EPISODES_PER_PLANT = 2        # top-2 stems per plant, then regenerate

TFDS_ROOT_DIR   = os.environ.get("TFDS_ROOT_DIR", "/home/myrtheiw/tfds_out")
DATASET_NAME    = os.environ.get("DATASET_NAME", "tomato_rlds")
DATASET_VERSION = os.environ.get("DATASET_VERSION", "0.0.40")

_DEFAULT_GOAL_IMAGE_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "outputs" / "goal_images"
)
GOAL_IMAGE_OUTPUT_DIR = os.environ.get(
    "GOAL_IMAGE_OUTPUT_DIR",
    str(_DEFAULT_GOAL_IMAGE_OUTPUT_DIR),
)

ENABLE_TFRECORD_HARVEST = bool(int(os.environ.get("ENABLE_TFRECORD_HARVEST", "0")))
def _ensure_goal_dir_exists():
    try:
        os.makedirs(GOAL_IMAGE_OUTPUT_DIR, exist_ok=True)
        return True
    except Exception as exc:
        print(f"[GOAL_IMG] unable to create directory '{GOAL_IMAGE_OUTPUT_DIR}': {exc}")
        return False

def _normalize_delta(delta: np.ndarray) -> np.ndarray:
    """
    Normalize a raw joint delta Δq (radians) into approx [-1, 1] using JOINT_DELTA_SCALE.
    """
    delta = np.asarray(delta, dtype=np.float32).reshape(7)
    scaled = delta / JOINT_DELTA_SCALE
    # Hard clamp to Octo-style range
    return np.clip(scaled, -1.0, 1.0).astype(np.float32, copy=False)


def _write_goal_image(path: str, image: np.ndarray) -> None:
    if image is None:
        return
    if _imageio is not None:
        _imageio.imwrite(path, image)
    elif _PILImage is not None:
        _PILImage.fromarray(image).save(path)
    else:
        try:
            encoded = tf.io.encode_png(image).numpy()
            with open(path, "wb") as f:
                f.write(encoded)
        except Exception as exc:
            print(f"[GOAL_IMG] skipping save for {path}; encode failed: {exc}")

# Debug
DEBUG_IK        = True
LIVE_RENDER     = True
LIVE_FPS        = 60.0
SHOW_DEBUG_VIZ  = False   # set True to drop mocap frames (pre/goal/retreat)

# --- Quality gate: only log successful episodes ---
MAX_FINAL_ERR = 0.010      # meters; tighten/loosen for your dataset quality
LIVE_RENDER_QC = True      # render the QC dry-run, like LIVE_RENDER
CAPTURE_IMAGES_DURING_QC = False  # speed up QC (don’t waste time rendering)


# Train/val/test ratios (edit as you like)
SPLIT_RATIOS = dict(train=0.90, val=0.10, test=0.0)

# Add near other config constants
DATASET_ACTION_SCALE = 1.0   # action labels store Δq / scale; match this with --dataset_action_scale during replay
DATASET_MIN_STEP_NORM = 1e-3   # drop frames whose joint delta is below this (radians)
DATASET_FINAL_HOLD_STEPS = 10  # small hold appended after deduplication for stability
DATASET_MAX_STEP_RAD = 2.0e-2  # clamp per-step joint deltas by inserting interpolated waypoints
JOINT_DELTA_SCALE = np.full(7, DATASET_MAX_STEP_RAD, dtype=np.float32)

# --------------------- Action representation (EE deltas) ----------------------

# EE action: [dx, dy, dz, dyaw]
EE_ACTION_DIM = 4

# Rough max per-step movement that should map to |a| ~= 1.0 after normalization.
# Tune if you see saturation or tiny effective actions.
EE_TRANS_SCALE = np.array([0.02, 0.02, 0.02], dtype=np.float32)  # 2 cm per step
EE_YAW_SCALE   = np.deg2rad(10.0)                                # 10 deg per step

def _ee_pos_yaw(model, data, ee_ref):
    """Return (pos[3], yaw) for the end-effector."""
    pos, R = _ee_pose(model, data, ee_ref)  # existing helper → pos, 3x3 rot matrix
    # Simple yaw extraction from rotation matrix (Z-rotation)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return pos.astype(np.float32), np.float32(yaw)

def _ee_delta(prev_pos, prev_yaw, cur_pos, cur_yaw):
    """Delta EE in world frame: [dx, dy, dz, dyaw] (raw, un-normalized)."""
    dpos = np.asarray(cur_pos, dtype=np.float32) - np.asarray(prev_pos, dtype=np.float32)
    dyaw = float(cur_yaw - prev_yaw)
    # Wrap to [-pi, pi] to avoid jumps
    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.concatenate([dpos, np.array([dyaw], dtype=np.float32)], axis=0)

def _normalize_ee_delta(delta: np.ndarray) -> np.ndarray:
    """
    Normalize raw EE delta [dx,dy,dz,dyaw] into approx [-1, 1].
    """
    delta = np.asarray(delta, dtype=np.float32).reshape(EE_ACTION_DIM)
    out = delta.copy()
    out[:3] /= EE_TRANS_SCALE
    out[3]  /= EE_YAW_SCALE
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

class ActionDeltaStatistics:
    """Accumulates raw Δq statistics for Octo-compatible normalization metadata."""

    def __init__(self, action_dim: int = 7):
        self.action_dim = int(action_dim)
        self._values: list[np.ndarray] = []
        self.num_transitions = 0

    def observe(self, delta: np.ndarray) -> None:
        arr = np.asarray(delta, dtype=np.float32).reshape(self.action_dim)
        self._values.append(arr.copy())
        self.num_transitions += 1

    def _stack(self) -> np.ndarray:
        if not self._values:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        return np.stack(self._values, axis=0)

    def compute_statistics(self) -> dict:
        values = self._stack()
        if values.size == 0:
            zero = [0.0] * self.action_dim
            return {
                "min": zero,
                "max": zero,
                "mean": zero,
                "std": zero,
                "p01": zero,
                "p99": zero,
                "mask": [True] * self.action_dim,
            }

        stats = {
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "p01": np.quantile(values, 0.01, axis=0).tolist(),
            "p99": np.quantile(values, 0.99, axis=0).tolist(),
            "mask": [True] * self.action_dim,
        }
        return stats

    def write_dataset_statistics(
        self,
        directory: str,
        dataset_name: str,
        dataset_version: str,
        num_trajectories: int,
    ) -> str:
        stats = self.compute_statistics()
        metadata = {
            "action": stats,
            "num_transitions": self.num_transitions,
            "num_trajectories": num_trajectories,
        }
        hash_components = (
            dataset_name,
            dataset_version,
            f"action_dim={self.action_dim}",
            "oracle_dynamic_norm",
        )
        unique_hash = hashlib.sha256(
            "".join(hash_components).encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        filename = f"dataset_statistics_{unique_hash}.json"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return path




# --------------------------- Model/Actuator mapping ---------------------------
def _post_write_repair(dataset_dir: str, dataset_name: str):
    """
    Make shard names TFDS-friendly and safely update dataset_info.json:
    - Never discard existing TFDS metadata (features/schema/etc.)
    - Only update: splits (numShards/shardLengths/numBytes), fileFormat (if missing)
    """
    import os as _os, glob as _glob, json as _json, tensorflow as _tf

    _os.makedirs(dataset_dir, exist_ok=True)

    # 1) Collect shards per split
    split_to_files = {}
    patterns = [
        _os.path.join(dataset_dir, f"{dataset_name}-*.tfrecord-*-of-*"),
        _os.path.join(dataset_dir, f"{dataset_name}-*.tfrecord-0000*"),  # legacy fallback
    ]
    seen = set()
    for pat in patterns:
        for p in sorted(_glob.glob(pat)):
            if p in seen:
                continue
            seen.add(p)
            base = _os.path.basename(p)
            # Expect tomato_rlds-<split>.tfrecord-...
            try:
                split = base.split("-")[1].split(".")[0]
            except Exception:
                continue
            split_to_files.setdefault(split, []).append(p)

    # 2) Canonicalize shard naming so TFDS can locate files deterministically
    canonical_split_files = {}
    for sp, files in split_to_files.items():
        files_sorted = sorted(files)
        n = len(files_sorted)
        new_paths = []
        for idx, path in enumerate(files_sorted):
            target = _os.path.join(
                dataset_dir,
                f"{dataset_name}-{sp}.tfrecord-{idx:05d}-of-{n:05d}",
            )
            if _os.path.abspath(path) != _os.path.abspath(target):
                _os.makedirs(_os.path.dirname(target), exist_ok=True)
                _os.replace(path, target)
            new_paths.append(target)
        canonical_split_files[sp] = new_paths

    def _count_records(path: str) -> int:
        n = 0
        for _ in _tf.data.TFRecordDataset(path):
            n += 1
        return max(1, n)  # be conservative

    # 3) Build new split entries (INT shardLengths)
    new_splits = []
    for sp, files in sorted(canonical_split_files.items()):
        files_sorted = sorted(files)
        shard_lengths = [_count_records(f) for f in files_sorted]
        num_bytes = sum(_os.path.getsize(f) for f in files_sorted)
        new_splits.append({
            "name": sp,
            "numShards": len(files_sorted),
            "shardLengths": shard_lengths,   # integers
            "numBytes": int(num_bytes),
        })

    # 4) Merge into existing dataset_info.json (preserve features/schema/etc.)
    info_path = _os.path.join(dataset_dir, "dataset_info.json")
    info = {}
    if _os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                info = _json.load(f)
        except Exception:
            info = {}

    info.setdefault("name", dataset_name)
    info["version"] = _os.path.basename(dataset_dir)  # e.g. "0.0.7"
    # Keep pre-existing "features" and other TFDS fields intact
    info["splits"] = new_splits
    # Ensure fileFormat is present for TFDS>=4 read_only builder
    info.setdefault("fileFormat", "tfrecord")

    with open(info_path, "w") as f:
        _json.dump(info, f, indent=2)


def _relocate_dataset_metadata(dataset_root: str, version_dir: str, dataset_version: str) -> None:
    """Move root-level metadata files into the active version directory."""

    for name in ("dataset_info.json", "features.json"):
        src = os.path.join(dataset_root, name)
        if not os.path.exists(src):
            continue

        meta_version = ""
        try:
            with open(src, "r") as fh:
                meta = json.load(fh)
            if isinstance(meta, dict):
                meta_version = str(meta.get("version", ""))
        except Exception:
            meta_version = ""

        if meta_version and meta_version != dataset_version:
            continue

        dest = os.path.join(version_dir, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.abspath(src) == os.path.abspath(dest):
            continue
        try:
            shutil.move(src, dest)
            print(f"[TFDS] Moved metadata {name} -> {dest}")
        except Exception as exc:
            print(f"[TFDS] Warning: unable to move {src} -> {dest}: {exc}")


def _ensure_min_train_split(version_dir: str, dataset_name: str) -> None:
    """Guarantee at least one train shard exists (tiny runs can skip all train episodes).

    If we only recorded validation shards (e.g., short collection where QC skipped train
    episodes), duplicate the first available val shard into the train split so downstream
    TFDS loaders see both splits. The copy keeps the val shard intact for evaluation.
    """

    train_pattern = os.path.join(version_dir, f"{dataset_name}-train.tfrecord-*")
    if glob.glob(train_pattern):
        return

    val_shards = sorted(glob.glob(os.path.join(version_dir, f"{dataset_name}-val.tfrecord-*")))
    if not val_shards:
        return

    src = val_shards[0]
    dest = src.replace(f"{dataset_name}-val", f"{dataset_name}-train")

    copy_idx = 1
    base_dest, ext = os.path.splitext(dest)
    while os.path.exists(dest):
        dest = f"{base_dest}_copy{copy_idx}{ext}"
        copy_idx += 1

    shutil.copy2(src, dest)
    print(f"[TFDS] Duplicated {os.path.basename(src)} -> {os.path.basename(dest)} to seed train split")


def choose_split_for_episode(ep_idx: int) -> str:
    """Deterministically assign episodes to train/val.

    For long runs we retain the historical "every Nth episode goes to val"
    behaviour (N≈1/val_ratio). For short runs, ensure we still log at least one
    validation shard without starving the train split.
    """

    val_ratio = float(SPLIT_RATIOS.get("val", 0.0))
    if val_ratio <= 0.0:
        return "train"

    # Aim for one val episode every `val_period`.
    val_period = max(2, int(round(1.0 / val_ratio)))  # >=2 so train keeps data

    if EPISODES_TOTAL <= 1:
        return "train"

    if EPISODES_TOTAL < val_period:
        # Short run: reserve final episode for validation, rest for train.
        return "val" if ep_idx == (EPISODES_TOTAL - 1) else "train"

    # Default stride for longer collections.
    return "val" if ((ep_idx + 1) % val_period == 0) else "train"


def find_gripper_actuator(model):
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if any(k in name.lower() for k in ("grip", "finger", "hand")):
            return aid
    return -1

def build_arm_mapping_from_model(model, prefer_position=True):
    """Return (arm_act_ids, arm_qpos_addr) for Panda joints (link1..link7)."""
    arm_body_names = [f"link{i}" for i in range(1, 8)]
    grip_keywords = ("grip", "finger", "hand")

    arm_joint_ids = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        b = int(model.jnt_bodyid[j])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bname in arm_body_names:
            arm_joint_ids.append(j)

    per_joint_candidates = {j: [] for j in arm_joint_ids}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid][0])
        if jid not in per_joint_candidates:
            continue
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if any(k in aname.lower() for k in grip_keywords):
            continue
        gaintype = int(model.actuator_gaintype[aid])  # 3=position, 0=general, etc.
        qadr = int(model.jnt_qposadr[jid])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])) or ""
        per_joint_candidates[jid].append((aid, jid, qadr, jname, aname, gaintype, bname))

    chosen = []
    for jid in arm_joint_ids:
        cands = per_joint_candidates.get(jid, [])
        if not cands:
            continue
        if prefer_position:
            cands.sort(key=lambda t: (0 if t[5] == 3 else 1, t[2]))
        else:
            cands.sort(key=lambda t: (0 if t[5] != 3 else 1, t[2]))
        chosen.append(cands[0])

    chosen = [c for c in chosen if c[6] in arm_body_names]
    by_jid = {}
    for c in chosen:
        by_jid.setdefault(c[1], c)
    chosen = sorted(by_jid.values(), key=lambda t: t[2])

    if len(chosen) != 7:
        sample = [(c[6], c[3], c[4], c[5], c[2]) for c in chosen]
        raise RuntimeError(
            f"Could not find 7 unique arm actuators (got {len(chosen)}). "
            f"Chosen sample: {sample}"
        )

    arm_act_ids  = np.array([c[0] for c in chosen], dtype=int)
    arm_qpos_adr = np.array([c[2] for c in chosen], dtype=int)
    return arm_act_ids, arm_qpos_adr

def build_arm_dof_indices(model, arm_act_ids):
    pairs = []
    for aid in arm_act_ids:
        jid = int(model.actuator_trnid[aid][0])
        dof = next(d for d in range(model.nv) if int(model.dof_jntid[d]) == jid)
        qadr = int(model.jnt_qposadr[jid])
        pairs.append((qadr, dof))
    pairs.sort(key=lambda t: t[0])
    return np.array([dof for (_, dof) in pairs], dtype=int)

# ------------------------------- IK utilities ---------------------------------

def _ee_pose(model, data, ee_ref):
    kind, name = ee_ref
    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid].copy()
        rot = data.site_xmat[sid].reshape(3, 3).copy()
    else:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[bid].copy()
        rot = data.xmat[bid].reshape(3, 3).copy()
    return pos, rot

def _jacobian_7(model, data, ee_ref, arm_dof_idx, with_orientation=False):
    mujoco.mj_forward(model, data)
    Jp = np.zeros((3, model.nv), dtype=float)
    Jr = np.zeros((3, model.nv), dtype=float)

    kind, name = ee_ref
    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        mujoco.mj_jacSite(model, data, Jp, Jr, sid)
    elif kind == "body":
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mujoco.mj_jacBody(model, data, Jp, Jr, bid)
    else:
        raise ValueError(f"ee_ref kind must be 'site' or 'body', got {kind}")

    Jp7 = Jp[:, arm_dof_idx]
    if not with_orientation:
        return Jp7
    Jr7 = Jr[:, arm_dof_idx]
    return np.vstack([Jp7, Jr7])

def lm_ik_step_position_only(
    model, data, target_pos_world, arm_dof_idx, arm_qpos_addr, ee_ref,
    step_size=LM_STEP, damping=LM_DAMP
):
    p_cur, _ = _ee_pose(model, data, ee_ref)
    e = target_pos_world - p_cur
    J = _jacobian_7(model, data, ee_ref, arm_dof_idx, with_orientation=False)
    H = J.T @ J + (damping * np.eye(7))
    dq7 = np.linalg.solve(H, J.T @ e)
    q = data.qpos[arm_qpos_addr].copy()
    q_new = q + step_size * dq7
    data.qpos[arm_qpos_addr] = q_new
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(e))

def lm_ik_step_position_with_avoidance(
    model, data, target_pos_world, arm_dof_idx, arm_qpos_addr, ee_ref,
    obstacles,
    avoid_links=AVOID_LINKS,
    step_size=LM_STEP, damping=LM_DAMP,
    d0=AVOID_D0, k_avoid=AVOID_GAIN, lam_avoid=AVOID_LAM,
    max_step=0.20,
    gate_r=0.05,        # start fading avoidance within 5 cm of goal
    gate_band=0.03,     # fade to zero over the next 3 cm
    posture_weight=0.02,# small posture prior in null space (0 disables)
    q_nom=None,         # nominal posture; if None, uses START_JOINTS if defined
    respect_limits=False,
    # Goal-conditioned corridor for the wrist/hand:
    goal_ctx=None,      # {"goal_pos": ..., "approach_dir": ..., "near_goal_name": ...}
    cone_deg=35.0,      # half-angle of allowed approach corridor for wrist/hand
):
    """
    One LM/DLS position step with null-space obstacle avoidance.

    Primary task: drive EE position -> target_pos_world.
    Secondary (in null space): repulse protected links from obstacle bodies,
    with stronger push for 'truss*' and added buffer for wrist/hand. Inside a
    narrow approach corridor near the goal, the wrist/hand keeps only sideways
    repulsion so it can advance toward the tomato.
    """
    p_cur, _ = _ee_pose(model, data, ee_ref)
    e = target_pos_world - p_cur                           # (3,)
    J_ee = _jacobian_7(model, data, ee_ref, arm_dof_idx, with_orientation=False)  # 3x7

    # LM/DLS task step
    H = J_ee.T @ J_ee + (damping * np.eye(7))
    dq_task = np.linalg.solve(H, J_ee.T @ e)               # (7,)

    # Null-space projector
    Jee_pinv = _damped_pinv(J_ee, lam=damping)             # (7x3)
    N = np.eye(7) - Jee_pinv @ J_ee                        # (7x7)

    # Fade avoidance near the goal
    dist_goal = np.linalg.norm(e)
    alpha_goal = np.clip((dist_goal - gate_r) / max(gate_band, 1e-9), 0.0, 1.0) if gate_band > 0 else 1.0

    # Corridor context
    goal_pos  = None
    a_dir     = None
    near_name = None
    if isinstance(goal_ctx, dict):
        goal_pos  = goal_ctx.get("goal_pos", None)
        a_dir     = goal_ctx.get("approach_dir", None)   # unit vector
        near_name = goal_ctx.get("near_goal_name", None)

    dq_avoid_sum = np.zeros(7)
    if obstacles:
        for obs_name in obstacles:
            c_obs, bid_obs = _body_pos(model, data, obs_name)
            R_obs = _approx_body_radius(model, bid_obs, default=0.03)  # safer "max" radius

            # Stronger push for truss bodies (tomatoes stick out)
            k_obs = k_avoid * (1.6 if ("truss" in obs_name) else 1.0)

            for link_name in (avoid_links or ()):
                p_link, _ = _body_pos(model, data, link_name)
                r = p_link - c_obs
                r_norm = np.linalg.norm(r)
                if r_norm < 1e-9:
                    continue
                dist = r_norm - R_obs  # signed dist to surface

                # Extra clearance/weight for wrist/hand
                is_wrist = link_name in ("link6", "hand", "wrist")
                d0_local = d0 + (0.03 if is_wrist else 0.0)
                k_local  = k_obs * (1.5 if is_wrist else 1.0)

                if dist < d0_local:
                    n = r / r_norm
                    v_avoid = alpha_goal * k_local * (d0_local - dist) * n

                    # Approach corridor shaping for wrist/hand
                    if is_wrist and (goal_pos is not None) and (a_dir is not None):
                        to_goal = goal_pos - p_link
                        tg_norm = np.linalg.norm(to_goal)
                        if tg_norm > 1e-9:
                            to_goal_u = to_goal / tg_norm
                            cosang = float(np.clip(np.dot(to_goal_u, a_dir), -1.0, 1.0))
                            ang_deg = np.degrees(np.arccos(cosang))
                            if ang_deg <= float(cone_deg):
                                v_avoid = v_avoid - (np.dot(v_avoid, a_dir) * a_dir)  # keep only sideways push
                                v_avoid *= 0.8  # damp

                    # Map spatial push to joints via link Jacobian
                    J_link  = _jacobian_body_point(model, data, link_name, arm_dof_idx)  # 3x7
                    dq_link = _damped_pinv(J_link, lam=AVOID_LAM) @ v_avoid              # (7,)
                    dq_avoid_sum += dq_link

    # Posture prior
    if posture_weight and posture_weight > 0.0:
        q = data.qpos[arm_qpos_addr].copy()
        if q_nom is None:
            q_nom = START_JOINTS
        dq_post = posture_weight * (q_nom - q)
    else:
        dq_post = 0.0

    # Compose and cap step
    dq = dq_task + N @ (dq_avoid_sum + dq_post)
    nrm = np.linalg.norm(dq)
    if nrm > max_step:
        dq *= (max_step / nrm)

    q = data.qpos[arm_qpos_addr].copy()
    q_new = q + step_size * dq

    if respect_limits:
        for i, dof in enumerate(arm_dof_idx):
            jid = int(model.dof_jntid[dof])
            if int(model.jnt_limited[jid]) == 1:
                lo, hi = model.jnt_range[jid]
                if hi > lo + 1e-6:
                    q_new[i] = np.clip(q_new[i], lo, hi)

    data.qpos[arm_qpos_addr] = q_new
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(e))

def solve_ik_LM_position(
    model, data, target_pos_world, arm_dof_idx, arm_qpos_addr, ee_ref,
    max_iters=IK_MAX_ITERS, pos_tol=IK_POS_TOL, step_size=LM_STEP, damping=LM_DAMP,
    debug=DEBUG_IK, use_avoidance=False, obstacles=None, goal_ctx=None
):
    if debug:
        Jp = _jacobian_7(model, data, ee_ref, arm_dof_idx)
        print("[J norms]", np.array2string(np.linalg.norm(Jp, axis=0), precision=3))

    last_err, stagnant = None, 0
    for it in range(max_iters):
        q_before = data.qpos[arm_qpos_addr].copy()

        if use_avoidance and obstacles:
            err = lm_ik_step_position_with_avoidance(
                model, data, target_pos_world, arm_dof_idx, arm_qpos_addr, ee_ref,
                obstacles=obstacles, step_size=step_size, damping=damping, goal_ctx=goal_ctx
            )
        else:
            err = lm_ik_step_position_only(
                model, data, target_pos_world, arm_dof_idx, arm_qpos_addr, ee_ref,
                step_size=step_size, damping=damping
            )

        dq_norm = np.linalg.norm(data.qpos[arm_qpos_addr] - q_before)

        if last_err is not None and (abs(err - last_err) < 1e-6 or dq_norm < 1e-8):
            stagnant += 1
        else:
            stagnant = 0
        last_err = err

        if debug and (it % 20 == 0 or err < pos_tol):
            print(f"[IK] it={it:3d}  err={err:.5f}  Δq={dq_norm:.2e}")

        if err < pos_tol or stagnant > 20:
            break
    return data.qpos[arm_qpos_addr].copy()

# -------------------------------- Planner -------------------------------------

def plan_cartesian_to_joint_traj(
    model, data, q_start, goal_pos, arm_dof_idx, arm_qpos_addr, ee_ref,
    n_cart=N_CART_WAYPOINTS,
    use_avoidance=USE_NULLSPACE_AVOID,
    obstacles=None,
    pregrasp_offset=PREGRASP_OFFSET,
    retreat_offset=RETREAT_OFFSET,
    interp_step=0.02,   # radians per interp step (~1.1°)
    goal_body_name=None,
):
    """
    Plan a joint trajectory through: pre-grasp (lateral) -> goal -> retreat.
    Returns (traj[T,7], (end_pre_idx, end_goal_idx, end_ret_idx)).
    """
    q_backup = data.qpos.copy()

    # Lateral approach normal from plant center to goal (in xy plane)
    n_app = _approach_normal_lateral(model, data, goal_pos)  # unit vector in xy
    orth_u = np.array([-n_app[1], n_app[0], 0.0], dtype=float); 
    orth_u /= (np.linalg.norm(orth_u) + 1e-9)
    pre_pos = goal_pos + float(pregrasp_offset) * n_app if pregrasp_offset > 0 else goal_pos
    ret_pos = goal_pos - float(retreat_offset)  * n_app if retreat_offset  > 0 else goal_pos

    # Sequence of cartesian sub-goals (dedupe if equal)
    cart_goals = []
    for p in (pre_pos, goal_pos, ret_pos):
        if not cart_goals or np.linalg.norm(p - cart_goals[-1]) > 1e-9:
            cart_goals.append(p)

    traj_all = []
    segment_end_indices = []
    try:
        # Seed state
        qs = q_start.copy()
        data.qpos[arm_qpos_addr] = qs
        mujoco.mj_forward(model, data)

        for gpos in cart_goals:
            goal_ctx = {
                "goal_pos": gpos,
                "approach_dir": n_app,
                "near_goal_name": goal_body_name,
            }
            # Solve IK to hit the sub-goal
            q_hit = solve_ik_LM_position(
                model, data, target_pos_world=gpos,
                arm_dof_idx=arm_dof_idx, arm_qpos_addr=arm_qpos_addr, ee_ref=ee_ref,
                max_iters=IK_MAX_ITERS, pos_tol=IK_POS_TOL,
                step_size=LM_STEP, damping=LM_DAMP,
                debug=DEBUG_IK, use_avoidance=use_avoidance, obstacles=obstacles,
                goal_ctx=goal_ctx,
            )

            # Interpolate qs -> q_hit into small steps for a smooth joint path
            d = np.linalg.norm(q_hit - qs)
            steps = max(5, int(d / float(interp_step)))
            if steps == 0:
                traj_all.append(q_hit.copy())
            else:
                for i in range(1, steps + 1):
                    traj_all.append(qs + (i / steps) * (q_hit - qs))

            # Next segment
            qs = q_hit.copy()
            data.qpos[arm_qpos_addr] = qs
            mujoco.mj_forward(model, data)
            segment_end_indices.append(len(traj_all) - 1)

        # Resample to ~n_cart waypoints (preserving phase indices)
        if n_cart is not None and len(traj_all) > n_cart:
            old_len = len(traj_all)
            idx = np.linspace(0, old_len - 1, num=n_cart, dtype=int)
            traj_all = [traj_all[i] for i in idx]
            segment_end_indices = [int(round(i * (n_cart - 1) / (old_len - 1)))
                                   for i in segment_end_indices]

        # Ensure we have 3 markers
        while len(segment_end_indices) < 3:
            segment_end_indices.append(len(traj_all) - 1)

        return np.asarray(traj_all, dtype=np.float32), tuple(segment_end_indices[:3])

    finally:
        # Restore state
        data.qpos[:] = q_backup
        mujoco.mj_forward(model, data)

# ------------------------------ Env (PD control) ------------------------------

class PandaOracleEnv(dm_env.Environment):
    """dm_env around MuJoCo Panda using joint waypoints + **PD torque control**."""
    def __init__(self, model, data, arm_act_ids, arm_qpos_addr, ee_ref,
                 language_instruction, substeps=40, gripper_idx=-1,
                 arm_dof_idx=None, kp=PD_KP, kd=PD_KD,
                 control_mode: str = "waypoints",
                 action_scale: float = 0.05):
        # Core sim references
        self.model = model
        self.data = data

        # Robot indexing / control
        self.arm_act_ids = arm_act_ids
        self.arm_qpos_addr = arm_qpos_addr
        self.arm_dof_idx = np.asarray(arm_dof_idx, dtype=int) if arm_dof_idx is not None else None
        self.ee_ref = ee_ref
        self.lang = language_instruction
        self.substeps = int(substeps)
        self.gripper_idx = gripper_idx
        self.kp = float(kp)
        self.kd = float(kd) if kd is not None else float(2.0 * np.sqrt(kp))
        self.control_mode = str(control_mode)
        self.action_scale = float(action_scale)

        # Runtime episode state
        self._capture_images = True
        self._waypoints = None
        self._T = 0
        self._t = 0
        self._last_gripper_cmd = 0.0
        self._goal_image_primary = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        self._goal_image_wrist   = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        # --- Camera resolution + renderers ---
        mujoco.mj_forward(self.model, self.data)

        self._r_primary = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self._r_wrist   = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

        cam_names: list[str] = []
        for cid in range(self.model.ncam):
            try:
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
            except Exception:
                nm = None
            if nm and nm not in cam_names:
                cam_names.append(nm)

        def _pick_camera(renderer: mujoco.Renderer, candidates: list[str]) -> tuple[str, int, float]:
            """Return (name, id, preview_mean) choosing the first camera with a non-dark frame.
            Falls back to the first valid camera even if dark."""
            first_valid: tuple[str, int, float] | None = None
            for name in candidates:
                if not name:
                    continue
                cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                if cid < 0:
                    continue
                try:
                    renderer.update_scene(self.data, camera=name)
                    frame = renderer.render()
                except Exception:
                    continue
                if frame.size == 0:
                    continue
                mean_val = float(frame.mean())
                if first_valid is None:
                    first_valid = (name, cid, mean_val)
                if mean_val > 1e-6:
                    return name, cid, mean_val
            if first_valid is not None:
                return first_valid
            return "unknown", -1, 0.0

        def _uniq(seq):
            out = []
            for item in seq:
                if item and item not in out:
                    out.append(item)
            return out

        primary_candidates = _uniq([
            PRIMARY_CAM_NAME,
            "front_cam" if PRIMARY_CAM_NAME != "front_cam" else None,
            "third_person_cam",
            *cam_names,
        ])
        self.cam_primary_name, self.cam_primary_id, primary_mean = _pick_camera(self._r_primary, primary_candidates)
        self.cam_primary = self.cam_primary_name

        wrist_candidates = _uniq([
            WRIST_CAM_NAME,
            "gripper_cam" if WRIST_CAM_NAME != "gripper_cam" else None,
            "wrist_cam" if WRIST_CAM_NAME != "wrist_cam" else None,
            self.cam_primary_name,
            "third_person_cam",
            *cam_names,
        ])
        self.cam_wrist_name, self.cam_wrist_id, wrist_mean = _pick_camera(self._r_wrist, wrist_candidates)
        self.cam_wrist = self.cam_wrist_name

        print(
            f"[camera] primary='{self.cam_primary_name}' (id={self.cam_primary_id}) mean={primary_mean:.3f}; "
            f"wrist='{self.cam_wrist_name}' (id={self.cam_wrist_id}) mean={wrist_mean:.3f}"
        )


    def set_language_instruction(self, text: str):
        self.lang = str(text)

    def set_capture_images(self, enabled: bool):
        self._capture_images = bool(enabled)

    def set_goal_images(self, primary, wrist=None):
        if primary is None:
            self._goal_image_primary = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            arr = np.asarray(primary, dtype=np.uint8)
            if arr.shape != (IMG_H, IMG_W, 3):
                raise ValueError(
                    f"goal image must have shape {(IMG_H, IMG_W, 3)}, got {arr.shape}"
                )
            self._goal_image_primary = arr.copy()

        if wrist is None:
            self._goal_image_wrist = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            warr = np.asarray(wrist, dtype=np.uint8)
            if warr.shape != (IMG_H, IMG_W, 3):
                raise ValueError(
                    f"goal wrist image must have shape {(IMG_H, IMG_W, 3)}, got {warr.shape}"
                )
            self._goal_image_wrist = warr.copy()

    def _render_images(self):
        if not getattr(self, "_capture_images", True):
            z = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            return z, z

        # Primary (front_cam fallback to third_person_cam was already resolved in __init__)
        self._r_primary.update_scene(self.data, camera=self.cam_primary_name)
        img_primary = self._r_primary.render().copy()

        # Wrist / gripper camera: only render if we actually resolved one
        if self.cam_wrist_id >= 0:
            self._r_wrist.update_scene(self.data, camera=self.cam_wrist_name)
            img_wrist = self._r_wrist.render().copy()
        else:
            img_wrist = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        return img_primary, img_wrist

   
    def set_waypoints(self, waypoints: np.ndarray):
        self._waypoints = waypoints.astype(np.float32)
        self._T = len(waypoints)
        self._t = 0


    def reset(self, waypoints=None):
        if waypoints is not None:
            self.set_waypoints(waypoints)
        elif self.control_mode == "waypoints" and self._waypoints is None:
            raise ValueError("reset(...): waypoints required in waypoints mode")
        else:
            # policy mode can reuse previously set waypoints (if any)
            self._T = len(self._waypoints) if self._waypoints is not None else 0

        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        # start pose: waypoint[0] if present, else default START_JOINTS
        if self._waypoints is not None and self._T > 0:
            self.data.qpos[self.arm_qpos_addr] = self._waypoints[0, :7]
        else:
            self.data.qpos[self.arm_qpos_addr] = START_JOINTS
        mujoco.mj_forward(self.model, self.data)
        self._t = 0
        if 0 <= self.gripper_idx < self.model.nu:
            if self._waypoints is not None and self._T > 0:
                self._last_gripper_cmd = float(self._waypoints[0, 7])
            else:
                self._last_gripper_cmd = float(self.data.ctrl[self.gripper_idx])
            self.data.ctrl[self.gripper_idx] = self._last_gripper_cmd
        else:
            self._last_gripper_cmd = 0.0

        img_primary, img_wrist = self._render_images()
        return TimeStep(
            dm_env.StepType.FIRST,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            observation={
                "proprio": self.data.qpos.astype(np.float32).copy(),
                "language_instruction": self.lang,
                "image_primary": img_primary,
                "image_wrist": img_wrist,
                "goal_image_primary": self._goal_image_primary.copy(),
                "goal_image_wrist": self._goal_image_wrist.copy(),
            },
        )

    def _clamp_to_limits(self, q_target):
        q_clamped = q_target.copy()
        for i, dof in enumerate(self.arm_dof_idx):
            jid = int(self.model.dof_jntid[dof])
            if int(self.model.jnt_limited[jid]) == 1:
                lo, hi = self.model.jnt_range[jid]
                if hi > lo + 1e-6:
                    q_clamped[i] = np.clip(q_clamped[i], lo, hi)
        return q_clamped

    # --- modify step() to consume policy actions when control_mode == "policy" ---
    def step(self, action):
        if self.control_mode == "waypoints":
            if self._waypoints is None or self._T == 0:
                raise RuntimeError("step() in waypoints mode before reset(waypoints=...)")
            if self.arm_dof_idx is None:
                raise RuntimeError("arm_dof_idx must be provided for PD control")

            idx = min(self._t, self._T - 1)
            q_target = self._waypoints[idx, :7]
            g_cmd    = self._waypoints[idx,  7]
            self._last_gripper_cmd = float(g_cmd)
        else:
            # POLICY MODE: use incoming action
            if self.arm_dof_idx is None:
                raise RuntimeError("arm_dof_idx must be provided for PD control")
            a = np.asarray(action, dtype=np.float32)
            # Octo often outputs (T, 8) over a future horizon; take first slice.
            if a.ndim == 2:
                a = a[0]
            a = a.reshape(-1)
            if a.shape[0] < 7:
                raise ValueError(f"Expected ≥7 DoF action, got shape {a.shape}")
            if a.shape[0] > 7:
                print(
                    f"[PandaOracleEnv] action has extra dims ({a.shape[0]}); "
                    "ignoring gripper component for now."
                )
                a = a[:7]

            q  = self.data.qpos[self.arm_qpos_addr].copy()
            # interpret policy actions as direct joint deltas
            q_target = q + a
            q_target = self._clamp_to_limits(q_target)
            if 0 <= self.gripper_idx < self.model.nu:
                if self._waypoints is not None and self._T > 0:
                    idx_next = min(self._t + 1, self._T - 1)
                    g_cmd = float(self._waypoints[idx_next, 7])
                else:
                    g_cmd = float(self._last_gripper_cmd)
            else:
                g_cmd = 0.0
            self._last_gripper_cmd = g_cmd

        # Common PD inner loop
        for _ in range(self.substeps):
            q  = self.data.qpos[self.arm_qpos_addr].copy()
            qd = self.data.qvel[self.arm_dof_idx].copy()
            e  = q_target - q
            u  = self.kp * e - self.kd * qd
            self.data.ctrl[self.arm_act_ids] = u
            if 0 <= self.gripper_idx < self.model.nu:
                self.data.ctrl[self.gripper_idx] = g_cmd
            mujoco.mj_step(self.model, self.data)

        self._t += 1
        last = (self._T > 0) and (self._t >= self._T)
        img_primary, img_wrist = self._render_images()
        return TimeStep(
            dm_env.StepType.LAST if last else dm_env.StepType.MID,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            observation={
                "proprio": self.data.qpos.astype(np.float32).copy(),
                "language_instruction": self.lang,
                "image_primary": img_primary,
                "image_wrist": img_wrist,
                "goal_image_primary": self._goal_image_primary.copy(),
                "goal_image_wrist": self._goal_image_wrist.copy(),
            },
        )

    def observation_spec(self):
        return {
            "proprio": specs.Array(shape=(self.model.nq,), dtype=np.float32, name="proprio"),
            "language_instruction": specs.Array(shape=(), dtype=object, name="language_instruction"),
            "image_primary": specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_primary"),
            "image_wrist":   specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_wrist"),
            "goal_image_primary": specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="goal_image_primary"),
            "goal_image_wrist":   specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="goal_image_wrist"),
        }

    def action_spec(self):
        # OLD: return specs.Array(shape=(7,), dtype=np.float32, name="action")
        return specs.Array(shape=(EE_ACTION_DIM,), dtype=np.float32, name="action")


# ------------------------------ Rollout & Logging -----------------------------

def _infer_goal_frame_idx(waypoints: np.ndarray) -> int:
    if waypoints is None or len(waypoints) == 0:
        return 0
    grip = waypoints[:, 7]
    closed = np.where(grip <= 0.05)[0]
    if closed.size == 0:
        return len(waypoints) - 1
    return int(closed[0])


def _render_goal_images_at_idx(base_env, model, data, arm_qpos_addr, waypoints, goal_idx):
    if waypoints is None or len(waypoints) == 0:
        return None, None
    goal_idx = int(np.clip(goal_idx, 0, len(waypoints) - 1))

    q_backup = data.qpos.copy()
    try:
        data.qpos[arm_qpos_addr] = waypoints[goal_idx, :7]
        mujoco.mj_forward(model, data)

        # PRIMARY
        base_env._r_primary.update_scene(base_env.data, camera=base_env.cam_primary)
        primary = base_env._r_primary.render().copy()

        # WRIST
        if base_env.cam_wrist_id >= 0:
            base_env._r_wrist.update_scene(base_env.data, camera=base_env.cam_wrist)
            wrist = base_env._r_wrist.render().copy()
        else:
            wrist = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    finally:
        data.qpos[:] = q_backup
        mujoco.mj_forward(model, data)

    return primary, wrist

def _compute_action_label(prev_row: np.ndarray, next_row: np.ndarray, scale: float) -> np.ndarray:
    """
    Convert consecutive waypoint rows into a dataset-scaled joint delta.
    """
    if scale == 0.0:
        raise ValueError("DATASET_ACTION_SCALE must be non-zero")
    prev_q = np.asarray(prev_row[:7], dtype=np.float32)
    next_q = np.asarray(next_row[:7], dtype=np.float32)
    delta = next_q - prev_q
    if not np.isclose(scale, 1.0, atol=1e-9):
        delta /= float(scale)
    return delta.astype(np.float32, copy=False)


def _dedupe_waypoints_for_logging(waypoints: np.ndarray) -> np.ndarray:
    """Remove near-duplicate joint targets to avoid long runs of zero actions."""

    if waypoints.size == 0:
        return waypoints

    keep = [0]
    last_q = waypoints[0, :7]
    last_g = waypoints[0, 7]
    for idx in range(1, len(waypoints)):
        q = waypoints[idx, :7]
        g = waypoints[idx, 7]
        if (np.linalg.norm(q - last_q) > DATASET_MIN_STEP_NORM) or (abs(g - last_g) > 1e-4):
            keep.append(idx)
            last_q = q
            last_g = g

    if keep[-1] != len(waypoints) - 1:
        keep.append(len(waypoints) - 1)

    trimmed = waypoints[np.array(keep, dtype=int)]

    if DATASET_FINAL_HOLD_STEPS > 0:
        hold = np.repeat(trimmed[-1][None, :], DATASET_FINAL_HOLD_STEPS, axis=0)
        trimmed = np.concatenate([trimmed, hold], axis=0)

    return trimmed.astype(np.float32, copy=False)


def _downsample_waypoints(traj: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep every `factor`-th row of `traj`, ensuring the final point remains."""

    if factor <= 1 or traj.size == 0:
        idx = np.arange(len(traj), dtype=int)
        return traj, idx

    indices = list(range(0, len(traj), factor))
    if indices[-1] != len(traj) - 1:
        indices.append(len(traj) - 1)
    idx = np.asarray(indices, dtype=int)
    return traj[idx], idx


def _limit_joint_step(waypoints: np.ndarray, max_step: float, goal_idx: int | None = None) -> tuple[np.ndarray, int]:
    """
    Insert interpolated waypoints so that each consecutive joint delta is bounded by `max_step`.
    Returns (dense_waypoints, adjusted_goal_idx).
    """
    if waypoints.size == 0 or not np.isfinite(max_step) or max_step <= 0.0:
        # Nothing to do; clamp goal_idx into range if provided.
        if goal_idx is None:
            return waypoints.astype(np.float32, copy=False), 0
        goal_idx = int(np.clip(goal_idx, 0, max(len(waypoints) - 1, 0)))
        return waypoints.astype(np.float32, copy=False), goal_idx

    dense: list[np.ndarray] = [np.asarray(waypoints[0], dtype=np.float32).copy()]
    index_map = [0]
    for src_idx in range(1, len(waypoints)):
        prev = dense[-1]
        target = np.asarray(waypoints[src_idx], dtype=np.float32)
        diff = target[:7] - prev[:7]
        max_abs = float(np.max(np.abs(diff)))
        segments = max(1, int(np.ceil(max_abs / float(max_step))))
        if segments > 1:
            for seg in range(1, segments):
                alpha = seg / segments
                interp = np.empty_like(prev)
                interp[:7] = prev[:7] + diff * alpha
                interp[7] = prev[7] + (target[7] - prev[7]) * alpha
                dense.append(interp.astype(np.float32, copy=False))
        dense.append(target.copy())
        index_map.append(len(dense) - 1)

    dense_arr = np.stack(dense, axis=0).astype(np.float32, copy=False)

    if goal_idx is None:
        adjusted_goal = 0
    else:
        goal_idx = int(np.clip(goal_idx, 0, len(index_map) - 1))
        adjusted_goal = index_map[goal_idx]

    return dense_arr, adjusted_goal

def _executed_delta_from_proprio(data, arm_qpos_addr):
    """Return the most recent executed joint delta (q_now - q_prev) for the arm."""
    # This helper assumes you snapshot q_prev outside, call a PD step, then read q_now here.
    # It only computes the diff; you manage the snapshots in the loop below.
    raise NotImplementedError  # (we'll compute deltas inline and not use this)


def run_oracle_once(
    env, base_env, model, data, arm_dof_idx, arm_qpos_addr, ee_ref,
    goal_pos, obstacles=None, goal_body_name=None,
    waypoints=None, dry_run=False, goal_frame_idx=None,
    action_delta_stats: ActionDeltaStatistics | None = None,
):
    """
    Plans and executes an oracle trajectory toward goal_pos.

    Builds a realistic waypoint sequence with:
      - Time-parameterized motion (EE limits)
      - Human-like pauses (pre-grasp, close, post-grasp/place, final)
      - Gripper ramp & hold
      - Optional micro-centering + wrist roll during grasp

    Returns:
        (err_final: float, waypoints: np.ndarray [N, 8], goal_frame_idx: int)
    """

    # ---- Reset state ----
    data.qpos[arm_qpos_addr] = START_JOINTS

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    q_start = data.qpos[arm_qpos_addr].copy()

    # ---- Tunables ----
    GOAL_SETTLE_STEPS    = 10
    GRASP_START_DIST      = 0.006
    MIN_CLOSE_IDX_MARGIN  = 4
    GRIP_OPEN, GRIP_CLOSE = 1.0, 0.0
    GRIPPER_RAMP_STEPS    = 25

    # ----------------------------------------------------------------------
    # 1) PLAN + BUILD WAYPOINTS (if not provided)
    # ----------------------------------------------------------------------
    if waypoints is None:
        # 1. Cartesian → Joint-space path
        traj_q, phase_idx = plan_cartesian_to_joint_traj(
            model, data, q_start, goal_pos,
            arm_dof_idx, arm_qpos_addr, ee_ref,
            n_cart=N_CART_WAYPOINTS,
            use_avoidance=USE_NULLSPACE_AVOID,
            obstacles=obstacles,
            goal_body_name=goal_body_name,
        )
        end_pre, end_goal, end_ret = phase_idx

        # 2. Time-parameterize by EE velocity/accel limits
        rate_hz = 1.0 / float(globals().get("TARGET_ACTION_DT", 0.10))
        traj_q_tp = _time_parameterize_by_ee_limits(
            model, data, arm_qpos_addr, ee_ref, traj_q,
            rate_hz=rate_hz,
            vmax_trans=V_MAX_TRANS, amax_trans=A_MAX_TRANS,
            vmax_rot=V_MAX_ROT,     amax_rot=A_MAX_ROT,
        )
        # Defensive: keep trajectory as pure 7-DoF joints here
        if traj_q_tp.shape[1] > 7:
            traj_q_tp = traj_q_tp[:, :7]

        # Rescale phase indices to new length
        scale_fac = (len(traj_q_tp) - 1) / max(1, (len(traj_q) - 1))
        end_pre, end_goal, end_ret = [int(round(x * scale_fac)) for x in (end_pre, end_goal, end_ret)]
        traj_q = traj_q_tp

        # Optional repeat for smoother motion
        rep = int(WAYPOINT_REPEAT)
        traj_q_rep = np.repeat(traj_q, repeats=rep, axis=0)
        end_pre  *= rep; end_goal *= rep; end_ret *= rep

        # 3. Convert dwell times (seconds → steps)
        sim_dt = float(model.opt.timestep) * float(getattr(base_env, "substeps", 40))
        to_steps = lambda sec: int(np.ceil(float(sec) / max(sim_dt, 1e-9)))

        dwell_pre_steps   = to_steps(PREGRASP_DWELL_SEC)
        dwell_close_steps = to_steps(GRIPPER_CLOSE_SEC)
        dwell_post_grasp  = to_steps(POST_GRASP_SETTLE_SEC)
        dwell_post_place  = to_steps(POST_PLACE_SETTLE_SEC)
        dwell_final_hold  = to_steps(FINAL_HOLD_SEC)

        print(f"[DWELL] pre={dwell_pre_steps} close={dwell_close_steps} "
              f"post_grasp={dwell_post_grasp} post_place={dwell_post_place} "
              f"final={dwell_final_hold} (dt={sim_dt:.3f}s)")

        # 4. Insert pre-grasp & post-grasp pauses
        dwell_pre  = np.repeat(traj_q_rep[end_pre:end_pre+1], repeats=dwell_pre_steps, axis=0)
        dwell_goal = np.repeat(traj_q_rep[end_goal:end_goal+1], repeats=dwell_post_grasp, axis=0)

        traj_q_full = np.concatenate([
            traj_q_rep[:end_pre+1],
            dwell_pre,
            traj_q_rep[end_pre+1:end_goal+1],
            dwell_goal,
            traj_q_rep[end_goal+1:end_ret+1],
        ], axis=0)

        # ---- Wrist roll ramp during the GOAL dwell (restore old behavior) ----
        # Figure out where the dwell_goal sits inside traj_q_full
        len_pre      = len(traj_q_rep[:end_pre+1])
        len_dpre     = len(dwell_pre)
        len_to_goal  = len(traj_q_rep[end_pre+1:end_goal+1])
        idx_dwell_goal_start = len_pre + len_dpre + len_to_goal
        idx_dwell_goal_end   = idx_dwell_goal_start + len(dwell_goal)

        # Ramp the 7th joint (index 6) over the tail of that dwell
        j_roll = 6
        i1 = int(idx_dwell_goal_end)
        i0 = int(max(idx_dwell_goal_start, i1 - int(ROLL_RAMP_STEPS)))
        if i1 > i0:
            roll0 = float(traj_q_full[i0 - 1, j_roll] if i0 > 0 else traj_q_full[0, j_roll])
            roll1 = roll0 + np.deg2rad(float(HAND_ROLL_AT_GRASP_DEG))
            steps = max(1, i1 - i0)
            for t, i in enumerate(range(i0, i1)):
                a = float(t + 1) / float(steps)
                traj_q_full[i, j_roll] = (1.0 - a) * roll0 + a * roll1
            print(f"[WRIST] roll ramp: {HAND_ROLL_AT_GRASP_DEG:.1f}° over {steps} steps")
        else:
            print("[WRIST] (skipped) dwell too short for roll ramp")
        # ---- End wrist roll ramp ----

        # 5. Distance-based timing for gripper closure
        _bak = data.qpos.copy()
        dists = []
        for q in traj_q_full:
            # joints only
            data.qpos[arm_qpos_addr] = q[:7]
            mujoco.mj_forward(model, data)
            ee, _ = _ee_pose(model, data, ee_ref)
            dists.append(float(np.linalg.norm(goal_pos - ee)))
        data.qpos[:] = _bak; mujoco.mj_forward(model, data)

        idx_close = next((i for i, d in enumerate(dists) if d <= GRASP_START_DIST),
                         len(dists) - 1)
        close_start = max(end_goal + MIN_CLOSE_IDX_MARGIN, idx_close)
        close_start = min(close_start, len(traj_q_full) - 1)

        # Insert dwell for gripper close
        dwell_close = np.repeat(traj_q_full[close_start:close_start+1],
                                repeats=dwell_close_steps, axis=0)
        traj_q_full = np.concatenate([
            traj_q_full[:close_start],
            dwell_close,
            traj_q_full[close_start:],
        ], axis=0)

        # 6. Post-place + Final holds
        traj_q_hold = np.concatenate([
            traj_q_full,
            np.repeat(traj_q_full[-1][None, :], repeats=dwell_post_place, axis=0),
            np.repeat(traj_q_full[-1][None, :], repeats=dwell_final_hold, axis=0),
        ], axis=0)

        # 7. Gripper ramp (after final length known)
        N_hold = len(traj_q_hold)
        close_start = int(np.clip(close_start, 0, N_hold - 1))
        end_ramp = int(min(close_start + GRIPPER_RAMP_STEPS, N_hold))
        grip = np.full(N_hold, GRIP_OPEN, np.float32)
        grip[close_start:end_ramp] = np.linspace(GRIP_OPEN, GRIP_CLOSE,
                                                 max(1, end_ramp - close_start), dtype=np.float32)
        grip[end_ramp:] = GRIP_CLOSE

        # 8. Downsample + Combine
        traj_q_proc, ds_indices = _downsample_waypoints(
            np.asarray(traj_q_hold, dtype=np.float32),
            int(max(1, WAYPOINT_DOWNSAMPLE))
        )
        ds_indices = np.clip(ds_indices, 0, len(grip) - 1)
        grip_proc = grip[ds_indices][:, None]
        waypoints = np.concatenate([traj_q_proc[:, :7], grip_proc], axis=1)
        waypoints = _dedupe_waypoints_for_logging(waypoints)

        # frame to evaluate success at (end of ramp or last)
        goal_frame_idx = int(max(0, min(end_ramp - 1, len(waypoints) - 1)))

    elif goal_frame_idx is None:
        goal_frame_idx = _infer_goal_frame_idx(waypoints)

    # ----------------------------------------------------------------------
    # 2) POST-PROCESS (safety clamping + index sanity)
    # ----------------------------------------------------------------------
    waypoints, goal_frame_idx = _limit_joint_step(waypoints, DATASET_MAX_STEP_RAD, goal_frame_idx)
    if waypoints.size == 0:
        goal_frame_idx = 0
    else:
        inferred = _infer_goal_frame_idx(waypoints)
        goal_frame_idx = int(np.clip(inferred, 0, len(waypoints) - 1))

    # ----------------------------------------------------------------------
    # 3) EXECUTION (dry-run vs logged)
    # ----------------------------------------------------------------------
    N = len(waypoints)
    t_idx = 0

    prev_control_mode = getattr(base_env, "control_mode", "waypoints")
    base_env.control_mode = "policy"

    def _step_once(stepper):
        """Advance one 10Hz step by commanding the delta from CURRENT q to the NEXT waypoint.
        This prevents drift from accumulating if PD tracking didn't fully reach the previous target."""
        nonlocal t_idx
        if N == 0:
            return True

        idx_cur  = min(t_idx,     N - 1)
        idx_next = min(t_idx + 1, N - 1)

        # NEXT waypoint we want to reach
        next_row = waypoints[idx_next]

        # Current joints from sim (what we actually reached)
        q_now = base_env.data.qpos[arm_qpos_addr].copy()

        # ACTION = joint delta from CURRENT to NEXT (joint_delta semantics)
        # Keep DATASET_ACTION_SCALE = 1.0 so this is "true radians".
        a_label = (next_row[:7] - q_now[:7]).astype(np.float32)

        # Optional: cap per-step delta (aligned with DATASET_MAX_STEP_RAD)
        max_step = float(DATASET_MAX_STEP_RAD)
        if max_step > 0.0:
            a_label = np.clip(a_label, -max_step, max_step)

        ts = stepper(action=a_label)
        t_idx += 1
        return ts.last()

    # --- DRY RUN (QC, no logging) ---
    if dry_run:
        prev_cap = getattr(base_env, "_capture_images", True)
        if hasattr(base_env, "set_capture_images"):
            base_env.set_capture_images(LIVE_RENDER_QC)

        base_env.set_waypoints(waypoints)
        base_env.reset(waypoints=waypoints)  # seed renderer if any

        try:
            if LIVE_RENDER_QC:
                # simple ~10 Hz live preview of the planned waypoints
                period = float(max(TARGET_ACTION_DT, 1e-3))
                t0 = time.time()
                # step through the whole plan once
                while not _step_once(base_env.step):
                    t0 += period
                    time.sleep(max(0.0, t0 - time.time()))
            else:
                # no rendering: set final joint pose so err is meaningful
                data.qpos[arm_qpos_addr] = waypoints[-1, :7]
                mujoco.mj_forward(model, data)

            # Evaluate final pose error and return
            ee_pos, _ = _ee_pose(model, data, ee_ref)
            err_final = float(np.linalg.norm(goal_pos - ee_pos))
            print(f"[QC] final |EE - goal| = {err_final:.4f} m")
            return err_final, waypoints, goal_frame_idx
        finally:
            if hasattr(base_env, "set_capture_images"):
                base_env.set_capture_images(prev_cap)

    # --- LOGGED RUN (executed-delta logging) ---
    if env is None:
        raise RuntimeError("Logged run requested but `env` is None. Use dry_run=True for QC or pass an EnvLogger.")


    # EXECUTE with waypoints (PD ignores the action argument),
    # LOG the *executed* EE delta from proprio (normalized).
    base_env.set_waypoints(waypoints)
    base_env.control_mode = "waypoints"
    base_env.reset(waypoints=waypoints)

    # 1) PRIME STEP (no logging yet)
    prev_pos, prev_yaw = _ee_pos_yaw(model, data, ee_ref)
    _ = base_env.step(action=np.zeros(EE_ACTION_DIM, dtype=np.float32))  # ignored in 'waypoints' mode
    cur_pos, cur_yaw = _ee_pos_yaw(model, data, ee_ref)
    a_exec_prev = _ee_delta(prev_pos, prev_yaw, cur_pos, cur_yaw)

    # Start EnvLogger from this state
    env.reset()

    # Optional live view
    if LIVE_RENDER:
        period = 1.0 / float(LIVE_FPS)
        with viewer.launch_passive(model, data) as v:
            last_t = time.perf_counter()
            done = False
            while v.is_running() and not done:
                # 2) Log raw EE stats and send normalized EE delta to the logger
                if action_delta_stats is not None:
                    action_delta_stats.observe(a_exec_prev)  # raw EE units
                a_log_prev = _normalize_ee_delta(a_exec_prev)  # normalized [-1, 1]
                done = env.step(a_log_prev).last()

                # 3) Measure executed EE delta for NEXT transition
                prev_pos, prev_yaw = cur_pos, cur_yaw
                cur_pos, cur_yaw = _ee_pos_yaw(model, data, ee_ref)
                a_exec_prev = _ee_delta(prev_pos, prev_yaw, cur_pos, cur_yaw)

                # pacing + viewer sync
                now = time.perf_counter()
                if now - last_t < period:
                    time.sleep(max(0.0, period - (now - last_t)))
                last_t = now
                v.sync()
    else:
        done = False
        while not done:
            if action_delta_stats is not None:
                action_delta_stats.observe(a_exec_prev)
            a_log_prev = _normalize_ee_delta(a_exec_prev)
            done = env.step(a_log_prev).last()

            prev_pos, prev_yaw = cur_pos, cur_yaw
            cur_pos, cur_yaw = _ee_pos_yaw(model, data, ee_ref)
            a_exec_prev = _ee_delta(prev_pos, prev_yaw, cur_pos, cur_yaw)

    # Done; report final error (unchanged)
    ee_pos, _ = _ee_pose(model, data, ee_ref)
    err_final = float(np.linalg.norm(goal_pos - ee_pos))
    print(f"[RUN] final |EE - goal| = {err_final:.4f} m")
    return err_final, waypoints, goal_frame_idx


# ----------------------------------- Main -------------------------------------

def _site_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0

def main():
    os.makedirs(TFDS_ROOT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    action_delta_stats = ActionDeltaStatistics(action_dim=EE_ACTION_DIM)

    episodes_done = 0
    plant_idx = 0

    while episodes_done < EPISODES_TOTAL:
        try:
            # --------- Build / reset plant ---------
            if USE_DYNAMIC_PLANT:
                model, data = _build_and_load_scene(MODEL_PATH)
                print(f"[SPLIT] (dynamic plant; split chosen per-episode)")
            else:
                model = mujoco.MjModel.from_xml_path(MODEL_PATH)
                data  = mujoco.MjData(model)
                mujoco.mj_forward(model, data)

            ee_ref = ("site", "tcp") if _site_exists(model, "tcp") else ("body", "hand")
            print(f"[EE_REF] using {ee_ref[0].upper()} '{ee_ref[1]}'")

            obstacles_all = _collect_plant_obstacles(model)
            print(f"[AVOID] using {len(obstacles_all)} plant bodies as obstacles")

            arm_act_ids, arm_qpos_addr = build_arm_mapping_from_model(model, prefer_position=True)
            arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)

            if AUTO_WIDEN_LIMITS:
                for i, dof in enumerate(arm_dof_idx):
                    jid = int(model.dof_jntid[dof])
                    model.jnt_limited[jid] = 1
                    lo, hi = TYPICAL_FRANKA_LIMITS[i]
                    model.jnt_range[jid][0] = lo
                    model.jnt_range[jid][1] = hi

            gripper_idx = find_gripper_actuator(model)

            # RLDS dataset config (proprio key!)
            ds_config = tfds.rlds.rlds_base.DatasetConfig(
                version=tfds.core.Version(DATASET_VERSION),
                name=DATASET_NAME,
                observation_info=tfds.features.FeaturesDict({
                    "proprio": tfds.features.Tensor(shape=(model.nq,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "image_primary": tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
                    "image_wrist":   tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
                    "goal_image_primary": tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
                    "goal_image_wrist":   tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
                }),
                # OLD: action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),
                action_info=tfds.features.Tensor(shape=(EE_ACTION_DIM,), dtype=np.float32),
                reward_info=tf.float32,
                discount_info=tf.float32,
            )

            # Choose substeps so each outer step advances ~0.1 s of simulated time.
            desired_dt   = float(TARGET_ACTION_DT)
            timestep     = float(model.opt.timestep)
            substeps_calc = max(1, int(round(desired_dt / max(timestep, 1e-6))))

            base_env = PandaOracleEnv(
                model, data, arm_act_ids, arm_qpos_addr, ee_ref,
                language_instruction="Pick the specified tomato by name.",
                substeps=substeps_calc,
                gripper_idx=gripper_idx,
                arm_dof_idx=arm_dof_idx, kp=PD_KP, kd=PD_KD,
                action_scale=1.0,           # <<< make actions = exact joint deltas
            )

            print(f"[RATE] timestep={timestep:.4g}s  substeps={substeps_calc}  → action_dt≈{timestep*substeps_calc:.3f}s (~{1.0/(timestep*substeps_calc):.1f} Hz)")


            # Precompute targets available on this plant
            top = _find_top_targets(model, data, k=EPISODES_PER_PLANT, s=0.66)
            for name, grasp_pos, approach_xy, truss_name in top:
                print(f"[Target(stem@0.66)] {name}  grasp={grasp_pos}  (Z={grasp_pos[2]:.3f})")

            if top:
                ordered = sorted(((name, float(np.linalg.norm(grasp[:2]))) for name, grasp, *_ in top), key=lambda t: t[1])
                mid_idx = max(1, len(ordered) // 2)
                front_targets = {name for name, _ in ordered[:mid_idx]}
                back_targets = {name for name, _ in ordered[mid_idx:]}
            else:
                front_targets, back_targets = set(), set()

            episodes_here = min(EPISODES_PER_PLANT, EPISODES_TOTAL - episodes_done)

            # --------- Episode loop ---------
            for epi in range(episodes_here):
                try:
                    # Pick a target for this episode
                    name, goal_pos, approach_xy, truss_name = top[epi % len(top)]
                    obstacles = obstacles_all

                    if name in front_targets or not back_targets:
                        language_text = "Pick the top tomato in front."
                    else:
                        language_text = "Pick the top tomato in the back."
                    base_env.set_language_instruction(language_text)

                    # Deterministic split per episode count
                    split_name = choose_split_for_episode(episodes_done)
                    print(f"[EP] {episodes_done} → split={split_name}  target={name}")

                    # Clear any stale goal image from prior episode
                    base_env.set_goal_images(None, None)

                    # ---- QC DRY RUN ----
                    err_final, waypoints, goal_frame_idx = run_oracle_once(
                        env=None, base_env=base_env, model=model, data=data,
                        arm_dof_idx=arm_dof_idx, arm_qpos_addr=arm_qpos_addr, ee_ref=ee_ref,
                        goal_pos=goal_pos, obstacles=obstacles, goal_body_name=truss_name,
                        waypoints=None, dry_run=True,
                    )
                    if err_final > MAX_FINAL_ERR:
                        print(f"[QC] ❌ Skip episode (err={err_final:.3f} > {MAX_FINAL_ERR})")
                        continue

                    goal_primary, goal_wrist = _render_goal_images_at_idx(
                        base_env=base_env,
                        model=model,
                        data=data,
                        arm_qpos_addr=arm_qpos_addr,
                        waypoints=waypoints,
                        goal_idx=goal_frame_idx,
                    )
                    base_env.set_goal_images(goal_primary, goal_wrist)

                    if goal_primary is not None and _ensure_goal_dir_exists():
                        stem_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
                        fname = f"ep{episodes_done:05d}_{split_name}_{stem_safe}.png"
                        goal_path = os.path.join(GOAL_IMAGE_OUTPUT_DIR, fname)
                        try:
                            _write_goal_image(goal_path, goal_primary)
                            if goal_wrist is not None:
                                wrist_name = f"ep{episodes_done:05d}_{split_name}_{stem_safe}_wrist.png"
                                wrist_path = os.path.join(GOAL_IMAGE_OUTPUT_DIR, wrist_name)
                                _write_goal_image(wrist_path, goal_wrist)
                        except Exception as exc:
                            print(f"[GOAL_IMG] failed to save goal images: {exc}")

                    # ---- LOG THE EPISODE ----
                    dataset_root = TFDS_ROOT_DIR
                    version_dir  = os.path.join(TFDS_ROOT_DIR, DATASET_NAME, DATASET_VERSION)
                    os.makedirs(dataset_root, exist_ok=True)
                    os.makedirs(version_dir,  exist_ok=True)

                    with envlogger.EnvLogger(
                        base_env,
                        backend=tfds_backend_writer.TFDSBackendWriter(
                            data_directory=dataset_root,                           # ROOT (robust)
                            split_name=split_name,                                 # "train"/"val"/"test"
                            max_episodes_per_file=8,
                            ds_config=ds_config,
                        ),
                        metadata={
                            "language_instruction": base_env.lang,
                            "action_type": "ee_delta_pos_yaw",
                            "action_scale": {
                                "trans": EE_TRANS_SCALE.tolist(),
                                "yaw": float(EE_YAW_SCALE),
                            },
                            "action_dt_sec": float(TARGET_ACTION_DT),
                        },

                    ) as env:
                        run_oracle_once(
                            env=env, base_env=base_env, model=model, data=data,
                            arm_dof_idx=arm_dof_idx, arm_qpos_addr=arm_qpos_addr, ee_ref=ee_ref,
                            goal_pos=goal_pos, obstacles=obstacles, goal_body_name=truss_name,
                            waypoints=waypoints, dry_run=False, goal_frame_idx=goal_frame_idx,
                            action_delta_stats=action_delta_stats,
                        )

                    episodes_done += 1
                    print(f"[PROGRESS] ✅ Episodes saved: {episodes_done}/{EPISODES_TOTAL}")

                    # Sweep any shards the writer dropped anywhere into the version dir, then repair splits
                    move_stray_shards_into_version_dir(TFDS_ROOT_DIR, ds_config)
                    if ENABLE_TFRECORD_HARVEST:
                        harvest_any_tfrecords(TFDS_ROOT_DIR, version_dir, DATASET_NAME, split_name)
                    # Repair shard naming / dataset_info.json after each write
                    _post_write_repair(version_dir, DATASET_NAME)
                    _relocate_dataset_metadata(TFDS_ROOT_DIR, version_dir, DATASET_VERSION)
                    _ensure_min_train_split(version_dir, DATASET_NAME)
                    repair_tfds_splits(version_dir, DATASET_NAME)
                except Exception as e:
                    print(f"[ERROR] Episode failed on plant {plant_idx}, epi {epi}: {e}")

        finally:
            plant_idx += 1


    # ---- Summary ----
    version_dir = os.path.join(TFDS_ROOT_DIR, DATASET_NAME, DATASET_VERSION)
    # safety sweep for all splits
    move_stray_shards_into_version_dir(TFDS_ROOT_DIR, ds_config)
    if ENABLE_TFRECORD_HARVEST:
        for split_name in ("train", "val", "test"):
            harvest_any_tfrecords(TFDS_ROOT_DIR, version_dir, DATASET_NAME, split_name)
    _post_write_repair(version_dir, DATASET_NAME)
    _relocate_dataset_metadata(TFDS_ROOT_DIR, version_dir, DATASET_VERSION)
    _ensure_min_train_split(version_dir, DATASET_NAME)
    repair_tfds_splits(version_dir, DATASET_NAME)

    stats_path = action_delta_stats.write_dataset_statistics(
        version_dir, DATASET_NAME, DATASET_VERSION, num_trajectories=episodes_done
    )
    print(f"[ACTION_NORM] saved Δq dataset_statistics to {stats_path}")

    print(f"✅ RLDS/TFDS episodes are under: {version_dir}")




if __name__ == "__main__":
    main()
