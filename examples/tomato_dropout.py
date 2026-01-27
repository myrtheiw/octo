"""
This script demonstrates how to load and rollout a finetuned Octo model.
UPDATED: Includes 'debug_use_dataset_actions' to replay dataset actions 
through the physics engine to verify control logic.
FIXED: 
1. step() returns 5-tuple
2. force_robot_pose() correctly locates .data
3. get_obs() dynamically matches dataset proprio shape (Fixes Shape mismatch 14 vs 9)
"""
from functools import partial
import csv
import sys
import os
import json
import random

# Ensure we use EGL for rendering (same as replay script)
#os.environ.setdefault("MUJOCO_GL", "egl") 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from absl import app, flags, logging
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from typing import Optional

# Explicitly import mujoco
import mujoco
from record_dataset.getlocation import get_side_stem_grasp_points

# =============================================================================
# TUNABLE THRESHOLDS (edit these instead of hunting through the file)
# =============================================================================
# Success/termination: EE-to-goal distance (meters) that counts as success.
# If this is too large, you may terminate before the gripper ever starts closing.
SUCCESS_DIST_M = 0.025

# Gripper closing trigger: start closing when EE is within this distance to goal.
# Oracle-style default was 0.006 (6mm), but 0.01–0.03 is often more robust.
GRIPPER_CLOSE_TRIGGER_DIST_M = 0.04

# Gripper ramp duration in outer env steps (oracle default: 25)
GRIPPER_RAMP_STEPS = 25

GRIP_OPEN_NORM = 1.0     # normalized "open"
GRIP_CLOSE_NORM = 0.0   # normalized "closed"

FREEZE_DIST_M = SUCCESS_DIST_M        # start holding pose when within this distance
FREEZE_HOLD_STEPS = 20        # hold for N outer env steps once triggered
POST_SUCCESS_HOLD_STEPS = 50
# Away-termination tuning: trigger if distance exceeds best-so-far by margin for N steps.
AWAY_MARGIN_M = 0.05
AWAY_STEPS = 8
# =============================================================================

MC_DROPOUT_SAMPLES = 10
SUCCESS_RETURN_THRESHOLD = 0.0  # Task-dependent fallback when no success flag exists.


try:
    # Prefer project-local oracle parameters when available.
    from record_dataset.oracle_dynamic_norm import (
        PD_KP,
        PD_KD,
        DATASET_MAX_STEP_RAD,
        TARGET_ACTION_DT as ORACLE_TARGET_ACTION_DT,
    )
except Exception:
    PD_KP = 100.0
    PD_KD = 2 * np.sqrt(PD_KP)
    DATASET_MAX_STEP_RAD = 2.0e-2
    ORACLE_TARGET_ACTION_DT = 0.1

sys.path.append("/home/myrtheiw/octo_ws/act")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# keep this to register ALOHA sim env
from envs.tomato_env import DEFAULT_MODEL_PATH, PandaTomatoSimEnv, TomatoGymEnv
from scripts import sim_env as sim_env_module

from octo.data.oxe.oxe_standardization_transforms import (
    tomato_rlds_dataset_transform,
)
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper, ResizeImageWrapper
from octo.utils.train_callbacks import supply_rng


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)
flags.DEFINE_integer(
    "finetuned_step",
    None,
    "Checkpoint step to load. If None, load the latest in finetuned_path.",
)
flags.DEFINE_string(
    "debug_trace_path",
    None,
    "If set, write JSONL traces containing commanded vs. actual joint deltas.",
)
flags.DEFINE_bool(
    "use_dataset_trajectory",
    False,
    "If true, run a dataset-based rollout instead of MuJoCo env rollouts.",
)
flags.DEFINE_string(
    "dataset_split",
    "train",
    "TFDS split to read from for dataset rollouts (e.g. train/val/test).",
)
flags.DEFINE_integer(
    "dataset_trajectory_index",
    0,
    "Index of the trajectory in the chosen split to evaluate.",
)
flags.DEFINE_integer(
    "dataset_max_steps",
    400,
    "Maximum number of timesteps to evaluate from the chosen trajectory.",
)
flags.DEFINE_string(
    "dataset_data_dir",
    "/home/myrtheiw/tfds_out",
    "Optional data_dir to pass to tfds.load for the tomato_rlds dataset.",
)
flags.DEFINE_integer(
    "num_dataset_episodes",
    1,
    "Number of dataset trajectories to evaluate when use_dataset_trajectory is True.",
)
flags.DEFINE_integer(
    "dataset_random_seed",
    0,
    "Random seed for sampling dataset trajectories when num_dataset_episodes > 1.",
)
flags.DEFINE_bool(
    "match_dataset_scene", 
    True, 
    "If True, sets the random seed to the episode index to recreate the exact training scene."
)
flags.DEFINE_bool(
    "debug_use_dataset_actions", 
    False, 
    "If True, ignores the model policy and feeds ground-truth dataset actions into the sim."
)

flags.DEFINE_integer(
    "policy_start_timestep",
    0,
    "In MuJoCo rollout mode, use dataset actions for t < policy_start_timestep, "
    "then switch to the policy for t >= policy_start_timestep. "
    "Ignored if debug_use_dataset_actions=True."
)
flags.DEFINE_float(
    "uncertainty_hz",
    50.0,
    "Frequency (Hz) for uncertainty logging.",
)
flags.register_validator(
    "uncertainty_hz",
    lambda value: value > 0.0,
    message="--uncertainty_hz must be > 0.",
)
flags.DEFINE_integer(
    "mc_dropout_samples",
    MC_DROPOUT_SAMPLES,
    "Number of MC Dropout samples per uncertainty estimate.",
)
flags.register_validator(
    "mc_dropout_samples",
    lambda value: value > 0,
    message="--mc_dropout_samples must be > 0.",
)

flags.DEFINE_integer(
    "target_success_episodes",
    5,
    "Number of successful episodes to collect (MuJoCo eval).",
)
flags.DEFINE_integer(
    "target_failure_episodes",
    5,
    "Number of failed episodes to collect (MuJoCo eval).",
)
flags.DEFINE_integer(
    "max_attempts",
    50,
    "Maximum rollout attempts when collecting success/failure quotas.",
)
flags.DEFINE_integer(
    "base_seed",
    0,
    "Base seed for deterministic rollout attempts; attempt i uses base_seed + i unless match_dataset_scene is True.",
)
flags.DEFINE_string(
    "output_dir",
    "./eval_uncertainty_out",
    "Directory to write CSV(s) and plots.",
)
flags.DEFINE_bool(
    "disable_wandb",
    False,
    "If true, do not init wandb and do not log any wandb metrics/videos.",
)
flags.DEFINE_bool(
    "log_video_to_wandb",
    False,
    "If true and wandb is enabled, log rollout video. Default false to avoid large uploads.",
)

# How to run (example):
# python tomato_dropout.py \
#   --finetuned_path=... \
#   --finetuned_step=... \
#   --use_dataset_trajectory=False \
#   --target_success_episodes=5 \
#   --target_failure_episodes=5 \
#   --max_attempts=50 \
#   --base_seed=0 \
#   --match_dataset_scene=False \
#   --mc_dropout_samples=10 \
#   --uncertainty_hz=50 \
#   --output_dir=./eval_uncertainty_out \
#   --disable_wandb=True \
#   --log_video_to_wandb=False



# --- HELPER FUNCTIONS FOR DATASET LOADING ---


TYPICAL_FRANKA_LIMITS = np.array([
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
], dtype=np.float64)

def find_gripper_actuator(model):
    """
    Finds the actuator index for the gripper.
    NOTE: The premade panda.xml you are using often names the gripper actuator
    something like "actuator8" and uses ctrlrange ~ [0,255]. Keyword matching
    alone will miss it, so we add a ctrlrange-based fallback.
    """
    # 1) Keyword match (legacy)
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if any(k in name.lower() for k in ("grip", "finger", "hand")):
            return aid
 
    # 2) Exact common name fallback
    try:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if aid >= 0:
            return aid
    except Exception:
        pass

    # 3) ctrlrange heuristic fallback (common for panda gripper: [0,255])
    # Pick the actuator with the largest ctrlrange span, preferring those with high max.
    best = -1
    best_span = -1.0
    for aid in range(model.nu):
        lo, hi = model.actuator_ctrlrange[aid]
        span = float(hi - lo)
        if hi >= 10.0 and span > best_span:
            best = aid
            best_span = span
    return best if best >= 0 else -1

def build_arm_mapping_from_model(model, prefer_position=True):
    """Return (arm_act_ids, arm_qpos_adr, arm_dof_idx) for Panda link1..link7."""
    arm_body_names = [f"link{i}" for i in range(1, 8)]
    grip_keywords = ("grip", "finger", "hand")

    # 1) Arm hinge joints by body name
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

        gaintype = int(model.actuator_gaintype[aid])
        qadr = int(model.jnt_qposadr[jid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])) or ""
        per_joint_candidates[jid].append((aid, jid, qadr, gaintype, bname))

    chosen = []
    for jid in arm_joint_ids:
        cands = per_joint_candidates.get(jid, [])
        if not cands:
            continue
        if prefer_position:
            cands.sort(key=lambda t: (0 if t[3] == 3 else 1, t[2]))
        else:
            cands.sort(key=lambda t: (0 if t[3] != 3 else 1, t[2]))
        chosen.append(cands[0])

    # Keep only on arm bodies, dedupe, sort by qpos adr
    chosen = [c for c in chosen if c[4] in arm_body_names]
    by_jid = {}
    for c in chosen:
        by_jid.setdefault(c[1], c)
    chosen = sorted(by_jid.values(), key=lambda t: t[2])

    if len(chosen) != 7:
        raise RuntimeError(f"Expected 7 arm actuators, found {len(chosen)} (chosen={chosen}).")

    arm_act_ids  = np.array([c[0] for c in chosen], dtype=int)
    arm_qpos_adr = np.array([c[2] for c in chosen], dtype=int)

    # dof indices for velocity terms (same ordering as arm_qpos_adr)
    arm_dof_idx = []
    for qadr in arm_qpos_adr:
        jid = None
        for j in range(model.njnt):
            if int(model.jnt_qposadr[j]) == int(qadr):
                jid = j
                break
        if jid is None:
            raise RuntimeError("Could not find joint for qpos address.")
        dof = None
        for d in range(model.nv):
            if int(model.dof_jntid[d]) == jid:
                dof = d
                break
        if dof is None:
            raise RuntimeError("Could not find dof for joint.")
        arm_dof_idx.append(dof)
    arm_dof_idx = np.array(arm_dof_idx, dtype=int)

    return arm_act_ids, arm_qpos_adr, arm_dof_idx

def _choose_ee_ref(model):
    # Prefer tcp site if present; else fall back to hand body
    try:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    except Exception:
        sid = -1
    if sid >= 0:
        return ("site", "tcp")
    return ("body", "hand")


def dump_distance_curve(distances, out_png_path, success_dist=SUCCESS_DIST_M, title=None):
    distances = np.asarray(distances, dtype=np.float32)
    os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)

    plt.figure()
    plt.plot(distances)
    plt.axhline(success_dist, linestyle="--")
    plt.xlabel("Env step")
    plt.ylabel("EE-to-goal distance (m)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()

def widen_arm_limits(model, arm_dof_idx):
    """Apply typical Franka limits to the arm joints."""
    for i, dof in enumerate(arm_dof_idx):
        jid = int(model.dof_jntid[dof])
        model.jnt_limited[jid] = 1
        lo, hi = TYPICAL_FRANKA_LIMITS[i]
        model.jnt_range[jid][0] = lo
        model.jnt_range[jid][1] = hi


def steps_to_np_trajectory(traj_dict):
    steps = list(traj_dict["steps"])
    if not steps:
        raise ValueError("Encountered empty trajectory in dataset.")

    obs = {}
    for k in steps[0]["observation"]:
        obs[k] = np.stack([s["observation"][k] for s in steps], axis=0)

    obs["timestep"] = np.arange(len(steps), dtype=np.int32)
    action = np.stack([s["action"] for s in steps], axis=0)
    language_instruction = np.asarray(
        [s["observation"]["language_instruction"] for s in steps]
    )

    return {
        "observation": obs,
        "action": action,
        "language_instruction": language_instruction,
    }

def get_trajectory_by_index(raw_ds, index: int):
    raw_iter = iter(tfds.as_numpy(raw_ds))
    for idx, traj_dict in enumerate(raw_iter):
        if idx == index:
            if "steps" in traj_dict:
                return steps_to_np_trajectory(traj_dict)
            ds = tf.data.Dataset.from_tensors(traj_dict)
            ds = ds.map(
                lambda traj: tomato_rlds_dataset_transform(traj),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            ds = ds.prefetch(1)
            return next(ds.as_numpy_iterator())
    raise IndexError(f"dataset_trajectory_index={index} is out of range.")

class EnsureOctoObsKeysWrapper(gym.ObservationWrapper):
    """
    Ensures rollout-time observations contain keys Octo commonly expects:
      - timestep: [H]
      - pad_mask_dict: per-key masks [H]
      - task_completed: [H,4] (dummy unless you have a real signal)

    Must be applied AFTER HistoryWrapper (so obs values are already [H,...]).
    """

    def __init__(self, env: gym.Env, task_completed_dim: int = 4):
        super().__init__(env)
        self.task_completed_dim = int(task_completed_dim)

    def observation(self, obs):
        # HistoryWrapper adds timestep_pad_mask: [H] with 0s for padded slots. :contentReference[oaicite:2]{index=2}
        tmask = obs.get("timestep_pad_mask", None)
        if tmask is None:
            # Fall back to "all valid" if not present
            H = obs["proprio"].shape[0]
            tmask = np.ones((H,), dtype=bool)
        else:
            tmask = np.asarray(tmask).astype(bool)
            H = tmask.shape[0]

        # 1) timestep: [0..H-1]
        if "timestep" not in obs:
            obs["timestep"] = np.arange(H, dtype=np.int32)

        # 2) pad_mask_dict: per-modality validity
        # Use timestep_pad_mask so padded history slots are masked out.
        pmd = dict(obs.get("pad_mask_dict", {}))

        # Only set masks for keys that actually exist in obs
        for k in ("image_primary", "image_wrist", "proprio", "timestep"):
            if k in obs:
                pmd.setdefault(k, tmask.copy())

        obs["pad_mask_dict"] = pmd

        # 3) task_completed: dummy zeros unless you have a real completion signal
        if "task_completed" not in obs:
            obs["task_completed"] = np.zeros((H, self.task_completed_dim), dtype=np.float32)

        return obs

# --- OVERRIDE CLASS FOR DIRECT CONTROL ---
class DirectControlPandaEnv(PandaTomatoSimEnv):
    """
    Direct-control wrapper that matches the successful replay logic:

    - action is interpreted as a 7D *delta* in "dataset action units"
    - convert to radians with ORACLE_SCALE (0.02)
    - integrate into a persistent virtual absolute joint target
    - command the correct actuator indices (arm_act_ids)
    - step multiple MuJoCo substeps per action
    - get_obs returns the same dict layout expected by Octo
    """
    ORACLE_SCALE = 0.02
    TARGET_ACTION_DT = float(ORACLE_TARGET_ACTION_DT)  # seconds per action (oracle used ~0.1s)

    def __init__(self, proprio_dim=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proprio_dim = proprio_dim

        # ... (Existing setup code) ...
        self.model = self._env.model
        self.data  = self._env.data

        # Debug: list all actuators and ctrlranges once at init.
        for aid in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
            lo, hi = self.model.actuator_ctrlrange[aid]
            # suppress actuator debug print

        # 1. FIND GRIPPER ACTUATOR ID
        self.gripper_act_id = find_gripper_actuator(self.model)

        self.arm_act_ids, self.arm_qpos_adr, self.arm_dof_idx = build_arm_mapping_from_model(
            self.model, prefer_position=True
        )
        widen_arm_limits(self.model, self.arm_dof_idx)

        # Substeps to match ~0.1s per action
        dt = float(self.model.opt.timestep)
        self.substeps = int(max(1, round(self.TARGET_ACTION_DT / max(dt, 1e-6))))
        self.max_step_rad = float(DATASET_MAX_STEP_RAD)

        # Persistent integrated target (absolute joint angles)
        self._q_target = None
        self._q_cmd = None

        self._debug_step_count = 0

        # Renderer (keep your existing approach)
        self._renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.success_dist = SUCCESS_DIST_M
        self.success_hold_steps = 3
        self._success_streak = 0
        self.ee_ref = _choose_ee_ref(self.model)
        # suppress EE reference debug print
        # --- Gripper control setup ---
        self._grip_state = "open"           # "open" | "ramping" | "closed"
        self._grip_ramp_step = 0
        self._grip_norm = GRIP_OPEN_NORM

        if self.gripper_act_id >= 0:
            lo, hi = self.model.actuator_ctrlrange[self.gripper_act_id]
            self._grip_close_cmd = float(lo)
            self._grip_open_cmd  = float(hi)
            # If behavior is reversed (closed at start, opens when "closing"), flip these:
            # self._grip_close_cmd, self._grip_open_cmd = self._grip_open
            
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_act_id) or ""
            # suppress gripper actuator debug print
        else:
            self._grip_close_cmd = 0.0
            self._grip_open_cmd  = 0.0
            # suppress gripper actuator debug print
        self._freeze_active = False
        self._freeze_steps_left = 0
        self._freeze_q_target = None
        self._last_q_target_cmd = None
        self._last_grip_cmd = None  # optional, but useful
        self._done_latched = False
        self._freeze_debug_enabled = True
        self._freeze_debug_window = 5
        self._freeze_debug_step = 0
        self._freeze_latch_step = None
        self._freeze_debug_buffer = []
        self._freeze_debug_printed_steps = set()
        self._freeze_last_dist = None
        self._freeze_last_qpos = None
        self._freeze_last_action_norm = None
        self._success_latched = False
        self._post_success_hold_left = 0
        self._post_success_hold_total = int(POST_SUCCESS_HOLD_STEPS)




    def _ee_pos_world(self):
        """Return EE xyz in world coords using ee_ref with fallbacks (same logic as success check)."""
        kind, name = getattr(self, "ee_ref", ("site", "tcp"))
        ee_pos = None
        if kind == "site":
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                ee_pos = self.data.site_xpos[sid].copy()
        elif kind == "body":
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ee_pos = self.data.xpos[bid].copy()
        if ee_pos is None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
            if sid >= 0:
                ee_pos = self.data.site_xpos[sid].copy()
        if ee_pos is None:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            if bid >= 0:
                ee_pos = self.data.xpos[bid].copy()
        return ee_pos

    def _grip_cmd_from_norm(self, g_norm: float) -> float:
        """Map normalized [0,1] open/close to actuator ctrlrange."""
        g = float(np.clip(g_norm, 0.0, 1.0))
        return self._grip_close_cmd + g * (self._grip_open_cmd - self._grip_close_cmd)
 


    def reset(self, **kwargs):
        self._success_streak = 0
        self._freeze_active = False
        self._freeze_steps_left = 0
        self._freeze_q_target = None
        self._done_latched = False
        self._freeze_debug_step = 0
        self._freeze_latch_step = None
        self._freeze_debug_buffer = []
        self._freeze_debug_printed_steps = set()
        self._freeze_last_dist = None
        self._freeze_last_qpos = None
        self._freeze_last_action_norm = None
        self._success_latched = False
        self._post_success_hold_left = 0
        self._post_success_hold_total = int(POST_SUCCESS_HOLD_STEPS)



        # 1. Standard reset (clears physics state)
        obs, info = super().reset(**kwargs)

        # 2. FORCE DATASET START POSE
        # These values match your "Data Joints" debug output exactly
        start_joints = np.array([
            0.0, -0.4948, 0.0, -1.5172, 0.0, 1.4902, 0.0
        ], dtype=np.float64)
        
        # Write to physics state directly
        self.data.qpos[self.arm_qpos_adr] = start_joints
        self.data.qvel[:] = 0  # Kill any momentum
        mujoco.mj_forward(self.model, self.data)

        # 3. SYNC CONTROLLER
        # Tell the P-controller: "Your target is exactly where we just put you"
        self._q_target = start_joints.copy()
        self._q_cmd = start_joints.copy().astype(np.float32)

        # 4. RE-CAPTURE OBSERVATION
        # Use your custom get_obs (which now has the gripper fix)
        # to ensure the first frame the model sees is correct.
        obs = self.get_obs()
 
        # Reset gripper state machine to OPEN (oracle-style: open until near goal)
        self._grip_state = "open"
        self._grip_ramp_step = 0
        self._grip_norm = GRIP_OPEN_NORM
        if self.gripper_act_id >= 0:
            self.data.ctrl[self.gripper_act_id] = self._grip_cmd_from_norm(self._grip_norm)

        return obs, info

    def step(self, action):
        """
        Option B (stateless): interpret action as normalized "error-to-go" in joint space.

        action: array-like with >=7 dims.
                Treated as normalized joint delta in [-1, 1] (dataset action space).
                Converted to radians by ORACLE_SCALE and applied relative to *current measured q*:
                    q_target = q_measured + dq

        Adds success termination:
        - If self.goal_pos is set (world xyz), terminate when EE is within 0.03 m
            for `self.success_hold_steps` consecutive outer steps.
        """
        # --- 0) Parse + validate ---
        a = np.asarray(action, dtype=np.float32)
        if a.ndim != 1:
            a = a.reshape(-1)
        if a.shape[0] < 7:
            raise ValueError(f"Expected action with >=7 dims, got shape {a.shape}")
        self._freeze_debug_step += 1
        step_idx = self._freeze_debug_step
        q_before = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float64)
        freeze_active_at_start = bool(getattr(self, "_freeze_active", False))
        freeze_left_at_start = int(getattr(self, "_freeze_steps_left", 0))

        # Keep arm dims and clamp to normalized range
        a7 = np.clip(a[:7], -1.0, 1.0).astype(np.float64, copy=False)

        # --- 1) Convert normalized delta -> radians ---
        dq = a7 * float(self.ORACLE_SCALE)  # radians
        max_step = float(self.max_step_rad)
        if max_step > 0.0:
            dq = np.clip(dq, -max_step, max_step)
        dq_norm = float(np.linalg.norm(dq))
        a_min = float(a7.min())
        a_max = float(a7.max())
        self._freeze_last_action_norm = dq_norm

        # --- 2) Compute absolute target from CURRENT measured q (no integration) ---
        q_meas = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float64)
        q_target = q_meas + dq
        q_target_candidate = q_target.copy()

        # --- 3) Build cache: qpos_adr -> joint id (once) ---
        if not hasattr(self, "_arm_qadr_to_jid"):
            qadr_to_jid = {}
            for jid in range(self.model.njnt):
                qadr_to_jid[int(self.model.jnt_qposadr[jid])] = jid
            self._arm_qadr_to_jid = qadr_to_jid

        # --- 4) Clamp target to joint limits ---
        for i, qadr in enumerate(self.arm_qpos_adr):
            jid = self._arm_qadr_to_jid.get(int(qadr), None)
            if jid is None:
                continue
            if int(self.model.jnt_limited[jid]) == 1:
                lo, hi = self.model.jnt_range[jid]
                q_target[i] = np.clip(q_target[i], lo, hi)
        # In measured mode we set _q_cmd for logging only.
        self._q_cmd = q_target.astype(np.float32)

        # --- 5) If success latched: HOLD (ignore action) ---
        if getattr(self, "_success_latched", False):
            dq = np.zeros_like(dq)
            q_target = getattr(self, "_q_hold", self._q_cmd).astype(np.float64, copy=False)

        # --- 6) Debug prints (safe) ---
        if getattr(self, "_dbg_printed", 0) < 10:
            self._dbg_printed = getattr(self, "_dbg_printed", 0) + 1
            print(
                f"[ENV-B] a_norm(min,max)=({float(a7.min()):.3f},{float(a7.max()):.3f}) "
                f"dq(rad)(min,max)=({float(dq.min()):.3e},{float(dq.max()):.3e}) "
                f"||dq||={float(np.linalg.norm(dq)):.3e}",
                flush=True,
            )

        # --- 7) Step MuJoCo: absolute target control ---
        # ----------------------------------------------------------------------
        # EE-to-goal distance (world frame), used for gripper schedule
        # ----------------------------------------------------------------------
        goal_pos = getattr(self, "goal_pos", None)
        dist_to_goal = None
        if goal_pos is not None:
            goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
            ee_pos = self._ee_pos_world()
            if ee_pos is not None:
                dist_to_goal = float(np.linalg.norm(ee_pos - goal_pos))

        # ----------------------------------------------------------------------
        # Gripper schedule: open until close-trigger, then ramp closed
        # ----------------------------------------------------------------------
        if self.gripper_act_id >= 0:
            if self._grip_state == "open":
                if dist_to_goal is not None and dist_to_goal <= float(GRIPPER_CLOSE_TRIGGER_DIST_M):
                    print(f"[GRIPPER] trigger dist={dist_to_goal:.4f} -> ramping", flush=True)
                    self._grip_state = "ramping"
                    self._grip_ramp_step = 0
                self._grip_norm = GRIP_OPEN_NORM

            elif self._grip_state == "ramping":
                t = float(self._grip_ramp_step) / float(max(1, GRIPPER_RAMP_STEPS - 1))
                # open (1.0) -> close (0.0)
                self._grip_norm = float(np.clip(1.0 - t, 0.0, 1.0))
                self._grip_ramp_step += 1
                if self._grip_ramp_step >= int(GRIPPER_RAMP_STEPS):
                    self._grip_state = "closed"
                    self._grip_norm = GRIP_CLOSE_NORM

            else:  # "closed"
                self._grip_norm = GRIP_CLOSE_NORM

            grip_cmd = self._grip_cmd_from_norm(self._grip_norm)
        else:
            grip_cmd = None

        self._last_q_target_cmd = q_target.copy()
        if grip_cmd is not None:
            self._last_grip_cmd = float(grip_cmd)
        else:
            self._last_grip_cmd = None


        for _ in range(int(self.substeps)):
            # Command the Arm to the absolute target computed this step
            self.data.ctrl[self.arm_act_ids] = q_target
            self._last_q_target_cmd = q_target.copy()
            self._last_grip_cmd = None if grip_cmd is None else float(grip_cmd)


            # Command gripper using oracle-style schedule (mapped to ctrlrange)
            if grip_cmd is not None:
                self.data.ctrl[self.gripper_act_id] = float(grip_cmd)
            mujoco.mj_step(self.model, self.data)

        # After physics sub-steps, compute post-step distance
        q_after = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float64)
        moved = float(np.linalg.norm(q_after - q_before))
        dist_post = None
        goal_pos = getattr(self, "goal_pos", None)
        if goal_pos is not None:
            goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
            ee_pos = self._ee_pos_world()
            if ee_pos is not None:
                dist_post = float(np.linalg.norm(ee_pos - goal_pos))
                if dist_post is not None and getattr(self, "_dbg_dist_post", 0) < 50:
                    self._dbg_dist_post = getattr(self, "_dbg_dist_post", 0) + 1
                    print(f"[DIST_POST] {dist_post:.4f} (freeze_th={FREEZE_DIST_M:.3f} success_th={SUCCESS_DIST_M:.3f})", flush=True)
        # --- 8) Success latch + post-success hold + termination ---
        terminated = False
        truncated = False
        info = {}

        success_dist = float(SUCCESS_DIST_M)
        success_hold_steps = int(getattr(self, "success_hold_steps", 3))
        if not hasattr(self, "_success_streak"):
            self._success_streak = 0
        if not hasattr(self, "_success_latched"):
            self._success_latched = False
        if not hasattr(self, "_post_success_hold_left"):
            self._post_success_hold_left = 0
        if not hasattr(self, "_post_success_hold_total"):
            self._post_success_hold_total = int(POST_SUCCESS_HOLD_STEPS)

        success_just_latched = False
        if dist_post is not None:
            info["ee_goal_dist"] = dist_post
            if not self._success_latched:
                if dist_post <= success_dist:
                    self._success_streak += 1
                else:
                    self._success_streak = 0

                if self._success_streak >= success_hold_steps:
                    self._success_latched = True
                    self._post_success_hold_total = int(POST_SUCCESS_HOLD_STEPS)
                    self._post_success_hold_left = int(POST_SUCCESS_HOLD_STEPS)
                    success_just_latched = True
                    self._q_hold = self._q_cmd.copy()
                    self._grip_state = "closed"
                    self._grip_norm = GRIP_CLOSE_NORM

        info["is_success"] = bool(self._success_latched)

        if self._success_latched:
            if not success_just_latched and self._post_success_hold_left > 0:
                self._post_success_hold_left -= 1
            if self._post_success_hold_left <= 0 and not success_just_latched:
                terminated = True
                info["terminated_by_success_hold"] = True
                info["is_success"] = True


        # --- 8) Return Gym API ---
        obs = self.get_obs()
        reward = 1.0 if info.get("is_success", False) else 0.0

        if getattr(self, "_debug_step_count", 0) < 5:
            self._debug_step_count = getattr(self, "_debug_step_count", 0) + 1
            q_sim = self.data.qpos[self.arm_qpos_adr]
            print(
                f"[STEP-B {self._debug_step_count}] "
                f"|q_target-q_sim|={np.linalg.norm(q_target - q_sim):.3e} "
                f"q0={q_sim[0]:.3f} "
                f"dist={info.get('ee_goal_dist', float('nan')):.4f} "
                f"streak={getattr(self, '_success_streak', 0)}",
                flush=True,
            )

        return obs, reward, terminated, truncated, info

    def set_goal_pos(self, goal_pos_world):
        self.goal_pos = np.asarray(goal_pos_world, dtype=np.float64).reshape(3)
        if hasattr(self, "_env") and hasattr(self._env, "set_goal_pos"):
            self._env.set_goal_pos(self.goal_pos)


    def get_obs(self):
            # ... (image rendering code remains the same) ...
            self._renderer.update_scene(self.data, camera="front_cam")
            image_primary = self._renderer.render()
            
            # ... (wrist camera code remains the same) ...
            try:
                self._renderer.update_scene(self.data, camera="wrist_cam")
                img_wrist_raw = self._renderer.render()
                image_wrist = img_wrist_raw[::2, ::2]
            except Exception:
                image_wrist = np.zeros((128, 128, 3), dtype=np.uint8)

            # --- FIX: FORCE GRIPPER TO OPEN (0.04) ---
            qpos_arm = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float32)
            # If we need more than 7 dims (e.g. 9), pad with gripper state.
            # We approximate finger joint positions using the same normalized open/close
            # used for actuator control: open≈0.04, closed≈0.0.
            if self.proprio_dim > 7:
                # We have 7 arm joints. The remaining (proprio_dim - 7) are gripper fingers.
                # 0.04 is the raw joint position for an OPEN Panda gripper.
                gripper_open_val = 0.04
                gripper_closed_val = 0.0
                g = float(np.clip(getattr(self, "_grip_norm", GRIP_OPEN_NORM), 0.0, 1.0))
                gripper_val = float(gripper_closed_val + g * (gripper_open_val - gripper_closed_val))
                gripper_state = np.full((self.proprio_dim - 7,), gripper_val, dtype=np.float32)

                proprio = np.concatenate([qpos_arm[:7], gripper_state], axis=0)
            else:
                proprio = qpos_arm[: self.proprio_dim]
            # ----------------------------------

            return {
                "image_primary": image_primary,
                "image_wrist": image_wrist,
                "proprio": proprio,
            }
def force_robot_pose(env, target_joints):
    """
    Forces the robot to a specific pose using raw MuJoCo data access.
    """
    base_env = env.unwrapped 
    internal_sim = None
    
    # Locate internal sim
    if hasattr(base_env, "panda_env") and hasattr(base_env.panda_env, "_env"):
        internal_sim = base_env.panda_env._env
    elif hasattr(base_env, "_env"):
        internal_sim = base_env._env

    # Apply Pose
    if internal_sim and hasattr(internal_sim, "data") and hasattr(internal_sim, "model"):
        data = internal_sim.data
        model = internal_sim.model
        
        current_qpos = data.qpos.copy()
        current_qpos[:7] = target_joints
        data.qpos[:] = current_qpos
        mujoco.mj_forward(model, data)
        print(f"Forcefully set robot joints to: {target_joints}")
    else:
        print("WARNING: Could not access env.data to force pose!")


DATASET_START_JOINTS = np.array([
        5.5947731e-19, -4.9482849e-01, 1.0067819e-18, -1.5171977e+00, 
        -4.1786848e-20, 1.4901806e+00, 4.3187124e-21
    ], dtype=float)

def _json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value)
    return str(value)

class ChunkLogger:
    def __init__(self, path: Optional[str]):
        self._fp = open(path, "w", encoding="utf-8") if path else None

    def log_chunk(self, **kwargs):
        if self._fp is None: return
        self._fp.write(json.dumps(kwargs, default=_json_default) + "\n")
        self._fp.flush()

    def close(self):
        if self._fp: self._fp.close()

def sample_mc_actions(model, obs, task, n_samples=MC_DROPOUT_SAMPLES, base_rng=None):
    if base_rng is None:
        base_rng = jax.random.PRNGKey(np.random.randint(0, 1_000_000))

    obs_batched = jax.tree_map(lambda x: x[None], obs)
    rngs = jax.random.split(base_rng, n_samples)

    def single_sample(rng_key):
        return model.sample_actions(
            obs_batched,
            task,
            train=True,
            rng=rng_key,
            unnormalization_statistics=model.dataset_statistics["action"],
        )[0]

    actions = jax.lax.map(single_sample, rngs)
    return actions

def _coerce_bool(value):
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return bool(arr.item())
    except Exception:
        pass
    return bool(value)

def detect_success(min_ee_goal_dist):
    if min_ee_goal_dist is None:
        return False
    return float(min_ee_goal_dist) <= float(SUCCESS_DIST_M)

def get_condition_name(info, default="default"):
    if isinstance(info, dict):
        for key in ("condition", "env_condition", "domain"):
            if key in info:
                return str(info[key])
    return default

def main(_):
    use_wandb = not FLAGS.disable_wandb
    if use_wandb:
        wandb.init(name="eval_tomato", project="octo")
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path, step=FLAGS.finetuned_step)
    
    # Dataset Statistics for un-normalization
    ACTION_MEAN = np.asarray(model.dataset_statistics["action"]["mean"], dtype=np.float32)
    ACTION_STD = np.asarray(model.dataset_statistics["action"]["std"], dtype=np.float32)
    PROPRIO_MEAN = np.asarray(model.dataset_statistics["proprio"]["mean"], dtype=np.float32)
    PROPRIO_STD = np.asarray(model.dataset_statistics["proprio"]["std"], dtype=np.float32)

    # CRITICAL: Detect actual proprio dimension from dataset stats (e.g. 9)
    EXPECTED_PROPRIO_DIM = PROPRIO_MEAN.shape[0]
    logging.info(f"Detected Proprio Dimension from Dataset Stats: {EXPECTED_PROPRIO_DIM}")

    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # -------------------------------------------------------------------------
    # LOAD DATASET (Common for both modes)
    # -------------------------------------------------------------------------
    raw_ds = tfds.load(
        name="tomato_rlds",
        split=FLAGS.dataset_split,
        data_dir=FLAGS.dataset_data_dir,
        shuffle_files=False,
    )

    # -------------------------------------------------------------------------
    # MODE 1: MUJOCO SIMULATION (with Policy OR Dataset Actions)
    # -------------------------------------------------------------------------
    if not FLAGS.use_dataset_trajectory:
        output_dir = FLAGS.output_dir
        os.makedirs(output_dir, exist_ok=True)
        target_success = int(FLAGS.target_success_episodes)
        target_failure = int(FLAGS.target_failure_episodes)
        max_attempts = int(FLAGS.max_attempts)
        traj_idx = int(FLAGS.dataset_trajectory_index)

        kept_rows = []
        attempt_rows = []
        success_rows = []
        failure_rows = []

        episode_uncertainties = []
        episode_successes = []
        episode_conditions = []
        episode_returns = []

        attempt_idx = 0
        kept_success = 0
        kept_failure = 0

        trace_logger = ChunkLogger(FLAGS.debug_trace_path)
        try:
            while attempt_idx < max_attempts and (
                kept_success < target_success or kept_failure < target_failure
            ):
                episode_index = attempt_idx
                print(
                    f"[ATTEMPT] {attempt_idx + 1}/{max_attempts} "
                    f"(kept_success={kept_success}/{target_success}, "
                    f"kept_failure={kept_failure}/{target_failure})",
                    flush=True,
                )

                if FLAGS.match_dataset_scene:
                    current_seed = traj_idx + attempt_idx
                    logging.info(
                        f"MATCHING DATASET: Regenerating scene for traj_idx={traj_idx} "
                        f"(seed={current_seed})"
                    )
                else:
                    current_seed = int(FLAGS.base_seed) + attempt_idx
                    logging.info(
                        f"DYNAMIC: Generating new random scene (Seed {current_seed})"
                    )

                np.random.seed(current_seed)
                random.seed(current_seed)
                print(
                    f"[SEED CHECK] traj_idx={traj_idx} seed={current_seed}",
                    flush=True,
                )

                # 2. PREPARE DEBUG ACTIONS (If flag is set)
                gt_actions_queue = []
                need_dataset_actions = (
                    FLAGS.debug_use_dataset_actions or (FLAGS.policy_start_timestep > 0)
                )

                if need_dataset_actions:
                    logging.warning(
                        f"Loading dataset actions for episode {episode_index} "
                        f"(debug_use_dataset_actions={FLAGS.debug_use_dataset_actions}, "
                        f"policy_start_timestep={FLAGS.policy_start_timestep})"
                    )
                    traj = get_trajectory_by_index(raw_ds, traj_idx)

                    actions = np.asarray(traj["action"])  # [T, >=7]
                    K = min(100, actions.shape[0])
                    a = actions[:K]
                    print(
                        "[DATA actions] shape=", a.shape,
                        "min=", float(a.min()),
                        "max=", float(a.max()),
                        "mean(abs)=", float(np.mean(np.abs(a))),
                        "p99(abs)=", float(np.quantile(np.abs(a), 0.99)),
                        flush=True,
                    )

                    gt_actions_queue = list(traj["action"])
                    logging.info(
                        f"Loaded {len(gt_actions_queue)} steps of ground-truth actions."
                    )

                # 3. REGENERATE SCENE XML
                sim_env_module.START_JOINTS = np.asarray(DATASET_START_JOINTS, dtype=float)
                model_xml_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
                try:
                    from record_dataset.helpers import build_and_load_scene, scene_dir_from_model_path
                    build_and_load_scene(str(model_xml_path))
                    scene_dir = Path(scene_dir_from_model_path(str(model_xml_path)))
                    dynamic_xml_path = scene_dir / "scene_dynamic.xml"
                except ImportError:
                    raise RuntimeError("record_dataset.helpers not found")

                # 4. BUILD ENV
                # Pass the detected proprio dim here!
                panda_env = DirectControlPandaEnv(
                    proprio_dim=EXPECTED_PROPRIO_DIM,
                    model_xml=str(dynamic_xml_path),
                    substeps=40,
                    action_scale=1.0,
                )
                panda_env.expects_absolute_action = True

                # --- SET GOAL FROM SIM (tomato pose) ---
                mujoco.mj_forward(panda_env.model, panda_env.data)  # ensure xpos/xmat are current

                grasp_dict = get_side_stem_grasp_points(panda_env.model, panda_env.data, s=0.66)
                if not grasp_dict:
                    raise RuntimeError("get_side_stem_grasp_points returned empty; scene naming mismatch?")

                # deterministic: pick the first stem by name
                stem_name = sorted(grasp_dict.keys())[0]
                goal_pos_world = np.asarray(grasp_dict[stem_name]["grasp_pos"], dtype=np.float64)

                panda_env.set_goal_pos(goal_pos_world)


                print("[CHECK] panda_env.goal_pos =", panda_env.goal_pos)
                # ---------------------------------------


                env = TomatoGymEnv(panda_env, max_steps=400)
                env.expects_absolute_action = True   # <-- must be BEFORE NormalizeProprio/History/RHC
                env = NormalizeProprio(env, model.dataset_statistics)
                env = ResizeImageWrapper(
                                env,
                                # REMOVE "image_" prefix here:
                                resize_size={"primary": (256, 256), "wrist": (128, 128)}, 
                                # KEEP "image_" prefix here (it matches the final key name):
                                #augmented_keys=["image_primary"],  
                                avg_scale=1.0,
                            )
                env = HistoryWrapper(env, horizon=2)
                env = EnsureOctoObsKeysWrapper(env)     
                env = RHCWrapper(env, exec_horizon=1)

                # --- RATE AUDIT (oracle-compatible outer-step pacing) ---
                timestep = float(panda_env.model.opt.timestep)
                substeps_calc = int(panda_env.substeps)
                action_dt = timestep * float(substeps_calc)
                hz = 1.0 / max(action_dt, 1e-9)
                print(
                    f"[RATE AUDIT] target_action_dt={float(panda_env.TARGET_ACTION_DT):.3f}s "
                    f"timestep={timestep:.6f}s substeps={substeps_calc} "
                    f"action_dt={action_dt:.6f}s hz={hz:.2f}",
                    flush=True,
                )
                print(
                    f"[RATE AUDIT] dataset metadata action_dt_sec={float(panda_env.TARGET_ACTION_DT):.3f}",
                    flush=True,
                )


                log_dt = 1.0 / float(FLAGS.uncertainty_hz)
                outer_dt = float(
                    getattr(env.unwrapped.panda_env, "TARGET_ACTION_DT", 0.1)
                )

                
                # 5. RUN ROLLOUT
                base = env.unwrapped
                panda = base.panda_env

                q_cmd = None if getattr(panda, "_last_q_target_cmd", None) is None else panda._last_q_target_cmd.copy()
                grip_cmd = getattr(panda, "_last_grip_cmd", None)
                freeze_active = bool(getattr(panda, "_freeze_active", False))

                # obs, info = env.reset()
                # # force_robot_pose(env, DATASET_START_JOINTS)

                
                # # Sync Virtual Target
                # internal_sim = None
                # base_env = env.unwrapped
                # if hasattr(base_env, "panda_env") and hasattr(base_env.panda_env, "_env"):
                #      internal_sim = base_env.panda_env._env
                
                # if internal_sim:
                #      raw_qpos = internal_sim.data.qpos[:7].copy()
                # else:
                #      raw_qpos = DATASET_START_JOINTS.copy()

                # virtual_target = raw_qpos.astype(np.float64)

                # # Update obs history manually
                # # NOTE: normalize using only the first 7 dims matching mean/std? 
                # # No, NormalizeProprio expects full vector. 
                # # We need to manually construct the normalized initial vector matching EXPECTED_PROPRIO_DIM
                
                # # init_proprio = np.zeros(EXPECTED_PROPRIO_DIM, dtype=np.float32)
                # # n = min(len(raw_qpos), EXPECTED_PROPRIO_DIM)
                # # init_proprio[:n] = raw_qpos[:n]
                
                # # norm_proprio = (init_proprio - PROPRIO_MEAN) / np.maximum(PROPRIO_STD, 1e-6)
                
                # # # Assign to history (Octo usually expects history of 2)
                # # # obs["proprio"] is shape (Window, Dim) -> (2, 9)
                # # obs["proprio"][-1] = norm_proprio

                # task = model.create_tasks(texts=env.get_task()["language_instruction"])
                # first = obs["image_primary"][-1] if obs["image_primary"].ndim == 4 else obs["image_primary"]
                # images = [first]

                # episode_return = 0.0

                # step_count = 0
                # while len(images) < 400:

                obs, info = env.reset()
                distance_curve = []

                traj = get_trajectory_by_index(raw_ds, traj_idx)

                ds = traj["observation"]["image_primary"][0].astype(np.float32)
                sim = (obs["image_primary"][-1] if obs["image_primary"].ndim == 4 else obs["image_primary"]).astype(np.float32)

                print("[IMG DIFF t=0] mean_abs =", float(np.mean(np.abs(ds - sim))), flush=True)

                task = model.create_tasks(texts=env.get_task()["language_instruction"])
                condition_name = get_condition_name(info, default="default")
                first = obs["image_primary"][-1] if obs["image_primary"].ndim == 4 else obs["image_primary"]
                images = [first]

                episode_return = 0.0
                step_uncertainties = []
                last_info = info
                step_count = 0
                away_steps = 0
                min_ee_goal_dist = None
                final_ee_goal_dist = None
                terminated_by_away = False
                unc_ticks = []
                unc_values = []
                unc_times = []
                tick_idx = 0
                t_sec = 0.0
                next_log_t = 0.0
                term_tick = None
                while len(images) < 400:
                    while next_log_t <= t_sec + 1e-9:
                        mc_actions = sample_mc_actions(
                            model,
                            obs,
                            task,
                            n_samples=int(FLAGS.mc_dropout_samples),
                            base_rng=jax.random.PRNGKey(np.random.randint(1e6)),
                        )
                        action_std = jnp.std(mc_actions, axis=0)
                        step_unc = float(jnp.mean(action_std))
                        unc_ticks.append(tick_idx)
                        unc_values.append(step_unc)
                        unc_times.append(next_log_t)
                        tick_idx += 1
                        next_log_t += log_dt
                    
                    # --- DECISION BLOCK ---
                    use_dataset_now = False
                    if FLAGS.debug_use_dataset_actions:
                        use_dataset_now = True
                    elif step_count < int(FLAGS.policy_start_timestep):
                        use_dataset_now = True

                    if use_dataset_now:
                        if step_count < len(gt_actions_queue):
                            action_delta = gt_actions_queue[step_count]
                            policy_chunk = np.array([action_delta], dtype=np.float32)
                        else:
                            logging.warning("Ran out of dataset actions, stopping.")
                            break
                    else:
                        mc_actions = sample_mc_actions(
                            model,
                            obs,
                            task,
                            n_samples=int(FLAGS.mc_dropout_samples),
                            base_rng=jax.random.PRNGKey(np.random.randint(1e6)),
                        )
                        action_mean = jnp.mean(mc_actions, axis=0)
                        action_std = jnp.std(mc_actions, axis=0)
                        step_uncertainties.append(float(jnp.mean(action_std)))
                        policy_chunk = np.asarray(action_mean, dtype=np.float32)  # shape (H, action_dim)


                        # print("--- DEBUG STEP ---")
                        # print(f"1. Raw Model Output (Normalized): {policy_chunk[0]}")
                        # print("------------------")

                        # If you were manually unnormalizing, what would it look like?
                        # Let's assume your 'real' max range is small, as you hinted.
                        # If your physical normalization bounds were [min_val, max_val]:
                        # min_val = ... # fill this in (e.g., -0.5 meters?)
                        # max_val = ... # fill this in (e.g., +0.5 meters?)
                        # manual_fix = (raw_action + 1) / 2 * (max_val - min_val) + min_val
                        # print(f"2. Theoretical Manual Fix: {manual_fix[0]}")
                       

                        # Temporarily STOP the robot to read the numbers without breaking hardware
                        # import sys; sys.exit()
                        
                    # policy_chunk: shape (H, 7) normalized deltas (same representation as dataset actions)
                    delta_norm = policy_chunk[0][:7].astype(np.float32)

                    if step_count < 10:
                        src = "DATASET" if use_dataset_now else "POLICY"
                        a = delta_norm
                        print(f"[{src} ACTION APPLY]", step_count,
                            "a_norm min/max:", float(a.min()), float(a.max()),
                            "dq(rad) min/max:", float((a*0.02).min()), float((a*0.02).max()),
                            flush=True)
   

                    obs, reward, done, trunc, info = env.step(delta_norm)
                    t_sec += outer_dt
                    last_info = info
                    # Collect EE-to-goal distances at the finest granularity we can find.
                    # Depending on wrappers, per-inner-step infos may be in info["infos"] (best),
                    # otherwise we fall back to a single scalar in info["ee_goal_dist"].
                    if isinstance(info, dict):
                        if "infos" in info and isinstance(info["infos"], (list, tuple)):
                            for ii in info["infos"]:
                                if isinstance(ii, dict) and "ee_goal_dist" in ii:
                                    d = ii["ee_goal_dist"]
                                    if isinstance(d, (list, tuple, np.ndarray)):
                                        for x in d:
                                            distance_curve.append(float(x))
                                    else:
                                        distance_curve.append(float(d))
                        elif "ee_goal_dist" in info:
                            d = info["ee_goal_dist"]
                            if isinstance(d, (list, tuple, np.ndarray)):
                                for x in d:
                                    distance_curve.append(float(x))
                            else:
                                distance_curve.append(float(d))

                    ee_goal_dist = info.get("ee_goal_dist", None) if isinstance(info, dict) else None
                    if isinstance(ee_goal_dist, (list, tuple, np.ndarray)):
                        ee_goal_dist = float(np.asarray(ee_goal_dist).reshape(-1)[-1])
                    if ee_goal_dist is None and distance_curve:
                        ee_goal_dist = float(distance_curve[-1])
                    if ee_goal_dist is not None and np.isfinite(ee_goal_dist):
                        final_ee_goal_dist = ee_goal_dist
                        if step_count % 10 == 0:
                            print(
                                f"[DIST] step={step_count} ee_goal_dist={ee_goal_dist:.6f}",
                                flush=True,
                            )
                        if min_ee_goal_dist is None or ee_goal_dist < min_ee_goal_dist:
                            min_ee_goal_dist = ee_goal_dist
                        if min_ee_goal_dist is not None and ee_goal_dist > (min_ee_goal_dist + float(AWAY_MARGIN_M)):
                            away_steps += 1
                        else:
                            away_steps = 0
                        ever_within_success = (min_ee_goal_dist is not None) and (min_ee_goal_dist <= float(SUCCESS_DIST_M))
                        if (not ever_within_success) and away_steps >= int(AWAY_STEPS):
                            info["terminated_by_away"] = True
                            terminated_by_away = True
                            print(
                                f"[EVAL TERM] t={step_count} terminated_by_away "
                                f"dist={ee_goal_dist:.4f} min_dist={float(min_ee_goal_dist):.4f}",
                                flush=True,
                            )
                            break

                    step_count += 1


                    # refresh per-step diagnostics from the real env
                    base = env.unwrapped
                    panda = base.panda_env
                    q_cmd = None if getattr(panda, "_last_q_target_cmd", None) is None else panda._last_q_target_cmd.copy()
                    grip_cmd = getattr(panda, "_last_grip_cmd", None)
                    freeze_active = bool(getattr(panda, "_freeze_active", False))

                    trace_logger.log_chunk(
                        episode_index=episode_index,
                        step=step_count,
                        language_instruction="DEBUG" if FLAGS.debug_use_dataset_actions else "Policy",
                        delta_norm=delta_norm,
                        q_target_cmd=q_cmd,
                        grip_cmd=grip_cmd,
                        freeze_active=freeze_active,
                        post_proprio=obs["proprio"][-1],
                        reward=reward,
                        done=done,
                        trunc=trunc,
                    )




                    
                    # Always record the latest image from the observation window.
                    # With HistoryWrapper(horizon=2), last element is "current".
                    img = obs["image_primary"][-1] if obs["image_primary"].ndim == 4 else obs["image_primary"]
                    images.append(img)

                    episode_return += reward
                    # print("[STOP CHECK] step_count=", step_count, "len(actions)=", len(gt_actions_queue), flush=True)

                    if done or trunc:
                        term_tick = int(round(t_sec / log_dt))
                        print(
                            f"[EP END] step={step_count} done={done} trunc={trunc} "
                            f"dist={info.get('ee_goal_dist', None)} "
                            f"is_success={info.get('is_success', None)} "
                            f"terminated_by_freeze={info.get('terminated_by_freeze', False)} "
                            f"success_streak={getattr(env.unwrapped.panda_env, '_success_streak', None)} "
                            f"freeze_active={getattr(env.unwrapped.panda_env, '_freeze_active', None)} "
                            f"freeze_left={getattr(env.unwrapped.panda_env, '_freeze_steps_left', None)}",
                            flush=True,
                        )
                        break

                
                print(f"Episode return: {episode_return}")
                episode_uncertainty_mean = (
                    float(np.mean(step_uncertainties)) if step_uncertainties else 0.0
                )
                episode_uncertainty_max = (
                    float(np.max(step_uncertainties)) if step_uncertainties else float("nan")
                )
                episode_success = detect_success(min_ee_goal_dist)
                want_success = episode_success and kept_success < target_success
                want_failure = (not episode_success) and kept_failure < target_failure
                kept = want_success or want_failure

                attempt_rows.append(
                    {
                        "attempt_index": attempt_idx,
                        "traj_idx": traj_idx,
                        "seed": current_seed,
                        "success": bool(episode_success),
                        "kept": bool(kept),
                        "episode_return": float(episode_return),
                        "episode_uncertainty_max": episode_uncertainty_max,
                        "steps_executed": step_count,
                    }
                )

                if kept:
                    if episode_success:
                        kept_success += 1
                    else:
                        kept_failure += 1
                    kept_index = kept_success + kept_failure - 1
                    outcome = "success" if episode_success else "failure"
                    kept_rows.append(
                        {
                            "kept_index": kept_index,
                            "attempt_index": attempt_idx,
                            "traj_idx": traj_idx,
                            "seed": current_seed,
                            "outcome": outcome,
                            "episode_return": float(episode_return),
                            "episode_uncertainty_max": episode_uncertainty_max,
                            "min_ee_goal_dist": float(min_ee_goal_dist)
                            if min_ee_goal_dist is not None
                            else float("nan"),
                            "final_ee_goal_dist": float(final_ee_goal_dist)
                            if final_ee_goal_dist is not None
                            else float("nan"),
                            "steps_executed": step_count,
                            "terminated_by_away": bool(terminated_by_away),
                            "terminated_by_success_hold": bool(
                                last_info.get("terminated_by_success_hold", False)
                            )
                            if isinstance(last_info, dict)
                            else False,
                        }
                    )
                    if episode_success:
                        success_rows.append(
                            {
                                "episode_return": float(episode_return),
                                "episode_uncertainty_max": episode_uncertainty_max,
                            }
                        )
                    else:
                        failure_rows.append(
                            {
                                "episode_return": float(episode_return),
                                "episode_uncertainty_max": episode_uncertainty_max,
                            }
                        )
                    print(
                        "[KEPT] "
                        f"kept_success={kept_success}/{target_success} "
                        f"kept_failure={kept_failure}/{target_failure}",
                        flush=True,
                    )
                    episode_uncertainties.append(episode_uncertainty_mean)
                    episode_successes.append(bool(episode_success))
                    episode_conditions.append(condition_name)
                    episode_returns.append(float(episode_return))

                logging.info(
                    "Episode %d metrics: return=%.3f success=%s uncertainty=%.6f",
                    episode_index,
                    episode_return,
                    episode_success,
                    episode_uncertainty_mean,
                )
                num_dataset_steps = min(step_count, FLAGS.policy_start_timestep)
                num_policy_steps = max(0, step_count - FLAGS.policy_start_timestep)

                print(
                    f"[ROLLOUT SUMMARY] steps={step_count} | "
                    f"dataset={num_dataset_steps} | policy={num_policy_steps}",
                    flush=True,
                )
                # Where to write plots: alongside the trace file if provided, else CWD.
                plot_dir = os.path.dirname(FLAGS.debug_trace_path) if FLAGS.debug_trace_path else "."
                plot_path = os.path.join(plot_dir, f"distance_curve_ep{episode_index:03d}.png")
                dump_distance_curve(
                    distance_curve,
                    plot_path,
                    success_dist=SUCCESS_DIST_M,
                    title=f"Episode {episode_index} | seed={current_seed}"
                )

                # Also dump raw numeric series for downstream ablations
                np.save(os.path.join(plot_dir, f"distance_curve_ep{episode_index:03d}.npy"),
                        np.asarray(distance_curve, dtype=np.float32))

                if unc_ticks and unc_values:
                    plt.figure(figsize=(8, 4))
                    ax = plt.gca()
                    ax.plot(unc_ticks, unc_values, color="tab:blue")
                    if term_tick is not None:
                        ax.axvline(term_tick, color="tab:red", linestyle="--", linewidth=1.0)
                    ax.set_xlabel(f"Controller ticks ({FLAGS.uncertainty_hz:.0f} Hz)")
                    ax.set_ylabel("MC Dropout uncertainty (mean action std)")
                    ax.set_title(
                        f"Uncertainty @ {FLAGS.uncertainty_hz:.0f} Hz vs controller steps"
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, "tomato_uncertainty_50hz_vs_steps.png"))
                    plt.close()
                    np.save(
                        os.path.join(plot_dir, f"uncertainty_50hz_ep{episode_index:03d}.npy"),
                        np.asarray(unc_values, dtype=np.float32),
                    )
                    np.save(
                        os.path.join(plot_dir, f"uncertainty_50hz_ticks_ep{episode_index:03d}.npy"),
                        np.asarray(unc_ticks, dtype=np.int32),
                    )

                trace_logger.log_chunk(
                    event="episode_end",
                    episode_index=episode_index,
                    episode_return=episode_return,
                    episode_success=bool(episode_success),
                    episode_uncertainty=episode_uncertainty_mean,
                    condition=condition_name,
                )

                if use_wandb:
                    wandb.log({
                        "rollout/episode_index": episode_index,
                        "rollout/episode_return": float(episode_return),
                        "rollout/episode_success": int(bool(episode_success)),
                        "rollout/episode_uncertainty": float(episode_uncertainty_mean),
                    })

                if use_wandb and FLAGS.log_video_to_wandb:
                    wandb.log(
                        {
                            "rollout_video": wandb.Video(
                                np.array(images).transpose(0, 3, 1, 2)[::2]
                            )
                        }
                    )

                attempt_idx += 1
        finally:
            trace_logger.close()

        if kept_success < target_success or kept_failure < target_failure:
            print(
                "[WARN] Max attempts reached before quotas met: "
                f"successes={kept_success}/{target_success}, "
                f"failures={kept_failure}/{target_failure}, attempts={attempt_idx}",
                flush=True,
            )

        kept_path = os.path.join(output_dir, "kept_episodes.csv")
        attempts_path = os.path.join(output_dir, "attempts.csv")

        with open(kept_path, "w", newline="") as kept_file:
            writer = csv.DictWriter(
                kept_file,
                fieldnames=[
                    "kept_index",
                    "attempt_index",
                    "traj_idx",
                    "seed",
                    "outcome",
                    "episode_return",
                    "episode_uncertainty_max",
                    "min_ee_goal_dist",
                    "final_ee_goal_dist",
                    "steps_executed",
                    "terminated_by_away",
                    "terminated_by_success_hold",
                ],
            )
            writer.writeheader()
            for row in kept_rows:
                writer.writerow(row)

        with open(attempts_path, "w", newline="") as attempts_file:
            writer = csv.DictWriter(
                attempts_file,
                fieldnames=[
                    "attempt_index",
                    "traj_idx",
                    "seed",
                    "success",
                    "kept",
                    "episode_return",
                    "episode_uncertainty_max",
                    "steps_executed",
                ],
            )
            writer.writeheader()
            for row in attempt_rows:
                writer.writerow(row)

        scatter_path = os.path.join(output_dir, "uncertainty_scatter.png")
        success_returns = [row["episode_return"] for row in success_rows]
        success_uncert = [row["episode_uncertainty_max"] for row in success_rows]
        failure_returns = [row["episode_return"] for row in failure_rows]
        failure_uncert = [row["episode_uncertainty_max"] for row in failure_rows]

        plt.figure(figsize=(7, 5))
        plt.scatter(success_returns, success_uncert, label="success")
        plt.scatter(failure_returns, failure_uncert, label="failure")
        plt.xlabel("episode_return")
        plt.ylabel("episode_uncertainty_max")
        plt.title("Episode Uncertainty (Max) vs Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()

        success_mean = (
            float(np.nanmean(success_uncert)) if success_uncert else float("nan")
        )
        failure_mean = (
            float(np.nanmean(failure_uncert)) if failure_uncert else float("nan")
        )

        print(f"[OUTPUT] output_dir={output_dir}", flush=True)
        print(f"[OUTPUT] kept_episodes.csv={kept_path}", flush=True)
        print(f"[OUTPUT] attempts.csv={attempts_path}", flush=True)
        print(f"[OUTPUT] uncertainty_scatter.png={scatter_path}", flush=True)
        print(
            f"[COUNTS] successes={kept_success} failures={kept_failure} attempts={attempt_idx}",
            flush=True,
        )
        print(
            "[MEAN UNCERTAINTY] success=%.6f failure=%.6f"
            % (success_mean, failure_mean),
            flush=True,
        )

        if episode_uncertainties:
            plot_dir = os.path.dirname(FLAGS.debug_trace_path) if FLAGS.debug_trace_path else "."
            condition_names = sorted(set(episode_conditions))
            color_map = plt.get_cmap("tab10")
            condition_colors = {
                name: color_map(i % color_map.N) for i, name in enumerate(condition_names)
            }

            plt.figure(figsize=(8, 4))
            ax = plt.gca()
            for idx, (uncertainty, success, condition) in enumerate(
                zip(episode_uncertainties, episode_successes, episode_conditions)
            ):
                marker = "o" if success else "x"
                ax.scatter(
                    idx + 1,
                    uncertainty,
                    marker=marker,
                    color=condition_colors.get(condition, "black"),
                )
            ax.set_xlabel("Episode Index")
            ax.set_ylabel("Per-Episode Uncertainty")
            ax.set_title("MC Dropout Uncertainty vs Episode Outcome")
            ax.set_ylim(bottom=0)

            condition_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=condition_colors[name],
                    label=name,
                    markersize=7,
                    linestyle="None",
                )
                for name in condition_names
            ]
            outcome_handles = [
                Line2D([0], [0], marker="o", color="k", label="Success", linestyle="None"),
                Line2D([0], [0], marker="x", color="k", label="Failure", linestyle="None"),
            ]
            if condition_handles:
                legend1 = ax.legend(handles=condition_handles, title="Condition", loc="upper right")
                ax.add_artist(legend1)
            ax.legend(handles=outcome_handles, title="Outcome", loc="lower right")

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "tomato_mc_uncertainty_scatter.png"))
            plt.close()

            boxplot_data = []
            boxplot_labels = []
            for condition in condition_names:
                success_vals = [
                    u for u, s, c in zip(episode_uncertainties, episode_successes, episode_conditions)
                    if c == condition and s
                ]
                failure_vals = [
                    u for u, s, c in zip(episode_uncertainties, episode_successes, episode_conditions)
                    if c == condition and not s
                ]
                if success_vals:
                    boxplot_labels.append(f"{condition} / Success")
                    boxplot_data.append(success_vals)
                if failure_vals:
                    boxplot_labels.append(f"{condition} / Failure")
                    boxplot_data.append(failure_vals)

            if boxplot_data:
                plt.figure(figsize=(10, 4))
                plt.boxplot(boxplot_data, labels=boxplot_labels, showfliers=False)
                plt.ylabel("Per-Episode Uncertainty")
                plt.title("MC Dropout Uncertainty by Outcome")
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "tomato_mc_uncertainty_boxplot.png"))
                plt.close()
        print("Finished MuJoCo eval")
            
    # -------------------------------------------------------------------------
    # MODE 2: PURE DATASET EVAL (No Physics)
    # -------------------------------------------------------------------------
    else:
        def count_trajectories(ds) -> int:
            count = 0
            for _ in tfds.as_numpy(ds):
                count += 1
            return count

        num_traj = count_trajectories(raw_ds)
        logging.info("Found %d trajectories in split=%s", num_traj, FLAGS.dataset_split)

        def sample_trajectory_indices(num_traj: int, num_episodes: int, seed: int):
            rng = np.random.default_rng(seed)
            num_episodes = min(num_episodes, num_traj)
            return rng.choice(num_traj, size=num_episodes, replace=False)

        if FLAGS.num_dataset_episodes <= 1:
            traj_indices = [FLAGS.dataset_trajectory_index]
        else:
            traj_indices = sample_trajectory_indices(
                num_traj=num_traj,
                num_episodes=FLAGS.num_dataset_episodes,
                seed=FLAGS.dataset_random_seed,
            )
            logging.info(
                "Randomly selected dataset trajectory indices: %s", traj_indices
            )

        def window_indices(t: int, window_size: int) -> np.ndarray:
            start = max(0, t - window_size + 1)
            idx = np.arange(start, t + 1, dtype=np.int32)
            if idx.shape[0] < window_size:
                pad = np.full((window_size - idx.shape[0],), idx[0], dtype=np.int32)
                idx = np.concatenate([pad, idx], axis=0)
            return idx

        mse_all_values = []

        for ep_idx, traj_idx in enumerate(traj_indices):
            logging.info(
                "Evaluating dataset trajectory %d/%d (index=%d)",
                ep_idx + 1,
                len(traj_indices),
                traj_idx,
            )

            traj = get_trajectory_by_index(raw_ds, int(traj_idx))

            lang_arr = traj["language_instruction"]
            if isinstance(lang_arr[0], bytes):
                language_instruction = lang_arr[0].decode("utf-8")
            else:
                language_instruction = str(lang_arr[0])
            task = model.create_tasks(texts=language_instruction)
            if isinstance(task.get("language_instruction"), dict):
                attn_mask = task["language_instruction"].get("attention_mask")
                if attn_mask is not None:
                    task["pad_mask_dict"]["language_instruction"] = np.asarray(
                        np.any(attn_mask, axis=-1), dtype=bool
                    )
            example_task = model.example_batch["task"]
            fixed_task = dict(task)
            pad_mask_dict = dict(fixed_task.get("pad_mask_dict", {}))
            for k, v in example_task.items():
                if k in ("pad_mask_dict", "language_instruction"):
                    continue
                target_shape = (1, *np.asarray(v).shape[1:])
                fixed_task[k] = np.zeros(target_shape, dtype=np.asarray(v).dtype)
                pad_mask_dict[k] = np.ones((target_shape[0],), dtype=bool)
            if "language_instruction" in fixed_task and isinstance(
                fixed_task["language_instruction"], dict
            ):
                if "language_instruction" not in pad_mask_dict:
                    pad_mask_dict["language_instruction"] = np.ones((1,), dtype=bool)
            fixed_task["pad_mask_dict"] = pad_mask_dict
            task = fixed_task

            H = 2
            pred_actions = []
            gt_actions = []
            T = min(
                traj["observation"]["image_primary"].shape[0],
                FLAGS.dataset_max_steps,
            )

            for t in range(T):
                idx = window_indices(t, H)
                obs_window = {
                    "image_primary": traj["observation"]["image_primary"][idx],
                    "proprio": traj["observation"]["proprio"][idx],
                    "timestep": traj["observation"]["timestep"][idx],
                }
                if "image_wrist" in traj["observation"]:
                    obs_window["image_wrist"] = traj["observation"]["image_wrist"][idx]

                obs_window["proprio"] = (
                    obs_window["proprio"] - PROPRIO_MEAN
                ) / np.maximum(PROPRIO_STD, 1e-6)

                if (
                    "image_wrist" in obs_window
                    and obs_window["image_wrist"].shape[1] != 128
                ):
                    obs_window["image_wrist"] = obs_window["image_wrist"][:, ::2, ::2, :]

                pad_mask_dict = {
                    "image_primary": np.ones((H,), dtype=bool),
                    "proprio": np.ones((H,), dtype=bool),
                    "timestep": np.ones((H,), dtype=bool),
                }
                if "image_wrist" in obs_window:
                    pad_mask_dict["image_wrist"] = np.ones((H,), dtype=bool)
                obs_window["pad_mask_dict"] = pad_mask_dict
                obs_window["timestep_pad_mask"] = np.ones((H,), dtype=bool)
                obs_window["task_completed"] = np.zeros((H, 4), dtype=np.float32)

                obs_batched = jax.tree_map(lambda x: x[None], obs_window)
                policy_out = policy_fn(obs_batched, task)
                policy_chunk = np.asarray(policy_out[0], dtype=np.float32)


                a_pred = policy_chunk[0]
                pred_actions.append(a_pred)
                gt_actions.append(traj["action"][t])

            preds = np.stack(pred_actions, axis=0)
            gts = np.stack(gt_actions, axis=0)
            mse_per_joint = np.mean((preds - gts) ** 2, axis=0)
            mse_all = float(np.mean((preds - gts) ** 2))
            mse_all_values.append(mse_all)

            logging.info(
                "Dataset rollout #%d on split=%s traj_idx=%d T=%d -> "
                "MSE_all=%.4f, MSE_per_joint=%s",
                ep_idx + 1,
                FLAGS.dataset_split,
                traj_idx,
                T,
                mse_all,
                mse_per_joint,
            )
            if use_wandb:
                wandb.log(
                    {
                        f"dataset_rollout/ep{ep_idx}_mse_all": mse_all,
                        **{
                            f"dataset_rollout/ep{ep_idx}_mse_joint_{j}": float(m)
                            for j, m in enumerate(mse_per_joint)
                        },
                    }
                )

                images = traj["observation"]["image_primary"][:T]
                if FLAGS.log_video_to_wandb:
                    wandb.log(
                        {
                            f"dataset_rollout_video/ep{ep_idx}": wandb.Video(
                                images.transpose(0, 3, 1, 2)[::2]
                            )
                        }
                    )

        if mse_all_values and use_wandb:
            wandb.log(
                {
                    "dataset_rollout/mse_all_mean_over_eps": float(
                        np.mean(mse_all_values)
                    )
                }
            )
        print(
            f"Finished dataset rollout on split={FLAGS.dataset_split}, "
            f"{len(traj_indices)} trajectories"
        )


if __name__ == "__main__":
    app.run(main)
