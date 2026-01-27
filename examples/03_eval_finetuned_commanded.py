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
import sys
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Ensure we use EGL for rendering (same as replay script)
os.environ.setdefault("MUJOCO_GL", "egl") 
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

DEBUG_TERM = True
DEBUG = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from absl import app, flags, logging
import gym
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from typing import Optional

# Explicitly import mujoco
import mujoco
from record_dataset.helpers import find_side_stem_targets

# =============================================================================
# TUNABLE THRESHOLDS (edit these instead of hunting through the file)
# =============================================================================
# Success/termination: EE-to-goal distance (meters) that counts as success.
# If this is too large, you may terminate before the gripper ever starts closing.
SUCCESS_DIST_M = 0.02

# Gripper closing trigger: start closing when EE is within this distance to goal.
# Oracle-style default was 0.006 (6mm), but 0.01–0.03 is often more robust.
GRIPPER_CLOSE_TRIGGER_DIST_M = 0.024
GRIPPER_RAMP_STEPS = 25

GRIP_OPEN_NORM = 1.0     # normalized "open"
GRIP_CLOSE_NORM = 0.0   # normalized "closed"

FREEZE_DIST_M = SUCCESS_DIST_M        # start holding pose when within this distance
FREEZE_HOLD_STEPS = 20        # hold for N outer env steps once triggered
POST_SUCCESS_HOLD_STEPS = 20
EPISODES_PER_PLANT = 2
# Away-termination tuning: trigger if distance exceeds best-so-far by margin for N steps.
AWAY_MARGIN_M = 0.05
AWAY_STEPS = 8
# =============================================================================


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
    None,
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
    True, 
    "If True, ignores the model policy and feeds ground-truth dataset actions into the sim."
)

flags.DEFINE_integer(
    "policy_start_timestep",
    0,
    "In MuJoCo rollout mode, use dataset actions for t < policy_start_timestep, "
    "then switch to the policy for t >= policy_start_timestep. "
    "Ignored if debug_use_dataset_actions=True."
)
flags.DEFINE_bool(
    "debug_compare_policy_to_dataset",
    True,
    "If true, log comparisons between policy and dataset actions per step.",
)
flags.DEFINE_integer(
    "debug_compare_steps",
    50,
    "Number of timesteps to log with detailed comparisons.",
)
flags.DEFINE_bool(
    "debug_save_debug_npz",
    True,
    "If true, save debug_rollout.npz with per-step rollout diagnostics.",
)
flags.DEFINE_string(
    "debug_out_dir",
    None,
    "Output directory for debug plots/npz. Defaults to dirname(debug_trace_path) or '.'.",
)
flags.DEFINE_enum(
    "debug_action_semantics",
    "command",
    ["command", "measured"],
    "Action semantics for A/B testing: 'command' integrates q_cmd; 'measured' uses q_meas + dq.",
)



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
        for k in ("image_primary", "image_wrist", "image_goal_primary", "image_goal_wrist", "proprio", "timestep"):
            if k in obs:
                pmd.setdefault(k, tmask.copy())

        obs["pad_mask_dict"] = pmd

        # 3) task_completed: dummy zeros unless you have a real completion signal
        if "task_completed" not in obs:
            obs["task_completed"] = np.zeros((H, self.task_completed_dim), dtype=np.float32)

        return obs

class GoalImageObsWrapper(gym.ObservationWrapper):
    """Inject goal images into the observation stream (history-aware)."""

    def __init__(self, env: gym.Env, goal_primary: Optional[np.ndarray], goal_wrist: Optional[np.ndarray]):
        super().__init__(env)
        self._goal_primary = self._prep_goal_image(goal_primary, size=(256, 256))
        self._goal_wrist = self._prep_goal_image(goal_wrist, size=(128, 128))

    @staticmethod
    def _prep_goal_image(img: Optional[np.ndarray], size: tuple[int, int]) -> Optional[np.ndarray]:
        if img is None:
            return None
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"goal image must be HxWx3 uint8, got {arr.shape}")
        if (arr.shape[0], arr.shape[1]) != size:
            arr = np.array(Image.fromarray(arr).resize(size, Image.BILINEAR))
        return arr

    def observation(self, obs):
        # HistoryWrapper produces [H, ...]; repeat goal images over H.
        H = obs["proprio"].shape[0] if "proprio" in obs and obs["proprio"].ndim >= 1 else 1
        if self._goal_primary is not None:
            obs["image_goal_primary"] = np.repeat(self._goal_primary[None, ...], H, axis=0)
        if self._goal_wrist is not None:
            obs["image_goal_wrist"] = np.repeat(self._goal_wrist[None, ...], H, axis=0)

        pmd = dict(obs.get("pad_mask_dict", {}))
        tmask = obs.get("timestep_pad_mask", None)
        if tmask is None:
            tmask = np.ones((H,), dtype=bool)
        else:
            tmask = np.asarray(tmask).astype(bool)
        if "image_goal_primary" in obs:
            pmd.setdefault("image_goal_primary", tmask.copy())
        if "image_goal_wrist" in obs:
            pmd.setdefault("image_goal_wrist", tmask.copy())
        obs["pad_mask_dict"] = pmd
        return obs

# --- OVERRIDE CLASS FOR DIRECT CONTROL ---
class DirectControlPandaEnv(PandaTomatoSimEnv):
    """
    Direct-control wrapper that matches oracle policy semantics:

    - action is interpreted as a 7D normalized delta
    - clamp to [-1, 1], then scale by ORACLE_SCALE to radians
    - target = measured_qpos + dq (no integrator drift)
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

        # 1. FIND GRIPPER ACTUATOR ID
        self.gripper_act_id = find_gripper_actuator(self.model)

        self.arm_act_ids, self.arm_qpos_adr, self.arm_dof_idx = build_arm_mapping_from_model(
            self.model, prefer_position=True
        )
        widen_arm_limits(self.model, self.arm_dof_idx)

        # Substeps to match ~0.1s per action
        dt = float(self.model.opt.timestep)
        self.substeps = int(max(1, round(self.TARGET_ACTION_DT / max(dt, 1e-6))))

        self._debug_step_count = 0
        self._step_count = 0

        # Renderer (keep your existing approach)
        self._renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.success_dist = SUCCESS_DIST_M
        self.success_hold_steps = 3
        self._success_streak = 0
        self._success_latched = False
        self._post_success_hold_left = 0
        self._post_success_hold_total = int(POST_SUCCESS_HOLD_STEPS)
        self.ee_ref = _choose_ee_ref(self.model)
        print(f"[EE_REF] using {self.ee_ref}", flush=True)
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
            print(
                f"[GRIPPER] act_id={self.gripper_act_id} name='{gname}' ctrlrange=({self._grip_close_cmd},{self._grip_open_cmd})",
                flush=True,
            )
        else:
            self._grip_close_cmd = 0.0
            self._grip_open_cmd  = 0.0
            print("[GRIPPER] No gripper actuator found (gripper_act_id=-1).", flush=True)
        self._last_q_target_cmd = None
        self._last_grip_cmd = None  # optional, but useful
        self._done_latched = False




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
        # ------------------------------------------------------------------
        # 0) Episode bookkeeping / termination + success latch state
        # ------------------------------------------------------------------
        self._success_streak = 0
        self._done_latched = False
        self._success_latched = False

        self._post_success_hold_total = int(POST_SUCCESS_HOLD_STEPS)
        self._post_success_hold_left = 0

        self._step_count = 0
        self._dbg_printed = 0
        self._dbg_dist_post = 0
        self._debug_step_count = 0
        
        # ------------------------------------------------------------------
        # 1) Standard reset (clears physics state)
        # ------------------------------------------------------------------
        obs, info = super().reset(**kwargs)

        # ------------------------------------------------------------------
        # 2) FORCE DATASET START POSE (deterministic)
        # ------------------------------------------------------------------
        start_joints = np.array(
            [0.0, -0.4948, 0.0, -1.5172, 0.0, 1.4902, 0.0],
            dtype=np.float64,
        )
        self.data.qpos[self.arm_qpos_adr] = start_joints
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # ------------------------------------------------------------------
        # 3) Build cache: qpos_adr -> joint id (once)
        # ------------------------------------------------------------------
        if not hasattr(self, "_arm_qadr_to_jid"):
            qadr_to_jid = {}
            for jid in range(self.model.njnt):
                qadr_to_jid[int(self.model.jnt_qposadr[jid])] = jid
            self._arm_qadr_to_jid = qadr_to_jid

        # ------------------------------------------------------------------
        # 4) Sync controller command space
        # ------------------------------------------------------------------
        q0 = start_joints.astype(np.float32, copy=False)
        self._q_cmd = q0.copy()
        self._q_target = self._q_cmd.copy()
        self._last_q_target_cmd = q0.copy()
        self._last_dq_cmd = np.zeros(7, dtype=np.float32)

        # ------------------------------------------------------------------
        # 5) Reset gripper state machine
        # ------------------------------------------------------------------
        self._grip_state = "open"
        self._grip_ramp_step = 0
        self._grip_norm = float(GRIP_OPEN_NORM)

        if getattr(self, "gripper_act_id", -1) >= 0:
            try:
                self.data.ctrl[self.gripper_act_id] = float(
                    self._grip_cmd_from_norm(self._grip_norm)
                )
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 6) Re-capture observation
        # ------------------------------------------------------------------
        obs = self.get_obs()

        info = dict(info) if info is not None else {}
        info["reset_q_start"] = q0.copy()
        info["is_success"] = False
        return obs, info


    def _goal_pos_world(self):
        goal_pos = getattr(self, "goal_pos", None)
        if goal_pos is None:
            return None
        return np.asarray(goal_pos, dtype=np.float64).reshape(3)


    def step(self, action):
            # --- 0) Parse + validate ---
            self._step_count += 1

            a = np.asarray(action, dtype=np.float32)
            if a.ndim == 2:
                a = a[0]
            a = a.reshape(-1)
            if a.shape[0] < 7:
                raise ValueError(f"Expected action with >=7 dims, got shape {a.shape}")

            # --- 1) Build cache: qpos_adr -> joint id (once) ---
            if not hasattr(self, "_arm_qadr_to_jid"):
                qadr_to_jid = {}
                for jid in range(self.model.njnt):
                    qadr_to_jid[int(self.model.jnt_qposadr[jid])] = jid
                self._arm_qadr_to_jid = qadr_to_jid

            # --- 2) SCALING & CLAMPING (The Fix) ---
            # Do NOT clamp normalized action to [-1, 1]. Allow policy overdrive.
            # This preserves the direction vector if the policy outputs [2.0, 0.5] vs [1.0, 0.25].
            a_raw = a[:7].astype(np.float64) 
            
            # Convert to radians (0.02 scale)
            dq_in = a_raw * float(self.ORACLE_SCALE)

            # Apply Safety Clamping in RADIANS (Physical Limits)
            # We allow 0.05 rad (~3 deg) per step, which is looser than the dataset (0.02)
            # to ensure the policy can correct errors without being clipped artificially.
            SAFE_LIMIT_RAD = 0.05 
            dq_in = np.clip(dq_in, -SAFE_LIMIT_RAD, SAFE_LIMIT_RAD)

            # --- 3) If success latched: HOLD (ignore action) ---
            if getattr(self, "_success_latched", False):
                # Hold the frozen target, not the current position
                q_target = getattr(self, "_q_hold", self._q_cmd).astype(np.float64, copy=False)
                dq_applied = np.zeros_like(dq_in)
            else:
                # --- INTEGRATOR LOGIC ---
                # 1. Take the previous COMMANDED target (ignoring physics lag)
                q_prev_cmd = self._q_cmd.astype(np.float64)
                
                # 2. Add the delta
                q_cmd = q_prev_cmd + dq_in

                # 3. Clamp to joint limits (safety only)
                for i, qadr in enumerate(self.arm_qpos_adr):
                    jid = self._arm_qadr_to_jid.get(int(qadr), None)
                    if jid is None:
                        continue
                    if int(self.model.jnt_limited[jid]) == 1:
                        lo, hi = self.model.jnt_range[jid]
                        q_cmd[i] = np.clip(q_cmd[i], lo, hi)

                # 4. Update the persistent command state for the next step
                self._q_cmd = q_cmd.astype(np.float32)
                q_target = q_cmd.astype(np.float64, copy=False)
                dq_applied = dq_in

            # Log the actually commanded target + delta
            self._last_q_target_cmd = q_target.astype(np.float32, copy=False).copy()
            self._last_dq_cmd = dq_applied.astype(np.float32, copy=False).copy()

            if getattr(self, "_dbg_printed", 0) < 10:
                self._dbg_printed = getattr(self, "_dbg_printed", 0) + 1
                print(
                    f"[ENV-B] a_norm(min,max)=({float(a_raw.min()):.3f},{float(a_raw.max()):.3f}) "
                    f"dq(rad)(min,max)=({float(dq_applied.min()):.3e},{float(dq_applied.max()):.3e}) "
                    f"||dq||={float(np.linalg.norm(dq_applied)):.3e}",
                    flush=True,
                )

            # ----------------------------------------------------------------------
            # EE-to-goal distance (world frame), used for gripper schedule
            # ----------------------------------------------------------------------
            ee_pos = self._ee_pos_world()
            goal_pos_world = self._goal_pos_world()
            dist_to_goal = None
            if ee_pos is not None and goal_pos_world is not None:
                dist_to_goal = float(np.linalg.norm(ee_pos - goal_pos_world))

            # ----------------------------------------------------------------------
            # Gripper schedule (Logic based on dist_to_goal)
            # ----------------------------------------------------------------------
            if self.gripper_act_id >= 0:
                if self._grip_state == "open":
                    if dist_to_goal is not None and dist_to_goal <= float(GRIPPER_CLOSE_TRIGGER_DIST_M):
                        self._grip_state = "ramping"
                        self._grip_ramp_step = 0
                    self._grip_norm = GRIP_OPEN_NORM
                elif self._grip_state == "ramping":
                    t = float(self._grip_ramp_step) / float(max(1, GRIPPER_RAMP_STEPS - 1))
                    self._grip_norm = float(np.clip(1.0 - t, 0.0, 1.0))
                    self._grip_ramp_step += 1
                    if self._grip_ramp_step >= int(GRIPPER_RAMP_STEPS):
                        self._grip_state = "closed"
                        self._grip_norm = GRIP_CLOSE_NORM
                else:
                    self._grip_norm = GRIP_CLOSE_NORM

                grip_cmd = self._grip_cmd_from_norm(self._grip_norm)
            else:
                grip_cmd = None

            # --- 4) Step MuJoCo: absolute target control ---
            self._last_grip_cmd = None if grip_cmd is None else float(grip_cmd)

            for _ in range(int(self.substeps)):
                self.data.ctrl[self.arm_act_ids] = q_target
                if grip_cmd is not None:
                    self.data.ctrl[self.gripper_act_id] = float(grip_cmd)
                mujoco.mj_step(self.model, self.data)

            # --- 5) Post-step Evaluation ---
            dist_post = None
            ee_pos_post = self._ee_pos_world()
            goal_pos_world_post = self._goal_pos_world()
            if ee_pos_post is not None and goal_pos_world_post is not None:
                dist_post = float(np.linalg.norm(ee_pos_post - goal_pos_world_post))

            terminated = False
            truncated = False
            info = {"ee_goal_dist": dist_post}

            # Success latching logic
            if not hasattr(self, "_success_streak"): self._success_streak = 0
            if not hasattr(self, "_success_latched"): self._success_latched = False
            
            if dist_post is not None and not self._success_latched:
                if dist_post <= float(SUCCESS_DIST_M):
                    self._success_streak += 1
                else:
                    self._success_streak = 0

                if self._success_streak >= int(getattr(self, "success_hold_steps", 3)):
                    self._success_latched = True
                    self._post_success_hold_left = int(POST_SUCCESS_HOLD_STEPS)
                    self._q_hold = q_target.copy() # Freeze current integrated target
                    self._grip_state = "closed"
                    self._grip_norm = GRIP_CLOSE_NORM

            info["is_success"] = bool(self._success_latched)

            if self._success_latched:
                if self._post_success_hold_left > 0:
                    self._post_success_hold_left -= 1
                else:
                    terminated = True
                    info["terminated_by_success_hold"] = True

            obs = self.get_obs()
            reward = 1.0 if info["is_success"] else 0.0

            return obs, reward, terminated, truncated, info
    
    def _goal_pos_world(self):
        goal_pos = getattr(self, "goal_pos", None)
        if goal_pos is not None:
            return np.asarray(goal_pos, dtype=np.float64).reshape(3)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_mocap")
        if bid >= 0:
            return self.data.xpos[bid].copy()
        return None


    def get_obs(self):
            # ... (image rendering code remains the same) ...
            self._renderer.update_scene(self.data, camera="front_cam")
            image_primary = self._renderer.render()
            
            # ... (wrist camera code remains the same) ...
            try:
                self._renderer.update_scene(self.data, camera="gripper_cam")
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
                # CHANGE: The dataset expects a value that normalizes to ~-0.79.
                # Forcing 0.04 (Open) resulted in ~8.26. 
                # Try 0.0 for the start val to align with the dataset's 'ds' mean.
                gripper_start_val = 0.00  # Adjust this to reduce the 'diff' in your logs
                gripper_closed_val = 0.0
                
                g = float(np.clip(getattr(self, "_grip_norm", GRIP_OPEN_NORM), 0.0, 1.0))
                # If starting at 0.0, ensure your linear interpolation reflects that
                gripper_val = float(gripper_closed_val + g * (gripper_start_val - gripper_closed_val))
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

def _extract_is_success(obs, info) -> bool:
    if isinstance(info, dict):
        if "is_success" in info:
            try:
                return bool(info["is_success"])
            except Exception:
                pass
        if "infos" in info and isinstance(info["infos"], (list, tuple)):
            for ii in info["infos"]:
                if isinstance(ii, dict) and "is_success" in ii:
                    try:
                        if bool(ii["is_success"]):
                            return True
                    except Exception:
                        continue
    if isinstance(obs, dict) and "is_success" in obs:
        try:
            return bool(np.any(np.asarray(obs["is_success"]) > 0))
        except Exception:
            return False
    return False

def _extract_step_type(info, done: bool, trunc: bool):
    if isinstance(info, dict):
        if "step_type" in info:
            return info["step_type"]
        if "infos" in info and isinstance(info["infos"], (list, tuple)) and info["infos"]:
            last = info["infos"][-1]
            if isinstance(last, dict) and "step_type" in last:
                return last["step_type"]
    if done:
        return "LAST"
    if trunc:
        return "TRUNC"
    return "MID"

def _hold_steps_used(env) -> int | None:
    base = env.unwrapped
    panda = getattr(base, "panda_env", None)
    if panda is None:
        panda = getattr(base, "_env", None)
    if panda is None:
        return None
    if hasattr(panda, "_post_success_hold_total"):
        total = int(getattr(panda, "_post_success_hold_total", 0))
        left = int(getattr(panda, "_post_success_hold_left", 0))
        return max(0, total - left)
    if hasattr(panda, "_freeze_steps_left"):
        total = int(globals().get("FREEZE_HOLD_STEPS", 0))
        left = int(getattr(panda, "_freeze_steps_left", 0))
        if total > 0:
            return max(0, total - left)
        return 0
    return None

def _resolve_debug_out_dir(debug_out_dir: Optional[str], debug_trace_path: Optional[str]) -> str:
    if debug_out_dir:
        return debug_out_dir
    if debug_trace_path:
        base = os.path.dirname(debug_trace_path)
        return base if base else "."
    return "."

def _min_ee_goal_dist(info: dict) -> float:
    if not isinstance(info, dict):
        return float("nan")
    if "infos" in info and isinstance(info["infos"], (list, tuple)):
        vals = []
        for ii in info["infos"]:
            if isinstance(ii, dict) and "ee_goal_dist" in ii:
                d = ii["ee_goal_dist"]
                if isinstance(d, (list, tuple, np.ndarray)):
                    vals.extend([float(x) for x in d])
                else:
                    vals.append(float(d))
        if vals:
            return float(np.min(vals))
    if "ee_goal_dist" in info:
        d = info["ee_goal_dist"]
        if isinstance(d, (list, tuple, np.ndarray)):
            return float(np.min(np.asarray(d, dtype=np.float32)))
        return float(d)
    return float("nan")

def main(_):
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
        import random 
        num_eval_episodes = 1 

        top_targets = None
        current_seed = None
        dynamic_xml_path = None

        for episode_index in range(num_eval_episodes):
            plant_index = episode_index // EPISODES_PER_PLANT
            episode_within_plant = episode_index % EPISODES_PER_PLANT

            traj_idx = int(FLAGS.dataset_trajectory_index)

            goal_primary_img = None
            goal_wrist_img = None
            ds_actions = None
            ds_proprio0 = None
            gt_actions_queue = []

            traj_for_eval = get_trajectory_by_index(raw_ds, traj_idx)
            if isinstance(traj_for_eval, dict) and "observation" in traj_for_eval:
                goal_primary_img = traj_for_eval["observation"].get("goal_image_primary")
                goal_wrist_img = traj_for_eval["observation"].get("goal_image_wrist")
                if goal_primary_img is not None and len(goal_primary_img) > 0:
                    goal_primary_img = np.asarray(goal_primary_img[0], dtype=np.uint8)
                if goal_wrist_img is not None and len(goal_wrist_img) > 0:
                    goal_wrist_img = np.asarray(goal_wrist_img[0], dtype=np.uint8)
                if "action" in traj_for_eval:
                    ds_actions = np.asarray(traj_for_eval["action"], dtype=np.float32)
                if "proprio" in traj_for_eval["observation"]:
                    ds_proprio0 = np.asarray(traj_for_eval["observation"]["proprio"][0], dtype=np.float32)

            need_dataset_actions = FLAGS.debug_use_dataset_actions or (FLAGS.policy_start_timestep > 0)
            if need_dataset_actions:
                logging.warning(
                    f"Loading dataset actions for episode {episode_index} "
                    f"(debug_use_dataset_actions={FLAGS.debug_use_dataset_actions}, "
                    f"policy_start_timestep={FLAGS.policy_start_timestep})"
                )
                gt_actions_queue = list(traj_for_eval["action"])
                logging.info(f"Loaded {len(gt_actions_queue)} steps of ground-truth actions.")

            if episode_within_plant == 0:
                if FLAGS.match_dataset_scene:
                    current_seed = traj_idx + plant_index
                    logging.info(
                        f"MATCHING DATASET: Regenerating scene for traj_idx={traj_idx} (seed={current_seed})"
                    )
                else:
                    current_seed = np.random.randint(0, 100000)
                    logging.info(f"DYNAMIC: Generating new random scene (Seed {current_seed})")

                np.random.seed(current_seed)
                random.seed(current_seed)
                print(f"[SEED CHECK] traj_idx={traj_idx} seed={current_seed}", flush=True)

            # 3. REGENERATE SCENE XML
            if episode_within_plant == 0:
                sim_env_module.START_JOINTS = np.asarray(DATASET_START_JOINTS, dtype=float)
                model_xml_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
                try:
                    from record_dataset.helpers import build_and_load_scene, scene_dir_from_model_path
                    build_and_load_scene(str(model_xml_path))
                    scene_dir = Path(scene_dir_from_model_path(str(model_xml_path)))
                    dynamic_xml_path = scene_dir / "scene_dynamic.xml"
                except ImportError:
                    raise RuntimeError("record_dataset.helpers not found")
            if dynamic_xml_path is None:
                raise RuntimeError("dynamic_xml_path not set; scene generation failed")

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

            if episode_within_plant == 0 or top_targets is None:
                top_targets = find_side_stem_targets(
                    panda_env.model,
                    panda_env.data,
                    k=EPISODES_PER_PLANT,
                    s=0.5,
                )
                if DEBUG:
                    print(f"[TARGETS] top={top_targets}", flush=True)
            if not top_targets:
                raise RuntimeError("find_side_stem_targets returned empty; scene naming mismatch?")

            stem_name, goal_pos_world, approach_xy, truss_name = top_targets[
                episode_within_plant % len(top_targets)
            ]
            goal_pos_world = np.asarray(goal_pos_world, dtype=np.float64)
            if DEBUG:
                print(
                    f"[TARGET] stem={stem_name} goal_pos_world={goal_pos_world}",
                    flush=True,
                )

            panda_env.set_goal_pos(goal_pos_world)
            panda_env.goal_pos = goal_pos_world
            bid_goal = mujoco.mj_name2id(panda_env.model, mujoco.mjtObj.mjOBJ_BODY, "goal_mocap")
            if bid_goal >= 0:
                mocap_id = int(panda_env.model.body_mocapid[bid_goal])
                if mocap_id >= 0:
                    panda_env.data.mocap_pos[mocap_id] = goal_pos_world
                    mujoco.mj_forward(panda_env.model, panda_env.data)

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
            env = GoalImageObsWrapper(env, goal_primary=goal_primary_img, goal_wrist=goal_wrist_img)
            env = EnsureOctoObsKeysWrapper(env)     
            env = RHCWrapper(env, exec_horizon=1)


            
            # 5. RUN ROLLOUT
            base = env.unwrapped
            panda = base.panda_env

            q_cmd = None if getattr(panda, "_last_q_target_cmd", None) is None else panda._last_q_target_cmd.copy()
            grip_cmd = getattr(panda, "_last_grip_cmd", None)
            freeze_active = bool(getattr(panda, "_freeze_active", False))

            trace_logger = ChunkLogger(FLAGS.debug_trace_path)
            out_dir = _resolve_debug_out_dir(FLAGS.debug_out_dir, FLAGS.debug_trace_path)
            os.makedirs(out_dir, exist_ok=True)
            compare_steps = int(FLAGS.debug_compare_steps)
            do_compare = bool(FLAGS.debug_compare_policy_to_dataset)
            p_diff_top_idx = None
            p_diff_top_vals = None
            p_diff_env = None
            p_diff_ds = None
            try:
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
                is_success_now = bool(info.get("is_success", False))
                traj_idx = int(FLAGS.dataset_trajectory_index)
                traj = get_trajectory_by_index(raw_ds, traj_idx)

                ds = traj["observation"]["image_primary"][0].astype(np.float32)
                sim = (obs["image_primary"][-1] if obs["image_primary"].ndim == 4 else obs["image_primary"]).astype(np.float32)

                print("[IMG DIFF t=0] mean_abs =", float(np.mean(np.abs(ds - sim))), flush=True)

                # --- TRUE PHYSICS CHECK ---
                # Access the raw MuJoCo data object hidden under the wrappers
                raw_physics_qpos = env.unwrapped.panda_env.data.qpos[env.unwrapped.panda_env.arm_qpos_adr]

                print("\n" + "="*30)
                print("--- TRUE PHYSICS CHECK (Raw Radians) ---")
                print(f"Goal Start Pose: {np.round(DATASET_START_JOINTS, 4)}")
                print(f"Actual Robot:    {np.round(raw_physics_qpos, 4)}")
                print(f"Difference:      {np.round(raw_physics_qpos - DATASET_START_JOINTS, 4)}")

                # Check Gripper too (Should be exactly 0.04 or 0.0 depending on your reset logic)
                # Note: reset() sets 0.04 (Open), get_obs() forces 0.0 (Closed) for the model.
                # The physics should likely be open (0.04) or closed (0.0) depending on the gripper command.
                # But checking the 7 joints is enough to prove the Start Pose Fix.
                print("="*30 + "\n")
                # -------------------------------

                if ds_proprio0 is not None:
                    p_env = obs["proprio"][-1] if obs["proprio"].ndim == 2 else obs["proprio"]
                    n = min(ds_proprio0.shape[0], PROPRIO_MEAN.shape[0], p_env.shape[0])
                    p_ds_norm = (ds_proprio0[:n] - PROPRIO_MEAN[:n]) / np.maximum(PROPRIO_STD[:n], 1e-6)
                    p_env_cmp = p_env[:n]
                    diff = np.abs(p_env_cmp - p_ds_norm)
                    idx = np.argsort(diff)[-3:]
                    p_diff_top_idx = idx.tolist()
                    p_diff_top_vals = diff[idx].tolist()
                    p_diff_env = p_env_cmp[idx].tolist()
                    p_diff_ds = p_ds_norm[idx].tolist()
                    print(
                        "[PROPRIO DIFF] max dims:",
                        idx.tolist(),
                        np.round(diff[idx], 6),
                        "env:",
                        np.round(p_env_cmp[idx], 6),
                        "ds:",
                        np.round(p_ds_norm[idx], 6),
                        flush=True,
                    )

                task = model.create_tasks(texts=env.get_task()["language_instruction"])
                first = obs["image_primary"][-1] if obs["image_primary"].ndim == 4 else obs["image_primary"]
                images = [first]

                episode_return = 0.0
                step_count = 0
                last_step_type = None
                last_ee_goal_dist = None
                min_ee_goal_dist = None
                away_steps = 0

                # --- DEBUG ARRAYS ---
                t_list = []
                ee_dist_list = []
                a_policy_list = []
                a_dataset_list = []
                dq_rad_policy_list = []
                dq_rad_dataset_list = []
                q_meas_list = []
                q_cmd_list = []
                track_err_list = []
                is_success_list = []
                while len(images) < 400:
                    
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
                        batched_obs = jax.tree_map(lambda x: x[None], obs)
                        policy_out = policy_fn(batched_obs, task)
                        policy_chunk = np.asarray(policy_out[0], dtype=np.float32)  # shape (H, action_dim)


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
                        
                    # policy_chunk: shape (H, 7) normalized deltas (same as dataset actions)
                    delta_norm = policy_chunk[0][:7].astype(np.float32)

                    a_pi = None
                    a_ds = None
                    if use_dataset_now:
                        a_ds = delta_norm.copy()
                        if ds_actions is not None and step_count < len(ds_actions):
                            a_ds = ds_actions[step_count][:7].astype(np.float32)
                    else:
                        a_pi = delta_norm.copy()
                        if ds_actions is not None and step_count < len(ds_actions):
                            a_ds = ds_actions[step_count][:7].astype(np.float32)

                    if step_count < 10:
                        src = "DATASET" if use_dataset_now else "POLICY"
                        a = delta_norm
                        print(f"[{src} ACTION APPLY]", step_count,
                            "a_norm min/max:", float(a.min()), float(a.max()),
                            f"dq(rad) min/max: {float((a*0.02).min()):.3e}/{float((a*0.02).max()):.3e}",
                            flush=True)
   

                    obs, reward, done, trunc, info = env.step(delta_norm)
                    #debug
                    panda = env.unwrapped.panda_env
                    ee = panda._ee_pos_world()
                    goal_world = panda.goal_pos
                    if goal_world is not None:
                        dist_true = np.linalg.norm(ee - goal_world)
                    else:
                        dist_true = float("nan")
                    print("[DIST TRUE]", step_count, dist_true, "grip_state", panda._grip_state, "grip_norm", panda._grip_norm)

                   
                    is_success_now = bool(info.get("is_success", False))
                    last_step_type = _extract_step_type(info, done, trunc)
                    if DEBUG_TERM:
                        should_log = (step_count % 20 == 0) or done or trunc
                        if should_log:
                            print(
                                f"[EVAL TERM] t={step_count} done={done} trunc={trunc} "
                                f"is_success_now={int(is_success_now)} last_step_type={last_step_type}",
                                flush=True,
                            )
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

                    ee_goal_dist = _min_ee_goal_dist(info)
                    if np.isnan(ee_goal_dist) and distance_curve:
                        ee_goal_dist = float(distance_curve[-1])
                    if np.isfinite(ee_goal_dist):
                        if min_ee_goal_dist is None or ee_goal_dist < min_ee_goal_dist:
                            min_ee_goal_dist = ee_goal_dist
                        if min_ee_goal_dist is not None and ee_goal_dist > (min_ee_goal_dist + float(AWAY_MARGIN_M)):
                            away_steps += 1
                        else:
                            away_steps = 0
                        last_ee_goal_dist = ee_goal_dist
                        ever_within_success = (min_ee_goal_dist is not None) and (min_ee_goal_dist <= float(SUCCESS_DIST_M))
                        if (not ever_within_success) and away_steps >= int(AWAY_STEPS):
                            info["terminated_by_away"] = True
                            print(
                                f"[EVAL TERM] t={step_count} terminated_by_away "
                                f"dist={ee_goal_dist:.4f} min_dist={float(min_ee_goal_dist):.4f}",
                                flush=True,
                            )
                            break

                    # --- PER-STEP DEBUG LOGGING ---
                    q_meas = env.unwrapped.panda_env.data.qpos[env.unwrapped.panda_env.arm_qpos_adr].copy()
                    q_cmd = getattr(env.unwrapped.panda_env, "_last_q_target_cmd", None)
                    if q_cmd is not None:
                        q_cmd = q_cmd.copy()
                    track_err = float(np.linalg.norm(q_cmd - q_meas)) if q_cmd is not None else float("nan")

                    if a_pi is None:
                        a_pi_arr = np.full((7,), np.nan, dtype=np.float32)
                    else:
                        a_pi_arr = a_pi.astype(np.float32)
                    if a_ds is None:
                        a_ds_arr = np.full((7,), np.nan, dtype=np.float32)
                    else:
                        a_ds_arr = a_ds.astype(np.float32)

                    dq_pi = a_pi_arr * 0.02
                    dq_ds = a_ds_arr * 0.02

                    t_list.append(int(step_count))
                    ee_dist_list.append(float(ee_goal_dist))
                    a_policy_list.append(a_pi_arr)
                    a_dataset_list.append(a_ds_arr)
                    dq_rad_policy_list.append(dq_pi)
                    dq_rad_dataset_list.append(dq_ds)
                    q_meas_list.append(q_meas.astype(np.float32))
                    q_cmd_list.append(q_cmd.astype(np.float32) if q_cmd is not None else np.full((7,), np.nan, dtype=np.float32))
                    track_err_list.append(float(track_err))
                    is_success_list.append(bool(is_success_now))

                    if do_compare and ((step_count < compare_steps) or (step_count % 10 == 0)):
                        src = "DATASET" if use_dataset_now else "POLICY"
                        a_pi_norm = float(np.linalg.norm(dq_pi)) if np.isfinite(dq_pi).all() else float("nan")
                        a_ds_norm = float(np.linalg.norm(dq_ds)) if np.isfinite(dq_ds).all() else float("nan")
                        pi_min = float(np.nanmin(a_pi_arr)) if np.isfinite(a_pi_arr).any() else float("nan")
                        pi_max = float(np.nanmax(a_pi_arr)) if np.isfinite(a_pi_arr).any() else float("nan")
                        ds_min = float(np.nanmin(a_ds_arr)) if np.isfinite(a_ds_arr).any() else float("nan")
                        ds_max = float(np.nanmax(a_ds_arr)) if np.isfinite(a_ds_arr).any() else float("nan")
                        print(
                            f"[HEALTH t={step_count:03d} {src}] "
                            f"pi(min/max)={pi_min:.3f}/{pi_max:.3f} "
                            f"ds(min/max)={ds_min:.3f}/{ds_max:.3f} "
                            f"||dq_pi||={a_pi_norm:.3e} ||dq_ds||={a_ds_norm:.3e} "
                            f"pi[:3]={np.round(a_pi_arr[:3],3)} ds[:3]={np.round(a_ds_arr[:3],3)} "
                            f"track_err={track_err:.3e} ee_dist={ee_goal_dist:.3f}",
                            flush=True,
                        )


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

                    if done or trunc: break
                
                print(f"Episode return: {episode_return}")
                hold_steps_used = _hold_steps_used(env)
                is_success_end = bool(is_success_list[-1]) if is_success_list else False
                print(
                    f"[EP END] done={done} trunc={trunc} "
                    f"is_success_now={int(is_success_end)} last_step_type={last_step_type} "
                    f"hold_steps_used={hold_steps_used if hold_steps_used is not None else -1}",
                    flush=True,
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

                # --- SAVE DEBUG NPZ + PLOTS ---
                t_arr = np.asarray(t_list, dtype=np.int32)
                ee_arr = np.asarray(ee_dist_list, dtype=np.float32)
                a_pi_arr = np.stack(a_policy_list, axis=0) if a_policy_list else np.zeros((0, 7), dtype=np.float32)
                a_ds_arr = np.stack(a_dataset_list, axis=0) if a_dataset_list else np.zeros((0, 7), dtype=np.float32)
                dq_pi_arr = np.stack(dq_rad_policy_list, axis=0) if dq_rad_policy_list else np.zeros((0, 7), dtype=np.float32)
                dq_ds_arr = np.stack(dq_rad_dataset_list, axis=0) if dq_rad_dataset_list else np.zeros((0, 7), dtype=np.float32)
                q_meas_arr = np.stack(q_meas_list, axis=0) if q_meas_list else np.zeros((0, 7), dtype=np.float32)
                q_cmd_arr = np.stack(q_cmd_list, axis=0) if q_cmd_list else np.zeros((0, 7), dtype=np.float32)
                track_err_arr = np.asarray(track_err_list, dtype=np.float32)
                is_success_arr = np.asarray(is_success_list, dtype=np.bool_)

                if FLAGS.debug_save_debug_npz:
                    np.savez(
                        os.path.join(out_dir, "debug_rollout.npz"),
                        t=t_arr,
                        ee_goal_dist=ee_arr,
                        a_policy=a_pi_arr,
                        a_dataset=a_ds_arr,
                        dq_rad_policy=dq_pi_arr,
                        dq_rad_dataset=dq_ds_arr,
                        q_meas=q_meas_arr,
                        q_cmd=q_cmd_arr,
                        track_err=track_err_arr,
                        is_success=is_success_arr,
                    )

                # Plot 1: distance vs step
                if ee_arr.size > 0:
                    plt.figure()
                    plt.plot(t_arr, ee_arr)
                    plt.axhline(SUCCESS_DIST_M, linestyle="--")
                    plt.xlabel("Step")
                    plt.ylabel("EE-to-goal distance (m)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, "distance_vs_step.png"), dpi=150)
                    plt.close()

                # Plot 2: tracking error vs step
                if track_err_arr.size > 0:
                    plt.figure()
                    plt.plot(t_arr, track_err_arr)
                    plt.xlabel("Step")
                    plt.ylabel("||q_cmd - q_meas||")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, "track_err_vs_step.png"), dpi=150)
                    plt.close()

                # Plot 3: action norm vs step (policy vs dataset)
                if dq_pi_arr.size > 0 or dq_ds_arr.size > 0:
                    plt.figure()
                    if dq_pi_arr.size > 0:
                        plt.plot(t_arr, np.linalg.norm(dq_pi_arr, axis=1), label="policy")
                    if dq_ds_arr.size > 0:
                        plt.plot(t_arr, np.linalg.norm(dq_ds_arr, axis=1), label="dataset")
                    plt.xlabel("Step")
                    plt.ylabel("||dq|| (rad)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, "action_norm_vs_step.png"), dpi=150)
                    plt.close()

                # Plot 4: per-joint overlay for first 2 joints
                if dq_pi_arr.size > 0 and dq_ds_arr.size > 0:
                    plt.figure()
                    plt.plot(t_arr, dq_pi_arr[:, 0], label="pi_j0")
                    plt.plot(t_arr, dq_ds_arr[:, 0], label="ds_j0")
                    plt.plot(t_arr, dq_pi_arr[:, 1], label="pi_j1")
                    plt.plot(t_arr, dq_ds_arr[:, 1], label="ds_j1")
                    plt.xlabel("Step")
                    plt.ylabel("dq (rad)")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, "per_joint_action_overlay.png"), dpi=150)
                    plt.close()

                # --- FINAL SUMMARY ---
                ee_valid = ee_arr[np.isfinite(ee_arr)]
                final_dist = float(ee_valid[-1]) if ee_valid.size else float("nan")
                min_dist = float(np.min(ee_valid)) if ee_valid.size else float("nan")
                min_dist_step = int(t_arr[int(np.argmin(ee_valid))]) if ee_valid.size else -1

                pi_norm = np.linalg.norm(dq_pi_arr, axis=1) if dq_pi_arr.size else np.asarray([], dtype=np.float32)
                ds_norm = np.linalg.norm(dq_ds_arr, axis=1) if dq_ds_arr.size else np.asarray([], dtype=np.float32)
                pi_norm_valid = pi_norm[np.isfinite(pi_norm)]
                ds_norm_valid = ds_norm[np.isfinite(ds_norm)]
                pi_norm_mean = float(np.mean(pi_norm_valid)) if pi_norm_valid.size else float("nan")
                pi_norm_median = float(np.median(pi_norm_valid)) if pi_norm_valid.size else float("nan")
                ds_norm_mean = float(np.mean(ds_norm_valid)) if ds_norm_valid.size else float("nan")
                norm_ratio = (pi_norm_mean / ds_norm_mean) if np.isfinite(ds_norm_mean) and ds_norm_mean > 0 else float("nan")

                track_valid = track_err_arr[np.isfinite(track_err_arr)]
                track_mean = float(np.mean(track_valid)) if track_valid.size else float("nan")
                track_median = float(np.median(track_valid)) if track_valid.size else float("nan")

                sat_mask = np.isfinite(a_pi_arr).all(axis=1) if a_pi_arr.size else np.zeros((0,), dtype=bool)
                sat_pct = float(
                    np.mean(np.any(np.abs(a_pi_arr[sat_mask]) > 0.95, axis=1))
                ) if sat_mask.any() else float("nan")

                print("\n[FINAL SUMMARY]")
                print(f"final_dist={final_dist:.4f} min_dist={min_dist:.4f} min_dist_step={min_dist_step}")
                print(
                    f"policy_action_norm mean/median={pi_norm_mean:.3e}/{pi_norm_median:.3e} "
                    f"ratio_vs_dataset_mean={norm_ratio:.3f}"
                )
                print(f"track_err mean/median={track_mean:.3e}/{track_median:.3e}")
                print(f"policy_action_saturation_pct(|a|>0.95)={sat_pct:.2%}")
                if p_diff_top_idx is not None:
                    print(
                        f"proprio_t0_topdiff idx={p_diff_top_idx} "
                        f"diff={np.round(p_diff_top_vals,6)} "
                        f"env={np.round(p_diff_env,6)} "
                        f"ds={np.round(p_diff_ds,6)}"
                    )

                wandb.log({"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])})

            finally:
                trace_logger.close()
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
            wandb.log(
                {
                    f"dataset_rollout_video/ep{ep_idx}": wandb.Video(
                        images.transpose(0, 3, 1, 2)[::2]
                    )
                }
            )

        if mse_all_values:
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
