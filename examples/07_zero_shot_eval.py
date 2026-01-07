"""
Zero-shot evaluation for PRETRAINED Octo on the existing Tomato MuJoCo environment.

Key differences vs 03_eval_finetuned.py:
- Loads pretrained model: hf://rail-berkeley/octo-small-1.5 (no finetuned_path)
- Removes dataset-action replay and dataset-only evaluation modes
- Runs N MuJoCo rollouts and reports success rate

This script reuses your existing environment construction and wrappers, including:
- DirectControlPandaEnv, TomatoGymEnv
- NormalizeProprio, ResizeImageWrapper, HistoryWrapper, EnsureOctoObsKeysWrapper, RHCWrapper
- Goal selection via get_side_stem_grasp_points
(from 03_eval_finetuned.py) :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

from functools import partial
import os
import sys
import random
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# CPU-only as in your script (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gym
import jax

from absl import app, flags, logging

import mujoco
import wandb

# --- Project path setup (as in your file) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep this to register ALOHA sim env / tomato env
sys.path.append("/home/myrtheiw/octo_ws/act")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from record_dataset.getlocation import get_side_stem_grasp_points

from envs.tomato_env import DEFAULT_MODEL_PATH, PandaTomatoSimEnv, TomatoGymEnv
from scripts import sim_env as sim_env_module

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper, ResizeImageWrapper
from octo.utils.train_callbacks import supply_rng


FLAGS = flags.FLAGS

# --- Evaluation controls ---
flags.DEFINE_integer("num_episodes", 20, "Number of MuJoCo episodes to evaluate.")
flags.DEFINE_integer("max_steps", 400, "Max steps per episode.")
flags.DEFINE_integer("seed", 0, "Base random seed.")
flags.DEFINE_integer(
    "log_video_every",
    1,
    "Log rollout video every N episodes (set <=0 to disable).",
)
flags.DEFINE_string(
    "dataset_stats_key",
    "bridge_dataset",
    "Key to select per-dataset statistics from pretrained Octo checkpoint.",
)
flags.DEFINE_bool(
    "match_dataset_scene",
    True,
    "If True, sets the random seed per episode index to recreate deterministic scenes "
    "(reusing your prior convention).",
)
flags.DEFINE_string(
    "debug_trace_path",
    None,
    "If set, writes a per-step JSONL trace. (Optional; keep your existing tooling if desired.)",
)

# Optional: use a fixed language instruction if you don't want env.get_task()
flags.DEFINE_string(
    "language_instruction",
    "Pick the tomato in the front",
    "If provided, overrides env.get_task()['language_instruction'].",
)

# --- Your tunable thresholds retained (optional) ---
SUCCESS_DIST_M = 0.07
GRIPPER_CLOSE_TRIGGER_DIST_M = 0.06
GRIPPER_RAMP_STEPS = 25
GRIP_OPEN_NORM = 1.0
GRIP_CLOSE_NORM = 0.0
FREEZE_DIST_M = SUCCESS_DIST_M
FREEZE_HOLD_STEPS = 20

# Dataset-start joints retained (used by DirectControlPandaEnv.reset)
DATASET_START_JOINTS = np.array(
    [0.0, -0.4948, 0.0, -1.5172, 0.0, 1.4902, 0.0], dtype=np.float64
)
sim_env_module.START_JOINTS = np.asarray(DATASET_START_JOINTS, dtype=float)

TYPICAL_FRANKA_LIMITS = np.array(
    [
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973),
    ],
    dtype=np.float64,
)


def find_gripper_actuator(model):
    """
    Finds the actuator index for the gripper with name + ctrlrange fallbacks.
    """
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if any(k in name.lower() for k in ("grip", "finger", "hand")):
            return aid

    try:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        if aid >= 0:
            return aid
    except Exception:
        pass

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

    chosen = [c for c in chosen if c[4] in arm_body_names]
    by_jid = {}
    for c in chosen:
        by_jid.setdefault(c[1], c)
    chosen = sorted(by_jid.values(), key=lambda t: t[2])

    if len(chosen) != 7:
        raise RuntimeError(f"Expected 7 arm actuators, found {len(chosen)} (chosen={chosen}).")

    arm_act_ids = np.array([c[0] for c in chosen], dtype=int)
    arm_qpos_adr = np.array([c[2] for c in chosen], dtype=int)

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
    try:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    except Exception:
        sid = -1
    if sid >= 0:
        return ("site", "tcp")
    return ("body", "hand")


def widen_arm_limits(model, arm_dof_idx):
    """Apply typical Franka limits to the arm joints."""
    for i, dof in enumerate(arm_dof_idx):
        jid = int(model.dof_jntid[dof])
        model.jnt_limited[jid] = 1
        lo, hi = TYPICAL_FRANKA_LIMITS[i]
        model.jnt_range[jid][0] = lo
        model.jnt_range[jid][1] = hi


# -----------------------------------------------------------------------------
# Minimal keep: EnsureOctoObsKeysWrapper (same logic as your file) :contentReference[oaicite:2]{index=2}
# -----------------------------------------------------------------------------
class EnsureOctoObsKeysWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, task_completed_dim: int = 4):
        super().__init__(env)
        self.task_completed_dim = int(task_completed_dim)

    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        tmask = obs.get("timestep_pad_mask", None)
        if tmask is None:
            H = obs["proprio"].shape[0]
            tmask = np.ones((H,), dtype=bool)
        else:
            tmask = np.asarray(tmask).astype(bool)
            H = tmask.shape[0]

        if "timestep" not in obs:
            obs["timestep"] = np.arange(H, dtype=np.int32)

        pmd = dict(obs.get("pad_mask_dict", {}))
        for k in ("image_primary", "image_wrist", "proprio", "timestep"):
            if k in obs:
                pmd.setdefault(k, tmask.copy())
        obs["pad_mask_dict"] = pmd

        if "task_completed" not in obs:
            obs["task_completed"] = np.zeros((H, self.task_completed_dim), dtype=np.float32)

        return obs


# -----------------------------------------------------------------------------
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
    EE_POS_SCALE_M = 0.05
    EE_ROT_SCALE_RAD = 0.25
    EE_DLS_LAMBDA = 1e-3
    EE_DQ_CLIP = 0.05
    TARGET_ACTION_DT = 0.1  # seconds per action (oracle used ~0.1s)

    def __init__(self, proprio_dim=14, action_mode: str = "ee_delta", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proprio_dim = proprio_dim
        self.action_mode = str(action_mode)
        if self.action_mode not in ("joint_delta", "ee_delta"):
            raise ValueError(f"Unknown action_mode '{self.action_mode}'")

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

        # Persistent integrated target (absolute joint angles)
        self._q_target = None

        self._debug_step_count = 0

        # Renderer (keep your existing approach)
        self._renderer = mujoco.Renderer(self.model, height=256, width=256)
        self.success_dist = SUCCESS_DIST_M
        self.success_hold_steps = 3
        self._success_streak = 0
        self.ee_ref = _choose_ee_ref(self.model)
        print(f"[EE_REF] using {self.ee_ref}", flush=True)
        self._ee_body_id = self._resolve_ee_body_id()
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
        self._freeze_active = False
        self._freeze_steps_left = 0
        self._freeze_q_target = None
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

    def _resolve_ee_body_id(self) -> int:
        kind, name = getattr(self, "ee_ref", ("site", "tcp"))
        body_id = -1
        if kind == "body":
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        elif kind == "site":
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                body_id = int(self.model.site_bodyid[sid])
        if body_id < 0:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        return int(body_id)

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
        action: array-like with >=7 dims.
        - action_mode="joint_delta": normalized joint delta in [-1, 1], scaled by ORACLE_SCALE
          and applied relative to *current measured q*.
        - action_mode="ee_delta": normalized EE delta in [-1, 1], scaled by EE_*_SCALE
          and converted to joint deltas via Jacobian DLS.

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

        # Keep arm dims and clamp to normalized range
        a7 = np.clip(a[:7], -1.0, 1.0).astype(np.float64, copy=False)

        q_meas = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float64)
        if self.action_mode == "joint_delta":
            dq = a7 * float(self.ORACLE_SCALE)  # radians
            q_target = q_meas + dq
        else:
            dpos = a7[:3] * float(self.EE_POS_SCALE_M)
            drot = a7[3:6] * float(self.EE_ROT_SCALE_RAD)
            if self._q_target is None:
                self._q_target = q_meas.copy()
            if int(self._ee_body_id) < 0:
                dq = np.zeros_like(q_meas)
            else:
                ee_pos = self.data.xpos[int(self._ee_body_id)].copy()
                ee_xmat = self.data.xmat[int(self._ee_body_id)].reshape(3, 3).copy()
                drot_world = ee_xmat @ drot
                dX = np.concatenate([dpos, drot_world], axis=0)

                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, int(self._ee_body_id))
                jacp = jacp[:, self.arm_dof_idx]
                jacr = jacr[:, self.arm_dof_idx]
                J = np.vstack([jacp, jacr])

                # Damped least squares IK: dq = J^T (J J^T + lambda I)^-1 dX
                JJt = J @ J.T
                lam = float(self.EE_DLS_LAMBDA)
                dq = J.T @ np.linalg.solve(JJt + lam * np.eye(6), dX)
                dq = np.clip(dq, -float(self.EE_DQ_CLIP), float(self.EE_DQ_CLIP))

                self._q_target = self._q_target + dq
            q_target = self._q_target

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
        if self.action_mode == "ee_delta":
            self._q_target = q_target.copy()

        # --- 5) Debug prints (safe) ---
        if getattr(self, "_dbg_printed", 0) < 10:
            self._dbg_printed = getattr(self, "_dbg_printed", 0) + 1
            print(
                f"[ENV-B/{self.action_mode}] a_norm(min,max)=({float(a7.min()):.3f},{float(a7.max()):.3f}) "
                f"dq(rad)(min,max)=({float(dq.min()):.3e},{float(dq.max()):.3e}) "
                f"||dq||={float(np.linalg.norm(dq)):.3e}",
                flush=True,
            )

        # --- 6) Step MuJoCo: absolute target control ---
        # ----------------------------------------------------------------------
        # EE-to-goal distance (world frame), used for freeze + gripper schedule
        # ----------------------------------------------------------------------
        goal_pos = getattr(self, "goal_pos", None)
        dist_to_goal = None
        if goal_pos is not None:
            goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
            ee_pos = self._ee_pos_world()
            if ee_pos is not None:
                dist_to_goal = float(np.linalg.norm(ee_pos - goal_pos))



        freeze_expired = False
        if getattr(self, "_freeze_active", False):
            if self._freeze_q_target is not None:
                q_target = self._freeze_q_target.copy()
            if self._freeze_steps_left > 0:
                self._freeze_steps_left -= 1
                if self._freeze_steps_left == 0:
                    freeze_expired = True

            # Recommended: keep freeze active until episode ends (do nothing here).

        # ----------------------------------------------------------------------
        # Gripper schedule: open until close-trigger, then ramp closed
        # ----------------------------------------------------------------------
        if self.gripper_act_id >= 0:
            if self._grip_state == "open":
                if dist_to_goal is not None and dist_to_goal <= float(GRIPPER_CLOSE_TRIGGER_DIST_M):
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

        # Freeze latch on POST-step distance
        if (not getattr(self, "_freeze_active", False)) and (dist_post is not None) and (dist_post <= float(FREEZE_DIST_M)):
            self._freeze_active = True
            self._freeze_steps_left = int(FREEZE_HOLD_STEPS)
            # Hold the *current measured* pose to prevent drift
            self._freeze_q_target = self.data.qpos[self.arm_qpos_adr].copy()
            print(f"[FREEZE] latch dist_post={dist_post:.4f}", flush=True)

        # --- 7) Success termination (distance-to-goal) ---
        terminated = False
        truncated = False
        info = {}

        # Use the distance you're testing
        success_dist = float(SUCCESS_DIST_M)

        # Initialize streak counter if missing
        if not hasattr(self, "_success_streak"):
            self._success_streak = 0

        # Default hold steps if missing
        success_hold_steps = int(getattr(self, "success_hold_steps", 3))

        # Only check if a goal was provided (now set via self.set_goal_pos(...))
        goal_pos = getattr(self, "goal_pos", None)
        if goal_pos is not None:
            goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)

            # Compute EE position from ee_ref (site/body), with fallbacks
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

            # Fallbacks if ee_ref lookup failed
            if ee_pos is None:
                sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
                if sid >= 0:
                    ee_pos = self.data.site_xpos[sid].copy()
            if ee_pos is None:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
                if bid >= 0:
                    ee_pos = self.data.xpos[bid].copy()

            if ee_pos is not None:
                dist = float(np.linalg.norm(ee_pos - goal_pos))
                info["ee_goal_dist"] = dist
                info["is_success"] = False  # explicit default

                if dist <= float(success_dist):
                    self._success_streak += 1
                else:
                    self._success_streak = 0

                if self._success_streak >= success_hold_steps:
                    terminated = True
                    info["is_success"] = True
                    self._done_latched = True

                    # Hold pose and keep gripper closed
                    self._freeze_active = True
                    self._freeze_steps_left = 0  # irrelevant now
                    self._freeze_q_target = self.data.qpos[self.arm_qpos_adr].copy()
                    self._grip_state = "closed"
                    self._grip_norm = GRIP_CLOSE_NORM

            else:
                # Could not resolve EE pose; do not terminate based on distance
                info["is_success"] = False
                info["ee_goal_dist"] = float("nan")
                self._success_streak = 0  # safest: don't carry streak
        else:
            info["is_success"] = False
            # leave ee_goal_dist absent unless you prefer info["ee_goal_dist"]=nan
            self._success_streak = 0
        
        if freeze_expired:
            terminated = True
            info["is_success"] = True  # or False if you want freeze-expire to be separate from success
            info["terminated_by_freeze"] = True


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
        self._renderer.update_scene(self.data, camera="front_cam")
        image_primary = self._renderer.render()

        try:
            self._renderer.update_scene(self.data, camera="wrist_cam")
            img_wrist_raw = self._renderer.render()
            image_wrist = img_wrist_raw[::2, ::2]
        except Exception:
            image_wrist = np.zeros((128, 128, 3), dtype=np.uint8)

        qpos_arm = self.data.qpos[self.arm_qpos_adr].copy().astype(np.float32)
        if self.proprio_dim > 7:
            gripper_open_val = 0.04
            gripper_closed_val = 0.0
            g = float(np.clip(getattr(self, "_grip_norm", GRIP_OPEN_NORM), 0.0, 1.0))
            gripper_val = float(gripper_closed_val + g * (gripper_open_val - gripper_closed_val))
            gripper_state = np.full((self.proprio_dim - 7,), gripper_val, dtype=np.float32)
            proprio = np.concatenate([qpos_arm[:7], gripper_state], axis=0)
        else:
            proprio = qpos_arm[: self.proprio_dim]

        return {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "proprio": proprio,
        }


def build_scene_xml() -> str:
    """
    Reuses your existing dynamic scene build path.
    Returns path to the generated scene_dynamic.xml.
    (Matches your eval script structure.) :contentReference[oaicite:3]{index=3}
    """
    scene_path = Path(
        "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene_ball.xml"
    )
    if not scene_path.exists():
        raise FileNotFoundError(f"scene_xml_path not found: {scene_path}")
    return str(scene_path)
    model_xml_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    try:
        from record_dataset.helpers import build_and_load_scene, scene_dir_from_model_path
        build_and_load_scene(str(model_xml_path))
        scene_dir = Path(scene_dir_from_model_path(str(model_xml_path)))
        dynamic_xml_path = scene_dir / "scene_dynamic.xml"
    except ImportError:
        raise RuntimeError("record_dataset.helpers not found; cannot build dynamic scene.")

    if not dynamic_xml_path.exists():
        raise FileNotFoundError(f"Expected dynamic scene at {dynamic_xml_path}, but not found.")
    return str(dynamic_xml_path)


def get_ball_goal_pos(model, data) -> np.ndarray:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_ball_geom")
    if geom_id < 0:
        raise RuntimeError("Ball geom 'red_ball_geom' not found in scene.")
    return data.geom_xpos[geom_id].copy()


def make_env(model: OctoModel, episode_index: int, dataset_stats: Optional[Dict[str, Any]]) -> gym.Env:
    """
    Builds your TomatoGymEnv with the same wrapper stack you used. :contentReference[oaicite:4]{index=4}
    """
    # Deterministic scene seeding (reuse your convention)
    if FLAGS.match_dataset_scene:
        current_seed = int(episode_index)
    else:
        current_seed = int(np.random.randint(0, 100000))

    np.random.seed(current_seed)
    random.seed(current_seed)
    logging.info("Episode %d seed=%d", episode_index, current_seed)

    # Regenerate dynamic XML
    dynamic_xml = build_scene_xml()

    # Detect proprio dim from per-dataset stats (or fallback to env default)
    expected_proprio_dim = None
    if isinstance(dataset_stats, dict):
        proprio_stats = dataset_stats.get("proprio")
        if isinstance(proprio_stats, dict) and "mean" in proprio_stats:
            proprio_mean = np.asarray(proprio_stats["mean"], dtype=np.float32)
            expected_proprio_dim = int(proprio_mean.shape[0])
            logging.info("Proprio dim from dataset stats: %d", expected_proprio_dim)
    if expected_proprio_dim is None:
        expected_proprio_dim = 14
        logging.info("Proprio dim fallback: %d", expected_proprio_dim)

    # Build base env
    panda_env = DirectControlPandaEnv(
        proprio_dim=expected_proprio_dim,
        model_xml=str(dynamic_xml),
        substeps=40,
        action_scale=1.0,
        action_mode="ee_delta",
    )
    panda_env.expects_absolute_action = True

    # Set goal from sim (as you do) :contentReference[oaicite:6]{index=6}
    mujoco.mj_forward(panda_env.model, panda_env.data)
    goal_pos_world = np.asarray(get_ball_goal_pos(panda_env.model, panda_env.data), dtype=np.float64)
    panda_env.set_goal_pos(goal_pos_world)

    env = TomatoGymEnv(panda_env, max_steps=int(FLAGS.max_steps))
    env.expects_absolute_action = True

    # Wrap like before :contentReference[oaicite:7]{index=7}
    if isinstance(dataset_stats, dict) and "proprio" in dataset_stats:
        env = NormalizeProprio(env, dataset_stats)
    env = ResizeImageWrapper(
        env,
        resize_size={"primary": (256, 256), "wrist": (128, 128)},
        avg_scale=1.0,
    )
    env = HistoryWrapper(env, horizon=2)
    env = EnsureOctoObsKeysWrapper(env)
    env = RHCWrapper(env, exec_horizon=1)
    return env


def get_instruction(env: gym.Env) -> str:
    if FLAGS.language_instruction.strip():
        return FLAGS.language_instruction.strip()
    try:
        task = env.get_task()
        if isinstance(task, dict) and "language_instruction" in task:
            return str(task["language_instruction"])
    except Exception:
        pass
    return "pick the tomato"


def evaluate_episode(
    model: OctoModel,
    env: gym.Env,
    policy_fn,
    episode_index: int,
    log_video: bool,
) -> Tuple[bool, float, int, Optional[np.ndarray]]:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info0 = out
    else:
        obs, info0 = out, {}

    instruction = get_instruction(env)
    task = model.create_tasks(texts=[instruction])

    ep_return = 0.0
    ep_len = 0
    success = False

    frames: Optional[List[np.ndarray]] = [] if log_video else None

    # Rollout
    for t in range(int(FLAGS.max_steps)):
        if frames is not None:
            frame = obs["image_primary"][-1]
            frames.append(np.asarray(frame, dtype=np.uint8))
        batched_obs = jax.tree_map(lambda x: x[None], obs)
        policy_out = policy_fn(batched_obs, task)
        policy_chunk = np.asarray(policy_out[0], dtype=np.float32)  # [chunk, action_dim]

        # IMPORTANT: your env expects a 7D delta_norm vector (it clips and scales internally) :contentReference[oaicite:8]{index=8}
        action = policy_chunk[0][:7].astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        ep_len += 1

        if isinstance(info, dict) and info.get("is_success", False):
            success = True

        if terminated or truncated:
            break

    logging.info(
        "Ep %d | len=%d return=%.3f success=%s instr='%s'",
        episode_index, ep_len, ep_return, success, instruction
    )
    frames_arr = None
    if frames is not None:
        frames_arr = np.stack(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return success, ep_return, ep_len, frames_arr


def main(_):
    wandb.init(name="eval_zero_shot_pretrained", project="octo")

    # --- Load PRETRAINED checkpoint (your requirement) ---
    logging.info("Loading pretrained model: hf://rail-berkeley/octo-small-1.5")
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

    dataset_stats = None
    if isinstance(model.dataset_statistics, dict):
        dataset_stats = model.dataset_statistics.get(FLAGS.dataset_stats_key)
        if dataset_stats is None:
            available = sorted(model.dataset_statistics.keys())
            raise KeyError(
                f"dataset_stats_key='{FLAGS.dataset_stats_key}' not found. "
                f"Available keys: {available}"
            )

    action_stats = None
    if isinstance(dataset_stats, dict):
        action_stats = dataset_stats.get("action")
        if action_stats is None:
            action_stats = dataset_stats.get("actions")
    if action_stats is None:
        logging.warning("No action stats for dataset_stats_key=%s; using raw model actions.", FLAGS.dataset_stats_key)

    # Policy sampler (same pattern as your script) :contentReference[oaicite:9]{index=9}
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=action_stats,
        ),
    )

    # Global seeding
    np.random.seed(int(FLAGS.seed))
    random.seed(int(FLAGS.seed))

    successes: List[float] = []
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(int(FLAGS.num_episodes)):
        env = make_env(model, episode_index=ep, dataset_stats=dataset_stats)
        log_video = int(FLAGS.log_video_every) > 0 and (ep % int(FLAGS.log_video_every) == 0)
        succ, ret, ln, frames = evaluate_episode(
            model, env, policy_fn, episode_index=ep, log_video=log_video
        )
        successes.append(1.0 if succ else 0.0)
        returns.append(float(ret))
        lengths.append(int(ln))

        wandb.log(
            {
                "eval/episode": ep,
                "eval/success": float(succ),
                "eval/return": float(ret),
                "eval/length": int(ln),
            }
        )
        if log_video and frames is not None and frames.shape[0] > 0:
            wandb.log(
                {
                    f"rollout_video/ep{ep}": wandb.Video(
                        frames.transpose(0, 3, 1, 2)[::2], fps=20, format="mp4"
                    )
                }
            )

    metrics = {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
    }

    print("\n=== Zero-shot Pretrained Octo Evaluation (Tomato Env) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    wandb.log({f"eval/{k}": v for k, v in metrics.items()})


if __name__ == "__main__":
    app.run(main)
