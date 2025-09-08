
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
import time
import random
import numpy as np
import mujoco
import dm_env
from dm_env import specs, TimeStep

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
    _choose_split_for_plant as choose_split_for_plant,  # new helper for train/val/test splits
)

# ------------------------------- Config ---------------------------------------

# Base model/scene: we will swap to a dynamic scene if USE_DYNAMIC_PLANT=True
MODEL_PATH = os.environ.get("MODEL_PATH", 
    "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"
)

# End-effector reference: prefer TCP site if present; else hand body
EE_REF = ("site", "tcp")   # will auto-fallback to ("body","hand") at runtime if tcp is missing

# Cameras
PRIMARY_CAM_NAME = "third_person_cam"
WRIST_CAM_NAME   = "gripper_cam"
IMG_H, IMG_W     = 256, 256

# Motion staging & pacing
START_JOINTS        = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=float)
PREGRASP_OFFSET     = 0.1   # meters along lateral normal (bigger is safer near foliage)
RETREAT_OFFSET      = 0.0     # meters opposite lateral normal (back out)
PREGRASP_DWELL_SEC = 1.0  # hold at pregrasp so the gripper can fully open

N_CART_WAYPOINTS    = 60       # resampled joint waypoints total (smoothness vs length)
WAYPOINT_REPEAT     = 2        # repeat each waypoint to slow motion (stability)
PAUSE_PREGRASP_STEPS= 40       # dwell at pregrasp while open
EXTRA_HOLD_STEPS    = 150      # hold final pose after retreat for logging stability

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
EPISODES_TOTAL     = int(os.environ.get("EPISODES_TOTAL", 200))
EPISODES_PER_PLANT = 2        # top-2 stems per plant, then regenerate

TFDS_ROOT_DIR   = os.environ.get("TFDS_ROOT_DIR", "/home/myrtheiw/tfds_out")
DATASET_NAME    = os.environ.get("DATASET_NAME", "tomato_rlds")
DATASET_VERSION = os.environ.get("DATASET_VERSION", "0.0.1")

# Debug
DEBUG_IK        = True
LIVE_RENDER     = False
LIVE_FPS        = 60.0
SHOW_DEBUG_VIZ  = False   # set True to drop mocap frames (pre/goal/retreat)

# --- Quality gate: only log successful episodes ---
MAX_FINAL_ERR = 0.010      # meters; tighten/loosen for your dataset quality
LIVE_RENDER_QC = False      # render the QC dry-run, like LIVE_RENDER
CAPTURE_IMAGES_DURING_QC = False  # speed up QC (don’t waste time rendering)


# --------------------------- Model/Actuator mapping ---------------------------

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
                 arm_dof_idx=None, kp=PD_KP, kd=PD_KD):
        self.model = model; self.data = data
        self.arm_act_ids = arm_act_ids
        self.arm_qpos_addr = arm_qpos_addr
        self.arm_dof_idx = np.asarray(arm_dof_idx, dtype=int) if arm_dof_idx is not None else None
        self.ee_ref = ee_ref
        self.lang = language_instruction
        self.substeps = int(substeps)
        self.gripper_idx = gripper_idx
        self._capture_images = True
        self.kp = float(kp)
        self.kd = float(kd) if kd is not None else float(2.0 * np.sqrt(kp))
        self._waypoints = None; self._T = 0; self._t = 0

        # Renderers
        self.cam_primary = PRIMARY_CAM_NAME
        self.cam_wrist   = WRIST_CAM_NAME
        self._r_primary = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self._r_wrist   = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

    def set_capture_images(self, enabled: bool):
        self._capture_images = bool(enabled)

    def _render_images(self):
        if not getattr(self, "_capture_images", True):
            z = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            return z, z
        self._r_primary.update_scene(self.data, camera=self.cam_primary)
        img_primary = self._r_primary.render().copy()
        self._r_wrist.update_scene(self.data, camera=self.cam_wrist)
        img_wrist = self._r_wrist.render().copy()
        return img_primary, img_wrist

   
    def set_waypoints(self, waypoints: np.ndarray):
        self._waypoints = waypoints.astype(np.float32)
        self._T = len(waypoints)
        self._t = 0

    def reset(self, waypoints=None):
        if waypoints is not None:
            self.set_waypoints(waypoints)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        if self._waypoints is not None and self._T > 0:
            self.data.qpos[self.arm_qpos_addr] = self._waypoints[0, :7]
        else:
            self.data.qpos[self.arm_qpos_addr] = START_JOINTS
        mujoco.mj_forward(self.model, self.data)
        self._t = 0
        img_primary, img_wrist = self._render_images()
        return TimeStep(
            dm_env.StepType.FIRST,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            observation={
                "state": self.data.qpos.astype(np.float32).copy(),
                "language_instruction": self.lang,
                "image_primary": img_primary,
                "image_wrist": img_wrist,
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

    def step(self, action):
        if self._waypoints is None or self._T == 0:
            raise RuntimeError("step() called before reset(waypoints=...)")
        if self.arm_dof_idx is None:
            raise RuntimeError("arm_dof_idx must be provided for PD control")

        idx = min(self._t, self._T - 1)
        q_target = self._waypoints[idx, :7]
        g_cmd    = self._waypoints[idx,  7]

        # Respect joint limits for the target
        q_target = self._clamp_to_limits(q_target)

        # PD torque control over several internal substeps
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
        last = (self._t >= self._T)
        img_primary, img_wrist = self._render_images()
        return TimeStep(
            dm_env.StepType.LAST if last else dm_env.StepType.MID,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            observation={
                "state": self.data.qpos.astype(np.float32).copy(),
                "language_instruction": self.lang,
                "image_primary": img_primary,
                "image_wrist": img_wrist,
            },
        )

    def _render_images(self):
        self._r_primary.update_scene(self.data, camera=self.cam_primary)
        img_primary = self._r_primary.render().copy()
        self._r_wrist.update_scene(self.data, camera=self.cam_wrist)
        img_wrist = self._r_wrist.render().copy()
        return img_primary, img_wrist

    def observation_spec(self):
        return {
            "state": specs.Array(shape=(self.model.nq,), dtype=np.float32, name="state"),
            "language_instruction": specs.Array(shape=(), dtype=object, name="language_instruction"),
            "image_primary": specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_primary"),
            "image_wrist":   specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_wrist"),
        }

    def action_spec(self):
        return specs.Array(shape=(7,), dtype=np.float32, name="action")  # unused but defined for RLDS

# ------------------------------ Rollout & Logging -----------------------------

def run_oracle_once(
    env, base_env, model, data, arm_dof_idx, arm_qpos_addr, ee_ref,
    goal_pos, obstacles=None, goal_body_name=None,
    waypoints=None, dry_run=False,
):
    """
    Plan to goal_pos, build waypoints (7 joints + gripper), then either:
      - DRY RUN (no logging): execute on base_env only and return (err_final, waypoints)
      - LOGGED RUN: execute through `env` (EnvLogger-wrapped) and return (err_final, waypoints)
    """
    # ---- Reset to start posture
    data.qpos[arm_qpos_addr] = START_JOINTS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    q_start = data.qpos[arm_qpos_addr].copy()

    # ---- tunables for robust grasp timing (add these near your config if you want)
    GOAL_SETTLE_STEPS   = 60            # extra frames holding exactly at the goal before closing
    GRASP_START_DIST    = 0.006         # meters; start closing when EE is within 6 mm of goal
    MIN_CLOSE_IDX_MARGIN= 4             # don't start closing sooner than a few frames after reaching goal
    # NOTE: if you see *very* slow tracking, increase GOAL_SETTLE_STEPS to 50–60.

    # ---- Plan & build waypoints if not supplied
    if waypoints is None:
        # --- plan cartesian → joint waypoints
        traj_q, phase_idx = plan_cartesian_to_joint_traj(
            model, data, q_start, goal_pos,
            arm_dof_idx, arm_qpos_addr, ee_ref,
            n_cart=N_CART_WAYPOINTS,
            use_avoidance=USE_NULLSPACE_AVOID,
            obstacles=obstacles,
            goal_body_name=goal_body_name,
        )
        end_pre, end_goal, end_ret = phase_idx
        rep = int(WAYPOINT_REPEAT)

        # repeat for slower motion
        traj_q_rep = np.repeat(traj_q, repeats=rep, axis=0)

        # pregrasp dwell (open) — make it ~1 second of sim time
        sim_dt = float(model.opt.timestep) * float(getattr(base_env, "substeps", 40))
        dwell_pre_steps = int(np.ceil(float(globals().get("PREGRASP_DWELL_SEC", 1.0)) / max(sim_dt, 1e-9)))
        dwell_pre = np.repeat(
            traj_q_rep[end_pre*rep + (rep-1) : end_pre*rep + rep],
            repeats=dwell_pre_steps, axis=0
        )
        print(f"[DWELL] pregrasp: {dwell_pre_steps} steps ≈ {dwell_pre_steps*sim_dt:.2f}s")

        # goal settle (to let PD stop “ringing”)
        GOAL_SETTLE_STEPS   = 60
        goal_frame = traj_q_rep[end_goal*rep + (rep-1) : end_goal*rep + rep]
        dwell_goal = np.repeat(goal_frame, repeats=GOAL_SETTLE_STEPS, axis=0)

        # stitch joints (no retreat): [..pre] + dwell_pre + (pre→goal] + dwell_goal + (goal→end]
        traj_q_full = np.concatenate(
            [
                traj_q_rep[: (end_pre+1)*rep],
                dwell_pre,
                traj_q_rep[(end_pre+1)*rep : (end_goal+1)*rep],
                dwell_goal,
                traj_q_rep[(end_goal+1)*rep :],
            ],
            axis=0
        )

        # ---------- MICRO-CENTERING SHIM (measures lateral bias during goal dwell) ----------
        idx_pre_end          = (end_pre+1)*rep
        idx_goal_start       = idx_pre_end + dwell_pre_steps

        idx_goal_end         = idx_goal_start + (end_goal - end_pre)*rep
        idx_dwell_goal_start = idx_goal_end
        idx_dwell_goal_end   = idx_goal_end + GOAL_SETTLE_STEPS

        n_app  = _approach_normal_lateral(model, data, goal_pos)
        orth_u = np.array([-n_app[1], n_app[0], 0.0], float)
        orth_u /= (np.linalg.norm(orth_u) + 1e-9)

        _bak = data.qpos.copy()
        ees = []
        for q in traj_q_full[idx_dwell_goal_start:idx_dwell_goal_end]:
            data.qpos[arm_qpos_addr] = q; mujoco.mj_forward(model, data)
            p, _ = _ee_pose(model, data, ee_ref); ees.append(p.copy())
        data.qpos[:] = _bak; mujoco.mj_forward(model, data)

        d_orth = float(np.mean([np.dot(p - goal_pos, orth_u) for p in ees])) if ees else 0.0
        if abs(d_orth) > 0.003:
            gpos_corr = goal_pos - d_orth * orth_u
            q_corr = solve_ik_LM_position(
                model, data, target_pos_world=gpos_corr,
                arm_dof_idx=arm_dof_idx, arm_qpos_addr=arm_qpos_addr, ee_ref=ee_ref,
                max_iters=80, pos_tol=5e-4, step_size=LM_STEP, damping=LM_DAMP,
                debug=False, use_avoidance=True, obstacles=obstacles,
                goal_ctx={"goal_pos": gpos_corr, "approach_dir": n_app}
            )
            last_goal_q = traj_q_full[idx_dwell_goal_end - 1]
            micro = [last_goal_q + t*(q_corr - last_goal_q) for t in np.linspace(0.0, 1.0, 10, dtype=np.float32)]
            traj_q_full = np.concatenate(
                [traj_q_full[:idx_dwell_goal_end], np.asarray(micro, np.float32), traj_q_full[idx_dwell_goal_end:]],
                axis=0
            )
            idx_dwell_goal_end += len(micro)
        # ---------- END SHIM ----------
        # ---- Roll the wrist (last joint) during the goal dwell, before closing ----
        j_roll = 6  # last of the 7 arm joints in traj_q_* arrays

        # We’ll apply the roll over the *end* of the goal-dwell window so it finishes before closing.
        i1 = int(idx_dwell_goal_end)                                 # end of dwell
        i0 = int(max(idx_dwell_goal_start, i1 - ROLL_RAMP_STEPS))    # start of ramp inside dwell
        if i1 > i0:
            roll0 = float(traj_q_full[i0 - 1, j_roll] if i0 > 0 else traj_q_full[0, j_roll])
            roll1 = roll0 + np.deg2rad(float(HAND_ROLL_AT_GRASP_DEG))
            for t, i in enumerate(range(i0, i1)):
                a = (t + 1) / max(1, (i1 - i0))
                traj_q_full[i, j_roll] = (1.0 - a) * roll0 + a * roll1
            print(f"[WRIST] roll ramp: {HAND_ROLL_AT_GRASP_DEG:.1f}° over {i1 - i0} steps")
        else:
            print("[WRIST] (skipped) dwell too short for roll ramp")

        # add final hold AFTER any splicing so lengths match
        traj_q_hold = np.concatenate(
            [traj_q_full, np.repeat(traj_q_full[-1][None, :], EXTRA_HOLD_STEPS, axis=0)],
            axis=0
        )

        # distance-based grasp timing (robust)
        _bak = data.qpos.copy()
        dists = []
        for q in traj_q_full:
            data.qpos[arm_qpos_addr] = q; mujoco.mj_forward(model, data)
            ee, _ = _ee_pose(model, data, ee_ref)
            dists.append(float(np.linalg.norm(goal_pos - ee)))
        data.qpos[:] = _bak; mujoco.mj_forward(model, data)

        GRASP_START_DIST      = 0.006
        MIN_CLOSE_IDX_MARGIN  = 4
        idx_close = next((i for i, d in enumerate(dists) if d <= GRASP_START_DIST), len(dists) - 1)
        close_start = max(idx_dwell_goal_end + MIN_CLOSE_IDX_MARGIN, idx_close)
        close_start = min(close_start, len(traj_q_full) - 1)

        # gripper ramp (reduces finger chatter)
        N = len(traj_q_hold)
        GRIP_OPEN, GRIP_CLOSE = 1.0, 0.0
        GRIPPER_RAMP_STEPS    = 25
        grip = np.full(N, GRIP_OPEN, np.float32)
        end_ramp = min(close_start + GRIPPER_RAMP_STEPS, N)
        grip[close_start:end_ramp] = np.linspace(GRIP_OPEN, GRIP_CLOSE, end_ramp - close_start, dtype=np.float32)
        grip[end_ramp:] = GRIP_CLOSE

        waypoints = np.concatenate([traj_q_hold, grip[:, None]], axis=1).astype(np.float32)

    # ---- DRY RUN: execute without logging (optional QC)
    if dry_run:
        try:
            prev_cap = getattr(base_env, "_capture_images", True)
            qc_cap = bool(globals().get("CAPTURE_IMAGES_DURING_QC", False))
            if hasattr(base_env, "set_capture_images"):
                base_env.set_capture_images(qc_cap)
        except Exception:
            prev_cap = True

        base_env.set_waypoints(waypoints)
        live_qc = bool(globals().get("LIVE_RENDER_QC", globals().get("LIVE_RENDER", True)))
        period = 1.0 / float(globals().get("LIVE_FPS", 60.0))

        if live_qc:
            base_env.reset()
            last_t = time.perf_counter()
            with viewer.launch_passive(model, data) as v:
                while v.is_running():
                    ts = base_env.step(np.zeros(7, dtype=np.float32))
                    now = time.perf_counter()
                    if now - last_t < period:
                        time.sleep(max(0.0, period - (now - last_t)))
                    last_t = now
                    v.sync()
                    if ts.last():
                        break
        else:
            base_env.reset()
            while True:
                ts = base_env.step(np.zeros(7, dtype=np.float32))
                if ts.last():
                    break

        ee_pos, _ = _ee_pose(model, data, ee_ref)
        err_final = float(np.linalg.norm(goal_pos - ee_pos))
        try:
            if hasattr(base_env, "set_capture_images"):
                base_env.set_capture_images(prev_cap)
        except Exception:
            pass
        print(f"[QC] final |EE - goal| = {err_final:.4f} m")
        return err_final, waypoints

    # ---- LOGGED RUN: execute through EnvLogger-wrapped env
    if env is None:
        raise RuntimeError("Logged run requested but `env` is None. Use dry_run=True for QC or pass an EnvLogger.")

    base_env.set_waypoints(waypoints)
    env.reset()
    env.step(np.zeros((7,), np.float32))  # warm-up for FIRST

    period = 1.0 / float(LIVE_FPS)
    if LIVE_RENDER:
        last_t = time.perf_counter()
        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                ts = env.step(action=np.zeros(7, dtype=np.float32))
                now = time.perf_counter()
                if now - last_t < period:
                    time.sleep(max(0.0, period - (now - last_t)))
                last_t = now
                v.sync()
                if ts.last():
                    break
    else:
        while True:
            ts = env.step(action=np.zeros(7, dtype=np.float32))
            if ts.last():
                break

    ee_pos, _ = _ee_pose(model, data, ee_ref)
    err_final = float(np.linalg.norm(goal_pos - ee_pos))
    print(f"[RUN] final |EE - goal| = {err_final:.4f} m")
    return err_final, waypoints

# ----------------------------------- Main -------------------------------------

def _site_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0

def main():
    os.makedirs(TFDS_ROOT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    episodes_done = 0
    plant_idx = 0

    while episodes_done < EPISODES_TOTAL:
        # Fresh plant + scene
        if USE_DYNAMIC_PLANT:
            model, data = _build_and_load_scene(MODEL_PATH)
            split_name = choose_split_for_plant(plant_idx)
            print(f"[SPLIT] Plant {plant_idx} → {split_name}")

        else:
            model = mujoco.MjModel.from_xml_path(MODEL_PATH)
            data  = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

        # Choose EE reference (prefer TCP site if present)
        ee_ref = ("site", "tcp") if _site_exists(model, "tcp") else ("body", "hand")
        print(f"[EE_REF] using {ee_ref[0].upper()} '{ee_ref[1]}'")

        # Obstacles (recomputed each plant)
        obstacles_all = _collect_plant_obstacles(model)
        print(f"[AVOID] using {len(obstacles_all)} plant bodies as obstacles")

        # Arm mapping
        arm_act_ids, arm_qpos_addr = build_arm_mapping_from_model(model, prefer_position=True)
        arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)

        # Optional: widen limits for the 7 arm joints
        if AUTO_WIDEN_LIMITS:
            for i, dof in enumerate(arm_dof_idx):
                jid = int(model.dof_jntid[dof])
                model.jnt_limited[jid] = 1
                lo, hi = TYPICAL_FRANKA_LIMITS[i]
                model.jnt_range[jid][0] = lo
                model.jnt_range[jid][1] = hi

        gripper_idx = find_gripper_actuator(model)

        # RLDS dataset config
        ds_config = tfds.rlds.rlds_base.DatasetConfig(
            version=tfds.core.Version(DATASET_VERSION),
            name=DATASET_NAME,
            observation_info=tfds.features.FeaturesDict({
                "state": tfds.features.Tensor(shape=(model.nq,), dtype=np.float32),
                "language_instruction": tfds.features.Text(),
                "image_primary": tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
                "image_wrist":   tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
            }),
            action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),
            reward_info=tf.float32,
            discount_info=tf.float32,
        )

        base_env = PandaOracleEnv(
            model, data, arm_act_ids, arm_qpos_addr, ee_ref,
            language_instruction="Pick the specified tomato by name.",
            substeps=80, gripper_idx=gripper_idx,
            arm_dof_idx=arm_dof_idx, kp=PD_KP, kd=PD_KD,
        )

        # Determine the top-2 targets for THIS plant (s=0.66 grasp along stem)
        top = _find_top_targets(model, data, k=EPISODES_PER_PLANT, s=0.66)
        for name, grasp_pos, approach_xy, truss_name in top:
            print(f"[Target(stem@0.66)] {name}  grasp={grasp_pos}  (Z={grasp_pos[2]:.3f})")

        episodes_here = min(EPISODES_PER_PLANT, EPISODES_TOTAL - episodes_done)

        # Open a logger around this base_env; close it after these episodes
        with envlogger.EnvLogger(
            base_env,
            backend=tfds_backend_writer.TFDSBackendWriter(
                data_directory=TFDS_ROOT_DIR,
                split_name="train",
                max_episodes_per_file=2,
                ds_config=ds_config,
            ),
        ) as env:
            try:
                for epi in range(episodes_here):
                    stem_name, grasp_pos, approach_xy, truss_name = top[epi]

                    # --- robust lateral side selection (keeps approach away from plant center) ---
                    STEM_R = 0.0025
                    CLEAR  = 0.0000
                    offset_mag = STEM_R + CLEAR

                    c_plant, _ = _body_pos(model, data, "tomato_plant")
                    orth = np.array([approach_xy[1], -approach_xy[0], 0.0], dtype=float)
                    cand1 = grasp_pos + offset_mag *  orth
                    cand2 = grasp_pos + offset_mag * (-orth)
                    goal_pos = cand1 if np.linalg.norm(cand1 - c_plant) > np.linalg.norm(cand2 - c_plant) else cand2
                    # (This exact pattern is what made the old oracle “approach better”.) :contentReference[oaicite:4]{index=4}

                    # Exclude the active stem & its truss from avoidance if you do that elsewhere
                    obstacles = [b for b in obstacles_all if b not in (stem_name, truss_name)]
                    
                    base_env.lang = f"Pick tomato truss at {stem_name} (plant={plant_idx}, split={split_name})"
                    # --- QC dry-run (no logging) ---
                    err_final, waypoints = run_oracle_once(
                        env=None, base_env=base_env, model=model, data=data,
                        arm_dof_idx=arm_dof_idx, arm_qpos_addr=arm_qpos_addr, ee_ref=ee_ref,
                        goal_pos=goal_pos, obstacles=obstacles, goal_body_name=truss_name,
                        waypoints=None, dry_run=True,
                    )
                    print(f"[QC] executed final |EE-goal| = {err_final:.4f} m")
                    if err_final > MAX_FINAL_ERR:
                        print("[QC] ❌ Skip episode")
                        print(f"[PROGRESS] ⏭️  Skipped. Still at {episodes_done}/{EPISODES_TOTAL} saved.")

                        continue
                    print("[QC] ✅ Keep episode")

                    # --- Log the successful episode with the *same* waypoints ---
                    # --- after QC passes ---
                    with envlogger.EnvLogger(
                        base_env,
                        backend=tfds_backend_writer.TFDSBackendWriter(
                            data_directory=TFDS_ROOT_DIR,
                            split_name=split_name,                # ← use plant-based split
                            max_episodes_per_file=2,
                            ds_config=tfds.rlds.rlds_base.DatasetConfig(
                                version=tfds.core.Version(DATASET_VERSION),
                                name=DATASET_NAME,
                                observation_info=tfds.features.FeaturesDict({
                                    "state": tfds.features.Tensor(shape=(model.nq,), dtype=np.float32),
                                    "language_instruction": tfds.features.Text(),
                                    "image_primary": tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
                                    "image_wrist":   tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
                                }),
                                action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                reward_info=tf.float32,
                                discount_info=tf.float32,
                            ),
                        ),
                    ) as env:
                        run_oracle_once(
                            env, base_env, model, data, arm_dof_idx, arm_qpos_addr, ee_ref,
                            goal_pos=goal_pos, obstacles=obstacles, goal_body_name=truss_name,
                            waypoints=waypoints, dry_run=False,
                        )
                        episodes_done += 1
                        print(f"[PROGRESS] ✅ Episodes saved: {episodes_done}/{EPISODES_TOTAL}")



            except Exception as e:
                print(f"[ERROR] Exception during rollout on plant {plant_idx}: {e}")

        plant_idx += 1

    # ---- Safe peek (optional) ----
    dataset_dir = os.path.join(TFDS_ROOT_DIR, DATASET_NAME, DATASET_VERSION)
    print(f"✅ RLDS/TFDS episodes written under {dataset_dir}")

if __name__ == "__main__":
    main()
