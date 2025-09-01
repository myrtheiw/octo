#!/usr/bin/env python3
"""
Panda oracle with Jacobian IK (LM/DLS), **PD joint servoing**, and RLDS logging.

Changes vs your previous version
- Use a PD (position) controller to compute actuator torques instead of
  writing desired joint positions directly to `data.ctrl`.
- Keep only essential prints (IK error progress + EE→goal error).
- Preserve the warm-up step for EnvLogger so `FIRST` has a valid action.
- (Optional) widen unrealistic joint limits for the 7 arm joints so waypoints
  are not clipped to tiny ranges. Toggle with `AUTO_WIDEN_LIMITS`.
"""

import os
import numpy as np
import mujoco
import dm_env
from dm_env import specs, TimeStep
import tensorflow as tf
import time
from mujoco import viewer

# RLDS / TFDS
import envlogger
from envlogger.backends import tfds_backend_writer
import tensorflow_datasets as tfds

# Target detection util 
from getlocation import get_side_stem_midpoints_and_quats

# collision avoidance helpers 
from helpers import (
    damped_pinv as _damped_pinv,
    body_pos as _body_pos,
    approx_body_radius_max as _approx_body_radius,   
    jacobian_body_point as _jacobian_body_point,
    collect_plant_obstacles as _collect_plant_obstacles,
    approach_normal_lateral as _approach_normal_lateral,  
)



# ------------------------------- Config ---------------------------------------

MODEL_PATH = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"

# End-effector reference (prefer a TCP site if present; else body "hand")
EE_REF = ("body", "hand")

# Cameras
PRIMARY_CAM_NAME = "third_person_cam"
WRIST_CAM_NAME   = "gripper_cam"
IMG_H, IMG_W     = 256, 256

# Initial joints and planning
START_JOINTS = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=float)
N_CART_WAYPOINTS = 30   # number of IK waypoints along the path
PREGRASP_OFFSET = 0.10
RETREAT_OFFSET  = 0.12

# IK tuning (LM/DLS)
IK_MAX_ITERS = 200
IK_POS_TOL   = 1e-3
LM_STEP      = 0.4
LM_DAMP      = 0.12

# PD controller gains (simple diagonal)
PD_KP = 150.0
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

# RLDS output
TFDS_ROOT_DIR   = "/home/myrtheiw/tfds_out"
DATASET_NAME    = "tomato_rlds"
DATASET_VERSION = "0.0.1"

# Which stems to go to (must exist in scene)
TARGET_NAMES = ["side_stem3", "side_stem4"]

# Show only error-related prints (IK error & EE→goal error)
DEBUG_IK = True

# --- Null-space collision avoidance (minimal) ---
USE_NULLSPACE_AVOID = True
AVOID_LINKS = ("link3", "link4", "link5", "link6", "hand")  
AVOID_D0    = 0.10   # start repelling when closer than 10 cm
AVOID_GAIN  = 0.6    # strength of repulsion (m/s-equivalent)
AVOID_LAM   = 0.05   # damping for the link Jacobian pinv

# Live rendering while logging
LIVE_RENDER = True      # set False to disable
LIVE_FPS    = 60.0      # target refresh rate for the viewer


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
    J = _jacobian_7(model, data, ("body", "hand"), arm_dof_idx, with_orientation=False)

    H = J.T @ J + (damping * np.eye(7))
    dq7 = np.linalg.solve(H, J.T @ e)

    q = data.qpos[arm_qpos_addr].copy()
    q_new = q + step_size * dq7

    data.qpos[arm_qpos_addr] = q_new
    mujoco.mj_forward(model, data)
    return np.linalg.norm(e)

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
    (tangential) repulsion so it can advance toward the tomato.

    Returns:
        float: pre-update ||position error|| (meters).
    """
    import numpy as np

    # --- Primary: hand position task ------------------------------------------
    p_cur, _ = _ee_pose(model, data, ee_ref)
    e = target_pos_world - p_cur                           # (3,)
    J_ee = _jacobian_7(model, data, ee_ref, arm_dof_idx,
                       with_orientation=False)             # 3x7

    # LM/DLS step for the task: dq_task = (J^T J + λI)^(-1) J^T e
    H = J_ee.T @ J_ee + (damping * np.eye(7))
    dq_task = np.linalg.solve(H, J_ee.T @ e)               # (7,)

    # Null-space projector for secondaries
    Jee_pinv = _damped_pinv(J_ee, lam=damping)             # (7x3)
    N = np.eye(7) - Jee_pinv @ J_ee                        # (7x7)

    # --- Secondary: obstacle avoidance in null space --------------------------
    # Fade avoidance near the goal to prevent end-game folding
    dist_goal = np.linalg.norm(e)
    if gate_band > 0.0:
        alpha_goal = np.clip((dist_goal - gate_r) / gate_band, 0.0, 1.0)
    else:
        alpha_goal = 1.0

    # Extract corridor context if present
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
            R_obs = _approx_body_radius(model, bid_obs, default=0.04)  # if aliased to max-radius helper, safer

            # Stronger push for truss bodies (tomatoes stick out)
            k_obs = k_avoid * (1.6 if ("truss" in obs_name) else 1.0)

            for link_name in (avoid_links or ()):
                p_link, _ = _body_pos(model, data, link_name)
                r = p_link - c_obs
                r_norm = np.linalg.norm(r)
                if r_norm < 1e-9:
                    continue
                dist = r_norm - R_obs                     # signed dist to obstacle surface

                # Extra clearance/weight for wrist/hand
                is_wrist = link_name in ("link6", "hand", "wrist")
                d0_local = d0 + (0.03 if is_wrist else 0.0)    # +3 cm buffer for wrist/hand
                k_local  = k_obs * (1.5 if is_wrist else 1.0)  # stronger push at wrist/hand

                if dist < d0_local:
                    n = r / r_norm
                    v_avoid = alpha_goal * k_local * (d0_local - dist) * n   # (3,)

                    # --- Approach corridor shaping for wrist/hand ------------
                    # Near the goal obstacle, allow forward motion but keep sideways repulsion.
                    if is_wrist and (goal_pos is not None) and (a_dir is not None):
                        # Treat as the "near goal" body if it matches, or if near_name is None.
                        near_goal_body = (near_name is None) or (obs_name == near_name)

                        # Is the wrist advancing roughly along the approach direction?
                        to_goal = goal_pos - p_link
                        tg_norm = np.linalg.norm(to_goal)
                        if near_goal_body and tg_norm > 1e-9:
                            to_goal_u = to_goal / tg_norm
                            cosang = float(np.clip(np.dot(to_goal_u, a_dir), -1.0, 1.0))
                            ang_deg = np.degrees(np.arccos(cosang))
                            if ang_deg <= float(cone_deg):
                                # Project out the component along the approach axis (keep sideways push)
                                v_avoid = v_avoid - (np.dot(v_avoid, a_dir) * a_dir)
                                v_avoid *= 0.8  # small damping to avoid oscillations

                    # convert to joint update via damped pinv of link Jacobian
                    J_link  = _jacobian_body_point(model, data, link_name, arm_dof_idx)  # 3x7
                    dq_link = _damped_pinv(J_link, lam=lam_avoid) @ v_avoid              # (7,)
                    dq_avoid_sum += dq_link

    # Optional posture prior (keeps elbow/wrist from collapsing) in null space
    if posture_weight and posture_weight > 0.0:
        q = data.qpos[arm_qpos_addr].copy()
        if q_nom is None:
            try:
                q_nom = START_JOINTS
            except NameError:
                q_nom = np.zeros_like(q)
        dq_post = posture_weight * (q_nom - q)
    else:
        dq_post = 0.0

    # Compose update: task + N*(avoid + posture)
    dq = dq_task + N @ (dq_avoid_sum + dq_post)

    # Guard: cap the size of the step
    nrm = np.linalg.norm(dq)
    if nrm > max_step:
        dq *= (max_step / nrm)

    # Apply and forward
    q = data.qpos[arm_qpos_addr].copy()
    q_new = q + step_size * dq

    # Optional: respect joint limits only if the joint is marked limited
    if respect_limits:
        for i, dof in enumerate(arm_dof_idx):
            jid = int(model.dof_jntid[dof])
            if int(model.jnt_limited[jid]) == 1:
                lo, hi = model.jnt_range[jid]
                if hi > lo + 1e-6:
                    q_new[i] = np.clip(q_new[i], lo, hi)

    data.qpos[arm_qpos_addr] = q_new
    mujoco.mj_forward(model, data)

    # Return pre-update task error (matches your other IK step API)
    return float(np.linalg.norm(e))



def solve_ik_LM_position(
    model, data, target_pos_world, arm_dof_idx, arm_qpos_addr, ee_ref,
    max_iters=IK_MAX_ITERS, pos_tol=IK_POS_TOL, step_size=LM_STEP, damping=LM_DAMP,
    debug=DEBUG_IK, use_avoidance=False, obstacles=None
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
                obstacles=obstacles, step_size=step_size, damping=damping
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
    interp_step=0.02,         # radians per interp step (≈ ~1.1°)
):
    """
    Plan a joint trajectory that passes through:
        pre-grasp (lateral) -> goal -> retreat (back out).

    Returns:
        np.ndarray with shape (T, 7) of joint waypoints.
    """
    q_backup = data.qpos.copy()

    # --- Build via points (lateral approach) ---
    n_app = _approach_normal_lateral(model, data, goal_pos)  # unit vector in xy plane
    pre_pos = goal_pos + float(pregrasp_offset) * n_app if pregrasp_offset > 0 else goal_pos
    ret_pos = goal_pos - float(retreat_offset)  * n_app if retreat_offset  > 0 else goal_pos

    # Sequence of cartesian sub-goals (dedupe if equal)
    cart_goals = []
    for p in (pre_pos, goal_pos, ret_pos):
        if not cart_goals or np.linalg.norm(p - cart_goals[-1]) > 1e-9:
            cart_goals.append(p)

    traj_all = []
    try:
        # Seed state from q_start
        qs = q_start.copy()
        data.qpos[arm_qpos_addr] = qs
        mujoco.mj_forward(model, data)

        # Optionally limit how many *internal* IK refinements you do overall
        # (We still call solve_ik once per sub-goal; this keeps the overall
        #  behavior similar to your original n_cart usage.)
        for gpos in cart_goals:
            # Solve IK to hit the cartesian sub-goal
            q_hit = solve_ik_LM_position(
                model, data, target_pos_world=gpos,
                arm_dof_idx=arm_dof_idx, arm_qpos_addr=arm_qpos_addr, ee_ref=ee_ref,
                max_iters=IK_MAX_ITERS, pos_tol=IK_POS_TOL,
                step_size=LM_STEP, damping=LM_DAMP,
                debug=DEBUG_IK, use_avoidance=use_avoidance, obstacles=obstacles,
            )

            # Interpolate qs -> q_hit into small steps for a smooth joint path
            d = np.linalg.norm(q_hit - qs)
            steps = max(5, int(d / float(interp_step)))  # at least a few steps
            if steps == 0:  # degenerate, still add the target
                traj_all.append(q_hit.copy())
            else:
                for i in range(1, steps + 1):
                    traj_all.append(qs + (i / steps) * (q_hit - qs))

            # Next segment starts from here
            qs = q_hit.copy()
            data.qpos[arm_qpos_addr] = qs
            mujoco.mj_forward(model, data)

        # As a final polish, you can sub-sample to ~n_cart waypoints if desired:
        if n_cart is not None and len(traj_all) > n_cart:
            idx = np.linspace(0, len(traj_all) - 1, num=n_cart, dtype=int)
            traj_all = [traj_all[i] for i in idx]

        return np.asarray(traj_all, dtype=np.float32)

    finally:
        # Always restore sim state
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
        self.kp = float(kp)
        self.kd = float(kd) if kd is not None else float(2.0 * np.sqrt(kp))
        self._waypoints = None; self._T = 0; self._t = 0

        # Renderers
        self.cam_primary = PRIMARY_CAM_NAME
        self.cam_wrist   = WRIST_CAM_NAME
        self._r_primary = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self._r_wrist   = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)

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
                if hi > lo + 1e-9:
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

def run_oracle_once(env, base_env, model, data, arm_dof_idx, arm_qpos_addr, ee_ref,
                    goal_pos, obstacles=None):
    """Plan to goal_pos, build waypoints (7 joints + gripper), and log ONE episode."""
    # Start posture
    data.qpos[arm_qpos_addr] = START_JOINTS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    q_start = data.qpos[arm_qpos_addr].copy()

    # Plan joint path (with optional null-space avoidance)
    traj_q = plan_cartesian_to_joint_traj(
        model, data, q_start, goal_pos,
        arm_dof_idx, arm_qpos_addr, ee_ref,
        n_cart=N_CART_WAYPOINTS,
        use_avoidance=USE_NULLSPACE_AVOID,
        obstacles=obstacles,
    )

    # Debug: where IK lands
    _bak = data.qpos.copy()
    data.qpos[arm_qpos_addr] = traj_q[-1]; mujoco.mj_forward(model, data)
    ee_pos, _ = _ee_pose(model, data, ee_ref)
    print(f"[DEBUG] EE final world pos: {ee_pos}, goal: {goal_pos}, |err|={np.linalg.norm(ee_pos-goal_pos):.4f} m")
    data.qpos[:] = _bak; mujoco.mj_forward(model, data)

    # Build waypoints block + simple open/close gripper schedule
    EXTRA_HOLD = 150
    PRE_OPEN_STEPS = 30
    traj_q_hold = np.concatenate(
        [traj_q, np.repeat(traj_q[-1][None, :], EXTRA_HOLD, axis=0)],
        axis=0,
    )
    N = len(traj_q_hold)
    GRIP_OPEN, GRIP_CLOSE = 1.0, 0.0
    grip = np.empty(N, dtype=np.float32)
    grip[: max(0, N - PRE_OPEN_STEPS)] = GRIP_CLOSE
    grip[max(0, N - PRE_OPEN_STEPS):]  = GRIP_OPEN
    waypoints = np.concatenate([traj_q_hold, grip[:, None]], axis=1).astype(np.float32)

    # Give waypoints to the *base* env, then start a logged episode
    base_env.set_waypoints(waypoints)   # base_env.reset(...) accepts this kwarg. :contentReference[oaicite:0]{index=0}
    env.reset()                         # EnvLogger.reset() takes no kwargs
    env.step(np.zeros((7,), np.float32))  # warm-up so FIRST has a valid action

    # Roll until LAST
    steps = 0
    if LIVE_RENDER:
        period = 1.0 / float(LIVE_FPS)
        last_t = time.perf_counter()
        from mujoco import viewer
        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                ts = env.step(action=np.zeros(7, dtype=np.float32))
                steps += 1
                if (steps % 20) == 0:
                    p, _ = _ee_pose(model, data, ee_ref)
                    print(f"[RUN] step={steps:4d}, |EE - goal| = {np.linalg.norm(goal_pos - p):.4f} m")
                now = time.perf_counter()
                if now - last_t < period:
                    time.sleep(max(0.0, period - (now - last_t)))
                last_t = time.perf_counter()
                v.sync()
                if ts.last():
                    break
    else:
        while True:
            ts = env.step(action=np.zeros(7, dtype=np.float32))
            steps += 1
            if (steps % 20) == 0:
                p, _ = _ee_pose(model, data, ee_ref)
                print(f"[RUN] step={steps:4d}, |EE - goal| = {np.linalg.norm(goal_pos - p):.4f} m")
            if ts.last():
                break


# ----------------------------------- Main -------------------------------------

def main():
    os.makedirs(TFDS_ROOT_DIR, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    # Obstacles
    obstacles = _collect_plant_obstacles(model)
    print(f"[AVOID] using {len(obstacles)} plant bodies as obstacles")

    # Targets
    stems = get_side_stem_midpoints_and_quats(model, data, prefix="side_stem")
    if not stems:
        raise RuntimeError("No stems found with prefix 'side_stem' — check scene names.")
    found = [(name, stems[name]) for name in TARGET_NAMES if name in stems]

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

    # Base env (PD control)
    base_env = PandaOracleEnv(
        model, data, arm_act_ids, arm_qpos_addr, EE_REF,
        language_instruction="Pick the specified tomato by name.",
        substeps=40, gripper_idx=gripper_idx,
        arm_dof_idx=arm_dof_idx, kp=PD_KP, kd=PD_KD,
    )  # PandaOracleEnv.reset supports waypoints kwarg. :contentReference[oaicite:1]{index=1}

    # Ensure TFRecord shards finalize
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
            for target_name, (pos, quat_wxyz) in found:
                print(f"[Target] Planning to {target_name} at pos={pos}, quat(wxyz)={quat_wxyz}")
                run_oracle_once(
                    env, base_env, model, data, arm_dof_idx, arm_qpos_addr, EE_REF,
                    goal_pos=pos.astype(float),
                    obstacles=obstacles,
                )
        except Exception as e:
            print(f"[ERROR] Exception during rollout: {e}")

    # ---- Safe peek (no SplitInfo construction) ----
    dataset_dir = os.path.join(TFDS_ROOT_DIR, DATASET_NAME, DATASET_VERSION)
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    print(f"✅ RLDS/TFDS episodes written under {dataset_dir}")

    if tf.io.gfile.exists(info_path):
        builder = tfds.builder_from_directory(dataset_dir)
        info = builder.info
        split_info = info.splits.get("train")
        n_train = int(getattr(split_info, "num_examples", 0) or 0) if split_info else 0
        print(f"[INFO] Episodes in 'train': {n_train}")
        if n_train > 0:
            ds = builder.as_dataset(split="train")
            sample_n = sum(1 for _ in ds.take(10))
            print(f"[INFO] Previewed {sample_n} example(s) from 'train'.")
        else:
            print("[WARN] 'train' has 0 examples (no finalized TFRecord shards).")
    else:
        print("[WARN] No dataset_info.json yet — nothing to preview.")


if __name__ == "__main__":
    main()
