#!/usr/bin/env python3
"""
oracle_recordings.py — A→B oracle using MuJoCo Jacobian IK (DLS), plus TFRecord logging.

Why this exists
---------------
We want a reliable, simulator-consistent "expert" to generate demos:
- Inverse kinematics solved with MuJoCo Jacobians (no external kinematics).
- Damped least-squares (DLS) update, stacking position + orientation error if enabled.
- Joint updates restricted to the 7 Panda arm dofs we actually control.

References
----------
- “Basic inverse kinematics in MuJoCo” (blog post) — iterative Jacobian method with DLS. 
- dm_control/utils/inverse_kinematics.py — well-documented MuJoCo IK utility.

This file follows that approach closely: compute Jp/Jr at the EE, stack a 6D error,
solve dq = J^T (J J^T + λ^2 I)^-1 * e, and integrate q.
"""

import os
import numpy as np
import tensorflow as tf
import mujoco

# ------------------------------- Config ---------------------------------------

MODEL_PATH  = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"

# End-effector reference used for IK *and* debug. Prefer a TCP site if your scene has one.
# Example alternatives: ("site","panda_hand_tcp")  or  ("body","panda_link8") / ("body","panda_hand")
EE_REF = ("body", "hand")

# Gripper actuator index (we’ll try to auto-detect; this is a fallback)
DEFAULT_GRIPPER_IDX = 7

# Bring-up start and goal (world frame)
START_JOINTS = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0], dtype=float)
GOAL_POS     = np.array([0.43, 0.00, 0.30], dtype=float)  # meters, world
USE_ORIENTATION = False                                   # start position-only (more robust)
GOAL_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # world orientation, if enabled
ORI_WEIGHT = 0.3                                          # weight for orientation rows

# IK tuning
DLS_DAMPING  = 2e-3     # λ in DLS
STEP_SCALE   = 0.5      # scales dq update
IK_MAX_ITERS = 300
IK_POS_TOL   = 1e-4     # stop when |pos_err| small
IK_ORI_TOL   = 1e-3     # rad; used if USE_ORIENTATION

# Planning & recording
N_CART_WAYPOINTS = 30
EP_LENGTH        = 400
REC_PATH         = "/home/myrtheiw/octo_ws/octo/record_dataset/tomato_dataset.tfrecord"


# --------------------------- Model introspection ------------------------------

def find_gripper_actuator(model, fallback_idx=DEFAULT_GRIPPER_IDX):
    """Return an actuator index likely to be the gripper."""
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
        if any(k in name.lower() for k in ("grip", "finger", "hand")):
            return aid
    return min(fallback_idx, model.nu - 1)


def build_arm_mapping_from_model(model, gripper_act_id):
    """
    Infer the 7 arm actuators and their qpos addresses from the model itself.
    Avoids brittle string-matching against URDF names.
    """
    cand_act = [aid for aid in range(model.nu) if aid != gripper_act_id]
    if len(cand_act) < 7:
        raise RuntimeError(f"Expected at least 7 arm actuators, found {len(cand_act)}")
    pairs = []
    for aid in cand_act:
        jid = int(model.actuator_trnid[aid][0])
        qadr = int(model.jnt_qposadr[jid])
        pairs.append((aid, jid, qadr))
    pairs.sort(key=lambda x: x[2])
    pairs = pairs[:7]
    arm_act_ids  = np.array([aid for (aid, _, _) in pairs], dtype=int)
    arm_qpos_adr = np.array([qadr for (_, _, qadr) in pairs], dtype=int)
    return arm_act_ids, arm_qpos_adr


def build_arm_dof_indices(model, arm_act_ids):
    """Return DOF indices (columns in Jacobian) corresponding to the 7 arm joints."""
    arm_joint_ids = [int(model.actuator_trnid[aid][0]) for aid in arm_act_ids]
    dof_idx = [d for d in range(model.nv) if int(model.dof_jntid[d]) in arm_joint_ids]
    return np.array(dof_idx, dtype=int)


# ------------------------------ Small helpers ---------------------------------

def quat_to_rotmat_wxyz(q_wxyz):
    """(w,x,y,z) -> 3x3 rotation."""
    w, x, y, z = q_wxyz
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=float)


def rotation_error_axis_angle(R_cur, R_tgt):
    """
    Map SO(3) error to R^3 via axis-angle (small-angle approx is fine for DLS):
      R_err = R_tgt * R_cur^T,  then  e_ori = axis * angle
    """
    R_err = R_tgt @ R_cur.T
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-9:
        return np.zeros(3)
    # axis from skew(R_err)
    axis = np.array([
        R_err[2,1] - R_err[1,2],
        R_err[0,2] - R_err[2,0],
        R_err[1,0] - R_err[0,1],
    ]) / (2*np.sin(angle))
    return axis * angle


def ee_world_pose(model, data, ee_ref):
    """Return (pos, rotmat) of EE in world frame for (kind,name)."""
    kind, name = ee_ref
    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid].copy()
        rot = data.site_xmat[sid].reshape(3,3).copy()
    else:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos = data.xpos[bid].copy()
        rot = data.xmat[bid].reshape(3,3).copy()
    return pos, rot


# ------------------------------- Jacobian IK ----------------------------------

def jacobian_ik_step(
    model, data,
    target_pos_world, target_rot_world,
    arm_dof_idx, arm_qpos_addr,
    ee_ref,
    use_orientation=False,
    dls_lambda=DLS_DAMPING, step_scale=STEP_SCALE, ori_weight=ORI_WEIGHT
):
    """
    One DLS step: build 3xnv (and 3xnv) Jacobians at EE, stack error, solve, update qpos.
    Matches the “blog-style” iterative IK in spirit (stacked pos+ori, DLS). 
    """
    # Current EE pose
    p_cur, R_cur = ee_world_pose(model, data, ee_ref)
    pos_err = (target_pos_world - p_cur)

    # Build Jacobians
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))

    kind, name = ee_ref
    if kind == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        mujoco.mj_jacSite(model, data, Jp, Jr, sid)
    else:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mujoco.mj_jacBody(model, data, Jp, Jr, bid)

    # Subselect the 7 arm columns
    Jp7 = Jp[:, arm_dof_idx]                      # (3,7)
    if use_orientation:
        R_tgt = target_rot_world
        ori_err = rotation_error_axis_angle(R_cur, R_tgt)   # (3,)
        Jr7 = Jr[:, arm_dof_idx]                            # (3,7)
        # Stack with a weight on orientation rows
        J = np.vstack([Jp7, ori_weight * Jr7])              # (6,7)
        e = np.concatenate([pos_err, ori_weight * ori_err]) # (6,)
    else:
        J = Jp7                                             # (3,7)
        e = pos_err                                         # (3,)

    # Damped least squares solve: dq = J^T (J J^T + λ^2 I)^-1 e
    A = J @ J.T + (dls_lambda ** 2) * np.eye(J.shape[0])
    dq7 = J.T @ np.linalg.solve(A, e)                       # (7,)

    # Integrate into qpos (hinge -> 1 dof per joint)
    dq_full_qpos = np.zeros(model.nq)
    dq_full_qpos[arm_qpos_addr] = step_scale * dq7
    data.qpos[:] = data.qpos + dq_full_qpos
    mujoco.mj_forward(model, data)

    # Return current position error norm (and ori if used) to check convergence outside
    if use_orientation:
        return np.linalg.norm(pos_err), np.linalg.norm(ori_err)
    return np.linalg.norm(pos_err), None


def solve_ik_to_pose(
    model, data, target_pos_world, target_quat_wxyz,
    arm_dof_idx, arm_qpos_addr, ee_ref,
    max_iters=IK_MAX_ITERS, pos_tol=IK_POS_TOL, ori_tol=IK_ORI_TOL, use_orientation=USE_ORIENTATION
):
    """Iteratively apply DLS steps until the error is small or we hit max_iters."""
    R_tgt = quat_to_rotmat_wxyz(target_quat_wxyz)
    for _ in range(max_iters):
        pos_err, ori_err = jacobian_ik_step(
            model, data,
            target_pos_world, R_tgt,
            arm_dof_idx, arm_qpos_addr, ee_ref,
            use_orientation=use_orientation
        )
        if use_orientation:
            if pos_err < pos_tol and (ori_err is not None and ori_err < ori_tol):
                break
        else:
            if pos_err < pos_tol:
                break
    # Return the 7 arm joints in our arm order
    return data.qpos[arm_qpos_addr].copy()


# -------------------------------- Planner -------------------------------------

def plan_AB_cartesian_to_joint_traj(model, data, q_start, goal_pos, goal_quat_wxyz,
                                    arm_dof_idx, arm_qpos_addr, ee_ref, n_cart=N_CART_WAYPOINTS):
    q_backup = data.qpos.copy()
    try:
        cart_targets = [goal_pos] * (n_cart + 1)
        path_q, qs = [], q_start.copy()

        for p in cart_targets:
            # seed IK from last solution
            data.qpos[arm_qpos_addr] = qs; mujoco.mj_forward(model, data)
            qj = solve_ik_to_pose(model, data, p, goal_quat_wxyz,
                                  arm_dof_idx, arm_qpos_addr, ee_ref,
                                  use_orientation=USE_ORIENTATION)
            path_q.append(qj)
            qs = qj

        # resample
        traj, qs = [], q_start.copy()
        for qg in path_q:
            steps = max(3, int(np.linalg.norm(qg - qs) / 0.02))
            for i in range(1, steps + 1):
                traj.append(qs + (i/steps)*(qg - qs))
            qs = qg
        return np.array(traj)
    finally:
        # restore original sim state so execution starts from q_start
        data.qpos[:] = q_backup
        mujoco.mj_forward(model, data)



# ------------------------------ Controller ------------------------------------
def get_arm_qvel(data, arm_dof_idx):
    """Return the 7 arm joint velocities (indexed by DOF)."""
    return data.qvel[arm_dof_idx].copy()

def oracle_step(model, data, waypoints, arm_act_ids, arm_qpos_addr, arm_dof_idx, gripper_idx, step):
    """
    Torque-mode oracle: PD on 7 joints + gripper command from waypoints[:,7].
    """
    idx = min(step, len(waypoints) - 1)
    q_target = waypoints[idx, :7]
    g_cmd    = waypoints[idx, 7]

    q_cur  = data.qpos[arm_qpos_addr]
    qd_cur = data.qvel[arm_dof_idx]

    Kp, Kd = 300.0, 10.0  # tune as you like
    tau = Kp * (q_target - q_cur) - Kd * qd_cur
    data.ctrl[arm_act_ids] = tau

    if 0 <= gripper_idx < model.nu:
        data.ctrl[gripper_idx] = g_cmd



# ----------------------------------- Main -------------------------------------

def main():
    # Load model/data
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # --- enumerate bodies/sites (trim or filter as you like) ---
    max_list = min(100, model.nbody)  # raise if you want everything
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(max_list)]
    sites  = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
    print("[BODIES] first", max_list, ":", bodies)
    print("[SITES] all   :", sites)

    # Gripper & arm mapping
    gripper_idx = find_gripper_actuator(model, DEFAULT_GRIPPER_IDX)
    arm_act_ids, arm_qpos_addr = build_arm_mapping_from_model(model, gripper_idx)
    arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)

    print("model.nu =", model.nu)
    print("Gripper actuator index:", gripper_idx,
          "name:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_idx))
    print("[MAP] ARM_ACT_IDS :", arm_act_ids)
    print("[MAP] ARM_QPOS_ADDR:", arm_qpos_addr)
    print("[MAP] ARM_DOF_IDX:", arm_dof_idx)

    # Check actuator types (0=general, 2=velocity, 3=position in recent MuJoCo; torque is 'general' with dyn params)
    print("[ACT TYPES]", [int(model.actuator_gaintype[aid]) for aid in arm_act_ids])


    # Quick sanity: teleporting should move the chosen EE
    mujoco.mj_forward(model, data)
    if EE_REF[0] == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_REF[1])
        p0 = data.site_xpos[sid].copy()
    else:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_REF[1])
        p0 = data.xpos[bid].copy()
    _bak = data.qpos.copy()
    bump = data.qpos.copy()
    bump[arm_qpos_addr[0]] += 0.3
    data.qpos[:] = bump; mujoco.mj_forward(model, data)
    if EE_REF[0] == "site":
        p1 = data.site_xpos[sid].copy()
    else:
        p1 = data.xpos[bid].copy()
    data.qpos[:] = _bak; mujoco.mj_forward(model, data)
    print(f"[SANITY] EE moved by {np.linalg.norm(p1 - p0):.4f} m (should be > 0.02)")

    # Reset to a known start
    data.qpos[arm_qpos_addr] = START_JOINTS
    mujoco.mj_forward(model, data)

    # Plan A→B with Jacobian IK (blog-style)
    q_start = data.qpos[arm_qpos_addr].copy()
    traj_q = plan_AB_cartesian_to_joint_traj(
        model, data, q_start, GOAL_POS, GOAL_QUAT_WXYZ,
        arm_dof_idx, arm_qpos_addr, EE_REF, n_cart=N_CART_WAYPOINTS
    )

    # Debug: does last q reach the goal?
    _bak = data.qpos.copy()
    data.qpos[arm_qpos_addr] = traj_q[-1]; mujoco.mj_forward(model, data)
    ee_pos, _ = ee_world_pose(model, data, EE_REF)
    err = np.linalg.norm(ee_pos - GOAL_POS)
    data.qpos[:] = _bak; mujoco.mj_forward(model, data)
    print(f"[DEBUG] EE final world pos: {ee_pos}, goal: {GOAL_POS}, |err|={err:.4f} m")

    # Build waypoints (+ gripper)
    g_cmd = 0.6 if True else 0.0  # open gripper; flip if needed
   
   
   # --- make the arm hold the final pose for settling ---
    EXTRA_HOLD = 150
    traj_q_hold = np.concatenate(
        [traj_q, np.repeat(traj_q[-1][None, :], EXTRA_HOLD, axis=0)],
        axis=0
    )

    # --- gripper schedule ---
    # semantics:
    #   - CLOSED during approach
    #   - OPEN for the last PRE_OPEN_STEPS before the goal (to get around the object)
    #   - CLOSE exactly at the goal and keep closed during the hold
    GRIP_OPEN  = 0.6
    GRIP_CLOSE = 0.0
    PRE_OPEN_STEPS = 30  # tweak: how many steps before the goal to open

    N = len(traj_q)  # index of the first "at goal" step is N-1
    grip = np.empty(len(traj_q_hold), dtype=float)
    # approach: closed
    grip[: max(0, N - PRE_OPEN_STEPS)] = GRIP_CLOSE
    # pre-open window: open
    grip[max(0, N - PRE_OPEN_STEPS): N] = GRIP_OPEN
    # at-goal and during hold: close
    grip[N:] = GRIP_CLOSE

    # pack waypoints: 7 joints + 1 gripper per row
    waypoints = np.concatenate([traj_q_hold, grip[:, None]], axis=1)

    # episode length matches waypoints
    EP_LENGTH = len(waypoints)


    # Record one episode
    episodes = []
    current_episode = []
    language_command = "Pick the tomato from the top truss closest to you."
    for step in range(EP_LENGTH):
        oracle_step(model, data, waypoints, arm_act_ids, arm_qpos_addr, arm_dof_idx, gripper_idx, step)
        action_snapshot = data.ctrl.copy()
        mujoco.mj_step(model, data)

        obs = {
            "state": data.qpos.copy(),
            "action": action_snapshot,
            "reward": 0,
            "is_terminal": step == (EP_LENGTH - 1),
            "is_first": step == 0,
            "language_command": language_command,
        }
        current_episode.append(obs)
    episodes.append(current_episode)

    # Save TFRecord
    os.makedirs(os.path.dirname(REC_PATH), exist_ok=True)
    with tf.io.TFRecordWriter(REC_PATH) as writer:
        for episode in episodes:
            for obs in episode:
                feature = {
                    "state": tf.train.Feature(float_list=tf.train.FloatList(value=obs["state"])),
                    "action": tf.train.Feature(float_list=tf.train.FloatList(value=obs["action"])),
                    "reward": tf.train.Feature(int64_list=tf.train.Int64List(value=[obs["reward"]])),
                    "is_terminal": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(obs["is_terminal"])])),
                    "is_first": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(obs["is_first"])])),
                    "language_command": tf.train.Feature(bytes_list=tf.train.BytesList(value=[obs["language_command"].encode()])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    print(f"✅ Dataset saved with {len(episodes)} episode(s) to {REC_PATH}")


if __name__ == "__main__":
    main()
