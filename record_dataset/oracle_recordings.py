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
import envlogger
from envlogger.backends import tfds_backend_writer
import tensorflow_datasets as tfds
import dm_env
from dm_env import specs, TimeStep

from path_planning.plan_to_tomato import plan_joint_traj_to_tomato

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
REC_PATH         = "/home/myrtheiw/octo_ws/octo/record_dataset/dataset/tomato_dataset.tfrecord"

IMG_H, IMG_W = 256, 256   # image size for both cameras
PRIMARY_CAM_NAME = "third_person_cam"
WRIST_CAM_NAME   = "gripper_cam"

# ---------------- Setup Envlogger ----------------

class PandaOracleEnv(dm_env.Environment):
    """dm_env wrapper around your MuJoCo Panda + oracle controller."""
    def __init__(self, model, data, arm_act_ids, arm_qpos_addr, arm_dof_idx,
                 ee_ref, language_instruction, kp=300.0, kd=10.0,
                 torque_mode=True, gripper_idx=-1): 
        self.model = model; self.data = data
        self.arm_act_ids = arm_act_ids
        self.arm_qpos_addr = arm_qpos_addr
        self.arm_dof_idx = arm_dof_idx
        self.ee_ref = ee_ref
        self.lang = language_instruction
        self.kp = kp; self.kd = kd
        self.torque_mode = torque_mode
        self._waypoints = None
        self._T = 0
        self._t = 0
        self.gripper_idx = gripper_idx
        self.cam_primary = PRIMARY_CAM_NAME
        self.cam_wrist   = WRIST_CAM_NAME
        # Reuse persistent GL contexts (faster than recreating)
        self._r_primary = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self._r_wrist   = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)


    def reset(self, waypoints=None):
        """Reset sim. If waypoints provided, load them; otherwise keep current waypoints."""
        self._t = 0
        if waypoints is not None:
            self._waypoints = waypoints.astype(np.float32)
            self._T = len(waypoints)
        # else: keep previously set self._waypoints / self._T as-is

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        img_primary, img_wrist = self._render_images()
        return TimeStep(
            dm_env.StepType.FIRST,
            reward=np.float32(0.0),
            discount=np.float32(1.0),
            observation={
                "state": self.data.qpos.astype(np.float32).copy(),
                "language_instruction": self.lang,
                "image_primary": img_primary,     # <— NEW
                "image_wrist": img_wrist,         # <— NEW
            },
        )




    def step(self, action):
        if self._waypoints is None or self._T == 0:
            raise RuntimeError("PandaOracleEnv.step() called before reset(waypoints=...)")
    
        # Drive sim using oracle’s PD to track current waypoint, but *log* the applied action you pass in.
        idx = min(self._t, self._T - 1)
        q_target = self._waypoints[idx, :7]
        g_cmd    = self._waypoints[idx,  7]

        q_cur  = self.data.qpos[self.arm_qpos_addr]
        qd_cur = self.data.qvel[self.arm_dof_idx]
        tau    = self.kp*(q_target - q_cur) - self.kd*qd_cur

        if self.torque_mode:
            self.data.ctrl[self.arm_act_ids] = tau
        else:
            self.data.ctrl[self.arm_act_ids] = q_target
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
                "image_primary": img_primary,     # <— NEW
                "image_wrist": img_wrist,         # <— NEW
            },
        )


    def _render_images(self):
        # Render third-person
        self._r_primary.update_scene(self.data, camera=self.cam_primary)
        img_primary = self._r_primary.render().copy()  # RGB uint8
        # Render wrist
        self._r_wrist.update_scene(self.data, camera=self.cam_wrist)
        img_wrist = self._r_wrist.render().copy()      # RGB uint8
        return img_primary, img_wrist

    def set_waypoints(self, waypoints: np.ndarray):
        """Set waypoints without resetting the logger wrapper."""
        self._waypoints = waypoints.astype(np.float32)
        self._T = len(waypoints)
        self._t = 0


    def observation_spec(self):
        return {
            "state": specs.Array(shape=(self.model.nq,), dtype=np.float32, name="state"),
            "language_instruction": specs.Array(shape=(), dtype=object, name="language_instruction"),
            "image_primary": specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_primary"),
            "image_wrist":   specs.Array(shape=(IMG_H, IMG_W, 3), dtype=np.uint8, name="image_wrist"),
        }

    def action_spec(self):
        # We’ll log what we *apply*: 7 torques (or targets) + 1 gripper cmd
        return specs.Array(shape=(7,), dtype=np.float32, name="action")

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

def eef_poses_for_waypoints(model, data, ee_ref, waypoints_q, arm_qpos_addr):
    """Returns lists of (pos, rot) for each q in waypoints_q without altering sim."""
    q_backup = data.qpos.copy()
    poses = []
    try:
        for q in waypoints_q:
            data.qpos[arm_qpos_addr] = q
            mujoco.mj_forward(model, data)
            p, R = ee_world_pose(model, data, ee_ref)
            poses.append((p.copy(), R.copy()))
    finally:
        data.qpos[:] = q_backup
        mujoco.mj_forward(model, data)
    return poses


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

# def oracle_step(model, data, waypoints, arm_act_ids, arm_qpos_addr, arm_dof_idx, gripper_idx, step):
#     """
#     Torque-mode oracle: PD on 7 joints + gripper command from waypoints[:,7].
#     """
#     idx = min(step, len(waypoints) - 1)
#     q_target = waypoints[idx, :7]
#     g_cmd    = waypoints[idx, 7]

#     q_cur  = data.qpos[arm_qpos_addr]
#     qd_cur = data.qvel[arm_dof_idx]

#     Kp, Kd = 300.0, 10.0  # tune as you like
#     tau = Kp * (q_target - q_cur) - Kd * qd_cur
#     data.ctrl[arm_act_ids] = tau

#     if 0 <= gripper_idx < model.nu:
#         data.ctrl[gripper_idx] = g_cmd

def run_oracle_policy(env, base_env, model, data, arm_dof_idx, arm_qpos_addr, ee_ref):
    """Plan A→B, build waypoints (7 joints + gripper), and roll one logged episode with Octo-style actions."""
    q_start = data.qpos[arm_qpos_addr].copy()

    # --- choose planner
    if PLANNER == "rrtstar":
        traj_q = plan_AB_rrtstar_joint_traj(
            model, data, q_start, GOAL_POS, GOAL_QUAT_WXYZ,
            arm_dof_idx, arm_qpos_addr, ee_ref, n_cart=None
        )
    else:
        traj_q = plan_AB_cartesian_to_joint_traj(
            model, data, q_start, GOAL_POS, GOAL_QUAT_WXYZ,
            arm_dof_idx, arm_qpos_addr, ee_ref, n_cart=N_CART_WAYPOINTS
        )

    # ---- hold + gripper schedule (unchanged)
    EXTRA_HOLD = 150
    traj_q_hold = np.concatenate(
        [traj_q, np.repeat(traj_q[-1][None, :], EXTRA_HOLD, axis=0)],
        axis=0
    )
    N = len(traj_q)
    PRE_OPEN_STEPS = 30
    GRIP_OPEN, GRIP_CLOSE = 1.0, 0.0

    grip = np.empty(len(traj_q_hold), dtype=np.float32)
    grip[: max(0, N - PRE_OPEN_STEPS)] = GRIP_CLOSE
    grip[max(0, N - PRE_OPEN_STEPS): N] = GRIP_OPEN
    grip[N:] = GRIP_CLOSE

    waypoints = np.concatenate([traj_q_hold, grip[:, None]], axis=1).astype(np.float32)

    # ---- precompute EEF poses for all joint waypoints
    q_seq = waypoints[:, :7]
    poses = eef_poses_for_waypoints(model, data, ee_ref, q_seq, arm_qpos_addr)  # [(p,R), ...]

    p_cur0, R_cur0 = ee_world_pose(model, data, ee_ref)
    eef_deltas = []
    p0, R0 = poses[0]
    dpos0 = (p0 - p_cur0)
    dori0 = rotation_error_axis_angle(R_cur0, R0)
    eef_deltas.append(np.concatenate([dpos0, dori0]))

    for i in range(1, len(poses)):
        p_prev, R_prev = poses[i-1]
        p_i, R_i = poses[i]
        dpos = (p_i - p_prev)
        dori = rotation_error_axis_angle(R_prev, R_i)
        eef_deltas.append(np.concatenate([dpos, dori]))
    eef_deltas = np.asarray(eef_deltas, dtype=np.float32)

    # ---- set waypoints, reset env, roll out
    base_env.set_waypoints(waypoints)
    ts = env.reset()

    g_cmd0 = np.float32(1.0 if waypoints[0, 7] > 0.5 else 0.0)
    first_action = np.concatenate([eef_deltas[0], [g_cmd0]]).astype(np.float32)
    ts = env.step(first_action)

    t = 1
    max_steps = len(waypoints) + 2
    while t < max_steps:
        idx = min(t, len(waypoints) - 1)
        q_target = waypoints[idx, :7]
        g_cmd    = waypoints[idx, 7]
        q_cur  = data.qpos[arm_qpos_addr]
        qd_cur = data.qvel[arm_dof_idx]
        tau = 300.0*(q_target - q_cur) - 10.0*qd_cur
        data.ctrl[base_env.arm_act_ids] = tau
        if 0 <= base_env.gripper_idx < model.nu:
            data.ctrl[base_env.gripper_idx] = g_cmd
        mujoco.mj_step(model, data)

        g_val = np.float32(1.0 if g_cmd > 0.5 else 0.0)
        a = np.concatenate([eef_deltas[idx], [g_val]]).astype(np.float32)
        ts = env.step(action=a)

        t += 1
        if ts.last():
            break

# ------------------------------ RRT pathplanning ------------------------------


# New: collision-aware RRT* planner with identical call shape.
def plan_AB_rrtstar_joint_traj(model, data, q_start, goal_pos, goal_quat_wxyz,
                               arm_dof_idx, arm_qpos_addr, ee_ref,
                               n_cart=None, *, rrt_cfg: RRTStarConfig | None = None, allow_contact=None):
    """
    Plan a collision-free joint trajectory to (goal_pos, goal_quat_wxyz) using RRT*.
    The q_start/n_cart args are accepted for signature compatibility with the IK planner.
    """
    return plan_joint_traj_to_tomato(
        model, data,
        arm_dof_idx=arm_dof_idx,
        arm_qpos_addr=arm_qpos_addr,
        ee_ref=ee_ref,
        tomato_pos_world=goal_pos,
        tomato_quat_wxyz=goal_quat_wxyz,
        allow_contact=allow_contact,
        rrt_cfg=rrt_cfg,
    )




# ----------------------------------- Main -------------------------------------

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    gripper_idx = find_gripper_actuator(model, DEFAULT_GRIPPER_IDX)
    arm_act_ids, arm_qpos_addr = build_arm_mapping_from_model(model, gripper_idx)
    arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)

    # Define RLDS/TFDS spec
    dataset_config = tfds.rlds.rlds_base.DatasetConfig(
        name="tomato_rlds",
        observation_info=tfds.features.FeaturesDict({
            "state": tfds.features.Tensor(shape=(model.nq,), dtype=np.float32),
            "language_instruction": tfds.features.Text(),
            "image_primary": tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
            "image_wrist":   tfds.features.Image(shape=(IMG_H, IMG_W, 3), encoding_format="jpeg"),
        }),
        action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),  # 6-DoF ΔEEF + gripper
        reward_info=tf.float32,
        discount_info=tf.float32,
    )


    TFDS_OUT_DIR = "/home/myrtheiw/tfds_out"
    os.makedirs(TFDS_OUT_DIR, exist_ok=True)

    # Wrap env with logger (TFDS backend)
    base_env = PandaOracleEnv(
        model, data, arm_act_ids, arm_qpos_addr, arm_dof_idx,
        EE_REF, language_instruction="Pick the tomato...", kp=300.0, kd=10.0,
        torque_mode=True, gripper_idx=gripper_idx,               # <— pass it in
    )

    env = envlogger.EnvLogger(
        base_env,
        backend=tfds_backend_writer.TFDSBackendWriter(
            data_directory=TFDS_OUT_DIR,
            split_name="train",
            max_episodes_per_file=1,   # <- was 500
            ds_config=dataset_config,
        ),
    )

    try:
        run_oracle_policy(env, base_env, model, data, arm_dof_idx, arm_qpos_addr, EE_REF)
    finally:
        env.close()  # single point of close

    print(f"✅ RLDS/TFDS episode written under {TFDS_OUT_DIR}")

    # -------------- Verify by loading with TFDS ------------------
    ds = tfds.builder_from_directory(TFDS_OUT_DIR).as_dataset(split="train")
    for ep in ds.take(1):
        steps = ep["steps"]
        for s in steps.take(3):
            print("state:", s["observation"]["state"].shape,
                  "action:", s["action"].shape,
                  "first/last:", s["is_first"].numpy(), s["is_last"].numpy())




if __name__ == "__main__":
    main()
