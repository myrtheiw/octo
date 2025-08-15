import mujoco
import numpy as np
import tensorflow as tf
import os
from ikpy.chain import Chain

from scipy.spatial.transform import Rotation as R

# ------------------ Setup ------------------
urdf_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/panda.urdf"
model_path = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"


# IK chain
panda_chain = Chain.from_urdf_file(
    urdf_path,
    base_elements=["panda_link0"],
    last_link_vector=[0, 0, 0.11],
    active_links_mask=[False, True, True, True, True, True, True, True, False, False, False, False]
)
# checkkkk
active_idx = [i for i, m in enumerate(panda_chain.active_links_mask) if m]
ik_names = [panda_chain.links[i].name for i in active_idx][:7]
print("[IK] active joint order:", ik_names)
print("[ARM] expected order     :", [f"panda_joint{i}" for i in range(1, 8)])

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Check gripper index
GRIPPER_IDX = 7
print("model.nu =", model.nu)
print("Gripper actuator index:", GRIPPER_IDX,
      "name:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_IDX))

#------------------- Helpers --------------------------
# Names in IKPy (URDF) for Panda arm joints
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

def get_mj_joint_addrs(model, joint_names):
    """Return qpos addresses for a list of MuJoCo joint names."""
    addrs = []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.array(addrs, dtype=int)

def get_mj_actuator_ids(model, joint_names):
    """
    Return actuator ids that drive the given joint names, in the SAME order.
    Assumes one actuator per joint. Falls back to name substring match.
    """
    act_ids = []
    for jn in joint_names:
        # try exact actuator with same name
        try:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            act_ids.append(int(aid)); continue
        except Exception:
            pass
        # fallback: find actuator whose joint is this joint
        found = None
        for aid in range(model.nu):
            # actuator -> joint id
            jid = int(model.actuator_trnid[aid][0])
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) == jn:
                found = aid; break
        if found is None:
            # last resort: substring
            for aid in range(model.nu):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
                if jn in name:
                    found = aid; break
        if found is None:
            raise RuntimeError(f"No actuator found for joint {jn}")
        act_ids.append(int(found))
    return np.array(act_ids, dtype=int)
# ------------------ Mujoco Ik ----------------
def build_arm_dof_indices(model, arm_joint_act_ids):
    """Return the DOF indices (columns in Jacobian) for the 7 arm joints."""
    jids = [int(model.actuator_trnid[aid][0]) for aid in arm_joint_act_ids]  # actuator -> joint id
    dof_idx = []
    for didx in range(model.nv):
        jid = int(model.dof_jntid[didx])
        if jid in jids:
            dof_idx.append(didx)
    return np.array(dof_idx, dtype=int)



def mj_inverse_kinematics_pos(model, data, target_pos_world, *,
                              site_name=None, body_name=None,
                              max_iters=200, tol=1e-4, damping=1e-3, step_scale=1.0):
    """
    Solve for q of the arm that reaches target_pos_world for a given site/body in MuJoCo.
    Position-only DLS IK on the 7 arm DOFs (consistent with execution).
    """
    assert site_name or body_name, "Pass site_name or body_name for the end-effector."
    import mujoco

    # pick end-effector function
    def ee_pos():
        if site_name:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            return data.site_xpos[sid].copy()
        else:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            return data.xpos[bid].copy()

    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        p = ee_pos()
        err = target_pos_world - p
        if np.linalg.norm(err) < tol:
            break

        # Jacobian at EE
        if site_name:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            mujoco.mj_jacSite(model, data, Jp, Jr, sid)
        else:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mujoco.mj_jacBody(model, data, Jp, Jr, bid)

        # Subselect the 7 arm DOFs
        J = Jp[:, ARM_DOF_IDX]  # 3 x 7
        # Damped least squares: dq = J^T (J J^T + λ^2 I)^-1 * err
        A = J @ J.T + (damping ** 2) * np.eye(3)
        dq_sub = J.T @ np.linalg.solve(A, err)  # (7,)

        # Apply update to full qpos via the arm DOFs
        # Map DOF increments to joint qpos increments (for hinge joints dof==joint)
        # We need to add dq_sub to the correct qpos indices
        # Build a small vector for all qpos, zeros elsewhere
        dq_full = np.zeros(model.nq)
        dq_full[ARM_QPOS_ADDR] = step_scale * dq_sub  # hinge: 1 dof per joint

        data.qpos[:] = data.qpos + dq_full

    mujoco.mj_forward(model, data)
    return data.qpos[ARM_QPOS_ADDR].copy()

# ------------------ mapping ------------------
def build_arm_mapping_from_model(model, gripper_act_id):
    """Infer 7 arm actuator IDs and their joint qpos addresses from the model."""
    # 1) candidate arm actuators = all except gripper
    cand_act = [aid for aid in range(model.nu) if aid != gripper_act_id]
    if len(cand_act) < 7:
        raise RuntimeError(f"Expected at least 7 arm actuators, found {len(cand_act)}")

    # 2) for each actuator, get the joint it drives and that joint's qpos address
    pairs = []
    for aid in cand_act:
        jid = int(model.actuator_trnid[aid][0])           # actuator -> joint id
        qadr = int(model.jnt_qposadr[jid])                # joint -> qpos address
        pairs.append((aid, jid, qadr))

    # 3) sort by qpos address to establish arm order
    pairs.sort(key=lambda x: x[2])

    # 4) keep the first 7 after sorting (in case there are extra non-arm actuators)
    pairs = pairs[:7]
    arm_act_ids  = np.array([aid for (aid, _, _) in pairs], dtype=int)
    arm_qpos_adr = np.array([qadr for (_, _, qadr) in pairs], dtype=int)

    # (optional) prints for sanity
    try:
        jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[aid][0])) for aid in arm_act_ids]
        anames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) for aid in arm_act_ids]
        print("[MAP] arm actuator names:", anames)
        print("[MAP] arm joint   names:", jnames)
    except Exception:
        pass
    return arm_act_ids, arm_qpos_adr

# First build mapping
ARM_ACT_IDS, ARM_QPOS_ADDR = build_arm_mapping_from_model(model, GRIPPER_IDX)
print("[MAP] ARM_ACT_IDS :", ARM_ACT_IDS)
print("[MAP] ARM_QPOS_ADDR:", ARM_QPOS_ADDR)

# Then build DOF index array
ARM_DOF_IDX = build_arm_dof_indices(model, ARM_ACT_IDS)
print("[MAP] ARM_DOF_IDX:", ARM_DOF_IDX)

def get_arm_qpos():
    return data.qpos[ARM_QPOS_ADDR].copy()

def set_arm_qpos(q):
    data.qpos[ARM_QPOS_ADDR] = q
    mujoco.mj_forward(model, data)

def set_arm_ctrl_positions(q_target):
    data.ctrl[ARM_ACT_IDS] = q_target

# ------------------ Oracle Function ------------------
def get_target_for_step(step):
    idx = min(step, len(waypoints) - 1)
    return waypoints[idx]


def oracle(step):
    target = get_target_for_step(step)
    q_target, g_target = target[:7], target[7]
    set_arm_ctrl_positions(q_target)
    data.ctrl[GRIPPER_IDX] = g_target



#-------------------Oracle controller with IKpy----------
def slerp(q1, q2, t):
    # q = [w, x, y, z], unit
    dot = np.dot(q1, q2)
    if dot < 0.0: q2, dot = -q2, -dot
    if dot > 0.9995:  # linear fallback
        q = q1 + t*(q2 - q1); return q/np.linalg.norm(q)
    theta0 = np.arccos(dot); sin0 = np.sin(theta0)
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    return s0*q1 + s1*q2

def interpolate_pose(pA, qA, pB, qB, n):
    poses = []
    for i in range(n+1):
        t = i / n
        p = (1-t)*pA + t*pB
        q = slerp(qA, qB, t)
        poses.append((p, q))
    return poses

def _unit_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])

def _feasible_full_init_from_qstart(chain, q_start7):
    """
    Build a full-length initial_position vector for IKPy from the current 7-DoF joints,
    clamped to URDF bounds and aligned with chain.active_links_mask.
    """
    # 1) Full vector length must equal number of links in the chain
    n_full = len(chain.links)
    x0_full = np.zeros(n_full, dtype=float)

    # 2) Active joint indices in the full vector
    active_idx = [i for i, m in enumerate(chain.active_links_mask) if m]

    # 3) Gather bounds for active joints (fallback to [-pi, pi] when missing)
    lowers, uppers = [], []
    for idx in active_idx:
        link = chain.links[idx]
        b = getattr(link, "bounds", (None, None))
        lo = -np.pi if b[0] is None else float(b[0])
        hi =  np.pi if b[1] is None else float(b[1])
        lowers.append(lo); uppers.append(hi)

    # 4) Clamp your 7-DoF seed into those bounds
    q_start7 = np.asarray(q_start7, dtype=float)
    kmax = min(len(q_start7), len(active_idx))
    q_clamped = np.clip(q_start7[:kmax], np.array(lowers[:kmax]), np.array(uppers[:kmax]))

    # 5) Write the clamped angles into the corresponding active slots
    for k in range(kmax):
        x0_full[active_idx[k]] = q_clamped[k]

    return x0_full

def _body_pose_world(model, data, body_name):
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mujoco.mj_forward(model, data)
    # MuJoCo Python uses xpos/xmat for bodies (shape: [nbody,3], [nbody,9])
    if hasattr(data, "xmat"):
        R = data.xmat[bid].reshape(3, 3).copy()
        t = data.xpos[bid].copy()
    else:
        # older bindings fallback
        R = data.body_xmat[bid].reshape(3, 3).copy()
        t = data.body_xpos[bid].copy()
    return R, t

def world_to_base(p_world, q_wxyz_world, R_base, t_base):
    # position
    p_base = R_base.T @ (np.asarray(p_world) - t_base)
    # orientation: q_base_world^-1 * q_world
    from scipy.spatial.transform import Rotation as R
    q_base_world = R.from_matrix(R_base)
    q_world = R.from_quat([q_wxyz_world[1], q_wxyz_world[2], q_wxyz_world[3], q_wxyz_world[0]])
    q_base = q_base_world.inv() * q_world
    q_wxyz_base = np.roll(q_base.as_quat(), 1)   # to (w,x,y,z)
    return p_base, q_wxyz_base

def ik_to_joints(panda_chain, p_world, q_wxyz_world, q_start7=None, use_orientation=False):
    # Get base pose of the Panda in the world (MuJoCo)
    R_base, t_base = _body_pose_world(model, data, "panda_link0")  # <- check base body name
    # Transform goal into the chain’s base frame
    p, q_wxyz = world_to_base(p_world, _unit_quat_wxyz(q_wxyz_world), R_base, t_base)

    # ------- same as you have now below -------
    if q_start7 is None:
        q_start7 = np.zeros(7)
    x0_full = _feasible_full_init_from_qstart(panda_chain, q_start7)

    if not use_orientation:
        q_sol_full = panda_chain.inverse_kinematics(
            target_position=np.asarray(p, dtype=float),
            initial_position=x0_full,
        )
    else:
        T = np.eye(4)
        T[:3,:3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        T[:3,3] = p
        q_sol_full = panda_chain.inverse_kinematics_frame(T, initial_position=x0_full)

    return np.asarray(q_sol_full[:7], dtype=float)


def plan_AB_cartesian_to_joint_traj(panda_chain, q_start, goal, n_cart=40):
    pB, qB = goal["pos"], goal["quat_wxyz"]

    # Build a minimal cartesian path (we can keep it just B for now)
    cart = [(pB, _unit_quat_wxyz(qB))] * (n_cart + 1)

    path_q = []
    qs = q_start.copy()
    for (p, q) in cart:
        # Position-only first: use_orientation=False
        # Move the sim to the current seed
        set_arm_qpos(qs)
        # Solve MuJoCo IK to target position (choose your EE site/body below)
        qj = mj_inverse_kinematics_pos(
            model, data, target_pos_world=p,
            # Prefer a TCP site if your scene has it; else use the hand body:
            # site_name="panda_hand_tcp",
            body_name="panda_link8",   # or "panda_hand" if that exists
            max_iters=200, tol=1e-4, damping=1e-3, step_scale=0.8
        )

        path_q.append(qj)
        qs = qj

    # Densify in joint space
    traj = []
    qs = q_start.copy()
    for qg in path_q:
        steps = max(3, int(np.linalg.norm(qg - qs) / 0.02))
        for i in range(1, steps + 1):
            traj.append(qs + (i / steps) * (qg - qs))
        qs = qg
    return traj

# ---------------- setup goal pose ------------
# Reset to a starting joint configuration
start_joints = np.array([0, -0.5, 0, -1.5, 0, 1.5, 0])
set_arm_qpos(start_joints)


goal = {
    "pos": np.array([0.43, 0, 0.3]),              # meters, world frame
    "quat_wxyz": np.array([ 0.70710678,  0., -0.70710678,  0.]),  # unit quaternion
    "gripper_open": True                     # or False
}

# --- plan & replace old waypoints ---
q_start = get_arm_qpos()

traj_q = plan_AB_cartesian_to_joint_traj(panda_chain, q_start, goal, n_cart=40)
# --- quick MuJoCo forward check ---
_state_backup = data.qpos.copy()
set_arm_qpos(traj_q[-1])
ee = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link8")].copy()
err = np.linalg.norm(ee - goal["pos"])
print(f"[DEBUG] EE final world pos: {ee}, goal: {goal['pos']}, |err|={err:.4f} m")
data.qpos[:] = _state_backup
mujoco.mj_forward(model, data)
# --------- deep debug: compare IK target vs FK vs MuJoCo EE frames ----------
def list_names(model, obj_type):
    import mujoco
    n = {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_SITE: model.nsite,
        mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
    }[obj_type]
    out = []
    for i in range(n):
        try:
            out.append(mujoco.mj_id2name(model, obj_type, i))
        except Exception:
            out.append(None)
    return out

def build_full_from_q7(chain, q7):
    """Embed 7-DoF arm into a full chain vector."""
    x = np.zeros(len(chain.links))
    active_idx = [i for i,m in enumerate(chain.active_links_mask) if m]
    for k, idx in enumerate(active_idx[:7]):
        x[idx] = q7[k]
    return x

# base transform used for IK
R_base, t_base = _body_pose_world(model, data, "panda_link0")  # <-- check your base body name
print("[DBG] base t:", t_base, "\n[DBG] base R row0:", R_base[0])

# the exact target we passed to IK, in base frame:
p_base, _ = world_to_base(goal["pos"], goal["quat_wxyz"], R_base, t_base)
print("[DBG] target (base frame):", p_base)

# FK from IKPy at last config
q_goal = traj_q[-1]
x_full = build_full_from_q7(panda_chain, q_goal)
T_fk = panda_chain.forward_kinematics(x_full)  # 4x4 in base frame
p_fk_base = T_fk[:3, 3]
p_fk_world = R_base @ p_fk_base + t_base
print("[DBG] IKPy FK world:", p_fk_world, " vs goal world:", goal["pos"],
      " |err|=", np.linalg.norm(p_fk_world - goal["pos"]))

# Also print a few likely EE frames from MuJoCo to see which matches:
import mujoco
candidates = {
    "body:panda_link8": ("body", "panda_link8"),
    "body:panda_hand":  ("body", "panda_hand"),
    "site:panda_hand_tcp": ("site", "panda_hand_tcp"),
    "site:attachment_site": ("site", "attachment_site"),
}
for label, (kind, name) in candidates.items():
    try:
        if kind == "body":
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = data.xpos[bid].copy()
        else:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            pos = data.site_xpos[sid].copy()
        print(f"[DBG] {label} world pos:", name, pos)
    except Exception:
        pass
# ---------------------------------------------------------------------------



# attach gripper command to each joint vector
g_cmd = 0.6 if goal["gripper_open"] else 0.0
waypoints = np.concatenate([np.array(traj_q), np.full((len(traj_q), 1), g_cmd)], axis=1)


# # ------------------ Waypoints ------------------
# # Each includes 8 values (7 arm joints + 1 gripper)
# waypoints = np.array([ 
#         [1.400000, 0.650000, -1.300000, -1.850000, -0.100000, 1.150000, 0.000000, 0.000000],
#         [1.400000, 0.650000, -1.300000, -2.000000, -0.100000, 1.450000, 0.600000, 0.000000],
#         [1.400000, 0.650000, -1.300000, -2.000000, -0.100000, 1.450000, 0.600000, 0.000000]
#     ])




# ------------------ Dataset Recording ------------------
episodes = []
current_episode = []
language_command = "Pick the tomato from the top truss closest to you."

for step in range(400):
    oracle(step)
    action_snapshot = data.ctrl.copy()
    mujoco.mj_step(model, data)

    obs = {
        "state": data.qpos.copy(),
        "action": action_snapshot,
        "reward": 0,
        "is_terminal": step == 399,
        "is_first": step == 0,
        "language_command": language_command,
    }
    current_episode.append(obs)

episodes.append(current_episode)

# ------------------ TFRecord Saving ------------------
output_path = "/home/myrtheiw/octo_ws/octo/record_dataset/tomato_dataset.tfrecord"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with tf.io.TFRecordWriter(output_path) as writer:
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

print(f"✅ Dataset saved with {len(episodes)} episode(s) to {output_path}")
