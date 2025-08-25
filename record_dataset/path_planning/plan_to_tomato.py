#plan_to_tomato.py

"""
High-level glue that: (1) computes a collision-free joint path with RRT*,
(2) resamples to dense joint waypoints, and (3) (optionally) hands waypoints
back to your existing oracle for execution and logging.
"""

from __future__ import annotations
import numpy as np
import mujoco


from path_planning.collision_checker import CollisionChecker, default_allow_contact_factory
from path_planning.rrt_star import RRTStar, RRTStarConfig, shortcut_smooth


# We reuse your IK from oracle_recordings to convert the tomato pose into a
# joint goal. If you prefer to avoid that import, copy the function over.
#from ..oracle_recordings import solve_ik_to_pose # noqa: E402


# ---------------------- Limits & helpers ----------------------


def panda_arm_limits_from_model(model: mujoco.MjModel, arm_dof_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) joint limits for the 7 arm joints in DOF order."""
    jids = [int(model.dof_jntid[d]) for d in arm_dof_idx]
    lo = np.array([model.jnt_range[j, 0] for j in jids], dtype=float)
    hi = np.array([model.jnt_range[j, 1] for j in jids], dtype=float)
    return lo, hi




def resample_dense(path: list[np.ndarray], max_step: float = 0.02) -> np.ndarray:
    """Interpolate a list of joint waypoints into a dense array [T,7]."""
    traj = []
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        dist = float(np.linalg.norm(b - a))
        steps = max(1, int(dist / max_step))
        for i in range(1, steps + 1):
            traj.append(a + (i / steps) * (b - a))
    return np.asarray(traj, dtype=np.float32)


# ---------------------- Main entrypoint ----------------------
def plan_joint_traj_to_tomato(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_dof_idx: np.ndarray,
    arm_qpos_addr: np.ndarray,
    ee_ref: tuple[str, str],
    tomato_pos_world: np.ndarray,
    tomato_quat_wxyz: np.ndarray,
    allow_contact=None,
    rrt_cfg: RRTStarConfig | None = None,
    ) -> np.ndarray:
    """
    Returns a [T,7] joint trajectory from the *current* qpos to an IK goal near
    the tomato, collision-checking against the full MuJoCo scene.
    """
    # Compute an IK goal at/near the tomato pose
    q_backup = data.qpos.copy()
    try:
        data.qpos[arm_qpos_addr] = data.qpos[arm_qpos_addr] # explicit no-op seed
        mujoco.mj_forward(model, data)
        q_goal = solve_ik_to_pose(
            model, data,
            target_pos_world=tomato_pos_world,
            target_quat_wxyz=tomato_quat_wxyz,
            arm_dof_idx=arm_dof_idx,
            arm_qpos_addr=arm_qpos_addr,
            ee_ref=ee_ref,
    )
    finally:
        data.qpos[:] = q_backup
        mujoco.mj_forward(model, data)


    q_start = data.qpos[arm_qpos_addr].copy()


# Build the collision checker (this is the RRT* "map")
    checker = CollisionChecker(
        model=model,
        data=data,
        arm_qpos_addr=arm_qpos_addr,
        step_rad=0.02,
        allow_contact=allow_contact or default_allow_contact_factory(model),
        )


    # Joint limits and planner
    lo, hi = panda_arm_limits_from_model(model, arm_dof_idx)
    planner = RRTStar(lo, hi, checker, cfg=rrt_cfg or RRTStarConfig())


    # Plan
    path = planner.plan(q_start, q_goal)
    if not path or len(path) < 2:
        raise RuntimeError("RRT* failed to find a path")


    # Smooth + densify
    path = shortcut_smooth(path, checker, attempts=300)
    traj = resample_dense(path, max_step=0.02)
    return traj.astype(np.float32)


# ---------------------- Tiny CLI test ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tomato_x", type=float, required=True)
    parser.add_argument("--tomato_y", type=float, required=True)
    parser.add_argument("--tomato_z", type=float, required=True)
    args = parser.parse_args()


    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)


    from oracle_recordings import (
    find_gripper_actuator, build_arm_mapping_from_model, build_arm_dof_indices, EE_REF
    )
    gripper_idx = find_gripper_actuator(model)
    arm_act_ids, arm_qpos_addr = build_arm_mapping_from_model(model, gripper_idx)
    arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)


    tomato_pos = np.array([args.tomato_x, args.tomato_y, args.tomato_z], dtype=float)
    tomato_quat = np.array([1,0,0,0], dtype=float)


    traj = plan_joint_traj_to_tomato(
    model, data, arm_dof_idx, arm_qpos_addr, EE_REF, tomato_pos, tomato_quat
    )
    print("Planned waypoints:", traj.shape)