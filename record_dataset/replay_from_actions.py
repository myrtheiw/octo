#!/usr/bin/env python3
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering

import argparse
import time
import numpy as np
import tensorflow as tf
import mujoco
import imageio.v2 as imageio  # pip install imageio

# Match the oracle’s scale and timing
ORACLE_SCALE       = 0.02        # joint delta scale used when logging
TARGET_ACTION_DT   = 0.1         # ~0.1 s per outer step in oracle
TYPICAL_FRANKA_LIMITS = [
    (-2.8973,  2.8973),
    (-1.7628,  1.7628),
    (-2.8973,  2.8973),
    (-3.0718, -0.0698),
    (-2.8973,  2.8973),
    (-0.0175,  3.7525),
    (-2.8973,  2.8973),
]

# Note: KP/KD are handled by the XML internal controller (gainprm), 
# so we do not manually calculate torque here.

def get_args():
    p = argparse.ArgumentParser(description="Replay RLDS episode from joint deltas and record video")
    p.add_argument("--dataset_path", type=str, required=True, help="Path to TFRecord file")
    p.add_argument("--model_xml",   type=str, required=True, help="Path to scene_dynamic.xml")
    p.add_argument("--episode",     type=int, default=0, help="Episode index to replay")
    p.add_argument("--video_path",  type=str, default="replay.gif", help="Output video (gif or mp4)")
    p.add_argument("--camera_name", type=str, default="front_cam", help="MuJoCo camera name to render from")
    return p.parse_args()


def iter_episodes(dataset_path):
    """Simple generator to read actions/proprio from TFRecord."""
    ds = tf.data.TFRecordDataset(dataset_path)
    for raw in ds:
        ex = tf.train.Example.FromString(raw.numpy())
        f = ex.features.feature

        proprio = np.array(
            f["steps/observation/proprio"].float_list.value, dtype=np.float32
        )
        action = np.array(
            f["steps/action"].float_list.value, dtype=np.float32
        )
        is_first = np.array(
            f["steps/is_first"].int64_list.value, dtype=np.int64
        )

        T = len(is_first)
        if T == 0:
            continue

        Dp = len(proprio) // T
        Da = len(action) // T

        yield {
            "proprio": proprio.reshape(T, Dp),
            "action":  action.reshape(T, Da),
            "is_first": is_first,
        }


# --------------------------- Model/Actuator mapping ---------------------------

def build_arm_mapping_from_model(model, prefer_position=True):
    """
    Return (arm_act_ids, arm_qpos_addr) for Panda joints (link1..link7).
    """
    arm_body_names = [f"link{i}" for i in range(1, 8)]
    grip_keywords = ("grip", "finger", "hand")

    # 1) Find hinge joints on link1..link7
    arm_joint_ids = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        b = int(model.jnt_bodyid[j])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bname in arm_body_names:
            arm_joint_ids.append(j)

    # 2) For each such joint, collect candidate actuators
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
        per_joint_candidates[jid].append(
            (aid, jid, qadr, jname, aname, gaintype, bname)
        )

    # 3) Choose one actuator per joint, preferring position-type if requested
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

    # 4) Keep only actuators on arm bodies, dedupe per joint, sort by qpos adr
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
    """Exactly the helper from oracle_dynamic_norm.py."""
    pairs = []
    for aid in arm_act_ids:
        jid = int(model.actuator_trnid[aid][0])
        dof = next(d for d in range(model.nv) if int(model.dof_jntid[d]) == jid)
        qadr = int(model.jnt_qposadr[jid])
        pairs.append((qadr, dof))
    pairs.sort(key=lambda t: t[0])
    return np.array([dof for (_, dof) in pairs], dtype=int)


# -----------------------------------------------------------------------------


def main():
    args = get_args()

    # 1. Load model and data
    try:
        from helpers import build_and_load_scene as _build_and_load_scene
        print(f"[-] Loading model via helpers.build_and_load_scene: {args.model_xml}")
        model, data = _build_and_load_scene(args.model_xml)
    except Exception as e:
        print(f"[!] helpers.build_and_load_scene failed ({e}), falling back to raw XML.")
        model = mujoco.MjModel.from_xml_path(args.model_xml)
        data  = mujoco.MjData(model)

    # 2. Build arm mapping and DOF indices
    arm_act_ids, arm_qpos_adr = build_arm_mapping_from_model(model, prefer_position=True)
    arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)
    print(f"[-] Arm actuators: {arm_act_ids.tolist()}")
    print(f"[-] Arm qpos addr: {arm_qpos_adr.tolist()}")

    # 3. Apply the same joint limit widening as oracle
    for i, dof in enumerate(arm_dof_idx):
        jid = int(model.dof_jntid[dof])
        model.jnt_limited[jid] = 1
        lo, hi = TYPICAL_FRANKA_LIMITS[i]
        model.jnt_range[jid][0] = lo
        model.jnt_range[jid][1] = hi

    # 4. Compute substeps to match oracle's action rate
    timestep = float(model.opt.timestep)
    substeps = max(1, int(round(TARGET_ACTION_DT / max(timestep, 1e-6))))
    print(f"[-] timestep={timestep:.4g}s  substeps={substeps}  outer_dt≈{timestep*substeps:.3f}s")

    # 5. Load episodes from TFRecord
    print(f"[-] Loading Episode {args.episode} from {args.dataset_path}...")
    episodes = list(iter_episodes(args.dataset_path))
    if args.episode >= len(episodes):
        print(f"[!] Episode {args.episode} not found. Dataset has {len(episodes)} episodes.")
        return

    ep = episodes[args.episode]
    actions = ep["action"]    # [T, Da] normalized Δq
    proprio = ep["proprio"]   # [T, nq] full qpos
    T = len(actions)

    if T < 2:
        print("[!] Episode too short to replay (T < 2)")
        return

    # 6. Extract arm joint trajectory from proprio
    q_arm_all = proprio[:, arm_qpos_adr]  # [T, 7]

    print(f"[-] Replaying {T} steps offscreen and recording to {args.video_path}")

    # 7. Initialize sim state from full proprio (t=0)
    q_full0 = proprio[0].copy()
    q_arm0  = q_arm_all[0].copy()

    mujoco.mj_resetData(model, data)
    nq = min(model.nq, q_full0.shape[0])
    data.qpos[:nq] = q_full0[:nq]
    mujoco.mj_forward(model, data)

    # Initialize the virtual target with the starting physical position
    virtual_target = q_arm0.copy()

    # 8. Set up offscreen renderer
    H, W = 480, 640
    renderer = mujoco.Renderer(model, height=H, width=W)

    # Try to resolve the camera name; fall back to free camera if missing
    try:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
        has_cam = cam_id >= 0
    except Exception:
        has_cam = False
        cam_id = -1
    if has_cam:
        print(f"[-] Using camera '{args.camera_name}' (id={cam_id})")
    else:
        print(f"[!] Camera '{args.camera_name}' not found; using free camera.")

    frames = []

    def render_frame():
        if has_cam:
            renderer.update_scene(data, camera=args.camera_name)
        else:
            renderer.update_scene(data)
        img = renderer.render()  # (H, W, 3), uint8
        return img

    # Record initial pose
    frames.append(render_frame())

    # 9. Replay loop
    for t in range(1, T):
        # 1) Un-normalize action to Δq (radians)
        a_norm = actions[t]
        if a_norm.shape[0] < 7:
            raise ValueError(f"Expected ≥7 action dims, got {a_norm.shape}")
        dq = a_norm[:7] * ORACLE_SCALE 

        # 2) Integrate into virtual joint target
        virtual_target += dq

        # 3) Send absolute target to the mapped actuators (Corrected Control)
        # We rely on the internal XML gainprm="4500" to generate the forces
        data.ctrl[arm_act_ids] = virtual_target

        # 4) Step physics for substeps
        for _ in range(substeps):
            # REMOVED: Manual torque calculation (tau = KP * ...)
            # The internal controller handles F = gainprm * (ctrl - qpos)
            mujoco.mj_step(model, data)

        # 5) (Optional) debug on a few steps
        if t <= 5 or t % 20 == 0:
            q_dataset = q_arm_all[t]
            q_sim = data.qpos[arm_qpos_adr].copy()
            print(
                f"[t={t:3d}] "
                f"dq={np.round(dq, 4)}, "
                f"q_target={np.round(virtual_target, 3)}, "
                f"q_sim={np.round(q_sim, 3)}"
            )

        # 6) Render and store frame
        frames.append(render_frame())

    # 10. Write video (GIF or MP4)
    out_path = args.video_path
    ext = os.path.splitext(out_path)[1].lower()
    fps = int(round(1.0 / TARGET_ACTION_DT))

    if ext == ".gif":
        imageio.mimsave(out_path, frames, fps=fps)
    else:
        # Use mp4 with ffmpeg if imageio has it
        imageio.mimsave(out_path, frames, fps=fps, codec="libx264")

    print(f"[-] Saved video to {out_path}")


if __name__ == "__main__":
    main()