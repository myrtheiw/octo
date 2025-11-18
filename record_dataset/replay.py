#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import tensorflow as tf
import mujoco
import imageio

# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser(description="Replay RLDS/EnvLogger episodes from TFRecord into MuJoCo video.")
    p.add_argument("--dataset_path", type=str,
                   default="/home/myrtheiw/tfds_out/tomato_rlds/0.0.34/tomato_rlds-train.tfrecord-00074-of-00090",
                   help="Path to TFRecord shard (uncompressed).")
    p.add_argument("--model_xml", type=str,
                   default="/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml",
                   help="MuJoCo XML model for replay.")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay from the shard.")
    p.add_argument("--fps", type=float, default=50.0, help="Output video FPS.")
    p.add_argument("--out", type=str, default="replay_episode.mp4", help="Output MP4 filename.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--sleep", type=float, default=0.0, help="Optional real-time sleep per frame (seconds).")
    p.add_argument("--camera", type=str, default=None, help="Optional camera name in the model.")
    p.add_argument("--qpos_dim", type=int, default=9, help="How many proprio dims map to qpos (prefix).")
    return p.parse_args()

# -----------------------------
# TFRecord episode reader (RLDS/EnvLogger, Example with float/int lists)
# -----------------------------
def iter_episodes(dataset_path):
    """Yield episodes as dict with numpy arrays:
       { 'proprio': [T, Dp], 'action': [T, Da], 'is_first': [T], 'is_terminal': [T] (optional) }"""
    ds = tf.data.TFRecordDataset(dataset_path)  # uncompressed (your shard is not gzip)
    for raw in ds:
        ex = tf.train.Example.FromString(raw.numpy())
        f = ex.features.feature

        def _floats(k):
            return np.array(f[k].float_list.value, dtype=np.float32) if k in f else None
        def _ints(k):
            return np.array(f[k].int64_list.value, dtype=np.int64) if k in f else None

        # Required keys in your shard
        fl_prop   = _floats("steps/observation/proprio")
        fl_action = _floats("steps/action")
        is_first  = _ints("steps/is_first")

        if fl_prop is None or fl_action is None or is_first is None or is_first.size == 0:
            # Skip malformed record
            continue

        T = int(is_first.size)
        if fl_prop.size % T != 0 or fl_action.size % T != 0:
            # Unexpected shapes; skip
            continue

        Dp = fl_prop.size // T
        Da = fl_action.size // T

        proprio = fl_prop.reshape(T, Dp)
        action  = fl_action.reshape(T, Da)
        is_term = _ints("steps/is_terminal")
        if is_term is not None and is_term.size == T:
            is_term = is_term.astype(bool)
        else:
            is_term = None

        yield {
            "proprio": proprio,
            "action": action,
            "is_first": is_first.astype(bool),
            "is_terminal": is_term,
        }

# -----------------------------
# MuJoCo setup and replay
# -----------------------------
def main():
    args = get_args()

    # Load Mujoco model & data
    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)

    # Optional: pick camera id if provided
    cam_id = None
    if args.camera is not None:
        # Find camera by name
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if cam_id < 0:
            print(f"[warn] Camera '{args.camera}' not found; using default free camera.")
            cam_id = None

    # Renderer
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Gather episodes
    eps = list(iter_episodes(args.dataset_path))
    if not eps:
        raise RuntimeError(f"No episodes read from {args.dataset_path} — check path/format.")

    if args.episode < 0 or args.episode >= len(eps):
        raise IndexError(f"Episode index {args.episode} out of range [0, {len(eps)-1}]")

    ep = eps[args.episode]
    proprio = ep["proprio"]   # [T, Dp], your Dp is 9
    actions = ep["action"]    # [T, Da], your Da is 7
    T, Dp = proprio.shape
    print(f"[info] Replaying episode {args.episode}: T={T}, proprio_dim={Dp}, action_dim={actions.shape[1]}")

    # Video frames
    frames = []
    qdim = min(args.qpos_dim, Dp, model.nq)  # map first qdim from proprio to qpos
    print(f"[info] Mapping proprio[:{qdim}] -> qpos[:{qdim}]")

    # Replay loop (set qpos from proprio; you can also set data.ctrl if desired)
    for t in range(T):
        # Reset velocities to avoid drift / ringing
        data.qvel[:] = 0.0
        # Map the first qdim proprio entries directly onto qpos
        data.qpos[:qdim] = proprio[t, :qdim]
        mujoco.mj_forward(model, data)

        # Render
        if cam_id is not None:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        rgb = renderer.render()
        frames.append(rgb)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Save MP4
    fps = max(1, int(round(args.fps)))
    imageio.mimsave(args.out, frames, fps=fps)
    print(f"[done] Saved {args.out} with {len(frames)} frames at {fps} FPS.")

if __name__ == "__main__":
    main()
