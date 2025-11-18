#!/usr/bin/env python3
import argparse
import time
import numpy as np
import tensorflow as tf
import mujoco
import imageio


def get_args():
    p = argparse.ArgumentParser(
        description="Replay RLDS/EnvLogger episodes from TFRecord into MuJoCo video (correctly handling joint-delta, joint-abs, or raw-ctrl actions)."
    )
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to a TFRecord shard with RLDS/EnvLogger episodes.")
    p.add_argument("--model_xml", type=str, required=True, help="MuJoCo XML model.")
    p.add_argument("--episode", type=int, default=0, help="Episode index to render.")
    p.add_argument("--out", type=str, default="replay_actions.mp4", help="Output MP4 path.")
    p.add_argument("--fps", type=int, default=50, help="Video frames per second.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--sleep", type=float, default=0.0, help="Optional real-time sleep per frame (seconds).")
    p.add_argument("--camera", type=str, default=None, help="Named camera in the model (optional).")

    # Replay semantics
    p.add_argument("--mode", type=str, default="auto",
                   choices=["auto", "joint_delta", "joint_abs", "ctrl"],
                   help="How to interpret actions. 'auto' tries to read EnvLogger metadata.")
    p.add_argument("--action_scale", type=float, default=None,
                   help="Override action scale. If not set, use metadata or default 1.0.")
    p.add_argument("--dof", type=int, default=7, help="Number of arm joints encoded in the action/proprio.")
    p.add_argument("--use_proprio_init", action="store_true",
                   help="Use proprio[0] once at t=0 to initialize qpos before action-only replay.")

    # For raw-ctrl fallback only
    p.add_argument("--nsteps_per_action", type=int, default=1,
                   help="Number of mj_step() integrations when --mode=ctrl.")
    p.add_argument("--zero_vel", action="store_true",
                   help="If set and --mode=ctrl, zero qvel before applying each action (helps with stability).")
    return p.parse_args()


def _try_read_metadata(ex):
    f = ex.features.feature
    m = {}
    # EnvLogger typically stores these in a 'metadata/...' namespace (float or bytes)
    if "metadata/action_type" in f and f["metadata/action_type"].bytes_list.value:
        m["action_type"] = f["metadata/action_type"].bytes_list.value[0].decode("utf-8")
    if "metadata/action_scale" in f:
        fl = f["metadata/action_scale"]
        if fl.float_list.value:
            m["action_scale"] = float(fl.float_list.value[0])
    if "metadata/action_dt_sec" in f:
        fl = f["metadata/action_dt_sec"]
        if fl.float_list.value:
            m["action_dt_sec"] = float(fl.float_list.value[0])
    return m


def iter_episodes(dataset_path):
    """Yield (episode_dict, metadata_dict). Each episode_dict has:
       { 'proprio': [T, Dp], 'action': [T, Da], 'is_first': [T], 'is_terminal': [T]|None }"""
    ds = tf.data.TFRecordDataset(dataset_path)  # expects uncompressed TFRecord
    meta = {}
    cur = {"proprio": [], "action": [], "is_first": [], "is_terminal": []}
    # Many EnvLogger exports pack one episode per tf.Example. We'll treat each Example as an episode.
    for raw in ds:
        ex = tf.train.Example.FromString(raw.numpy())
        # Try to fill metadata (only first time needed)
        if not meta:
            meta = _try_read_metadata(ex)

        f = ex.features.feature
        def _floats(k): return np.array(f[k].float_list.value, dtype=np.float32) if k in f else None
        def _ints(k):   return np.array(f[k].int64_list.value, dtype=np.int64) if k in f else None

        fl_prop   = _floats("steps/observation/proprio")
        fl_action = _floats("steps/action")
        is_first  = _ints("steps/is_first")
        is_term   = _ints("steps/is_terminal") if "steps/is_terminal" in f else None

        if fl_prop is None or fl_action is None or is_first is None or is_first.size == 0:
            # skip malformed records
            continue

        T = is_first.size
        Dp = fl_prop.size // T
        Da = fl_action.size // T
        proprio = fl_prop.reshape(T, Dp)
        action  = fl_action.reshape(T, Da)
        yield {
            "proprio": proprio,
            "action": action,
            "is_first": is_first.astype(np.bool_),
            "is_terminal": (is_term.astype(np.bool_) if is_term is not None else None),
        }, meta


def resolve_mode_and_scale(cli_mode, cli_scale, meta):
    # Mode
    if cli_mode != "auto":
        mode = cli_mode
    else:
        mode = meta.get("action_type", "joint_delta")
        if mode == "joint_absolute":
            mode = "joint_abs"

    # Scale
    if cli_scale is not None:
        scale = float(cli_scale)
    else:
        scale = float(meta.get("action_scale", 1.0))

    return mode, scale


def main():
    args = get_args()

    # Load model & data
    model = mujoco.MjModel.from_xml_path(args.model_xml)
    data = mujoco.MjData(model)

    # Optional camera id
    cam_id = None
    if args.camera is not None:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if cid >= 0:
            cam_id = cid
        else:
            print(f"[warn] Camera '{args.camera}' not found; using default free camera.")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Gather episodes
    eps = list(iter_episodes(args.dataset_path))
    eps_only = [e for e, _ in eps]
    metas    = [m for _, m in eps]
    if not eps_only:
        raise RuntimeError(f"No episodes read from {args.dataset_path} — check path/format.")  # :contentReference[oaicite:1]{index=1}

    if args.episode < 0 or args.episode >= len(eps_only):
        raise IndexError(f"Episode index {args.episode} out of range [0, {len(eps_only)-1}]")  # :contentReference[oaicite:2]{index=2}

    ep   = eps_only[args.episode]
    meta = metas[args.episode] if args.episode < len(metas) else {}
    actions = ep["action"]
    proprio = ep["proprio"]
    is_first = ep["is_first"]
    T, Da = actions.shape
    Dp = proprio.shape[1]

    mode, scale = resolve_mode_and_scale(args.mode, args.action_scale, meta)
    dof = int(args.dof)

    print(f"[info] Episode {args.episode}: T={T}, action_dim={Da}, proprio_dim={Dp}")
    print(f"[info] Mode={mode}, scale={scale}, dof={dof}")
    print(f"[info] use_proprio_init={args.use_proprio_init} (only t=0)")
    if mode == "joint_delta":
        print("[info] joint_delta action stats (min/max/mean/std per dim):")
        for j in range(Da):
            dim = actions[:, j]
            print(
                f"  j={j}: min={dim.min():+.5f}, max={dim.max():+.5f}, "
                f"mean={dim.mean():+.5f}, std={dim.std():.5f}"
            )

    frames = []

    # Reset to the model default pose; optional proprio init will occur inside the loop at t=0.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Helper for rendering one frame
    def render_frame():
        if cam_id is not None:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        frames.append(renderer.render())

    # Main replay loop — after the optional t=0 alignment the robot evolves purely under actions.
    for t in range(T):
        if is_first[t]:
            mujoco.mj_resetData(model, data)
            # Only the very first timestep may use proprio to align the initial robot pose.
            if args.use_proprio_init and t == 0:
                data.qpos[:dof] = proprio[0, :dof]
            mujoco.mj_forward(model, data)

        if mode == "joint_delta":
            # Interpret actions as Δq over 1 step; apply to current joints
            q_now = data.qpos[:dof].copy()
            dq = actions[t, :dof]
            q_next = q_now + scale * dq
            data.qpos[:dof] = q_next
            mujoco.mj_forward(model, data)

        elif mode == "joint_abs":
            # Interpret actions as absolute joint targets
            q_next = scale * actions[t, :dof]
            data.qpos[:dof] = q_next
            mujoco.mj_forward(model, data)

        elif mode == "ctrl":
            # Fallback: treat actions as raw ctrl inputs (original behavior)
            if args.zero_vel:
                data.qvel[:] = 0.0
            if Da > model.nu:
                raise RuntimeError(f"Action dim ({Da}) exceeds model.nu ({model.nu}) in ctrl mode.")
            data.ctrl[:Da] = actions[t]
            nsub = max(1, int(args.nsteps_per_action))
            for _ in range(nsub):
                mujoco.mj_step(model, data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        render_frame()
        if args.sleep > 0:
            time.sleep(args.sleep)

    fps = max(1, int(round(args.fps)))
    imageio.mimsave(args.out, frames, fps=fps)
    print(f"[done] Saved {args.out} with {len(frames)} frames at {fps} FPS.")


if __name__ == "__main__":
    main()
