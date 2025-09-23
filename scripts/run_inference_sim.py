
# ======================== run_inference_sim.py ========================
#!/usr/bin/env python3
import os, argparse, time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax, jax.numpy as jnp
from functools import partial
import mujoco
from mujoco import viewer

from octo.model.octo_model import OctoModel

from sim_env import (
    PandaSimEnv, build_arm_mapping_from_model, build_arm_dof_indices,
    find_gripper_actuator, auto_ee_ref,
)

# --- dynamic plant builder from your project (octo/record_dataset/helpers.py) ---
try:
    from record_dataset.helpers import build_and_load_scene as _build_and_load_scene
except Exception:
    # Fallback if PYTHONPATH doesn't include the project root:
    # try to add the parent of this script (…/octo) so record_dataset.* is importable
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from record_dataset.helpers import build_and_load_scene as _build_and_load_scene
    except Exception:
        _build_and_load_scene = None

# --- NEW: resolve experiment directories when a parent checkpoints dir is passed ---
import glob, os, json


def resolve_exp_dirs(path: str):
    """
    Return a list of experiment dirs (each containing config.json).
    If `path` itself is an experiment dir, return [path]. If it is a parent,
    return all children (recursive) that have a config.json, sorted by mtime
    (newest first).
    """
    path = os.path.expanduser(path)
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        exps = [path]
    else:
        # Find all config.json files under the parent
        candidates = glob.glob(os.path.join(path, "**", "config.json"), recursive=True)
        # Filter out the parent itself if matched
        candidates = [c for c in candidates if os.path.dirname(c) != path]
        # Sort newest first by file mtime
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        exps = [os.path.dirname(c) for c in candidates]

    if not exps:
        raise SystemExit(
            f"Could not find any config.json under: {path}. Pass --exp to a specific experiment directory."
        )

    # Pretty-print the top few for clarity (safe single-line strings)
    head = "\n  ".join(exps[:5])
    tail = "\n  ..." if len(exps) > 5 else ""
    print("[resolve] Found experiments (newest first):\n  " + head + tail)
    return exps


def _synth_namespaced_timestep(flat: dict):
    if "observation/pad_mask_dict/timestep" in flat and "observation/timestep_pad_mask" not in flat:
        flat["observation/timestep_pad_mask"] = flat["observation/pad_mask_dict/timestep"]
    if "observation/timestep_pad_mask" in flat and "observation/pad_mask_dict/timestep" not in flat:
        flat["observation/pad_mask_dict/timestep"] = flat["observation/timestep_pad_mask"]


def _add_legacy_aliases(d: dict):
    for k in ("proprio", "timestep", "task_completed", "image_primary", "image_wrist"):
        nk = f"observation/{k}"
        if nk in d and k not in d:
            d[k] = d[nk]
    for k in ("proprio", "timestep", "image_primary", "image_wrist"):
        nk = f"observation/pad_mask_dict/{k}"
        lk = f"pad_mask_dict/{k}"
        if nk in d and lk not in d:
            d[lk] = d[nk]
    src = d.get("observation/timestep_pad_mask") or d.get("observation/pad_mask_dict/timestep")
    if src is not None:
        d.setdefault("timestep_pad_mask", src)
        d.setdefault("pad_mask_dict/timestep", src)


def supply_rng(fn, seed=0):
    rng = jax.random.PRNGKey(int(seed))

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, sub = jax.random.split(rng)
        try:
            return fn(*args, rng=sub, **kwargs)
        except TypeError:
            try:
                return fn(*args, prng_key=sub, **kwargs)
            except TypeError:
                return fn(*args, **kwargs)

    return wrapped


def _nearest_downsample(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    H, W = img.shape[:2]
    if (H, W) == (Ht, Wt):
        return img
    # integer-stride nearest neighbor (fast, no deps)
    sh = max(1, int(round(H / max(1, Ht))))
    sw = max(1, int(round(W / max(1, Wt))))
    return img[::sh, ::sw]

def build_obs_for_octo(ts_obs, t_idx: int, img_shapes: dict | None = None):
    """
    Wrap env obs into Octo format and provide BOTH namespaced and top-level pad masks.
    This silences 'No pad_mask_dict found' and avoids inputs being ignored.
    """
    img_p = np.asarray(ts_obs["image_primary"], np.uint8)
    img_w = np.asarray(ts_obs["image_wrist"],   np.uint8)
    if img_shapes:
        if "image_primary" in img_shapes:
            img_p = _nearest_downsample(img_p, img_shapes["image_primary"])
        if "image_wrist" in img_shapes:
            img_w = _nearest_downsample(img_w, img_shapes["image_wrist"])

    img_p = img_p[None, None, ...]  # (1,1,H,W,3)
    img_w = img_w[None, None, ...]
    proprio = np.asarray(ts_obs["proprio"], np.float32)[None, None, ...]
    timestep = np.array([[t_idx]], np.int32)

    # --- base namespaced keys ---
    obs = {
        "observation/image_primary":  img_p,
        "observation/image_wrist":    img_w,
        "observation/proprio":        proprio,
        "observation/task_completed": np.zeros((1, 1, 4), np.float32),
        "observation/timestep":       timestep,
        "observation/pad_mask_dict/image_primary": np.ones((1, 1), bool),
        "observation/pad_mask_dict/image_wrist":   np.ones((1, 1), bool),
        "observation/pad_mask_dict/proprio":       np.ones((1, 1), bool),
        "observation/pad_mask_dict/timestep":      np.ones((1, 1), bool),
    }

    # --- add legacy/toplevel aliases the checkpoint expects ---
    _synth_namespaced_timestep(obs)
    _add_legacy_aliases(obs)
    # Critically, also inject TOP-LEVEL pad_mask_dict/* (the warnings complain about these)
    obs["pad_mask_dict/image_primary"] = np.ones((1, 1), bool)
    obs["pad_mask_dict/image_wrist"]   = np.ones((1, 1), bool)
    obs["pad_mask_dict/proprio"]       = np.ones((1, 1), bool)
    obs["pad_mask_dict/timestep"]      = np.ones((1, 1), bool)
    return obs

def expected_shapes_for(model):
    """
    Infer expected (H,W) for primary and wrist from model.example_batch.
    Defaults to (256,256) and (128,128) if missing.
    """
    shapes = {}
    ex = getattr(model, "example_batch", {}) or {}
    for k, out in (("image_primary", "observation/image_primary"),
                   ("image_wrist",   "observation/image_wrist")):
        arr = ex.get(out)
        if isinstance(arr, np.ndarray) and arr.ndim == 5:
            shapes[k] = (arr.shape[2], arr.shape[3])
    shapes.setdefault("image_primary", (256, 256))
    shapes.setdefault("image_wrist", (128, 128))
    return shapes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Octo experiment directory OR a parent containing experiment_*/config.json")
    ap.add_argument("--per_exp_steps", type=int, default=None, help="If multiple experiments are found under --exp, run this many steps per experiment (defaults to --max_steps)")
    ap.add_argument("--model_xml", default=os.environ.get("MODEL_XML", ""), help="Path to MuJoCo scene xml")
    ap.add_argument("--dynamic_plant", action="store_true", help="Regenerate random plant (requires helpers.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_language", action="store_true", default=False)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--substeps", type=int, default=80)
    ap.add_argument("--kp", type=float, default=120.0)
    ap.add_argument("--action_scale", type=float, default=0.05)
    ap.add_argument("--task_mode", choices=["language", "goal"], default="language",
                help="Use language prompt or a blank goal image task (matches many finetunes).")
    ap.add_argument("--debug_constant_action", type=float, default=None,
                help="If set, ignore policy and apply a constant delta on joint 1 (radians).")

    args = ap.parse_args()

    exp_dirs = resolve_exp_dirs(args.exp)

    # Determine expected image shapes from example_batch to avoid shape mismatches
    def expected_shapes_for(model):
        shapes = {}
        ex = getattr(model, "example_batch", {}) or {}
        for k, out in (("image_primary", "observation/image_primary"), ("image_wrist", "observation/image_wrist")):
            arr = ex.get(out)
            if isinstance(arr, np.ndarray) and arr.ndim == 5:
                shapes[k] = (arr.shape[2], arr.shape[3])
        shapes.setdefault("image_primary", (256, 256))
        shapes.setdefault("image_wrist", (128, 128))
        return shapes

    steps_per = int(args.per_exp_steps or args.max_steps)

    # Build sim once
    if args.dynamic_plant:
        if _build_and_load_scene is None:
            print("[scene] --dynamic_plant requested but record_dataset.helpers.build_and_load_scene not found.")
            if not args.model_xml:
                raise SystemExit("--model_xml is required when dynamic plant builder is unavailable.")
            print("[scene] Falling back to static --model_xml.")
            sim_model = mujoco.MjModel.from_xml_path(args.model_xml)
            sim_data  = mujoco.MjData(sim_model)
            mujoco.mj_forward(sim_model, sim_data)
        else:
            print("[scene] Using dynamic plant via record_dataset.helpers.build_and_load_scene(...)")
            sim_model, sim_data = _build_and_load_scene(args.model_xml or "")
    else:
        if not args.model_xml:
            raise SystemExit("--model_xml is required (or use --dynamic_plant with helpers.py).")
        sim_model = mujoco.MjModel.from_xml_path(args.model_xml)
        sim_data  = mujoco.MjData(sim_model)
        mujoco.mj_forward(sim_model, sim_data)

    # --- NEW: build mapping/env after model+data exist (for BOTH branches) ---
    arm_act_ids, arm_qpos_addr, arm_gain_type = build_arm_mapping_from_model(
        sim_model, prefer_position=True
    )
    arm_dof_idx = build_arm_dof_indices(sim_model, arm_act_ids)
    gripper_idx = find_gripper_actuator(sim_model)
    ee_ref = auto_ee_ref(sim_model, sim_data)

    env = PandaSimEnv(
        sim_model, sim_data,
        arm_act_ids, arm_qpos_addr, ee_ref,
        substeps=args.substeps, gripper_idx=gripper_idx, arm_dof_idx=arm_dof_idx,
        kp=args.kp, kd=None, action_scale=args.action_scale,
        arm_gain_type=arm_gain_type,
    )

    # (optional: tiny debug to confirm position-mode detection)
    print(f"[actuators] position_mode={getattr(env, '_all_position_act', False)} "
        f"gain_types={arm_gain_type.tolist()}")


    period = 1.0 / max(1e-6, float(args.fps))

    with viewer.launch_passive(sim_model, sim_data) as v:
        for i, exp_dir in enumerate(exp_dirs, 1):
            print(f"[load] ({i}/{len(exp_dirs)}) Octo from: {exp_dir}")
            model = OctoModel.load_pretrained(exp_dir)
            img_shapes = expected_shapes_for(model)
            print(f"[shapes] primary={img_shapes['image_primary']} wrist={img_shapes['image_wrist']}")
            # BEFORE building policy, right after loading 'model'
            if args.task_mode == "language":
                task = model.create_tasks(texts=["Pick the tomato."])
            else:
                # simple blank goal image; replace with your own if you logged goal frames
                task = model.create_tasks(goal_images=np.zeros((1, 256, 256, 3), np.uint8))

            policy = supply_rng(
                partial(model.sample_actions,
                        unnormalization_statistics=model.dataset_statistics["action"]),
                seed=args.seed,
            )
            ts = env.reset()
            t_last = time.perf_counter()
            for step in range(steps_per):
                if not v.is_running():
                    break
                obs_for_octo = build_obs_for_octo(ts.observation, t_idx=step, img_shapes=img_shapes)                
                act_tree = policy(obs_for_octo, task)
                leaves = jax.tree_util.tree_leaves(act_tree)
                a = np.zeros((7,), np.float32) if not leaves else np.asarray(leaves[0]).reshape(-1)[:7].astype(np.float32)
                ts = env.step(a)
                now = time.perf_counter()
                dt = now - t_last
                if dt < period:
                    time.sleep(max(0.0, period - dt))
                t_last = now
                v.sync()
            print(f"[done] steps={step+1} for {exp_dir}")

    print("[all done]")


if __name__ == "__main__":
    main()
