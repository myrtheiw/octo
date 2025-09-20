#!/usr/bin/env python3
import os, argparse
import numpy as np
import jax, jax.numpy as jnp
from functools import partial
from octo.model.octo_model import OctoModel

# ---- utils to match Octo’s legacy shims ----
def _synth_namespaced_timestep(flat: dict):
    # keep namespaced pads in sync
    if "observation/pad_mask_dict/timestep" in flat and "observation/timestep_pad_mask" not in flat:
        flat["observation/timestep_pad_mask"] = flat["observation/pad_mask_dict/timestep"]
    if "observation/timestep_pad_mask" in flat and "observation/pad_mask_dict/timestep" not in flat:
        flat["observation/pad_mask_dict/timestep"] = flat["observation/timestep_pad_mask"]

def _add_legacy_aliases(d: dict):
    # top-level values mirrored from observation/*
    for k in ("proprio", "timestep", "task_completed", "image_primary", "image_wrist"):
        nk = f"observation/{k}"
        if nk in d and k not in d:
            d[k] = d[nk]
    # top-level masks mirrored from observation/pad_mask_dict/*
    for k in ("proprio", "timestep", "image_primary", "image_wrist"):
        nk = f"observation/pad_mask_dict/{k}"
        lk = f"pad_mask_dict/{k}"
        if nk in d and lk not in d:
            d[lk] = d[nk]
    # timestep_pad_mask alias
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Experiment directory (contains config.json)")
    ap.add_argument("--use_language", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    print(f"[nogl] Loading model from: {args.exp}")
    model = OctoModel.load_pretrained(args.exp)

    # Build a one-step dummy observation batch that matches example_batch shapes
    ex = getattr(model, "example_batch", {}) or {}
    # Fallback shapes if example_batch is missing (based on your dump)
    H = ex.get("observation/timestep", np.zeros((1,1), np.int32)).shape[1] if "observation/timestep" in ex else 1
    Dp = ex.get("observation/proprio", np.zeros((1,1,9), np.float32)).shape[-1]

    obs = {
        "observation/image_primary":  np.zeros((1, 1, 256, 256, 3), np.uint8),
        "observation/image_wrist":    np.zeros((1, 1, 128, 128, 3), np.uint8),
        "observation/proprio":        np.zeros((1, 1, Dp), np.float32),
        "observation/task_completed": np.zeros((1, 1, 4), np.float32),
        "observation/timestep":       np.zeros((1, 1), np.int32),
        "observation/pad_mask_dict/image_primary": np.ones((1, 1), bool),
        "observation/pad_mask_dict/image_wrist":   np.ones((1, 1), bool),
        "observation/pad_mask_dict/proprio":       np.ones((1, 1), bool),
        "observation/pad_mask_dict/timestep":      np.ones((1, 1), bool),
    }
    _synth_namespaced_timestep(obs)
    _add_legacy_aliases(obs)

    # Build a task (language or goal image)
    if args.use_language:
        task = model.create_tasks(texts=["Pick the tomato."])
    else:
        task = model.create_tasks(goal_images=np.zeros((1, 256, 256, 3), np.uint8))

    # Policy
    policy = supply_rng(
        partial(model.sample_actions,
                unnormalization_statistics=model.dataset_statistics["action"]),
        seed=args.seed,
    )

    print("[nogl] Running a forward pass...")
    out = policy(obs, task)
    try: jax.block_until_ready(out)
    except Exception: pass

    leaves = jax.tree_util.tree_leaves(out)
    print("[nogl] action leaves:", [tuple(jnp.shape(x)) for x in leaves] if leaves else "<none>")
    if leaves:
        a = np.asarray(leaves[0])
        print("[nogl] first action leaf (sample):", a.reshape(-1)[:8])

if __name__ == "__main__":
    main()
