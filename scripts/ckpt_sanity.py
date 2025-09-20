#!/usr/bin/env python3
import os, argparse, numpy as np

# CPU/headless
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

def _flatten(d, prefix=""):
    for k, v in d.items():
        kk = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten(v, kk)
        else:
            yield kk, v

def _norm(k: str) -> str:
    # normalize observations -> observation
    return "observation/" + k[len("observations/"):] if k.startswith("observations/") else k

def _synth_namespaced_timestep(flat: dict):
    # ensure both namespaced timestep masks exist if either does
    if ("observation/pad_mask_dict/timestep" in flat and
        "observation/timestep_pad_mask" not in flat):
        flat["observation/timestep_pad_mask"] = flat["observation/pad_mask_dict/timestep"]
    if ("observation/timestep_pad_mask" in flat and
        "observation/pad_mask_dict/timestep" not in flat):
        flat["observation/pad_mask_dict/timestep"] = flat["observation/timestep_pad_mask"]

def _add_legacy_aliases(warm_obs: dict):
    """Mirror namespaced observation keys into legacy top-level keys."""
    # values
    for k in ("proprio", "timestep", "task_completed", "image_primary", "image_wrist"):
        nk = f"observation/{k}"
        if nk in warm_obs and k not in warm_obs:
            warm_obs[k] = warm_obs[nk]
    # masks
    for k in ("proprio", "timestep", "image_primary", "image_wrist"):
        nk = f"observation/pad_mask_dict/{k}"
        lk = f"pad_mask_dict/{k}"
        if nk in warm_obs and lk not in warm_obs:
            warm_obs[lk] = warm_obs[nk]
    # special alias for timestep_pad_mask
    src = warm_obs.get("observation/timestep_pad_mask") \
          or warm_obs.get("observation/pad_mask_dict/timestep")
    if src is not None:
        warm_obs.setdefault("timestep_pad_mask", src)
        warm_obs.setdefault("pad_mask_dict/timestep", src)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()

    from octo.model.octo_model import OctoModel
    import jax, jax.numpy as jnp

    print(f"[ckpt] Loading {args.exp}")
    model = OctoModel.load_pretrained(args.exp)

    ex = getattr(model, "example_batch", {}) or {}
    ex_flat = { _norm(k): v for k, v in _flatten(ex) }
    obs_keys = [k for k in ex_flat if k.startswith("observation/")]
    if not obs_keys:
        raise RuntimeError("example_batch has no observation/* keys after flatten.")

    # zeros with exact shapes for observation/* keys
    warm_obs = {k: np.zeros_like(ex_flat[k]) for k in obs_keys}
    _synth_namespaced_timestep(warm_obs)
    _add_legacy_aliases(warm_obs)  # <- mirror legacy values + masks

    # language (or goal image) task
    if "task/language_instruction/input_ids" in ex_flat:
        task = model.create_tasks(texts=["warmup"])
    else:
        gi = ex_flat.get("task/image_primary", np.zeros((1, 256, 256, 3), np.uint8))
        task = model.create_tasks(goal_images=gi)

    # call policy once (rng arg name can vary)
    key = jax.random.PRNGKey(0)
    kwargs = dict(unnormalization_statistics=model.dataset_statistics["action"])
    try:
        out = model.sample_actions(warm_obs, task, rng=key, **kwargs)
    except TypeError:
        try:
            out = model.sample_actions(warm_obs, task, prng_key=key, **kwargs)
        except TypeError:
            out = model.sample_actions(warm_obs, task, **kwargs)

    print("[ckpt] Running one forward pass (CPU JIT may compile)...")
    try: jax.block_until_ready(out)
    except Exception: pass

    leaves = jax.tree_util.tree_leaves(out)
    print("[ckpt] OK: action leaf shapes:",
          [tuple(jnp.shape(x)) for x in leaves] if leaves else "<none>")
    print("[ckpt] Success: checkpoint loads and produces actions.")

if __name__ == "__main__":
    main()
