#!/usr/bin/env python3
"""
Offline plot of GT vs predicted actions for one tomato RLDS episode,
aligned to Octo's tomato transform pipeline.

Key alignment choices vs naive TFDS evaluation:
- Applies tomato_rlds_dataset_transform (same as OXE pipeline).
- Builds task from episode language_instruction when present.
- Normalizes proprio using model.dataset_statistics['proprio'].
- Unnormalizes model outputs using model.dataset_statistics['action'] (optional).
- Builds windowed observations with correct timestep_pad_mask for left padding.

Outputs:
- episode_predictions.csv
- episode_plot_all_dims.png (or episode_plot_dimX.png)
- episode_metrics.json
"""

import argparse
import json
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import jax

from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng
import matplotlib.pyplot as plt

# Same import used by plot_actions.py (adjust if your repo differs)
from octo.data.oxe.oxe_standardization_transforms import tomato_rlds_dataset_transform

def build_obs_matching_example(model, obs_full, idx, window_size, tmask):
    """
    Build an observation dict that matches model.example_batch["observations"] exactly:
    - only keys present in example_batch
    - exact shapes per key
    """
    ex_obs = model.example_batch["observations"]

    # Start with an empty dict and populate only what the model expects
    obs_window = {}

    for k, ex_val in ex_obs.items():
        # ex_val includes batch dim in example_batch; we build window (H, ...) and later add batch dim
        # Example shapes: (1, H, ...)

        if k == "image_primary":
            obs_window[k] = obs_full["image_primary"][idx]

        elif k == "image_wrist":
            if "image_wrist" in obs_full:
                w = obs_full["image_wrist"][idx]
                # If needed, downsample to 128x128 like your prior code
                if w.shape[1] != ex_val.shape[2]:
                    # crude but consistent with your earlier approach
                    factor = w.shape[1] // ex_val.shape[2]
                    w = w[:, ::factor, ::factor, :]
                obs_window[k] = w
            else:
                # dummy wrist if model expects it
                H = window_size
                H_img = ex_val.shape[2]
                W_img = ex_val.shape[3]
                C = ex_val.shape[4]
                obs_window[k] = np.zeros((H, H_img, W_img, C), dtype=obs_full["image_primary"].dtype)

        elif k == "proprio":
            obs_window[k] = obs_full["proprio"][idx]
            normalize_proprio_inplace(obs_window, model)  # reuses your existing function

        elif k == "timestep":
            # Use absolute timesteps (as you’ve been doing)
            obs_window[k] = np.asarray(idx, dtype=np.int32)

        elif k == "timestep_pad_mask":
            obs_window[k] = tmask.astype(bool)

        elif k == "task_completed":
            # IMPORTANT: create zeros with the exact per-timestep shape expected by the model
            # example shape is (1, H, 4) so we create (H, 4)
            per_timestep_shape = tuple(ex_val.shape[2:])  # e.g. (4,)
            obs_window[k] = np.zeros((window_size, *per_timestep_shape), dtype=bool)

        elif k == "pad_mask_dict":
            # Only include pad_mask_dict if the model expects it (your log suggests it might NOT)
            # Build only the subkeys that exist in ex_obs["pad_mask_dict"]
            obs_window[k] = {}
            for subk in ex_val.keys():
                obs_window[k][subk] = tmask.astype(bool)

        else:
            # For any other expected key, create zeros matching the expected per-window shape.
            # example value is (1, H, ...) -> we create (H, ...)
            target_shape = tuple(ex_val.shape[1:])
            obs_window[k] = np.zeros(target_shape, dtype=np.asarray(ex_val).dtype)

    return obs_window


def compute_window_indices(t: int, window_size: int) -> np.ndarray:
    """Left-padded indices for a fixed window."""
    start = max(0, t - window_size + 1)
    idxs = list(range(start, t + 1))
    while len(idxs) < window_size:
        idxs.insert(0, idxs[0])
    return np.asarray(idxs, dtype=np.int32)


def build_timestep_pad_mask(idxs: np.ndarray) -> np.ndarray:
    """
    Mask padded history slots as False.
    With left-padding by repeating idxs[0]==0, we treat repeated-0s at the start as padding
    until the first strictly-increasing point.
    """
    # Example: [0,0,0,1] -> [False,False,False,True]
    #          [0,0,1,2] -> [False,False,True,True]
    mask = np.ones_like(idxs, dtype=bool)
    # Mark left repeats of the first index as padding, but keep the last slot always valid.
    first = idxs[0]
    for i in range(len(idxs) - 1):
        if idxs[i] == first and idxs[i + 1] == first:
            mask[i] = False
        else:
            # once we start moving, remaining are valid
            break
    mask[-1] = True
    return mask


def rlds_episode_to_numpy(raw_traj) -> dict:
    steps = list(raw_traj["steps"].as_numpy_iterator())
    first_obs = steps[0]["observation"]

    obs = {}
    for key in ["image_primary", "image_wrist", "proprio", "language_instruction"]:
        if key in first_obs:
            obs[key] = np.stack([s["observation"][key] for s in steps], axis=0)

    actions = np.stack([s["action"] for s in steps], axis=0)
    return {"observation": obs, "action": actions}


def apply_transform(traj_np: dict) -> dict:
    tf_traj = tf.nest.map_structure(tf.convert_to_tensor, traj_np)
    tf_traj = tomato_rlds_dataset_transform(tf_traj)
    return tf.nest.map_structure(lambda x: x.numpy(), tf_traj)


def extract_language_text(traj: dict, fallback_text: str) -> str:
    # After transform, language may be at traj["language_instruction"] or inside observation;
    # plot_actions.py assumes traj["language_instruction"] exists. :contentReference[oaicite:9]{index=9}
    if "language_instruction" in traj:
        lang_arr = traj["language_instruction"]
    else:
        lang_arr = traj.get("observation", {}).get("language_instruction", None)

    if lang_arr is None:
        return fallback_text

    # Use first element (robust to bytes / np.bytes_ / object arrays)
    x0 = lang_arr[0]
    if isinstance(x0, (bytes, np.bytes_)):
        return x0.decode("utf-8")
    # Some pipelines return 0-d arrays or object-wrapped strings
    if isinstance(x0, np.ndarray) and x0.shape == ():
        x0 = x0.item()
    if isinstance(x0, (bytes, np.bytes_)):
        return x0.decode("utf-8")
    return str(x0)



def normalize_proprio_inplace(obs_window: dict, model: OctoModel):
    stats = model.dataset_statistics.get("proprio", None)
    if stats is None:
        return
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.maximum(std, 1e-6)
    obs_window["proprio"] = (obs_window["proprio"].astype(np.float32) - mean) / std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned_path", required=True)
    ap.add_argument("--finetuned_step", type=int, default=None)
    ap.add_argument("--dataset_name", default="tomato_rlds")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--fallback_task_text", default="Pick the tomato")

    ap.add_argument("--unnormalize_action", action="store_true",
                    help="Use model.dataset_statistics['action'] to unnormalize sample_actions output. "
                         "WARNING: only valid if GT actions are in the same unnormalized space.")
    ap.add_argument("--output_dir", default="./offline_eval_outputs_fixed")

    ap.add_argument("--plot_all_dims", action="store_true")
    ap.add_argument("--plot_dim", type=int, default=0)

    # Controller-aligned option: clip predictions to [-1, 1] when computing metrics/plots
    ap.add_argument("--clip_pred", action="store_true",
                    help="Clip predicted actions to [-1, 1] for controller-aligned metrics and plots.")
    args = ap.parse_args()

    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load model
    model = OctoModel.load_pretrained(args.finetuned_path, step=args.finetuned_step)

    # Your checkpoint contract is H=2; enforce it.
    H = 2

    # 2) Load dataset split and pick one random trajectory
    raw_ds = tfds.load(
        name=args.dataset_name,
        split=args.split,
        data_dir=args.data_dir,
        shuffle_files=False,
    )
    raw_traj = next(iter(raw_ds.shuffle(1024, seed=args.seed, reshuffle_each_iteration=False).take(1)))

    # 3) RLDS -> numpy -> apply tomato transform
    traj_np = rlds_episode_to_numpy(raw_traj)
    traj = apply_transform(traj_np)

    obs_full = traj["observation"]
    gt_actions = np.asarray(traj["action"], dtype=np.float32)

    def _slice_tree(x, T):
        if isinstance(x, dict):
            return {kk: _slice_tree(vv, T) for kk, vv in x.items()}
        if isinstance(x, np.ndarray):
            return x[:T]
        return x

    if args.max_steps is not None:
        T = min(len(gt_actions), args.max_steps)
        obs_full = _slice_tree(obs_full, T)
        gt_actions = gt_actions[:T]
    else:
        T = len(gt_actions)

    # 4) Task from dataset language (fallback if missing)
    task_text = extract_language_text(traj, args.fallback_task_text)
    task = model.create_tasks(texts=[task_text])  # batch size = 1

    # 5) Policy wrapper
    unnorm_stats = model.dataset_statistics["action"] if args.unnormalize_action else None
    policy_fn = supply_rng(partial(model.sample_actions, unnormalization_statistics=model.dataset_statistics['action']))

    preds = []

    for t in range(T):
        idx = compute_window_indices(t, H)          # (H,)
        tmask = build_timestep_pad_mask(idx).astype(bool)  # (H,)

        obs_window = {
            "image_primary": obs_full["image_primary"][idx],
            "proprio": obs_full["proprio"][idx],
            "timestep": np.asarray(idx, dtype=np.int32),
            "timestep_pad_mask": tmask,
            # Checkpoint expects (B,H,4); we provide (H,4) then batch later
            "task_completed": np.zeros((H, 4), dtype=np.float32),
            # Checkpoint expects these 4 keys only
            "pad_mask_dict": {
                "image_primary": tmask.copy(),
                "image_wrist":   tmask.copy(),
                "proprio":       tmask.copy(),
                "timestep":      tmask.copy(),
            },
        }

        # Wrist camera (real or dummy) with correct resolution 128x128x3
        if "image_wrist" in obs_full:
            w = obs_full["image_wrist"][idx]
            if w.shape[1] != 128:
                factor = max(1, w.shape[1] // 128)
                w = w[:, ::factor, ::factor, :]
            obs_window["image_wrist"] = w
        else:
            obs_window["image_wrist"] = np.zeros((H, 128, 128, 3), dtype=obs_window["image_primary"].dtype)

        # Normalize proprio using model stats
        normalize_proprio_inplace(obs_window, model)

        obs_batched = jax.tree_map(lambda x: x[None], obs_window)  # add batch dim
        policy_out = policy_fn(obs_batched, task)

        a_pred = np.asarray(policy_out[0, 0], dtype=np.float32)  # first horizon step
        preds.append(a_pred)

    preds = np.stack(preds, axis=0)  # (T, A)

    # Optionally clip predictions (controller-aligned diagnostics)
    preds_for_metrics = np.clip(preds, -1.0, 1.0) if args.clip_pred else preds

    # 6) Metrics (same space only!)
    # If you pass --unnormalize_action, you must also ensure gt_actions are unnormalized identically.
    mse_per_dim = np.mean((preds_for_metrics - gt_actions) ** 2, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)

    mse_all = float(np.mean(mse_per_dim))
    rmse_all = float(np.sqrt(mse_all))

    print(f"MSE(all): {mse_all:.6f}  RMSE(all): {rmse_all:.6f}")
    print(f"Pred clipping for metrics: {args.clip_pred}")
    print(f"Task text: {task_text}")

    metrics = {
        "T": int(T),
        "action_dim": int(gt_actions.shape[-1]),
        "mse_all": mse_all,
        "rmse_all": rmse_all,
        "mse_per_dim": mse_per_dim.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
        "task_text": task_text,
        "window_size": H,
        "unnormalize_action": bool(args.unnormalize_action),
        "clip_pred": bool(args.clip_pred),
        "seed": int(args.seed),
        "split": args.split,
        "dataset_name": args.dataset_name,
    }

    # 7) Save CSV (always save raw preds; you can post-process)
    csv_path = outdir / "episode_predictions.csv"
    A = gt_actions.shape[-1]
    with csv_path.open("w") as fp:
        fp.write(",".join(["t"] + [f"pred_{i}" for i in range(A)] + [f"gt_{i}" for i in range(A)]) + "\n")
        for t in range(T):
            row = [str(t)] + [f"{x:.6f}" for x in preds[t]] + [f"{x:.6f}" for x in gt_actions[t]]
            fp.write(",".join(row) + "\n")

    # 8) Save plots (do not plt.show(); your backend is non-interactive)
    t_axis = np.arange(T)
    plot_pred = preds_for_metrics  # plot what you used for metrics

    if args.plot_all_dims:
        fig, axes = plt.subplots(A, 1, figsize=(10, 3 * A), sharex=True)
        if A == 1:
            axes = [axes]
        for i in range(A):
            axes[i].plot(t_axis, gt_actions[:, i], label="GT")
            axes[i].plot(t_axis, plot_pred[:, i], label="Pred" + (" (clipped)" if args.clip_pred else ""))
            axes[i].set_ylabel(f"action[{i}]")
            axes[i].legend()
        axes[-1].set_xlabel("timestep")
        fig.suptitle("GT vs Pred (offline)")
        fig.savefig(outdir / "episode_plot_all_dims.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        i = int(args.plot_dim)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_axis, gt_actions[:, i], label=f"GT dim {i}")
        ax.plot(t_axis, plot_pred[:, i], label=f"Pred dim {i}" + (" (clipped)" if args.clip_pred else ""))
        ax.set_xlabel("timestep")
        ax.set_ylabel(f"action[{i}]")
        ax.legend()
        ax.set_title("GT vs Pred (offline)")
        fig.savefig(outdir / f"episode_plot_dim{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 9) Save metrics JSON
    with (outdir / "episode_metrics.json").open("w") as fp:
        json.dump(metrics, fp, indent=2)

    print(f"Done. Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
