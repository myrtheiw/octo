#!/usr/bin/env python3
"""
Evaluate Octo on a single (random) tomato episode per split.

For every state–action pair in that episode:
- Build a windowed observation (history of length window_size).
- Run the trained Octo policy to get a predicted action.
- Unnormalize the predicted action to physical units using model.dataset_statistics["action"].
- Compare against the ground-truth action from the transformed dataset (also in physical units).

Outputs (per run) in **radians**:
- CSV with per-timestep predictions, ground truth, and errors.
- Plot of GT vs prediction per action dimension (or a single dimension).
- JSON with summary metrics (MSE/RMSE, per-dim and overall) and some metadata.
"""

import os
import json
from pathlib import Path
from functools import partial
import time

from absl import app, flags, logging

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp

# Env/env-logger share a single joint-delta scale; reuse it here instead of hard-coding 0.02 rad.
from octo.examples.envs.tomato_env import JOINT_DELTA_SCALE as ENV_JOINT_DELTA_SCALE
# --- TODO: adapt these imports to your repo layout ---
from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng
from octo.data.oxe.oxe_standardization_transforms import (
    tomato_rlds_dataset_transform,
)

FLAGS = flags.FLAGS

# Basic flags (adapt as needed)
flags.DEFINE_string("config_path", None, "Path to Octo config, if required.")
flags.DEFINE_string("finetuned_path", None, "Directory with finetuned checkpoints.")
flags.DEFINE_integer("finetuned_step", None, "Checkpoint step of finetuned model.")
flags.DEFINE_string("split", "train", "Dataset split to evaluate on (e.g., 'train', 'val', 'test').")

flags.DEFINE_string("output_dir", "./offline_eval_outputs", "Directory to write CSV/plots/metrics.")
flags.DEFINE_integer("seed", 0, "Random seed for episode selection, etc.")
flags.DEFINE_integer("window_size", 8, "Temporal window size for observations.")

flags.DEFINE_bool("use_gpu", False, "If False, force CPU-only execution.")
flags.DEFINE_bool("plot_all_dims", False, "If True, plot all action dims; otherwise just the first.")
flags.DEFINE_bool("plot_separate_dims", False, "If True, also write one figure per joint dim in radians.")
flags.DEFINE_integer("max_lang_len", 64, "Optional max length for language tokens (if applicable).")
flags.DEFINE_integer(
    "plot_dim", 0,
    "Action dimension to plot when plot_all_dims is False."
)
flags.DEFINE_string(
    "dataset_data_dir",
    None,
    "Optional override for tfds.load(data_dir=...). If unset, use the path from the finetune config.",
)

# sample_actions only undoes dataset mean/std; it does NOT apply the JOINT_DELTA_SCALE used by the env.
# The tomato env (tomato_env.TomatoGymEnv.step) multiplies normalized Δq by JOINT_DELTA_SCALE (0.02 rad)
# to execute in MuJoCo. We import that same scale here so preds/gts/metrics are reported in radians.
ACTION_SCALE_RAD = np.asarray(ENV_JOINT_DELTA_SCALE, dtype=np.float32)


def configure_device():
    """Configure CPU/GPU usage based on FLAGS.use_gpu."""
    if not FLAGS.use_gpu:
        # Disable GPUs for TF and JAX so it stays CPU-only.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception as e:  # noqa: BLE001
            logging.warning("Could not disable TF GPUs: %s", e)
        logging.info("Running on CPU only (use_gpu=False).")
    else:
        logging.info("GPU usage allowed (use_gpu=True).")


def load_dataset(model):
    """Load the tomato RLDS dataset split as a tf.data.Dataset of trajectories."""
    ds_kwargs = dict(model.config.get("dataset_kwargs", {}))

    # Finetuned checkpoints save a separate finetune_config.json with the
    # tomato dataset kwargs; the main config.json from pretraining often
    # contains only the large pretrain mixture (no "name" field), which is
    # what caused the KeyError. Fall back to the finetune config when needed.
    if "name" not in ds_kwargs:
        finetune_cfg_path = Path(FLAGS.finetuned_path) / "finetune_config.json"
        if finetune_cfg_path.exists():
            with finetune_cfg_path.open() as fp:
                finetune_cfg = json.load(fp)
            ds_kwargs = dict(finetune_cfg.get("dataset_kwargs", {}))
            logging.info(
                "Loaded dataset kwargs from %s: name=%s, data_dir=%s",
                finetune_cfg_path,
                ds_kwargs.get("name"),
                ds_kwargs.get("data_dir"),
            )
        elif "dataset_kwargs_list" in ds_kwargs and ds_kwargs["dataset_kwargs_list"]:
            # If only one dataset was used, grab it; otherwise ask the user to disambiguate.
            if len(ds_kwargs["dataset_kwargs_list"]) == 1:
                ds_kwargs = dict(ds_kwargs["dataset_kwargs_list"][0])
                logging.info(
                    "Using the single dataset entry from dataset_kwargs_list: %s",
                    ds_kwargs.get("name"),
                )
            else:
                raise KeyError(
                    "Dataset config has a dataset_kwargs_list but no single 'name'; "
                    "please specify which dataset to load."
                )
        else:
            raise KeyError(
                "Dataset kwargs missing 'name'. Provide a finetune_config.json or "
                "add a --dataset name override."
            )

    name = ds_kwargs.get("name")
    data_dir = FLAGS.dataset_data_dir or ds_kwargs.get("data_dir", None)

    raw_ds = tfds.load(
        name=name,
        split=FLAGS.split,
        data_dir=data_dir,
        shuffle_files=False,
    )
    logging.info(
        "Loaded dataset split '%s' (name=%s, data_dir=%s)",
        FLAGS.split,
        name,
        data_dir,
    )
    return raw_ds


def pick_random_trajectory(raw_ds, seed):
    """
    Shuffle the trajectory dataset and take a single episode.

    This is not perfect global shuffling for huge datasets, but good enough for
    "one random episode" and avoids iterating the entire dataset.
    """
    # We shuffle the trajectory dataset (not steps) and take(1).
    # NOTE: buffer_size can be tuned; big enough to get decent randomness.
    shuffled = raw_ds.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=False)
    single = shuffled.take(1)

    # Extract that single raw trajectory
    raw_traj = None
    for elem in single:
        raw_traj = elem
        break

    if raw_traj is None:
        raise RuntimeError("Dataset split is empty; could not sample an episode.")
    return raw_traj


def rlds_to_numpy_trajectory(raw_traj):
    """
    Convert a single RLDS trajectory (with 'steps') to a NumPy-based dict
    matching the expected input for tomato_rlds_dataset_transform.
    """
    steps = list(raw_traj["steps"].as_numpy_iterator())

    # Build an RLDS-like dict of numpy arrays
    obs = {}
    first_obs = steps[0]["observation"]

    # Conservative: only include keys that exist
    for key in ["image_primary", "image_wrist", "proprio", "language_instruction"]:
        if key in first_obs:
            obs[key] = np.stack([s["observation"][key] for s in steps], axis=0)

    # Actions
    if "action" not in steps[0]:
        raise KeyError("Expected 'action' field in steps, but not found.")
    actions = np.stack([s["action"] for s in steps], axis=0)

    # RLDS-like structure:
    traj_np = {
        "observation": obs,
        "action": actions,
    }
    return traj_np


def apply_tomato_transform(traj_np):
    """
    Apply the tomato-specific dataset transform and return a pure-NumPy dict.

    Assumes tomato_rlds_dataset_transform returns a structure of tf.Tensors.
    """
    # Convert every leaf to a tf.Tensor while preserving the nested dict structure.
    tf_traj = tf.nest.map_structure(tf.convert_to_tensor, traj_np)
    tf_traj = tomato_rlds_dataset_transform(tf_traj)  # type: ignore[name-defined]

    # Convert back to NumPy
    traj_np_out = tf.nest.map_structure(lambda x: x.numpy(), tf_traj)
    return traj_np_out


def compute_window_indices(t, window_size, T):
    """
    For timestep t in [0, T-1], return indices for a window of size window_size,
    left-padded with the first frame index as needed.

    Example: T=10, window_size=4
      t = 0 -> [0, 0, 0, 0]
      t = 1 -> [0, 0, 0, 1]
      t = 2 -> [0, 0, 1, 2]
      t = 3 -> [0, 1, 2, 3]
      ...
    """
    start = max(0, t - window_size + 1)
    idxs = list(range(start, t + 1))
    while len(idxs) < window_size:
        idxs.insert(0, idxs[0])
    return np.array(idxs, dtype=np.int32)


def build_task_from_traj(traj, model):
    """
    Build a task dict matching model.example_batch['task'] from a transformed trajectory.
    Mirrors the logic in 03_eval_finetuned.py (dataset rollout path).
    """
    # 1) Extract language text from traj["language_instruction"]
    lang_arr = traj["language_instruction"]
    # lang_arr is typically a 1D array/list of bytes/str
    if isinstance(lang_arr[0], bytes):
        language_instruction = lang_arr[0].decode("utf-8")
    else:
        language_instruction = str(lang_arr[0])

    # 2) Let the model create its task structure
    task = model.create_tasks(texts=language_instruction)

    # 3) Align language pad mask with tokenizer attention mask
    if isinstance(task.get("language_instruction"), dict):
        attn_mask = task["language_instruction"].get("attention_mask")
        if attn_mask is not None:
            task["pad_mask_dict"]["language_instruction"] = np.asarray(
                np.any(attn_mask, axis=-1), dtype=bool
            )
            logging.info(
                "Using language pad mask shape %s",
                task["pad_mask_dict"]["language_instruction"].shape,
            )

    # 4) Force task tensors to match example_batch shapes
    example_task = model.example_batch["task"]
    fixed_task = dict(task)
    pad_mask_dict = dict(fixed_task.get("pad_mask_dict", {}))

    for k, v in example_task.items():
        if k in ("pad_mask_dict", "language_instruction"):
            continue
        target_shape = (1, *np.asarray(v).shape[1:])
        fixed_task[k] = np.zeros(target_shape, dtype=np.asarray(v).dtype)
        pad_mask_dict[k] = np.ones((target_shape[0],), dtype=bool)

    if "language_instruction" in fixed_task and isinstance(
        fixed_task["language_instruction"], dict
    ):
        if "language_instruction" not in pad_mask_dict:
            pad_mask_dict["language_instruction"] = np.ones((1,), dtype=bool)

    fixed_task["pad_mask_dict"] = pad_mask_dict
    task = fixed_task

    logging.info(
        "Task shapes: %s",
        {
            k: (
                np.shape(v)
                if not isinstance(v, dict)
                else {dk: np.shape(dv) for dk, dv in v.items()}
            )
            for k, v in task.items()
        },
    )
    return task


def build_observation_traj(traj):
    """
    Build a dict of observation arrays from the transformed trajectory.

    We assume 'traj' already has keys like 'image_primary', 'image_wrist',
    'proprio', etc., as NumPy arrays with leading time dimension.
    """
    T = traj["proprio"].shape[0]

    obs_traj = {}
    # Required modalities
    obs_traj["proprio"] = traj["proprio"]
    obs_traj["timestep"] = np.arange(T, dtype=np.int32)
    obs_traj["task_completed"] = np.zeros((T,), dtype=np.int32)

    # Optional image modalities
    for img_key in ("image_primary", "image_wrist"):
        if img_key in traj:
            obs_traj[img_key] = traj[img_key]

    # Pad masks (1 = valid)
    obs_traj["proprio_pad_mask"] = np.ones_like(obs_traj["proprio"][..., 0], dtype=np.int32)
    obs_traj["timestep_pad_mask"] = np.ones_like(obs_traj["timestep"], dtype=np.int32)
    obs_traj["task_completed_pad_mask"] = np.ones_like(obs_traj["task_completed"], dtype=np.int32)

    for img_key in ("image_primary", "image_wrist"):
        if img_key in obs_traj:
            obs_traj[f"{img_key}_pad_mask"] = np.ones(
                (T,), dtype=np.int32
            )

    return obs_traj


def normalize_proprio(obs_traj, model):
    """
    Normalize proprio using model.dataset_statistics["proprio"].
    """
    stats = model.dataset_statistics.get("proprio", None)
    if stats is None:
        raise ValueError("model.dataset_statistics['proprio'] is missing.")

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    # avoid division by zero
    std = np.where(std == 0, 1.0, std)

    obs_traj["proprio"] = (obs_traj["proprio"] - mean) / std
    return obs_traj


def get_action_stats(model):
    """
    Retrieve action normalization stats from the model.
    """
    action_stats = model.dataset_statistics.get("action", None)
    if action_stats is None:
        raise ValueError("model.dataset_statistics['action'] is missing.")

    mean = np.asarray(action_stats["mean"], dtype=np.float32)
    std = np.asarray(action_stats["std"], dtype=np.float32)
    std = np.where(std == 0, 1.0, std)

    return {"mean": mean, "std": std}


def unnormalize_actions(pred_norm, action_stats):
    """Convert normalized action predictions back to physical units."""
    return pred_norm * action_stats["std"] + action_stats["mean"]


def eval_episode(traj, model, action_stats):
    """
    Evaluate the model on a single transformed tomato trajectory.

    Returns:
    - preds_phys: (T, A) predicted actions (radians)
    - gts_phys:   (T, A) ground truth actions (radians)
    - metrics:    dict with per-dim and overall MSE/RMSE
    """
    # --- 1) Get observation + actions from transformed traj ---
    obs_full = traj["observation"]
    # traj["action"] is already in physical units (radians Δq)
    actions_rad = np.asarray(traj["action"], dtype=np.float32)
    T, action_dim = actions_rad.shape

    scale = ACTION_SCALE_RAD.reshape(-1)
    if scale.size not in (1, action_dim):
        raise ValueError(
            f"JOINT_DELTA_SCALE shape {scale.shape} incompatible with action dim {action_dim}"
        )
    scale = np.broadcast_to(scale, (action_dim,))

    # --- 2) Proprio stats for normalization ---
    proprio_stats = model.dataset_statistics.get("proprio", {})
    PROPRIO_MEAN = np.asarray(proprio_stats.get("mean"), dtype=np.float32)
    PROPRIO_STD = np.asarray(proprio_stats.get("std"), dtype=np.float32)

    # --- 3) Build task using the helper above ---
    task = build_task_from_traj(traj, model)

    # --- 4) Policy function that returns UNNORMALIZED (physical) actions ---
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        )
    )

    preds_phys = []
    gts_phys = []

    H = FLAGS.window_size

    logging.info(
        "Running offline consistency: split=%s T=%d window=%d",
        FLAGS.split,
        T,
        H,
    )

    for t in range(T):
        idx = compute_window_indices(t, H, T)  # shape (H,)

        # 4a) Build observation window (time-major)
        obs_window = {
            "image_primary": obs_full["image_primary"][idx],  # (H, 256, 256, 3)
            "proprio": obs_full["proprio"][idx],              # (H, 9)
            "timestep": obs_full["timestep"][idx],            # (H,)
        }
        if "image_wrist" in obs_full:
            obs_window["image_wrist"] = obs_full["image_wrist"][idx]

        # 4b) Normalize proprio
        obs_window["proprio"] = (
            obs_window["proprio"] - PROPRIO_MEAN
        ) / np.maximum(PROPRIO_STD, 1e-6)

        # 4c) Downsample wrist to 128x128 if needed (safety)
        if "image_wrist" in obs_window and obs_window["image_wrist"].shape[1] != 128:
            obs_window["image_wrist"] = obs_window["image_wrist"][:, ::2, ::2, :]

        # 4d) pad_mask_dict + timestep_pad_mask + task_completed
        pad_mask_dict = {
            "image_primary": np.ones((H,), dtype=bool),
            "proprio": np.ones((H,), dtype=bool),
            "timestep": np.ones((H,), dtype=bool),
        }
        if "image_wrist" in obs_window:
            pad_mask_dict["image_wrist"] = np.ones((H,), dtype=bool)
        obs_window["pad_mask_dict"] = pad_mask_dict
        obs_window["timestep_pad_mask"] = np.ones((H,), dtype=bool)
        obs_window["task_completed"] = np.zeros((H, 4), dtype=np.float32)

        # 4e) Add batch dimension -> shapes like (1, H, ...)
        obs_batched = jax.tree_map(lambda x: x[None], obs_window)

        # 4f) Run policy; output is already in physical units
        policy_out = policy_fn(obs_batched, task)
        policy_chunk = np.asarray(policy_out[0], dtype=np.float32)
        a_pred = policy_chunk[0]

        preds_phys.append(a_pred)        # a_pred is still in normalized units here
        gts_phys.append(actions_rad[t])  # already radians; don't rescale

    # Convert predicted normalized actions -> radians using JOINT_DELTA_SCALE
    preds_phys = np.stack(preds_phys, axis=0) # radians

    # Ground-truth is already in radians; no extra scaling
    gts_phys = np.stack(gts_phys, axis=0)    # radians

    # --- 5) Metrics in physical units ---
    mse_per_dim = np.mean((preds_phys - gts_phys) ** 2, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    mse_all = float(np.mean(mse_per_dim))
    rmse_all = float(np.sqrt(mse_all))

    metrics = {
        "mse_per_dim": mse_per_dim.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
        "mse_all": mse_all,
        "rmse_all": rmse_all,
        "T": int(T),
        "action_dim": int(action_dim),
    }

    actions = np.asarray(traj["action"], dtype=np.float32)
    max_abs = np.max(np.abs(actions))

    if max_abs < 0.1:  # likely radians (small)
        actions_rad = actions
        logging.info("Treating traj['action'] as radians (max |action|=%.4f).", max_abs)
    else:              # likely normalized in [-1, 1]
        actions_rad = actions * scale
        logging.info("Treating traj['action'] as normalized; scaling by JOINT_DELTA_SCALE.")

    return preds_phys, gts_phys, metrics


def save_csv(output_dir: Path, preds, gts):
    T, A = preds.shape
    csv_path = output_dir / "episode_predictions.csv"
    with csv_path.open("w") as fp:
        header = ["t"] + [f"pred_rad_{i}" for i in range(A)] + [f"gt_rad_{i}" for i in range(A)]
        fp.write(",".join(header) + "\n")
        for t in range(T):
            row = [str(t)] + [f"{x:.6f}" for x in preds[t]] + [f"{x:.6f}" for x in gts[t]]
            fp.write(",".join(row) + "\n")
    logging.info("Wrote CSV to %s", csv_path)


def save_plot(output_dir: Path, preds, gts):
    T, A = preds.shape
    t_axis = np.arange(T)

    if FLAGS.plot_all_dims:
        fig, axes = plt.subplots(A, 1, figsize=(10, 3 * A), sharex=True)
        if A == 1:
            axes = [axes]
        for i in range(A):
            axes[i].plot(t_axis, gts[:, i], label="GT")
            axes[i].plot(t_axis, preds[:, i], label="Pred")
            axes[i].set_ylabel(f"Δq[{i}] [rad]")
            axes[i].legend()
        axes[-1].set_xlabel("timestep")
        fig.suptitle("Ground Truth vs Prediction (radians, all dims)")

        png_path = output_dir / "episode_plot_all_dims.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info("Wrote plot to %s", png_path)

    else:
        i = int(FLAGS.plot_dim)
        if i < 0 or i >= A:
            raise ValueError(f"plot_dim {i} is out of range for action dim {A}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_axis, gts[:, i], label=f"GT dim {i}")
        ax.plot(t_axis, preds[:, i], label=f"Pred dim {i}")
        ax.set_xlabel("timestep")
        ax.set_ylabel(f"Δq[{i}] [rad]")
        ax.legend()
        ax.set_title(f"Ground Truth vs Prediction (radians, dim {i})")

        png_path = output_dir / f"episode_plot_dim{i}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logging.info("Wrote plot to %s", png_path)

    if FLAGS.plot_separate_dims:
        for i in range(A):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t_axis, gts[:, i], label=f"GT dim {i}")
            ax.plot(t_axis, preds[:, i], label=f"Pred dim {i}")
            ax.set_xlabel("timestep")
            ax.set_ylabel(f"Δq[{i}] [rad]")
            ax.legend()
            ax.set_title(f"Ground Truth vs Prediction (radians, dim {i})")
            png_path = output_dir / f"episode_plot_dim{i}_separate.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        logging.info("Wrote per-dim plots to %s (one file per joint)", output_dir)


def save_metrics(output_dir: Path, metrics, extra_meta=None):
    metrics_out = dict(metrics)
    if extra_meta is not None:
        metrics_out.update(extra_meta)
    metrics_path = output_dir / "episode_metrics.json"
    with metrics_path.open("w") as fp:
        json.dump(metrics_out, fp, indent=2)
    logging.info("Wrote metrics to %s", metrics_path)


def main(argv):
    del argv  # unused
    configure_device()

    # Use the user-provided output_dir as an absolute path to avoid writing to the
    # current working directory (e.g., octo/examples) unintentionally.
    output_dir = Path(FLAGS.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load finetuned model (same as 03_eval_finetuned.py)
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(
        FLAGS.finetuned_path,
        step=FLAGS.finetuned_step,
    )

    # You DO NOT need to reassign model here.
    # The finetuned config you care about is already in model.config.
    config = model.config

    # 2) Load dataset using the model's dataset_kwargs
    raw_ds = load_dataset(model)

    # 3) Pick a random trajectory/episode from this split
    raw_traj = pick_random_trajectory(raw_ds, seed=FLAGS.seed)

    # 4) RLDS -> NumPy
    traj_np = rlds_to_numpy_trajectory(raw_traj)

    # 5) Apply tomato transform (same transform as in finetune & eval script)
    traj = apply_tomato_transform(traj_np)

    # 6) Get action stats (for unnormalization)
    action_stats = get_action_stats(model)

    # 7) Run evaluation on this single episode
    t0 = time.time()
    preds, gts, metrics = eval_episode(traj, model, action_stats)
    elapsed = time.time() - t0

    # 8) Save artifacts
    save_csv(output_dir, preds, gts)
    save_plot(output_dir, preds, gts)
    save_metrics(
        output_dir,
        metrics,
        extra_meta={
            "units": "rad",
            "joint_delta_scale_rad": ACTION_SCALE_RAD.tolist(),
            "scale_source": "octo.examples.envs.tomato_env.JOINT_DELTA_SCALE",
            "split": FLAGS.split,
            "seed": FLAGS.seed,
            "window_size": FLAGS.window_size,
            "runtime_sec": elapsed,
        },
    )

    logging.info("Done. Runtime: %.2f s", elapsed)

if __name__ == "__main__":
    app.run(main)
