#!/usr/bin/env python3
"""Quick dataset inspector for tomato RLDS shards."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _save_png(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    tensor = tf.convert_to_tensor(arr, dtype=tf.uint8)
    png = tf.io.encode_png(tensor).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def inspect_dataset(
    dataset_root: Path,
    dataset_name: str,
    version: str,
    split: str,
    episodes: int,
    output_dir: Path,
) -> None:
    dataset_identifier = f"{dataset_name}:{version}"
    builder = tfds.builder(dataset_identifier, data_dir=str(dataset_root))
    builder.download_and_prepare()
    ds = builder.as_dataset(split=split, read_config=tfds.ReadConfig(shuffle_seed=0))

    output_dir.mkdir(parents=True, exist_ok=True)

    for epi, traj in enumerate(tfds.as_numpy(ds.take(episodes))):
        steps = traj.get("steps", {})
        obs = steps.get("observation", {})
        acts = steps.get("action")

        if acts is None:
            print(f"Episode {epi}: missing action data; skipping")
            continue

        episode_dir = output_dir / f"episode_{epi:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        num_steps = int(acts.shape[0])

        def _maybe_get(key):
            val = obs.get(key)
            return val if val is not None else None

        primary = _maybe_get("image_primary")
        wrist = _maybe_get("image_wrist")
        goal_primary = _maybe_get("goal_image_primary")
        goal_wrist = _maybe_get("goal_image_wrist")

        def _save_series(name, series):
            if series is None:
                return
            series = np.asarray(series)
            if series.ndim == 3:
                _save_png(episode_dir / f"{name}.png", series)
            elif series.ndim == 4 and series.shape[0] > 0:
                indices = [0, series.shape[0] // 2, series.shape[0] - 1]
                for idx in indices:
                    _save_png(episode_dir / f"{name}_{idx:04d}.png", series[idx])

        _save_series("primary", primary)
        _save_series("wrist", wrist)

        if goal_primary is not None:
            first_goal = goal_primary[0] if goal_primary.ndim == 4 else goal_primary
            _save_series("goal_primary", first_goal)

        if goal_wrist is not None:
            first_goal_wrist = goal_wrist[0] if goal_wrist.ndim == 4 else goal_wrist
            _save_series("goal_wrist", first_goal_wrist)

        stats = {
            "num_steps": num_steps,
            "action_min": np.min(acts, axis=0, initial=np.inf).tolist(),
            "action_max": np.max(acts, axis=0, initial=-np.inf).tolist(),
            "action_mean": np.mean(acts, axis=0).tolist(),
        }

        if primary is not None and primary.ndim == 4:
            stats["image_primary_mean"] = float(np.mean(primary))
        if wrist is not None and wrist.ndim == 4:
            stats["image_wrist_mean"] = float(np.mean(wrist))

        with open(episode_dir / "summary.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Wrote inspection artifacts for episode {epi} to {episode_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RLDS tomato dataset shards")
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/myrtheiw/tfds_out"))
    parser.add_argument("--dataset_name", type=str, default="tomato_rlds")
    parser.add_argument("--version", type=str, default="0.0.15")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("octo/outputs/dataset_inspect"))

    args = parser.parse_args()


    version_dir = args.dataset_root / args.dataset_name / args.version
    if not version_dir.exists():
        raise SystemExit(f"Dataset directory not found: {version_dir}")

    inspect_dataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        version=args.version,
        split=args.split,
        episodes=args.episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
