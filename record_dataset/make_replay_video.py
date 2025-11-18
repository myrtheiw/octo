#!/usr/bin/env python3
"""Export Octo RLDS episode replays to MP4 videos."""

from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import imageio
import imageio.v3 as iio
import tensorflow as tf
from tqdm import tqdm


CAMERA_KEY_MAP: Dict[str, Tuple[str, ...]] = {
    "primary": (
        "steps/image_primary",
        "steps/observation/image_primary",
        "steps.observation.image_primary",
    ),
    "wrist": (
        "steps/image_wrist",
        "steps/observation/image_wrist",
        "steps.observation.image_wrist",
    ),
}


@dataclass
class EpisodeFrames:
    """Container for decoded frames and basic stats."""

    name: str
    frames: List[np.ndarray]


def list_shards(path: str) -> List[str]:
    """Return sorted list of TFRecord shard paths under ``path``.

    Args:
        path: Directory or file path.

    Raises:
        FileNotFoundError: If no shards are found.
        ValueError: If ``path`` is invalid.
    """
    if not path:
        raise ValueError("Expected --path to be provided.")
    if os.path.isfile(path):
        basename = os.path.basename(path)
        if ".tfrecord" in basename:
            return [path]
        raise ValueError(f"File '{path}' does not look like a TFRecord shard.")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path '{path}' not found.")

    shards: List[str] = []
    for root, _, files in os.walk(path):
        for fname in files:
            if ".tfrecord" in fname:
                shards.append(os.path.join(root, fname))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards found under '{path}'.")
    return sorted(shards)


def extract_feature_list(
    example: tf.train.Example, key_options: Sequence[str]
) -> List[bytes]:
    """Extract a list of raw bytes from the first populated feature key."""
    for key in key_options:
        feature = example.features.feature.get(key)
        if feature is None:
            continue
        if feature.bytes_list.value:
            return list(feature.bytes_list.value)
    return []


def decode_frame(buffer: bytes, default_hw: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Decode a raw or encoded RGB frame into uint8 HxWx3 array."""
    default_h, default_w = default_hw
    expected_len = default_h * default_w * 3
    arr = np.frombuffer(buffer, dtype=np.uint8)

    frame: Optional[np.ndarray] = None
    if arr.size == expected_len:
        frame = arr.reshape((default_h, default_w, 3))
    else:
        # Fall back to encoded image decoding (e.g., PNG/JPEG).
        with io.BytesIO(buffer) as stream:
            frame = iio.imread(stream)

    if frame.ndim == 2:  # Grayscale -> RGB
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:  # Drop alpha channel if present
        frame = frame[..., :3]

    if frame.dtype != np.uint8:
        frame = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    return frame


def resize_keep_aspect(frame: np.ndarray, target_short_side: int) -> np.ndarray:
    """Resize frame so its shorter side equals ``target_short_side``."""
    h, w = frame.shape[:2]
    short_side = min(h, w)
    if short_side == 0:
        raise ValueError("Frame has zero dimension, cannot resize.")
    scale = target_short_side / float(short_side)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
    resized = tf.image.resize(tensor, (new_h, new_w), method="bilinear", antialias=True)
    resized = tf.clip_by_value(resized, 0.0, 255.0)
    return tf.cast(tf.round(resized), tf.uint8).numpy()


def stack_cameras(frames: Sequence[np.ndarray], target_short_side: int) -> np.ndarray:
    """Resize, pad, and horizontally stack frames from multiple cameras."""
    if not frames:
        raise ValueError("Cannot stack empty frame sequence.")
    resized = [resize_keep_aspect(frame, target_short_side) for frame in frames]
    max_h = max(frame.shape[0] for frame in resized)
    padded: List[np.ndarray] = []
    for frame in resized:
        pad_h = max_h - frame.shape[0]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        padded.append(
            np.pad(
                frame,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        )
    return np.concatenate(padded, axis=1)


def interp_frames(frames: Sequence[np.ndarray], n_hold: int) -> np.ndarray:
    """Interpolate frames so each input frame spans ``n_hold`` steps."""
    if not frames:
        raise ValueError("No frames provided for interpolation.")
    if n_hold <= 1:
        return np.stack(frames, axis=0)

    float_frames = [frame.astype(np.float32) for frame in frames]
    interpolated: List[np.ndarray] = []
    last_index = len(frames) - 1

    for idx, frame in enumerate(float_frames):
        if idx == last_index:
            for _ in range(n_hold):
                interpolated.append(frames[idx])
            continue
        next_frame = float_frames[idx + 1]
        for hold_idx in range(n_hold):
            alpha = hold_idx / float(n_hold)
            blended = frame * (1.0 - alpha) + next_frame * alpha
            interpolated.append(np.clip(np.rint(blended), 0, 255).astype(np.uint8))
    return np.stack(interpolated, axis=0)


def detect_compression(shards: Sequence[str]) -> Optional[str]:
    """Infer TFRecord compression type from shard suffix."""
    if not shards:
        return None
    all_gz = all(shard.endswith(".gz") for shard in shards)
    if all_gz:
        return "GZIP"
    any_gz = any(shard.endswith(".gz") for shard in shards)
    if any_gz:
        raise ValueError("Cannot mix compressed and uncompressed TFRecord shards.")
    return None


def iter_examples(shards: Sequence[str]) -> Iterator[tf.train.Example]:
    """Yield parsed tf.train.Example messages from shards."""
    compression = detect_compression(shards)
    dataset = tf.data.TFRecordDataset(
        shards,
        compression_type=compression,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    for raw in dataset:
        example = tf.train.Example()
        example.ParseFromString(bytes(raw.numpy()))
        yield example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Octo RLDS replays to MP4.")
    parser.add_argument("--path", required=True, type=str, help="Shard directory or file.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./replays",
        help="Output directory for MP4 files.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument(
        "--action_dt",
        type=float,
        default=0.10,
        help="Seconds per dataset step.",
    )
    parser.add_argument(
        "--max_eps",
        type=int,
        default=5,
        help="Maximum number of episodes to export.",
    )
    parser.add_argument(
        "--cams",
        type=str,
        default="primary",
        help="Comma-separated list of camera keys (e.g., primary,wrist).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Resize shorter side to this many pixels before stacking.",
    )
    parser.add_argument(
        "--limit_frames",
        type=int,
        default=None,
        help="Optional limit on steps to include per episode.",
    )
    parser.add_argument("--codec", type=str, default="libx264", help="FFmpeg codec.")
    parser.add_argument("--bitrate", type=str, default="6M", help="Target bitrate.")
    return parser.parse_args()


def warn_black_camera(cam_name: str, mean_value: float) -> None:
    """Emit warning if a camera stream appears black."""
    if mean_value < 5.0:
        tqdm.write(f"[warn] camera '{cam_name}' appears black (mean {mean_value:.2f}).")


def parse_bitrate(bitrate: Optional[str]) -> Optional[int]:
    """Convert bitrate strings like '6M' to bits-per-second ints."""
    if not bitrate:
        return None
    text = bitrate.strip()
    if not text:
        return None
    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    lower = text.lower()
    if lower[-1] in suffixes:
        factor = suffixes[lower[-1]]
        number = lower[:-1].strip()
        try:
            value = float(number)
        except ValueError as exc:
            raise ValueError(f"Invalid bitrate value: '{bitrate}'") from exc
        return int(value * factor)
    try:
        return int(float(lower))
    except ValueError as exc:
        raise ValueError(f"Invalid bitrate value: '{bitrate}'") from exc


def main() -> None:
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.action_dt <= 0:
        raise ValueError("--action_dt must be positive.")
    if args.resize <= 0:
        raise ValueError("--resize must be positive.")

    cameras = [cam.strip() for cam in args.cams.split(",") if cam.strip()]
    if not cameras:
        raise ValueError("No cameras specified via --cams.")
    for cam in cameras:
        if cam not in CAMERA_KEY_MAP:
            valid = ", ".join(sorted(CAMERA_KEY_MAP))
            raise ValueError(f"Unknown camera '{cam}'. Valid options: {valid}.")

    shards = list_shards(args.path)
    os.makedirs(args.out_dir, exist_ok=True)

    n_hold = max(1, round(args.fps * args.action_dt))
    action_dt_ms = int(round(args.action_dt * 1000.0))

    exported = 0
    for example_idx, example in enumerate(tqdm(iter_examples(shards), desc="Episodes", unit="ep")):
        if exported >= args.max_eps:
            break

        available: List[EpisodeFrames] = []
        missing: List[str] = []

        for cam in cameras:
            raw_frames = extract_feature_list(example, CAMERA_KEY_MAP[cam])
            if not raw_frames:
                missing.append(cam)
                continue
            decoded_frames = [decode_frame(buffer) for buffer in raw_frames]
            available.append(EpisodeFrames(cam, decoded_frames))

        if missing:
            tqdm.write(
                "[info] episode %d missing cameras: %s"
                % (example_idx, ", ".join(missing))
            )

        if not available:
            tqdm.write(f"[skip] episode {example_idx} has no requested camera frames.")
            continue

        lengths = [len(ep.frames) for ep in available]
        if not lengths or min(lengths) == 0:
            tqdm.write(f"[skip] episode {example_idx} contains empty frame lists.")
            continue
        steps = min(lengths)
        if args.limit_frames is not None:
            steps = min(steps, args.limit_frames)
        if steps == 0:
            tqdm.write(f"[skip] episode {example_idx} limit produced zero frames.")
            continue

        stacked_frames: List[np.ndarray] = []
        per_cam_mean: Dict[str, float] = {}

        for ep in available:
            mean_total = 0.0
            for frame in ep.frames[:steps]:
                mean_total += float(frame.mean())
            per_cam_mean[ep.name] = mean_total / float(steps)

        for step_idx in range(steps):
            frames_to_stack = [ep.frames[step_idx] for ep in available]
            stacked = stack_cameras(frames_to_stack, args.resize)
            stacked_frames.append(stacked)

        video_frames = interp_frames(stacked_frames, n_hold)
        outfile = os.path.join(
            args.out_dir,
            f"replay_ep{exported:05d}_T{steps}_fps{args.fps}_dt{action_dt_ms}ms.mp4",
        )
        bitrate_value = parse_bitrate(args.bitrate)
        writer_kwargs = {"fps": args.fps, "codec": args.codec}
        if bitrate_value is not None:
            writer_kwargs["bitrate"] = bitrate_value
        with imageio.get_writer(outfile, **writer_kwargs) as writer:
            for frame in video_frames:
                writer.append_data(frame)

        duration = steps * args.action_dt
        total_frames = video_frames.shape[0]
        means_formatted = ", ".join(
            f"{cam}:{per_cam_mean[cam]:.1f}" for cam in cameras if cam in per_cam_mean
        )

        tqdm.write(
            "[done] episode {idx} -> {path} | T={steps} | duration≈{duration:.2f}s | "
            "frames={total} | means[{means}]".format(
                idx=example_idx,
                path=os.path.relpath(outfile),
                steps=steps,
                duration=duration,
                total=total_frames,
                means=means_formatted or "n/a",
            )
        )

        for cam_name, mean_value in per_cam_mean.items():
            warn_black_camera(cam_name, mean_value)

        exported += 1

    if exported == 0:
        tqdm.write("No episodes exported.")
    else:
        tqdm.write(f"Exported {exported} episode(s) to '{args.out_dir}'.")


if __name__ == "__main__":
    main()
