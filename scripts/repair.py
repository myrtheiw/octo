#!/usr/bin/env python3
"""Recompute action deltas for an RLDS dataset version.

Reads an existing tomato_rlds/<SRC_VERSION> directory, recomputes
steps/action from the stored proprio sequence, and writes a new version
under tomato_rlds/<DST_VERSION> with the updated action tensors.
"""

import glob
import json
import os
import shutil
from typing import Iterable

import numpy as np
import tensorflow as tf


SRC_VERSION = "0.0.24"
DST_VERSION = "0.0.25"
DATASET_NAME = "tomato_rlds"
DATA_DIR = "/home/myrtheiw/tfds_out"
ACTION_SCALE = 1.0


def recompute_actions(proprio: np.ndarray) -> np.ndarray:
    """Return (proprio[t+1] - proprio[t]) / ACTION_SCALE, padded to length T."""

    dq = proprio[1:, :7] - proprio[:-1, :7]
    if dq.size == 0:
        last = np.zeros((1, 7), dtype=np.float32)
    else:
        last = dq[-1:]
    actions = np.concatenate([dq, last], axis=0) / float(ACTION_SCALE)
    return actions.astype(np.float32)


def _load_proprio(seq_example: tf.train.SequenceExample) -> np.ndarray:
    fl = seq_example.feature_lists.feature_list["steps/observation/proprio"].feature
    if not fl:
        return np.zeros((0, 7), dtype=np.float32)
    step_dim = len(fl[0].float_list.value)
    proprio = np.array([feat.float_list.value for feat in fl], dtype=np.float32)
    return proprio.reshape(len(fl), step_dim)


def _write_actions(seq_example: tf.train.SequenceExample, actions: np.ndarray) -> None:
    action_list = seq_example.feature_lists.feature_list["steps/action"]
    del action_list.feature[:]
    for vec in actions:
        feat = action_list.feature.add()
        feat.float_list.value.extend(vec.tolist())


def _rewrite_shard(src_path: str, dst_path: str) -> None:
    dataset = tf.data.TFRecordDataset(src_path)
    with tf.io.TFRecordWriter(dst_path) as writer:
        for raw in dataset:
            seq_example = tf.train.SequenceExample.FromString(raw.numpy())
            proprio = _load_proprio(seq_example)
            if proprio.shape[0] == 0:
                writer.write(raw.numpy())
                continue
            actions = recompute_actions(proprio)
            _write_actions(seq_example, actions)
            writer.write(seq_example.SerializeToString())


def _copy_dataset_metadata(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    info_src = os.path.join(src_dir, "dataset_info.json")
    info_dst = os.path.join(dst_dir, "dataset_info.json")
    if os.path.isfile(info_src):
        with open(info_src, "r") as fh:
            info = json.load(fh)
        info["version"] = DST_VERSION
        with open(info_dst, "w") as fh:
            json.dump(info, fh, indent=2)
    features_src = os.path.join(src_dir, "features.json")
    if os.path.isfile(features_src):
        shutil.copy(features_src, os.path.join(dst_dir, "features.json"))


def process_split(split: str) -> None:
    src_dir = os.path.join(DATA_DIR, DATASET_NAME, SRC_VERSION)
    dst_dir = os.path.join(DATA_DIR, DATASET_NAME, DST_VERSION)

    shards = sorted(
        glob.glob(os.path.join(src_dir, f"{DATASET_NAME}-{split}.tfrecord-*"))
    )
    if not shards:
        print(f"[skip] split '{split}' has no shards under {SRC_VERSION}")
        return

    os.makedirs(dst_dir, exist_ok=True)
    for src_shard in shards:
        shard_name = os.path.basename(src_shard)
        dst_shard = os.path.join(dst_dir, shard_name)
        print(f"[rewrite] {split}: {shard_name}")
        _rewrite_shard(src_shard, dst_shard)


def main() -> None:
    src_dir = os.path.join(DATA_DIR, DATASET_NAME, SRC_VERSION)
    dst_dir = os.path.join(DATA_DIR, DATASET_NAME, DST_VERSION)
    if not os.path.isdir(src_dir):
        raise SystemExit(f"Source dataset version not found: {src_dir}")

    if os.path.isdir(dst_dir):
        raise SystemExit(f"Destination version already exists: {dst_dir}")

    _copy_dataset_metadata(src_dir, dst_dir)

    for split in ("train", "val", "test"):
        process_split(split)


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    main()
