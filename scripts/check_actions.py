#!/usr/bin/env python3
"""
check_actions.py — verify that recorded actions ~= (q_{t+1} - q_t) / action_scale

Works with TFDS RLDS datasets produced by EnvLogger.
Assumes:
  - observations contain 'proprio' (full qpos vector per step),
  - actions are 7-DoF joint deltas (Franka arm),
  - you want to validate the first 7 joints (links 1..7),
  - your logging scale matches the CLI --action_scale (default 0.05).

Usage example:
  python check_actions.py \
    --tfds_name tomato_pick  \
    --data_dir /path/to/tfds_dir \
    --split train \
    --episodes 10 \
    --action_scale 0.05 \
    --arm_joints 7

If your dataset uses different key paths, the script will try common fallbacks;
use --print_keys to dump available keys for an episode.
"""

import argparse
import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.file_adapters import FileFormat
import os
import glob
import json
from typing import Iterable, Dict, Any

def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _first_present(d, keys):
    """Return the first present key in dict d from a list of candidates (supports nested via '/')."""
    for k in keys:
        cur = d
        ok = True
        for part in k.split("/"):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return k, cur
    return None, None


def _episode_to_numpy(ep):
    """Convert a TFDS RLDS episode to a plain numpy dict of sequences."""
    # Common RLDS structure: ep = {'steps': { 'observation': {...}, 'action': ..., 'is_terminal': ...}, 'episode_metadata': ...}
    # Sometimes it's already flattened (no 'steps'); handle both.
    d = {}
    for k, v in ep.items():
        if isinstance(v, dict):
            d[k] = {kk: _np(vv) for kk, vv in v.items()}
        else:
            d[k] = _np(v)
    return d


def _extract_steps(ep_dict, print_keys=False):
    """Return (proprio[T, nq], actions[T,7]) arrays from a numpyfied episode dict."""
    # Try to find the 'steps' dict
    steps = ep_dict.get("steps", None)
    if steps is None:
        # some builders expose flattened keys at top level; collect them
        # Heuristics: collect entries whose first dim matches others
        steps = {k: v for k, v in ep_dict.items() if isinstance(v, np.ndarray)}

    if print_keys:
        def _list_keys(prefix, obj):
            if isinstance(obj, dict):
                for kk, vv in obj.items():
                    _list_keys(prefix + kk + "/", vv)
            else:
                print(prefix[:-1], obj.shape, obj.dtype)
        print("=== Available step keys & shapes ===")
        _list_keys("", steps)

    # Candidate keys for proprio and action
    proprio_keys = [
        "observation/proprio",
        "proprio",
        "observation/qpos",
        "qpos",
    ]
    action_keys = [
        "action",
        "actions",
        "policy/action",
        "observation/action",  # uncommon, but check
    ]

    pk, proprio = _first_present(steps, proprio_keys)
    ak, actions = _first_present(steps, action_keys)
    if proprio is None or actions is None:
        raise KeyError(
            f"Could not find proprio/actions in episode. "
            f"Looked for proprio in {proprio_keys}, action in {action_keys}. "
            f"Use --print_keys to inspect available keys."
        )

    proprio = _np(proprio)
    actions = _np(actions)
    if proprio.ndim != 2:
        raise ValueError(f"Expected proprio to be rank-2 [T, nq], got shape {proprio.shape} (key='{pk}')")
    if actions.ndim == 1:
        actions = actions[:, None]
    if actions.ndim != 2:
        raise ValueError(f"Expected action to be rank-2 [T, na], got shape {actions.shape} (key='{ak}')")

    return proprio, actions, pk, ak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds_name", required=True, help="TFDS dataset name (the builder name you used in EnvLogger)")
    ap.add_argument("--data_dir", required=True, help="TFDS data_dir where the dataset is stored")
    ap.add_argument("--split", default="train", help="TFDS split to load (default: train)")
    ap.add_argument("--episodes", type=int, default=10, help="Number of episodes to check")
    ap.add_argument("--arm_joints", type=int, default=7, help="Number of arm joints to validate from the start of qpos")
    ap.add_argument("--action_scale", type=float, default=0.05, help="Scale used when logging actions (Δq/scale)")
    ap.add_argument("--tol_mae", type=float, default=0.05, help="Per-joint MAE tolerance on (Δq/scale) - action")
    ap.add_argument("--print_keys", action="store_true", help="Print available step keys and shapes for the first episode")
    args = ap.parse_args()

    tf.config.set_visible_devices([], "GPU")  # avoid GPU warnings
    # 1) Try normal registry load (works if your dataset is registered/importable)
    # try normal TFDS load
    try:
        ds = tfds.load(args.tfds_name, data_dir=args.data_dir, split=args.split)
        use_tfds = True
    except Exception:
        use_tfds = False

    if not use_tfds:
        # try read-only builder if dataset_info.json has TFDS features
        # Resolve version_dir robustly
        if "/" in args.tfds_name:
            version_dir = os.path.join(args.data_dir, args.tfds_name)
            ds_name = args.tfds_name.split("/")[0]    # <-- "tomato_rlds"
        else:
            root = os.path.join(args.data_dir, args.tfds_name)
            versions = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            if not versions:
                print(f"[error] no versions under {root}")
                sys.exit(2)
            versions.sort(key=lambda s: tuple(int(x) for x in s.split(".")))
            version_dir = os.path.join(root, versions[-1])
            ds_name = args.tfds_name
        info_path = os.path.join(version_dir, "dataset_info.json")        
        features_present = False
        if os.path.exists(info_path):
            with open(info_path, "r") as fh:
                try:
                    j = json.load(fh)
                    features_present = "features" in j  # TFDS metadata preserved?
                except Exception:
                    pass
        if features_present:
            builder = tfds.builder_from_directory(version_dir)  # no file_format override
            ds = builder.as_dataset(split=args.split)
            use_tfds = True

    if not use_tfds:
        # === RAW TFRecord fallback: parse Example float_list/int64_list ===
        pat = os.path.join(version_dir, f"{ds_name}-{args.split}.tfrecord-*")
        shards = sorted(glob.glob(pat))
        if not shards:
            print(f"[error] no shards matching {pat}")
            sys.exit(2)

        def _iter_envlogger_simple(shard_paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
            for sp in shard_paths:
                for raw in tf.data.TFRecordDataset(sp):  # uncompressed
                    ex = tf.train.Example.FromString(raw.numpy())
                    f = ex.features.feature

                    def _floats(k: str):
                        return np.array(f[k].float_list.value, dtype=np.float32) if k in f else None

                    def _ints(k: str):
                        return np.array(f[k].int64_list.value, dtype=np.int64) if k in f else None

                    # Required keys
                    fl_prop = _floats("steps/observation/proprio")
                    fl_act = _floats("steps/action")
                    il_first = _ints("steps/is_first")  # to infer T

                    if fl_prop is None or fl_act is None or il_first is None or il_first.size == 0:
                        continue

                    T = int(il_first.size)
                    if fl_prop.size % T != 0 or fl_act.size % T != 0:
                        continue

                    Dp = fl_prop.size // T
                    Da = fl_act.size // T

                    proprio = fl_prop.reshape(T, Dp)
                    actions = fl_act.reshape(T, Da)

                    # Optional terminal flags
                    term = _ints("steps/is_terminal")
                    if term is not None and term.size == T:
                        term = term.astype(bool)

                    steps = {"observation": {"proprio": proprio}, "action": actions}
                    if term is not None and term.size == T:
                        steps["is_terminal"] = term

                    yield {"steps": steps}

        iter_source = ("tfrecord", _iter_envlogger_simple(shards))


    ep_count = 0
    total_steps = 0
    all_mae = []
    all_norm_action = []
    all_norm_target = []
    all_zero_rate = []

    src_kind, src_iter = iter_source
    if src_kind == "tfds":
        episode_iter = tfds.as_numpy(src_iter)
    else:
        episode_iter = src_iter

    for ep in episode_iter:
        ep_count += 1
        np_ep = _episode_to_numpy(ep) if src_kind == "tfds" else ep
        proprio, actions, pk, ak = _extract_steps(np_ep, print_keys=(args.print_keys and ep_count == 1))

        T = proprio.shape[0]
        if T < 2:
            print(f"[warn] episode {ep_count} too short (T={T}), skipping")
            continue

        nq = proprio.shape[1]
        na = actions.shape[1]
        if na < args.arm_joints:
            print(f"[warn] episode {ep_count} has action dim {na} < arm_joints {args.arm_joints}, skipping")
            continue

        # Compute targets: Δq / scale on first arm_joints
        dq = proprio[1:, :args.arm_joints] - proprio[:-1, :args.arm_joints]
        target = dq / float(args.action_scale)  # [T-1, J]
        act = actions[:-1, :args.arm_joints]   # align lengths

        # Metrics
        mae = np.mean(np.abs(target - act), axis=0)           # per-joint
        mae_all = float(np.mean(np.abs(target - act)))        # scalar
        zero_rate = float(np.mean(np.all(np.isclose(act, 0.0, atol=1e-6), axis=1)))
        norm_act = float(np.mean(np.linalg.norm(act, axis=1)))
        norm_tgt = float(np.mean(np.linalg.norm(target, axis=1)))

        all_mae.append(mae)
        all_zero_rate.append(zero_rate)
        all_norm_action.append(norm_act)
        all_norm_target.append(norm_tgt)
        total_steps += (T - 1)

        ok = bool(np.all(mae <= args.tol_mae))
        status = "OK" if ok else "MISMATCH"
        print(f"[ep {ep_count:03d}] steps={T-1:4d}  "
              f"MAE_per_joint={np.array2string(mae, precision=3)}  "
              f"MAE_mean={mae_all:.3f}  "
              f"|a|_mean={norm_act:.3f}  |Δq/scale|_mean={norm_tgt:.3f}  "
              f"zero_rate={zero_rate:.3f}  => {status}")

        if ep_count >= args.episodes:
            break

    if ep_count == 0:
        print("[error] no episodes found; check --tfds_name and --data_dir (or shard parsing). "
              "Tip: try --print_keys to see what was decoded.")
        sys.exit(2)

    # Summary
    all_mae = np.stack(all_mae, axis=0) if all_mae else np.zeros((0, args.arm_joints))
    mean_mae = np.mean(all_mae, axis=0) if all_mae.size else np.zeros(args.arm_joints)
    print("\n===== SUMMARY =====")
    print(f"episodes_checked: {ep_count}")
    print(f"steps_total:      {total_steps}")
    print(f"mean_MAE_per_joint: {np.array2string(mean_mae, precision=3)} (tol={args.tol_mae})")
    print(f"mean_|a|:           {np.mean(all_norm_action) if all_norm_action else 0.0:.3f}")
    print(f"mean_|Δq/scale|:    {np.mean(all_norm_target) if all_norm_target else 0.0:.3f}")
    print(f"mean_zero_rate:     {np.mean(all_zero_rate) if all_zero_rate else 0.0:.3f}")

    if np.any(mean_mae > args.tol_mae):
        print("\n[!] Action labels deviate from Δq/scale beyond tolerance. "
              "Verify DATASET_ACTION_SCALE during logging and --action_scale here match, "
              "and that the first 'arm_joints' proprio entries correspond to the Panda arm joints.")
    else:
        print("\n[✓] Actions look consistent with Δq/scale within tolerance.")


if __name__ == "__main__":
    main()
