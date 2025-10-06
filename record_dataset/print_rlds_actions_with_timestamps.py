#!/usr/bin/env python3
"""
print_rlds_actions_with_timestamps.py

Print per-step timestamps and actions from RLDS TFRecord episodes.

Usage:
  python print_rlds_actions_with_timestamps.py /path/to/file.tfrecord \
      [--limit 3] [--compression GZIP|ZLIB|NONE] [--guess-compression] [--iso]

Notes:
- Assumes actions are shape (7,) float32 per step (as in your config).
- Timestamp is searched in feature_lists (preferred) or context; if it's an
  epoch-like integer it will be shown raw and (optionally) rendered as ISO-8601.
"""

import argparse
import os
from typing import List, Optional, Tuple

import tensorflow as tf
from datetime import datetime, timezone

import base64
import itertools

def _bytes_preview(b: bytes, max_len: int = 64):
    # Try UTF-8; else fall back to base64 (truncated)
    try:
        s = b.decode("utf-8")
        return s if len(s) <= max_len else s[:max_len] + "…"
    except UnicodeDecodeError:
        enc = base64.b64encode(b).decode("ascii")
        return f"<b64:{len(b)}:{(enc[:max_len] + '…') if len(enc) > max_len else enc}>"

def _feature_to_python(feat, max_items=16):
    if feat.bytes_list.value:
        vals = list(feat.bytes_list.value)
        return [ _bytes_preview(v) for v in vals[:max_items] ] + (["…"] if len(vals) > max_items else [])
    if feat.float_list.value:
        vals = list(feat.float_list.value)
        return vals[:max_items] + (["…"] if len(vals) > max_items else [])
    if feat.int64_list.value:
        vals = list(feat.int64_list.value)
        return vals[:max_items] + (["…"] if len(vals) > max_items else [])
    return None

def _summarize_sequence_example(se: tf.train.SequenceExample, sample_steps: int = 5):
    ctx = se.context.feature
    fl = se.feature_lists.feature_list

    # --- Header: lists & lengths ---
    print("=== Episode (schema) ===")
    if fl:
        print("feature_lists:")
        for k, flist in fl.items():
            n = len(flist.feature)
            # peek the first non-empty feature to guess dtype
            dtype = "unknown"
            for feat in flist.feature:
                if feat.bytes_list.value: dtype = "bytes"; break
                if feat.float_list.value: dtype = "float"; break
                if feat.int64_list.value: dtype = "int64"; break
            print(f"  - {k}: len={n}, dtype={dtype}")
    else:
        print("feature_lists: <none>")

    # --- Context ---
    print("context:")
    if ctx:
        for k, v in ctx.items():
            pv = _feature_to_python(v, max_items=8)
            print(f"  - {k}: {pv}")
    else:
        print("  <none>")

    # --- Sample a few steps across all lists ---
    if not fl:
        return
    max_len = max(len(v.feature) for v in fl.values())
    if max_len == 0:
        return
    print(f"\n=== Sample steps (0..{min(sample_steps, max_len)-1}) ===")
    for i in range(min(sample_steps, max_len)):
        print(f"[step {i}]")
        for k, flist in fl.items():
            if i >= len(flist.feature):
                continue
            vals = _feature_to_python(flist.feature[i], max_items=16)
            print(f"  {k}: {vals}")

def _guess_options(path: str, guess: bool) -> Optional[tf.io.TFRecordOptions]:
    if not guess:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in (".gz", ".gzip"):
        return tf.io.TFRecordOptions(compression_type="GZIP")
    if ext in (".z", ".zz", ".zlib"):
        return tf.io.TFRecordOptions(compression_type="ZLIB")
    return None


def _as_seconds_and_unit(ts: int) -> Tuple[float, str]:
    """
    Heuristic to convert epoch-like integers to seconds.
    """
    abs_ts = abs(ts)
    if abs_ts >= 10**17:  # nanoseconds (e.g., 1_6xx_xxx_xxx_xxx_xxx_xxx)
        return ts / 1e9, "ns"
    if abs_ts >= 10**14:  # microseconds
        return ts / 1e6, "us"
    if abs_ts >= 10**11:  # milliseconds
        return ts / 1e3, "ms"
    return float(ts), "s"  # seconds or smaller / non-epoch counters


def _to_iso8601(ts_int: int) -> Optional[str]:
    secs, unit = _as_seconds_and_unit(ts_int)
    # Only attempt ISO if it looks like epoch time (>= year ~2000)
    try:
        if secs > 946684800 - 60:  # 2000-01-01
            return datetime.fromtimestamp(secs, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        pass
    return None


def _seq_len(se: tf.train.SequenceExample) -> int:
    # Determine the number of steps from the length of the 'action' list if present,
    # otherwise take the max length across all feature_lists.
    fl = se.feature_lists.feature_list
    if "action" in fl:
        return len(fl["action"].feature)
    return max((len(v.feature) for v in fl.values()), default=0)


def _get_feature_list_ints(se: tf.train.SequenceExample, key: str) -> Optional[List[int]]:
    fl = se.feature_lists.feature_list
    if key not in fl:
        return None
    out = []
    for feat in fl[key].feature:
        if feat.int64_list.value:
            out.append(int(feat.int64_list.value[0]))
        else:
            out.append(0)
    return out


def _find_timestamp_list(se: tf.train.SequenceExample) -> Optional[Tuple[str, List[int]]]:
    # Preferred exact key
    ts = _get_feature_list_ints(se, "timestamp")
    if ts is not None:
        return "timestamp", ts
    # Try a few common variants
    for k in list(se.feature_lists.feature_list.keys()):
        lk = k.lower()
        if "timestamp" in lk or lk in ("ts", "time", "tstamp"):
            vals = _get_feature_list_ints(se, k)
            if vals is not None:
                return k, vals
    # As a last resort, some datasets store a single context timestamp (not per-step)
    ctx = se.context.feature
    for k in ("timestamp", "episode_timestamp", "start_time", "ts"):
        if k in ctx and ctx[k].int64_list.value:
            val = int(ctx[k].int64_list.value[0])
            return f"context/{k}", [val] * _seq_len(se)
    return None


def _get_actions(se: tf.train.SequenceExample) -> Optional[List[List[float]]]:
    fl = se.feature_lists.feature_list
    if "action" not in fl:
        return None
    out: List[List[float]] = []
    for feat in fl["action"].feature:
        if feat.float_list.value:
            out.append(list(feat.float_list.value))
        elif feat.int64_list.value:
            out.append([float(x) for x in feat.int64_list.value])
        else:
            out.append([])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to RLDS .tfrecord file")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N episodes")
    ap.add_argument("--compression", choices=["GZIP", "ZLIB", "NONE"], default=None,
                    help="Compression type override")
    ap.add_argument("--guess-compression", action="store_true",
                    help="Guess compression from file extension (.gz/.gzip => GZIP, .zlib => ZLIB)")
    ap.add_argument("--iso", action="store_true",
                    help="Also print ISO-8601 UTC times when timestamps look like epochs")
    args = ap.parse_args()
    ap.add_argument("--dump", action="store_true",
                help="Print a generic dump of all features if actions are missing (or always, if used)")


    if args.compression and args.compression != "NONE":
        options = tf.io.TFRecordOptions(compression_type=args.compression)
        compression = args.compression
    elif args.compression == "NONE":
        options = None
        compression = None
    else:
        options = _guess_options(args.path, args.guess_compression)
        compression = options.compression_type if options else None

    ds = tf.data.TFRecordDataset(args.path, compression_type=compression)

    episodes_printed = 0
    for epi_idx, rec in enumerate(ds):
        rec_bytes = bytes(rec.numpy())

        # Parse as SequenceExample (RLDS writes episodes this way)
        se = tf.train.SequenceExample()
        try:
            se.ParseFromString(rec_bytes)
        except Exception as e:
            print(f"[episode {epi_idx}] ERROR: not a valid SequenceExample ({e})")
            continue

        actions = _get_actions(se)
        if actions is None:
            print(f"[episode {epi_idx}] WARNING: no 'action' feature_list found. Dumping contents…")
            _summarize_sequence_example(se, sample_steps=5)
            continue



        ts_found = _find_timestamp_list(se)
        ts_key, timestamps = (ts_found if ts_found is not None else ("<none>", [0] * len(actions)))

        print(f"=== Episode {epi_idx} ===")
        print(f"steps: {len(actions)}  | timestamp_key: {ts_key}")
        for step_idx, (a, ts) in enumerate(zip(actions, timestamps)):
            iso = _to_iso8601(ts) if args.iso else None
            if iso:
                print(f"{step_idx:06d}  ts={ts}  iso={iso}  action={a}")
            else:
                # Show unit guess to aid interpretation
                _, unit = _as_seconds_and_unit(ts)
                print(f"{step_idx:06d}  ts={ts}({unit})  action={a}")

        episodes_printed += 1
        if args.limit is not None and episodes_printed >= args.limit:
            break


if __name__ == "__main__":
    main()
