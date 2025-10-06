#!/usr/bin/env python3
import argparse, os, json, glob, random, shutil, struct

import tensorflow as tf

def is_gzip(path):
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"

def collect_records(tfrecord_paths):
    """Return list of raw serialized examples from one or more shards (kept raw)."""
    records = []
    for p in tfrecord_paths:
        comp = "GZIP" if is_gzip(p) else ""
        for rec in tf.data.TFRecordDataset(p, compression_type=comp):
            records.append(bytes(rec.numpy()))
    return records

def write_single_shard(path, records, compress=False):
    """Write all records to a single TFRecord shard; return file size (bytes) & count."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opt = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None
    with tf.io.TFRecordWriter(path, options=opt) as w:
        for r in records:
            w.write(r)
    return os.path.getsize(path), len(records)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="TFDS root, e.g. /home/you/tfds_out")
    ap.add_argument("--name", default="tomato_rlds", help="dataset name")
    ap.add_argument("--src_version", required=True, help="e.g. 0.0.9")
    ap.add_argument("--dst_version", required=True, help="e.g. 0.0.10")
    ap.add_argument("--from_split", default="val", help="existing split to repartition")
    ap.add_argument("--train_frac", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_dir = os.path.join(args.root, args.name, args.src_version)
    dst_dir = os.path.join(args.root, args.name, args.dst_version)
    os.makedirs(dst_dir, exist_ok=True)

    # 1) find shards that belong to the single existing split
    pattern = os.path.join(src_dir, f"{args.name}-{args.from_split}.tfrecord-*")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"No shards matching {pattern}")

    # 2) read all records as raw bytes (preserve exact examples)
    print(f"Reading {len(shards)} shard(s) from {args.from_split} …")
    records = collect_records(shards)
    n = len(records)
    if n == 0:
        raise SystemExit("No records found in source shards")

    # 3) shuffle deterministically and split
    random.Random(args.seed).shuffle(records)
    n_train = int(round(args.train_frac * n))
    train_recs = records[:n_train]
    val_recs   = records[n_train:]

    # 4) decide compression for output based on the first input shard
    compress_out = is_gzip(shards[0])

    # 5) write single-shard outputs in TFDS naming scheme
    train_path = os.path.join(dst_dir, f"{args.name}-train.tfrecord-00000-of-00001")
    val_path   = os.path.join(dst_dir, f"{args.name}-val.tfrecord-00000-of-00001")

    print(f"Writing train ({len(train_recs)}) → {train_path}")
    train_bytes, train_count = write_single_shard(train_path, train_recs, compress_out)

    print(f"Writing val   ({len(val_recs)}) → {val_path}")
    val_bytes, val_count = write_single_shard(val_path, val_recs, compress_out)

    # 6) copy features.json verbatim (schema)
    for fname in ("features.json",):
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))

    # 7) build a minimal dataset_info.json with both splits
    #    Use same 'version' & 'name' as source to stay consistent with your builder.
    src_info_path = os.path.join(src_dir, "dataset_info.json")
    version = "0.0.1"
    if os.path.exists(src_info_path):
        with open(src_info_path, "r") as f:
            try:
                version = json.load(f).get("version", version)
            except Exception:
                pass

    info = {
        "fileFormat": "tfrecord",
        "name": args.name,
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}",
                "name": "train",
                "numBytes": str(train_bytes),
                "shardLengths": [str(train_count)],
            },
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}",
                "name": "val",
                "numBytes": str(val_bytes),
                "shardLengths": [str(val_count)],
            },
        ],
        "version": version,
    }

    with open(os.path.join(dst_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote dataset_info.json with splits train({train_count})/val({val_count}).")

    # 8) Optional: remove stale dataset_statistics_* so Octo recomputes for the new split
    for p in glob.glob(os.path.join(dst_dir, "dataset_statistics_*.json")):
        os.remove(p)

    print("Done. Point Octo to the new version with --config.dataset_kwargs.name "
          f"'{args.name}:{args.dst_version}'")

if __name__ == "__main__":
    main()
