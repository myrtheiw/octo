# scripts/fix_tfds_info.py
import json, os, re, sys
from pathlib import Path

try:
    import tensorflow as tf
except Exception:
    tf = None  # if you don't want to count examples, that's ok

PAT = re.compile(r"^(?P<name>.+)-(?P<split>train|val|test)\.tfrecord-(?P<shard>\d{5})(?:-of-(?P<nshards>\d{5}))?$")

def count_examples(path):
    if tf is None:
        return 1  # fallback: stub length
    # Fast & memory-light counter
    n = 0
    for _ in tf.data.TFRecordDataset(str(path)):
        n += 1
    return n

def main(root, ds_name, version):
    ds_dir = Path(root) / ds_name / version
    assert ds_dir.is_dir(), f"Not found: {ds_dir}"

    # Discover shards
    splits = {}
    for p in sorted(ds_dir.glob("*.tfrecord-*")):
        m = PAT.match(p.name)
        if not m: 
            continue
        split = m["split"]
        shard = int(m["shard"])
        nshards = m["nshards"]
        splits.setdefault(split, dict(files=[], nshards=None))
        splits[split]["files"].append((shard, p))
        if nshards is not None:
            splits[split]["nshards"] = int(nshards)

    if not splits:
        raise SystemExit(f"No shards found in {ds_dir}")

    # Build new dataset_info.json
    info_path = ds_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
    else:
        info = {}

    info["fileFormat"] = "tfrecord"
    info["name"] = ds_name
    info["version"] = version
    # Match files like tomato_rlds-train.tfrecord-00012-of-00100
    info["filepathTemplate"] = "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_INDEX}-of-{NUM_SHARDS}"

    new_splits = []
    for split, meta in splits.items():
        files = sorted(meta["files"], key=lambda t: t[0])
        # Infer nshards from filename; if absent, deduce from count
        n_shards = meta["nshards"] or len(files)

        # Compute sizes & lengths
        num_bytes = 0
        shard_lengths = []
        for _, path in files:
            num_bytes += path.stat().st_size
            shard_lengths.append(str(count_examples(path)))

        # If you don't want to count, replace the line above with:
        # shard_lengths = [ "1" ] * len(files)

        new_splits.append({
            "name": split,
            "numBytes": str(num_bytes),
            "shardLengths": shard_lengths,
            # TFDS uses filepathTemplate + NUM_SHARDS to render names;
            # it doesn't store NUM_SHARDS per-split here, so shardLengths length is key.
        })

        # Sanity: warn if lengths don't match declared nshards
        if len(files) != n_shards:
            print(f"[warn] split {split}: found {len(files)} files but filenames say {n_shards} shards")

    info["splits"] = new_splits

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {info_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/fix_tfds_info.py <TFDS_ROOT_DIR> <DATASET_NAME> <VERSION>")
        sys.exit(1)
    main(*sys.argv[1:])

