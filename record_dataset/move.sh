#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/home/myrtheiw/tfds_out"

# Read dataset name + version from the top-level dataset_info.json (fallbacks if missing)
read -r NAME VERSION < <(python3 - <<'PY'
import json, os, sys
d = os.environ.get("DATA_DIR", "/home/myrtheiw/tfds_out")
p = os.path.join(d, "dataset_info.json")
name, ver = "tomato_rlds", "0.0.1"
try:
    with open(p, "r") as f:
        info = json.load(f)
        name = info.get("name", name)
        ver = info.get("version", ver)
except Exception:
    pass
print(name, ver)
PY
)

DST="${DATA_DIR}/${NAME}/${VERSION}"
mkdir -p "$DST"

echo "Moving TFDS files into: $DST"

# Move the metadata files if they exist
for f in dataset_info.json features.json; do
  if [[ -f "${DATA_DIR}/${f}" ]]; then
    mv "${DATA_DIR}/${f}" "$DST/"
  fi
done

# Move split shards and other writer outputs sitting in the parent
shopt -s nullglob
for f in "${DATA_DIR}/"*.tfrecord* "${DATA_DIR}/"*.jsonl "${DATA_DIR}/"*.pb "${DATA_DIR}/"*.index; do
  mv "$f" "$DST/"
done
shopt -u nullglob

echo "Done. Current tree:"
find "$DATA_DIR" -maxdepth 3 -type f | sed "s|$DATA_DIR/||"
