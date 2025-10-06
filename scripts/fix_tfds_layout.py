# fix_tfds_layout.py
import json, os, re, sys
from pathlib import Path

def strip_of_suffix(p: Path):
    """
    Rename files like:  name-00000-of-00001  ->  name-00000
    Works for any ...-XXXXX-of-YYYYY pattern.
    """
    m = re.match(r"^(.*\.tfrecord-\d{5})-of-\d{5}$", p.name)
    if not m:
        return None
    new = p.with_name(m.group(1))
    if new.exists():
        print(f"[skip] {new} already exists")
        return None
    print(f"[rename] {p.name} -> {new.name}")
    p.rename(new)
    return new

def fix_dataset_info(info_path: Path):
    data = json.loads(info_path.read_text())
    if "filepathTemplate" in data:
        print("[edit] Removing unsupported 'filepathTemplate' from dataset_info.json")
        data.pop("filepathTemplate", None)
        info_path.write_text(json.dumps(data, indent=2))
    else:
        print("[ok] dataset_info.json has no 'filepathTemplate'")
    # sanity: ensure splits exist
    names = [s.get("name") for s in data.get("splits", [])]
    print(f"[splits] {names}")

def main(root_dir):
    root = Path(root_dir).expanduser().resolve()
    # Accept either .../tomato_rlds/<ver>/ or the parent dir
    if (root / "dataset_info.json").exists():
        version_dir = root
    else:
        # pick the newest version folder inside root
        cands = sorted([p for p in root.iterdir() if p.is_dir() and (p / "dataset_info.json").exists()])
        if not cands:
            print("No dataset versions found under:", root)
            sys.exit(1)
        version_dir = cands[-1]

    print("[version_dir]", version_dir)

    # 1) Rename shard files
    for p in version_dir.glob("*.tfrecord-*"):
        strip_of_suffix(p)

    # 2) Fix dataset_info.json
    info = version_dir / "dataset_info.json"
    if info.exists():
        fix_dataset_info(info)
    else:
        print("ERROR: dataset_info.json not found at", info)
        sys.exit(1)

    # Optional: show what files exist now
    print("\n[files]")
    for p in sorted(version_dir.glob("*.tfrecord-*")):
        print(" -", p.name)

if __name__ == "__main__":
    # Usage: python fix_tfds_layout.py /home/<you>/tfds_out/tomato_rlds/0.0.10
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
