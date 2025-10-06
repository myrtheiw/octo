import argparse
import os
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import orbax.checkpoint

from octo.utils.train_utils import TrainState


def iter_experiments(root: str) -> Iterable[str]:
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            yield path


def export_experiment(path: str, force: bool = False) -> None:
    state_dir = os.path.join(path, "state")
    if not os.path.isdir(state_dir):
        print(f"[skip] {path}: no state directory")
        return

    already_exported = os.path.isfile(os.path.join(path, "config.json"))
    if already_exported and not force:
        print(f"[skip] {path}: already exported")
        return

    manager = orbax.checkpoint.CheckpointManager(
        state_dir,
        orbax.checkpoint.PyTreeCheckpointer(),
    )
    step = manager.latest_step()
    if step is None:
        print(f"[skip] {path}: no checkpoint steps")
        return

    print(f"[export] {path}: step {step}")
    train_state: TrainState = manager.restore(step)
    train_state.model.save_pretrained(step, checkpoint_path=path)
    print(f"[done] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export finetuned Octo experiments")
    parser.add_argument(
        "--root",
        default="octo/outputs/checkpoints/octo_finetune",
        help="Root directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing exports (re-run save_pretrained)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory not found: {root}")

    for exp_path in iter_experiments(root):
        export_experiment(exp_path, force=args.force)


if __name__ == "__main__":
    main()
