#!/usr/bin/env python3
"""Batch evaluation of multiple Octo checkpoints using evaluate_checkpoint."""

import argparse
import glob
import json
import os
from typing import Iterable

from evaluate_checkpoint import evaluate, evaluate_pretrained  # type: ignore


def _find_checkpoints(root: str, include_root: bool = True) -> list[str]:
    root = os.path.abspath(root)
    candidates: list[str] = []
    if include_root and os.path.isfile(os.path.join(root, "config.json")):
        candidates.append(root)

    pattern = os.path.join(root, "*", "config.json")
    for cfg in glob.glob(pattern):
        ckpt_dir = os.path.dirname(cfg)
        candidates.append(ckpt_dir)

    def sort_key(path: str):
        base = os.path.basename(path)
        try:
            return (0, int(base))
        except ValueError:
            return (1, base)

    return sorted(set(candidates), key=sort_key)


def _resolve_dirs(exp_root: str, explicit: Iterable[str] | None) -> list[str]:
    if explicit:
        return [os.path.abspath(p) for p in explicit]
    return _find_checkpoints(exp_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a sequence of checkpoints (and optionally the base pretrained model)."
    )
    parser.add_argument(
        "--exp",
        required=True,
        help="Root experiment directory (contains step subdirectories and finetune_config.json)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        help="Optional explicit list of checkpoint directories to evaluate. Defaults to all children of --exp.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Override dataset data_dir (passed through to evaluate_checkpoint).",
    )
    parser.add_argument(
        "--num_val_batches",
        type=int,
        default=None,
        help="Override number of validation batches for every evaluation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write full results JSON.",
    )
    parser.add_argument(
        "--include_pretrained",
        action="store_true",
        help="Also evaluate the base pretrained checkpoint referenced in finetune_config.json.",
    )
    parser.add_argument(
        "--mc_dropout_samples",
        type=int,
        default=1,
        help="Number of dropout-enabled passes to average per evaluation batch (>=1).",
    )
    args = parser.parse_args()

    if args.mc_dropout_samples < 1:
        raise SystemExit("--mc_dropout_samples must be >= 1")

    ckpt_dirs = _resolve_dirs(args.exp, args.checkpoints)
    if not ckpt_dirs:
        raise SystemExit("No checkpoints found to evaluate.")

    results = []

    if args.include_pretrained:
        cfg_path = os.path.join(os.path.abspath(args.exp), "finetune_config.json")
        if not os.path.isfile(cfg_path):
            raise SystemExit("Cannot locate finetune_config.json to determine pretrained_path.")
        with open(cfg_path, "r") as fh:
            cfg_json = json.load(fh)
        pretrained_path = cfg_json.get("pretrained_path")
        if not pretrained_path:
            raise SystemExit("'pretrained_path' missing in finetune_config.json")
        base_metrics = evaluate_pretrained(
            args.exp,
            pretrained_path,
            override_batches=args.num_val_batches,
            data_dir_override=args.data_dir,
            quiet=True,
            mc_dropout_samples=args.mc_dropout_samples,
        )
        results.append({"checkpoint": pretrained_path, "metrics": base_metrics})

    for directory in ckpt_dirs:
        metrics = evaluate(
            directory,
            override_batches=args.num_val_batches,
            output=None,
            data_dir_override=args.data_dir,
            quiet=True,
            mc_dropout_samples=args.mc_dropout_samples,
        )
        results.append({"checkpoint": directory, "metrics": metrics})

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
