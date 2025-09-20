#!/usr/bin/env python3
"""
dump_shapes.py — print key paths and shapes from nested dict-like data.

Supports:
  • --pickle FILE.pkl      (Python pickle; dict or object with dict-like fields)
  • --npz FILE.npz         (NumPy .npz; treated as a dict of arrays)
  • --json FILE.json       (JSON; lists become shape len, strings show as <str>)
  • --module MOD --expr EXPR  (import a module and eval EXPR in its namespace)

Optionally select a sub-tree using --key with slash-separated path, e.g.:
  --key observations/proprio
"""

import argparse
import importlib
import io
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA to prevent TensorFlow from using GPU memory

import pickle
import sys
from typing import Any, Dict

def _shape_tree(x):
    import numpy as np
    try:
        return tuple(int(d) for d in np.shape(x))
    except Exception:
        # Fallback for scalars/strings/objects without shape
        if isinstance(x, (str, bytes)):
            return f"<{type(x).__name__}>"
        return f"<{type(x).__name__}>"

def _to_plain(obj: Any) -> Any:
    """Try to coerce JAX/FLAX/torch tensors to numpy, leave others as-is."""
    try:
        import jax.numpy as jnp  # noqa
        import numpy as np
        import torch
        if hasattr(obj, "shape") and hasattr(obj, "__array__"):
            return np.asarray(obj)
        if "DeviceArray" in type(obj).__name__:
            return np.asarray(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        pass
    return obj

def dump_dict_shapes(name: str, d: Any) -> None:
    flat: Dict[str, Any] = {}

    def _rec(prefix: str, obj: Any):
        obj = _to_plain(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                _rec(f"{prefix}{k}/", v)
        else:
            flat[prefix[:-1]] = _shape_tree(obj)

    _rec("", d)
    print(f"\n--- {name} ---")
    for k in sorted(flat):
        print(f"{k}: {flat[k]}")

def _walk_keypath(root: Any, keypath: str) -> Any:
    if not keypath:
        return root
    cur = root
    for seg in keypath.split("/"):
        if seg == "":
            continue
        if isinstance(cur, dict):
            cur = cur[seg]
        else:
            # try attribute access for simple objects/namespaces
            cur = getattr(cur, seg)
    return cur

def load_from_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_from_npz(path: str) -> Dict[str, Any]:
    import numpy as np
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}

def load_from_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def load_from_module(modname: str, expr: str) -> Any:
    mod = importlib.import_module(modname)
    # evaluate expr in the module's global namespace
    return eval(expr, vars(mod), vars(mod))

def main():
    p = argparse.ArgumentParser(description="Dump shapes of nested dicts/arrays.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pickle", type=str, help="Path to .pkl file")
    src.add_argument("--npz", type=str, help="Path to .npz file")
    src.add_argument("--json", type=str, help="Path to .json file")
    src.add_argument("--module", type=str, help="Python module to import (e.g., inference)")

    p.add_argument("--expr", type=str,
                   help="Python expression to eval in module namespace (with --module)")
    p.add_argument("--key", type=str, default="",
                   help="Slash-separated key path to select sub-tree (e.g., observations/proprio)")
    p.add_argument("--title", type=str, default="DUMP", help="Optional title for the dump")
    args = p.parse_args()

    if args.module:
        if not args.expr:
            print("Error: --expr is required when using --module", file=sys.stderr)
            sys.exit(2)
        data = load_from_module(args.module, args.expr)
    elif args.pickle:
        data = load_from_pickle(args.pickle)
    elif args.npz:
        data = load_from_npz(args.npz)
    elif args.json:
        data = load_from_json(args.json)
    else:
        p.error("No valid source selected.")

    try:
        if args.key:
            data = _walk_keypath(data, args.key)
    except Exception as e:
        print(f"Failed to apply --key '{args.key}': {e}", file=sys.stderr)
        sys.exit(1)

    dump_dict_shapes(args.title, data)

if __name__ == "__main__":
    # Be nice on headless boxes; don’t force any backends here.
    sys.exit(main())
