"""Shared checkpoint inspection utilities for Octo runners and tools."""

from collections.abc import Mapping


def _summarize_pytree(tree, name="obj", max_leaves=50):
    import jax
    import numpy as np

    try:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
    except Exception as exc:
        print(f"[ckpt] {name}: tree_flatten failed: {exc}")
        return

    tree_type = type(tree)
    extra = ""
    if isinstance(tree, Mapping):
        keys = list(tree.keys())
        extra = f" keys={keys[:8]}" + (" ..." if len(keys) > 8 else "")
    elif hasattr(tree, "__dict__"):
        attrs = list(vars(tree).keys())
        extra = f" attrs={attrs[:8]}" + (" ..." if len(attrs) > 8 else "")

    print(f"[ckpt] {name}: type={tree_type} treedef={treedef} leaves={len(leaves)}{extra}")
    for i, leaf in enumerate(leaves[:max_leaves]):
        try:
            shape = getattr(leaf, "shape", None)
            dtype = getattr(leaf, "dtype", None)
            ltype = type(leaf).__name__
            if dtype is None and shape is None:
                try:
                    arr = np.asarray(leaf)
                    shape = arr.shape
                    dtype = arr.dtype
                except Exception:
                    pass
            print(f"[ckpt]  leaf[{i:03d}] type={ltype} shape={shape} dtype={dtype}")
        except Exception as exc:
            print(f"[ckpt]  leaf[{i:03d}] <unprintable> err={exc}")
    if len(leaves) > max_leaves:
        print(f"[ckpt]  ... ({len(leaves) - max_leaves} more leaves)")


def _find_params_paths(obj):
    """
    Return a list of candidate paths to 'params' inside obj.
    Search common layouts: obj.params, obj['params'], obj['state']['params'],
    obj['target']['params'], train_state-like dataclasses, etc.
    """
    paths = []

    def _append(kind, path, value):
        try:
            paths.append((kind, path, value))
        except Exception:
            paths.append((kind, path, None))

    if hasattr(obj, "params"):
        try:
            _append("attr", "params", getattr(obj, "params"))
        except Exception:
            _append("attr", "params", None)

    if isinstance(obj, Mapping):
        if "params" in obj:
            _append("dict", "params", obj.get("params"))
        if "state" in obj and isinstance(obj.get("state"), Mapping):
            state = obj.get("state")
            if "params" in state:
                _append("dict", "state/params", state.get("params"))
        if "target" in obj and isinstance(obj.get("target"), Mapping):
            target = obj.get("target")
            if "params" in target:
                _append("dict", "target/params", target.get("params"))

    if hasattr(obj, "__dict__"):
        try:
            data = vars(obj)
        except Exception:
            data = {}
        if "params" in data:
            _append("vars", "params", data.get("params"))
        if isinstance(data.get("state"), Mapping) and "params" in data.get("state"):
            _append("vars", "state/params", data["state"].get("params"))
        if isinstance(data.get("target"), Mapping) and "params" in data.get("target"):
            _append("vars", "target/params", data["target"].get("params"))

    for key in ("target", "ema_target"):
        sub = None
        try:
            if hasattr(obj, key):
                sub = getattr(obj, key)
            elif isinstance(obj, Mapping):
                sub = obj.get(key)
        except Exception:
            sub = None
        if isinstance(sub, Mapping) and "params" in sub:
            _append("maybe-train-state", f"{key}/params", sub.get("params"))

    try:
        members = getattr(obj, "_fields", None) or getattr(obj, "__slots__", None)
        if members:
            for member in members:
                try:
                    sub = getattr(obj, member)
                except Exception:
                    continue
                if isinstance(sub, Mapping) and "params" in sub:
                    _append("slots", f"{member}/params", sub.get("params"))
    except Exception:
        pass

    return paths


def _debug_probe_params_msgpack(ckpt_path: str):
    import glob
    import os

    print(f"[ckpt] Probing for params.msgpack near: {ckpt_path}")
    for probe in (
        ckpt_path,
        os.path.dirname(ckpt_path),
        os.path.dirname(os.path.dirname(ckpt_path)),
    ):
        cand = os.path.join(probe, "params.msgpack")
        print(f"[ckpt]  - exists({cand}) = {os.path.exists(cand)}")
        try:
            matches = glob.glob(os.path.join(probe, "*.msgpack"))
        except Exception as exc:
            print(f"[ckpt]    glob failed at {probe}: {exc}")
            continue
        for file_name in matches:
            print(f"[ckpt]    found: {file_name}")


def _debug_print_ckpt_summary(ckpt_path: str, restored_obj=None, params=None):
    print(f"[ckpt] Inspecting checkpoint at: {ckpt_path}")
    if restored_obj is not None:
        print("[ckpt] restore_checkpoint returned an object; summarizing...")
        _summarize_pytree(restored_obj, "restored_obj")
        candidates = _find_params_paths(restored_obj)
        if candidates:
            print("[ckpt] Candidate params paths in restored_obj:")
            for kind, path, node in candidates:
                print(f"[ckpt]  - {kind}:{path}")
                if node is not None:
                    _summarize_pytree(node, f"restored_obj.{path}", max_leaves=10)
                else:
                    print(f"[ckpt]    (value unavailable due to access error)")
        else:
            print("[ckpt] No obvious 'params' found in restored_obj (search tried attr/dict/common TS paths).")
    else:
        print("[ckpt] restore_checkpoint returned None.")

    if params is not None:
        print("[ckpt] Summarizing resolved params...")
        _summarize_pytree(params, "params")
    else:
        print("[ckpt] No params resolved to summarize.")


__all__ = [
    "_summarize_pytree",
    "_find_params_paths",
    "_debug_probe_params_msgpack",
    "_debug_print_ckpt_summary",
]

