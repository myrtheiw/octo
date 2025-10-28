
# ======================== run_inference_sim.py ========================
#!/usr/bin/env python3
import os, argparse, time, json, hashlib, csv, logging
import numpy as np
from collections.abc import Mapping
import dm_env
from dm_env import TimeStep
from contextlib import nullcontext

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax, jax.numpy as jnp
import flax.serialization
from collections import deque
from functools import partial
from typing import Any, Optional, Tuple
import mujoco
from mujoco import viewer
try:
    import torch
except ImportError:
    torch = None
try:
    import imageio.v2 as _imageio
except ModuleNotFoundError:
    _imageio = None
    try:
        from PIL import Image as _PILImage
    except ModuleNotFoundError:
        _PILImage = None
else:
    _PILImage = None

from octo.model.octo_model import OctoModel
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training import checkpoints as flax_checkpoints
from octo.utils.spec import ModuleSpec
from octo.scripts.ckpt_debug_utils import (
    _debug_print_ckpt_summary,
    _debug_probe_params_msgpack,
)

from sim_env import (
    PandaSimEnv, build_arm_mapping_from_model, build_arm_dof_indices,
    find_gripper_actuator, auto_ee_ref,
)

GRIPPER_OPEN_CMD = 1.0
GRIPPER_CLOSE_CMD = 0.0
GRIPPER_RAMP_STEP = 0.05
GRIPPER_CLOSE_DIST = 0.006

DEBUG_CKPT_INSPECT = False

# --- dynamic plant builder from your project (octo/record_dataset/helpers.py) ---
try:
    from record_dataset.helpers import (
        build_and_load_scene as _build_and_load_scene,
        find_side_stem_targets as _find_top_targets,
    )
except Exception:
    # Fallback if PYTHONPATH doesn't include the project root:
    # try to add the parent of this script (…/octo) so record_dataset.* is importable
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from record_dataset.helpers import (
            build_and_load_scene as _build_and_load_scene,
            find_side_stem_targets as _find_top_targets,
        )
    except Exception:
        _build_and_load_scene = None

# --- NEW: resolve experiment directories when a parent checkpoints dir is passed ---
import glob, os


def resolve_exp_dirs(path: str):
    """
    Return a list of experiment dirs (each containing config.json).
    If `path` itself is an experiment dir, return [path]. If it is a parent,
    return all children (recursive) that have a config.json, sorted by mtime
    (newest first).
    """
    path = os.path.expanduser(path)
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        exps = [path]
    else:
        # Find all config.json files under the parent
        candidates = glob.glob(os.path.join(path, "**", "config.json"), recursive=True)
        # Filter out the parent itself if matched
        candidates = [c for c in candidates if os.path.dirname(c) != path]
        # Sort newest first by file mtime
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        exps = [os.path.dirname(c) for c in candidates]

    if not exps:
        raise SystemExit(
            f"Could not find any config.json under: {path}. Pass --exp to a specific experiment directory."
        )

    # Pretty-print the top few for clarity (safe single-line strings)
    head = "\n  ".join(exps[:5])
    tail = "\n  ..." if len(exps) > 5 else ""
    print("[resolve] Found experiments (newest first):\n  " + head + tail)
    return exps


def _load_model_from_exp_dir(exp_dir: str, ckpt_file: Optional[str]) -> OctoModel:
    # If no override checkpoint is given, load the pretrained experiment as-is.
    if not ckpt_file:
        return OctoModel.load_pretrained(exp_dir)

    # --- Validate inputs & gather experiment artifacts ---
    exp_dir = os.path.expanduser(exp_dir)
    ckpt_path = os.path.expanduser(ckpt_file)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"--ckpt_file '{ckpt_file}' does not exist or is not accessible.")

    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(config_path):
        raise SystemExit(f"config.json not found under experiment dir: {exp_dir}")
    example_batch_path = os.path.join(exp_dir, "example_batch.msgpack")
    if not os.path.isfile(example_batch_path):
        raise SystemExit(f"example_batch.msgpack not found under experiment dir: {exp_dir}")
    dataset_stats_path = os.path.join(exp_dir, "dataset_statistics.json")
    if not os.path.isfile(dataset_stats_path):
        raise SystemExit(f"dataset_statistics.json not found under experiment dir: {exp_dir}")

    with open(config_path, "r", encoding="utf-8") as fp:
        config = json.load(fp)

    with open(example_batch_path, "rb") as fp:
        example_batch = flax.serialization.msgpack_restore(fp.read())
    # Backward-compat: some runs used "tasks" at the top-level; normalize to "task".
    if "tasks" in example_batch:
        example_batch["task"] = example_batch.pop("tasks")
    # Backward-compat: mirror legacy observation.pad_mask -> observation.timestep_pad_mask
    if "timestep_pad_mask" not in example_batch.get("observation", {}):
        pad_mask = example_batch.get("observation", {}).get("pad_mask")
        if pad_mask is not None:
            example_batch["observation"]["timestep_pad_mask"] = pad_mask

    with open(dataset_stats_path, "r", encoding="utf-8") as fp:
        raw_stats = json.load(fp)
    dataset_statistics = jax.tree_map(
        lambda x: np.array(x),
        raw_stats,
        is_leaf=lambda x: not isinstance(x, dict),
    )

    if config.get("text_processor") is not None:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()
    else:
        text_processor = None

    model = OctoModel.from_config(
        config=config,
        example_batch=example_batch,
        text_processor=text_processor,
        dataset_statistics=dataset_statistics,
    )

    # --- Restore checkpoint or params.msgpack ---
    params = None
    restored_obj = None
    restore_error = None

    # Try Flax/Orbax restore first.
    try:
        restored_obj = flax_checkpoints.restore_checkpoint(ckpt_path, target=None)
    except Exception as exc:
        restore_error = exc
        logging.warning(
            "Failed to restore Flax checkpoint from %s: %s. Falling back to params.msgpack if available.",
            ckpt_path,
            exc,
        )

    # Helper to extract params from common train_state layouts.
    def _extract_params(obj: Any) -> Optional[Any]:
        if obj is None:
            return None

        # 1) Direct attribute
        cand = getattr(obj, "params", None)
        if cand is not None:
            return cand

        # 2) Mapping-like at top level
        if isinstance(obj, Mapping):
            if "params" in obj:
                return obj["params"]
            # Common nested layouts
            for top in ("state", "target", "ema_target"):
                sub = obj.get(top)
                if isinstance(sub, Mapping) and "params" in sub:
                    return sub["params"]

        # 3) Dataclass / objects with __dict__
        if hasattr(obj, "__dict__"):
            d = vars(obj)
            if "params" in d:
                return d["params"]
            for top in ("state", "target", "ema_target"):
                sub = d.get(top)
                if sub is None:
                    continue
                # attr path
                sub_params = getattr(sub, "params", None)
                if sub_params is not None:
                    return sub_params
                # dict path
                if isinstance(sub, Mapping) and "params" in sub:
                    return sub["params"]

        # 4) Fallback: some train_states put everything in a 'target' PyTree directly
        tgt = getattr(obj, "target", None)
        if tgt is not None:
            tgt_params = getattr(tgt, "params", None)
            if tgt_params is not None:
                return tgt_params
            if isinstance(tgt, Mapping) and "params" in tgt:
                return tgt["params"]

        return None

    # Try to pull params from the restored train_state-like object.
    if restored_obj is not None:
        params = _extract_params(restored_obj)

    # Optional debug hook to print nearby params.msgpack candidates
    if params is None and DEBUG_CKPT_INSPECT:
        _debug_probe_params_msgpack(ckpt_path)

    # Fallback to params.msgpack in ckpt_path or its parents.
    if params is None:
        msgpack_candidates = [
            os.path.join(ckpt_path, "params.msgpack"),
            os.path.join(os.path.dirname(ckpt_path), "params.msgpack"),
            os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "params.msgpack"),
        ]
        for candidate in msgpack_candidates:
            if not os.path.isfile(candidate):
                continue
            with open(candidate, "rb") as fp:
                try:
                    params = flax.serialization.msgpack_restore(fp.read())
                    logging.info("Loaded params from fallback msgpack: %s", candidate)
                    break
                except Exception as exc:
                    logging.warning("Failed to load params.msgpack at %s: %s", candidate, exc)
        if params is None and restore_error is not None:
            raise SystemExit(
                f"Tiny-overfit checkpoint at '{ckpt_file}' could not be restored ({restore_error}) "
                "and no params.msgpack fallback was found."
            ) from restore_error

    # Final guard & debug summary.
    if params is None:
        if DEBUG_CKPT_INSPECT:
            _debug_print_ckpt_summary(ckpt_path, restored_obj, params)
        raise SystemExit(f"Tiny-overfit checkpoint at '{ckpt_file}' is missing 'params'.")

    if DEBUG_CKPT_INSPECT:
        _debug_print_ckpt_summary(ckpt_path, restored_obj, params)

    # Swap params into the model and return.
    model = model.replace(params=params)
    logging.info("Loaded params from ckpt_file=%s", ckpt_path)
    return model


def _extract_model_spec(model: OctoModel) -> tuple[int, int, int]:
    """Infer (history, action_horizon, action_dim) from a loaded Octo model."""

    history = None
    example_batch = getattr(model, "example_batch", {})
    if isinstance(example_batch, Mapping):
        observation = example_batch.get("observation", {})
    else:
        observation = {}
    if isinstance(observation, Mapping):
        pad = observation.get("timestep_pad_mask")
        if pad is not None:
            try:
                history = int(np.asarray(pad).shape[1])
            except Exception:
                history = None
        if history is None:
            for key in ("image_primary", "image_wrist", "proprio"):
                arr = observation.get(key)
                if arr is None:
                    continue
                try:
                    arr_np = np.asarray(arr)
                except Exception:
                    continue
                if arr_np.ndim >= 2:
                    history = int(arr_np.shape[1])
                    break
    if history is None:
        try:
            history = int(model.config["model"]["window_size"])
        except Exception:
            history = 1

    action_horizon = None
    action_dim = None
    try:
        head_cfg = model.config["model"]["heads"]["action"]["kwargs"]
    except Exception:
        head_cfg = None
    if isinstance(head_cfg, Mapping):
        action_horizon = head_cfg.get("action_horizon") or head_cfg.get("pred_horizon")
        action_dim = head_cfg.get("action_dim")

    action_head = None
    try:
        heads = getattr(model.module, "heads", None)
        if isinstance(heads, Mapping):
            action_head = heads.get("action")
    except Exception:
        action_head = None
    if action_head is not None:
        if action_horizon is None and hasattr(action_head, "action_horizon"):
            try:
                action_horizon = int(action_head.action_horizon)
            except Exception:
                pass
        if action_dim is None and hasattr(action_head, "action_dim"):
            try:
                action_dim = int(action_head.action_dim)
            except Exception:
                pass

    if action_horizon is None:
        action_horizon = 1
    if action_dim is None:
        action_dim = 1
    return int(history), int(action_horizon), int(action_dim)


def _count_dropout_hints(config: Any) -> int:
    """Heuristically count dropout-like knobs present in the config dict."""

    keys_of_interest = {
        "dropout_rate",
        "attention_dropout_rate",
        "image_dropout_prob",
        "dropout_prob",
    }
    count = 0

    def _walk(node: Any) -> None:
        nonlocal count
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in keys_of_interest:
                    count += 1
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    if isinstance(config, Mapping):
        _walk(config)
    return count



def _summarize_torch_trainables(objs: tuple[Any, ...], freeze: bool = False) -> int:
    """Count (and optionally freeze) trainable torch parameters across objects."""

    if torch is None:
        return 0

    seen: set[int] = set()
    trainable = 0

    for obj in objs:
        if obj is None:
            continue

        param_iter = None
        if hasattr(obj, "parameters"):
            try:
                param_iter = obj.parameters()
            except Exception:
                param_iter = None
        if param_iter is None and hasattr(obj, "named_parameters"):
            try:
                param_iter = (p for _, p in obj.named_parameters())
            except Exception:
                param_iter = None
        if param_iter is None:
            continue

        for param in param_iter:
            try:
                pid = id(param)
            except Exception:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            requires_grad = getattr(param, "requires_grad", False)
            if requires_grad:
                try:
                    trainable += int(param.numel())
                except Exception:
                    trainable += 1
                if freeze:
                    try:
                        param.requires_grad_(False)
                    except Exception:
                        param.requires_grad = False

    return trainable


def _resolve_dataset_version_dir(data_dir: str, dataset_name: str) -> str:
    data_dir = os.path.expanduser(data_dir)
    if "/" in dataset_name:
        return os.path.join(data_dir, dataset_name)
    root = os.path.join(data_dir, dataset_name)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root '{root}' not found")
    versions = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not versions:
        raise FileNotFoundError(f"No dataset versions found under '{root}'")
    try:
        versions.sort(key=lambda s: tuple(int(x) for x in s.split(".")))
    except Exception:
        versions.sort()
    return os.path.join(root, versions[-1])


def load_rlds_dataset(name: str, data_dir: str, split: str, shuffle: bool = False, seed: int = 0):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    version_dir = None
    try:
        ds = tfds.load(name, data_dir=data_dir, split=split)
        version_dir = _resolve_dataset_version_dir(data_dir, name)
    except Exception:
        version_dir = _resolve_dataset_version_dir(data_dir, name)
        builder = tfds.builder_from_directory(version_dir)
        ds = builder.as_dataset(split=split)

    if version_dir is not None:
        print(f"[dataset] Using TFDS builder directory: {version_dir}")

    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=True)

    return tfds.as_numpy(ds)


def supply_rng(fn, seed=0):
    rng = jax.random.PRNGKey(int(seed))

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, sub = jax.random.split(rng)
        try:
            return fn(*args, rng=sub, **kwargs)
        except TypeError:
            try:
                return fn(*args, prng_key=sub, **kwargs)
            except TypeError:
                return fn(*args, **kwargs)

    return wrapped


def _nearest_downsample(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    Ht, Wt = target_hw
    H, W = img.shape[:2]
    if (H, W) == (Ht, Wt):
        return img
    # integer-stride nearest neighbor (fast, no deps)
    sh = max(1, int(round(H / max(1, Ht))))
    sw = max(1, int(round(W / max(1, Wt))))
    return img[::sh, ::sw]

def build_example_meta(example_batch) -> dict:
    """
    Extract observation metadata (shape, dtype) from example_batch.
    Normalizes keys to the ('observation', ...) namespace and synthesizes
    missing entries for vision/proprio so runtime aggregation won't prune them.
    """
    meta: dict = {}

    # 1) Collect any observation entries from a flattened example_batch
    for path, value in flatten_dict(example_batch).items():
        if not path:
            continue
        head, *rest = path
        if head not in ("observations", "observation"):
            continue
        if not rest:
            continue
        key = ("observation", *rest)
        try:
            shape = tuple(value.shape)
            dtype = np.dtype(value.dtype)
        except Exception:
            arr = np.asarray(value)
            shape, dtype = arr.shape, arr.dtype
        meta[key] = (shape, dtype)

    # 2) Legacy structure support: example_batch["observations"][k] = array
    if not meta and isinstance(example_batch, dict) and "observations" in example_batch:
        for k, v in example_batch["observations"].items():
            try:
                shape = tuple(v.shape)
                dtype = np.dtype(v.dtype)
            except Exception:
                arr = np.asarray(v)
                shape, dtype = arr.shape, arr.dtype
            meta[("observation", k)] = (shape, dtype)

    # 3) Infer history/window size H to synthesize defaults when absent
    def _infer_history(m: dict) -> int:
        # Prefer an explicit timestep mask shape: (B, H, ...)
        info = m.get(("observation", "timestep_pad_mask")) or m.get(("timestep_pad_mask",))
        if info is not None:
            shp = info[0]
            if isinstance(shp, (tuple, list)) and len(shp) >= 2:
                try:
                    return int(shp[1])
                except Exception:
                    pass
        # Else, use any observation entry that looks like (B, H, ...)
        for (k0, *rest), (shp, _dt) in m.items():
            if k0 == "observation" and isinstance(shp, (tuple, list)) and len(shp) >= 2:
                try:
                    return int(shp[1])
                except Exception:
                    continue
        return 1

    H = _infer_history(meta)

    # 4) Synthesize expected observation entries if missing
    # Defaults align with your setup: primary 256^2, wrist 128^2, proprio dim=9
    defaults = {
        ("observation", "image_primary"): ((1, H, 256, 256, 3), np.dtype(np.uint8)),
        ("observation", "image_wrist"):   ((1, H, 128, 128, 3), np.dtype(np.uint8)),
        ("observation", "proprio"):       ((1, H, 9),           np.dtype(np.float32)),
    }
    for key, (shape, dtype) in defaults.items():
        if key not in meta:
            meta[key] = (shape, dtype)

    return meta



def _read_image_file(path: str) -> np.ndarray:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Goal image not found at '{path}'")
    if _imageio is not None:
        img = _imageio.imread(path)
    elif _PILImage is not None:
        with _PILImage.open(path) as im:
            img = np.array(im.convert("RGB"))
    else:
        raise RuntimeError(
            "Unable to load goal image. Install imageio (`pip install imageio`) or Pillow (`pip install pillow`)."
        )
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return np.asarray(img, dtype=np.uint8)


def _normalize_language_value(value) -> Optional[str]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.dtype == object:
        arr = arr.flat[0]
    while isinstance(arr, np.ndarray):
        if arr.size == 0:
            return None
        arr = arr.flat[0]
    if isinstance(arr, bytes):
        try:
            return arr.decode("utf-8")
        except Exception:
            return arr.decode("utf-8", errors="ignore")
    if isinstance(arr, str):
        return arr
    return str(arr)


def _resize_goal_image(img: np.ndarray, target_shape: tuple[int, ...], dtype) -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if len(target_shape) < 3:
        raise ValueError(f"Expected goal image shape (*, H, W, C) with C>=1; got {target_shape!r}")
    target_h, target_w = target_shape[0], target_shape[1]
    target_c = target_shape[2]
    if (img.shape[0], img.shape[1]) != (target_h, target_w):
        if _PILImage is None and _imageio is None:
            raise RuntimeError(
                "Pillow is required to resize goal images; please install it (pip install pillow)."
            )
        if _PILImage is not None:
            img_resized = _PILImage.fromarray(img).resize((target_w, target_h), resample=_PILImage.BILINEAR)
            img = np.array(img_resized)
        else:
            # fallback to nearest-neighbor using numpy slicing
            img = _nearest_downsample(img, (target_h, target_w))
    if img.shape[-1] != target_c:
        if img.shape[-1] == 3 and target_c == 1:
            img = np.mean(img, axis=-1, keepdims=True)
        elif img.shape[-1] == 1 and target_c == 3:
            img = np.repeat(img, 3, axis=-1)
        else:
            raise ValueError(f"Cannot reshape goal image channels from {img.shape[-1]} to {target_c}")
    return img.astype(dtype, copy=False)


def _prepare_goal_payload(model: OctoModel, primary: Optional[np.ndarray], wrist: Optional[np.ndarray]):
    task_template = model.example_batch.get("task", {})
    payload = {}
    missing = []
    provided = set()
    for key, value in task_template.items():
        if key in ("pad_mask_dict", "language_instruction"):
            continue
        target_shape = tuple(value.shape[1:])
        dtype = np.dtype(value.dtype)
        img = None
        if "primary" in key and primary is not None:
            img = primary
        elif "wrist" in key and wrist is not None:
            img = wrist
        if img is None:
            payload[key] = np.zeros((1, *target_shape), dtype=dtype)
            missing.append(key)
        else:
            resized = _resize_goal_image(img, target_shape, dtype)
            payload[key] = resized[None, ...]
            provided.add(key)
    # Ensure at least primary image is filled from whichever source is available
    if "image_primary" in task_template and "image_primary" not in payload:
        if primary is not None:
            target_shape = tuple(task_template["image_primary"].shape[1:])
            dtype = np.dtype(task_template["image_primary"].dtype)
            payload["image_primary"] = _resize_goal_image(primary, target_shape, dtype)[None, ...]
            provided.add("image_primary")
        else:
            target_shape = tuple(task_template["image_primary"].shape[1:])
            dtype = np.dtype(task_template["image_primary"].dtype)
            payload["image_primary"] = np.zeros((1, *target_shape), dtype=dtype)
            missing.append("image_primary")
    if not provided:
        raise ValueError("No goal imagery could be provided for any of the expected task keys.")
    return payload, missing


def _first_frame(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr.astype(np.uint8, copy=False)


def _load_goal_images_from_dataset_episode(
    dataset_name: str,
    data_dir: str,
    split: str,
    episode_index: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    ds_iter = load_rlds_dataset(dataset_name, data_dir, split, shuffle=False)
    for idx, episode in enumerate(ds_iter):
        if idx == episode_index:
            episode = _structured_to_dict(episode)
            steps = episode.get("steps")
            steps = _materialize_steps(steps)
            steps = _structured_to_dict(steps)
            primary = None
            wrist = None
            language = None
            if isinstance(steps, Mapping):
                obs = steps.get("observation")
                if isinstance(obs, Mapping):
                    primary = obs.get("goal_image_primary")
                    wrist = obs.get("goal_image_wrist")
                    language = _normalize_language_value(obs.get("language_instruction"))
                if primary is None:
                    primary = steps.get("goal_image_primary")
                if wrist is None:
                    wrist = steps.get("goal_image_wrist")
                if language is None:
                    language = _normalize_language_value(steps.get("language_instruction"))
            return _first_frame(primary), _first_frame(wrist), language
    raise IndexError(
        f"Episode index {episode_index} not found in dataset '{dataset_name}' split '{split}'."
    )


def _synth_namespaced_timestep(flat: dict):
    if "observation/pad_mask_dict/timestep" in flat and "observation/timestep_pad_mask" not in flat:
        flat["observation/timestep_pad_mask"] = flat["observation/pad_mask_dict/timestep"]
    if "observation/timestep_pad_mask" in flat and "observation/pad_mask_dict/timestep" not in flat:
        flat["observation/pad_mask_dict/timestep"] = flat["observation/timestep_pad_mask"]


def _add_legacy_aliases(d: dict):
    for k in ("proprio", "timestep", "task_completed", "image_primary", "image_wrist"):
        nk = f"observation/{k}"
        if nk in d and k not in d:
            d[k] = d[nk]
    for k in ("proprio", "timestep", "image_primary", "image_wrist"):
        nk = f"observation/pad_mask_dict/{k}"
        lk = f"pad_mask_dict/{k}"
        if nk in d and lk not in d:
            d[lk] = d[nk]
    src = d.get("observation/timestep_pad_mask") or d.get("observation/pad_mask_dict/timestep")
    if src is not None:
        d.setdefault("timestep_pad_mask", src)
        d.setdefault("pad_mask_dict/timestep", src)


def _structured_to_dict(value):
    if isinstance(value, Mapping):
        return {k: _structured_to_dict(v) for k, v in value.items()}
    if hasattr(value, "items") and callable(getattr(value, "items")):
        try:
            return {k: _structured_to_dict(v) for k, v in value.items()}
        except TypeError:
            pass
    if hasattr(value, "keys") and callable(getattr(value, "keys")):
        try:
            return {k: _structured_to_dict(value[k]) for k in value.keys()}
        except TypeError:
            pass
    # numpy structured scalars/arrays (dtype with named fields)
    names = getattr(getattr(value, "dtype", None), "names", None)
    if names:
        return {name: _structured_to_dict(value[name]) for name in names}
    return value


def _stack_step_samples(samples):
    if not samples:
        return samples
    first = samples[0]
    if isinstance(first, Mapping):
        keys = first.keys()
        return {k: _stack_step_samples([s[k] for s in samples]) for k in keys}
    arrays = [np.asarray(s) for s in samples]
    try:
        return np.stack(arrays, axis=0)
    except Exception:
        return np.asarray(arrays)


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    collapsed = "".join(safe).strip("_")
    return collapsed or "run"


def _to_python_scalar(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return str(value)


def _array_summary(value: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(value)
    except Exception:
        return {"type": str(type(value))}

    info: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }

    if arr.size == 0:
        info["empty"] = True
        return info

    if arr.dtype.kind in ("O", "U", "S"):
        flat = arr.reshape(-1)
        first = flat[0]
        info["sample"] = _to_python_scalar(first)
        unique = np.unique(flat[: min(flat.size, 32)])
        info["unique_in_sample"] = int(unique.size)
        return info

    arr_float = arr.astype(np.float64)
    info["min"] = _to_python_scalar(np.nanmin(arr_float))
    info["max"] = _to_python_scalar(np.nanmax(arr_float))
    info["mean"] = _to_python_scalar(np.nanmean(arr_float))
    info["std"] = _to_python_scalar(np.nanstd(arr_float))
    info["nan_count"] = int(np.isnan(arr_float).sum())
    info["inf_count"] = int(np.isinf(arr_float).sum())

    flat = arr.reshape(-1)
    if flat.size <= 4096:
        digest = hashlib.sha1(arr.tobytes()).hexdigest()
    else:
        sample = flat[:4096]
        digest = hashlib.sha1(sample.tobytes()).hexdigest()
        info["sampled_for_hash"] = True
    info["sha1"] = digest[:16]
    return info


def _tree_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _tree_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tree_summary(v) for v in value]
    return _array_summary(value)


def _meta_summary(meta: dict[tuple[str, ...], tuple[tuple[int, ...], np.dtype]]) -> dict[str, Any]:
    summary = {}
    for key, (shape, dtype) in meta.items():
        summary["/".join(key)] = {
            "shape": list(shape),
            "dtype": str(dtype),
        }
    return summary


def _vector_payload(arr: Any) -> dict[str, Any]:
    arr_np = np.asarray(arr, dtype=np.float32).reshape(-1)
    payload = {
        "vector": arr_np.tolist(),
        "summary": _array_summary(arr_np),
    }
    try:
        payload["norm"] = float(np.linalg.norm(arr_np))
    except Exception:
        pass
    return payload


def _log_joint_motion_event(
    trace_logger: "TraceLogger",
    tag: str,
    step_idx: int,
    *,
    commanded: Any = None,
    actual: Any = None,
    q_before: Any = None,
    q_after: Any = None,
    raw_action: Any = None,
    misc: Optional[Mapping[str, Any]] = None,
) -> None:
    enabled = (
        trace_logger is not None
        and isinstance(trace_logger, TraceLogger)
        and trace_logger.enabled
        and trace_logger._fp is not None  # type: ignore[attr-defined]
    )

    if not enabled:
        return

    payload: dict[str, Any] = {
        "tag": str(tag),
        "step": int(step_idx),
    }

    if commanded is not None:
        payload["commanded_delta"] = _vector_payload(commanded)
    if actual is not None:
        payload["actual_delta"] = _vector_payload(actual)
    if q_before is not None:
        payload["q_before"] = _vector_payload(q_before)
    if q_after is not None:
        payload["q_after"] = _vector_payload(q_after)
    if raw_action is not None:
        payload["raw_action"] = _vector_payload(raw_action)

    if misc:
        misc_payload: dict[str, Any] = {}
        for key, value in misc.items():
            if isinstance(value, (np.ndarray, list, tuple)):
                try:
                    misc_payload[str(key)] = _vector_payload(value)
                    continue
                except Exception:
                    pass
            misc_payload[str(key)] = _to_python_scalar(value)
        payload["context"] = misc_payload

    trace_logger.log_event("joint_motion", payload)

def _find_action_statistics(stats: Any) -> Optional[Mapping[str, Any]]:
    """Locate an 'action' stats dict in potentially nested dataset statistics."""
    if isinstance(stats, Mapping):
        action = stats.get("action")
        if isinstance(action, Mapping):
            return action
        for value in stats.values():
            found = _find_action_statistics(value)
            if found is not None:
                return found
    return None


def _vec_norms(arr: Any) -> list[float]:
    """Return vector norms for 1D or 2D inputs."""
    a = np.asarray(arr)
    if a.ndim == 1:
        return [float(np.linalg.norm(a))]
    if a.ndim == 2:
        return [float(x) for x in np.linalg.norm(a, axis=-1)]
    return []


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


class TraceLogger:
    def __init__(self, root: Optional[str], every: int = 1):
        self.enabled = bool(root)
        self.root = os.path.abspath(root) if root else None
        self.every = max(1, int(every))
        self.run_dir: Optional[str] = None
        self._fp = None
        self._step_counter = 0
        self._example_meta: Optional[dict[tuple[str, ...], tuple[tuple[int, ...], np.dtype]]] = None

    def start_run(self, exp_dir: str) -> Optional[str]:
        if not self.enabled:
            return None
        if self._fp is not None:
            self.close_run()
        os.makedirs(self.root, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = os.path.basename(exp_dir.rstrip(os.sep)) or exp_dir
        safe_name = _sanitize_name(exp_name)
        run_dir = os.path.join(self.root, f"{timestamp}_{safe_name}")
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "trace.jsonl")
        self._fp = open(path, "w", encoding="utf-8")
        self.run_dir = run_dir
        self._step_counter = 0
        self.log_event("start_run", {"exp_dir": exp_dir, "trace_file": path})
        return run_dir

    def set_example_meta(self, meta: dict[tuple[str, ...], tuple[tuple[int, ...], np.dtype]]):
        if not self.enabled:
            return
        self._example_meta = meta
        self.log_event("example_meta", _meta_summary(meta))

    def log_event(self, event: str, payload: dict[str, Any]):
        if not self.enabled or self._fp is None:
            return
        record = {
            "time": time.time(),
            "event": event,
            "payload": payload,
        }
        self._fp.write(json.dumps(record, default=_json_default) + "\n")
        self._fp.flush()

    def log_model_template(self, model: OctoModel):
        if not self.enabled:
            return
        template = {
            "example_batch": _tree_summary(model.example_batch),
        }
        if model.dataset_statistics is not None:
            template["dataset_statistics"] = _tree_summary(model.dataset_statistics)
        template["config_keys"] = sorted(model.config.keys()) if isinstance(model.config, Mapping) else None
        self.log_event("model_template", template)

    def log_action_statistics(self, stats: Any):
        if not self.enabled:
            return
        self.log_event("action_unnormalization_statistics", _tree_summary(stats))

    def log_task(self, tag: str, task: Any):
        if not self.enabled:
            return
        self.log_event(tag, _tree_summary(task))

    def log_step(
        self,
        step_idx: int,
        timestep: TimeStep,
        obs_for_model: Mapping[str, Any],
        action_tree: Any,
        action_vector: np.ndarray,
        gripper_cmd: float,
        history: "ObservationHistory",
        mc_samples: Optional[np.ndarray] = None,
        policy_mode: str = "policy",
    ):
        if not self.enabled or self._fp is None:
            return

        should_log = (self._step_counter % self.every) == 0
        self._step_counter += 1
        if not should_log:
            return

        history_info = {
            "/".join(key): {
                "len": len(buf),
                "gap": max(0, history.window_size - len(buf)),
            }
            for key, buf in history.storage.items()
        }

        payload: dict[str, Any] = {
            "step_index": int(step_idx),
            "policy_mode": policy_mode,
            "raw_observation": _tree_summary(timestep.observation),
            "model_input": _tree_summary(obs_for_model),
            "history_status": {
                "window_size": history.window_size,
                "buffers": history_info,
            },
            "action": {
                "vector": np.asarray(action_vector, dtype=np.float32).tolist(),
                "summary": _array_summary(action_vector),
                "gripper_cmd": float(gripper_cmd),
            },
        }

        if action_tree is not None:
            payload["action_tree"] = _tree_summary(action_tree)
        if mc_samples is not None:
            payload["mc_samples"] = _array_summary(mc_samples)
        if hasattr(timestep, "reward") and timestep.reward is not None:
            payload["reward"] = _to_python_scalar(timestep.reward)
        if hasattr(timestep, "discount") and timestep.discount is not None:
            payload["discount"] = _to_python_scalar(timestep.discount)

        try:
            pad_mask = obs_for_model.get("timestep_pad_mask")
            if pad_mask is not None:
                payload["timestep_pad_mask"] = _array_summary(pad_mask)
        except AttributeError:
            pass

        self.log_event("step", payload)

    def close_run(self):
        if not self.enabled or self._fp is None:
            return
        self.log_event("end_run", {})
        self._fp.close()
        self._fp = None
        self.run_dir = None
        self._example_meta = None


class StepLogger:
    """Lightweight per-step logger producing a rolling CSV and optional NPZ dumps."""

    VECTOR_DIM = 7
    _ARRAY_KEYS = ("q", "qdot", "a_raw", "a_post", "dq_meas", "q_ref", "ctrl_cmd", "clamped")
    _HEADER = (
        ["episode", "step", "mode", "policy_mode", "action_scale", "tokens_hash"]
        + [f"q{i}" for i in range(VECTOR_DIM)]
        + [f"qdot{i}" for i in range(VECTOR_DIM)]
        + [f"a_raw{i}" for i in range(VECTOR_DIM)]
        + [f"a_post{i}" for i in range(VECTOR_DIM)]
        + [f"dq_meas{i}" for i in range(VECTOR_DIM)]
        + [f"q_ref{i}" for i in range(VECTOR_DIM)]
        + [f"ctrl_cmd{i}" for i in range(VECTOR_DIM)]
        + [f"clamped{i}" for i in range(VECTOR_DIM)]
    )

    def __init__(self, csv_path: Optional[str] = None, npz_dir: Optional[str] = None):
        self.enabled = bool(csv_path or npz_dir)
        self.csv_path = os.path.abspath(csv_path) if csv_path else None
        self.npz_dir = os.path.abspath(npz_dir) if npz_dir else None
        self._csv_file = None
        self._csv_writer = None
        self._header_written = False
        self._episode_counter = 0
        self._current: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _ensure_csv(self):
        if not self.csv_path or self._csv_writer is not None:
            return
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        file_exists = os.path.isfile(self.csv_path)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists or os.path.getsize(self.csv_path) == 0:
            self._csv_writer.writerow(self._HEADER)
            self._header_written = True

    @staticmethod
    def _vector_to_fixed(value: Any, fill_value: float = float("nan")) -> np.ndarray:
        vec = np.full((StepLogger.VECTOR_DIM,), fill_value, dtype=np.float32)
        if value is None:
            return vec
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        n = min(arr.size, StepLogger.VECTOR_DIM)
        if n > 0:
            vec[:n] = arr[:n]
        return vec

    @staticmethod
    def _flags_to_fixed(value: Any) -> np.ndarray:
        vec = np.full((StepLogger.VECTOR_DIM,), -1, dtype=np.int8)
        if value is None:
            return vec
        arr = np.asarray(value).reshape(-1)
        n = min(arr.size, StepLogger.VECTOR_DIM)
        if n > 0:
            vec[:n] = (arr[:n] != 0).astype(np.int8)
        return vec

    def start_episode(self, tag: str, metadata: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        if not self.enabled:
            return None
        self._ensure_csv()
        self._episode_counter += 1
        episode_id = f"{_sanitize_name(tag) or 'episode'}_{self._episode_counter:03d}"
        episode_meta = dict(metadata or {})
        arrays = {key: [] for key in self._ARRAY_KEYS}
        self._current = {
            "id": episode_id,
            "meta": episode_meta,
            "arrays": arrays,
            "steps": [],
            "policy_mode": [],
            "action_scale": [],
            "mode": None,
            "tokens_hash": "",
        }
        return episode_id

    def log_step(
        self,
        episode_id: Optional[str],
        step_idx: int,
        *,
        mode: str = "",
        policy_mode: str = "",
        action_scale: Optional[float] = None,
        tokens_hash: str = "",
        q: Any = None,
        qdot: Any = None,
        a_raw: Any = None,
        a_post: Any = None,
        dq_meas: Any = None,
        q_ref: Any = None,
        ctrl_cmd: Any = None,
        clamped_flags: Any = None,
    ) -> None:
        if not self.enabled or episode_id is None:
            return
        if not self._current or self._current.get("id") != episode_id:
            return
        cur = self._current
        if mode and cur["mode"] is None:
            cur["mode"] = mode
        row_mode = mode or cur["mode"] or ""
        if tokens_hash:
            cur["tokens_hash"] = tokens_hash
        cur["steps"].append(int(step_idx))
        cur["policy_mode"].append(str(policy_mode or ""))
        cur["action_scale"].append(float(action_scale) if action_scale is not None else float("nan"))

        arrays = cur["arrays"]
        arrays["q"].append(self._vector_to_fixed(q))
        arrays["qdot"].append(self._vector_to_fixed(qdot))
        arrays["a_raw"].append(self._vector_to_fixed(a_raw))
        arrays["a_post"].append(self._vector_to_fixed(a_post))
        arrays["dq_meas"].append(self._vector_to_fixed(dq_meas))
        arrays["q_ref"].append(self._vector_to_fixed(q_ref))
        arrays["ctrl_cmd"].append(self._vector_to_fixed(ctrl_cmd))
        arrays["clamped"].append(self._flags_to_fixed(clamped_flags))

        if self._csv_writer:
            row = [
                episode_id,
                int(step_idx),
                row_mode,
                str(policy_mode or ""),
                "" if action_scale is None else str(float(action_scale)),
                cur.get("tokens_hash", ""),
            ]
            row.extend(arrays["q"][-1].tolist())
            row.extend(arrays["qdot"][-1].tolist())
            row.extend(arrays["a_raw"][-1].tolist())
            row.extend(arrays["a_post"][-1].tolist())
            row.extend(arrays["dq_meas"][-1].tolist())
            row.extend(arrays["q_ref"][-1].tolist())
            row.extend(arrays["ctrl_cmd"][-1].tolist())
            row.extend(arrays["clamped"][-1].tolist())
            self._csv_writer.writerow(row)
            if self._csv_file:
                self._csv_file.flush()

    def end_episode(self, episode_id: Optional[str], extra_meta: Optional[Mapping[str, Any]] = None) -> None:
        if not self.enabled or episode_id is None:
            return
        if not self._current or self._current.get("id") != episode_id:
            return
        cur = self._current
        if extra_meta:
            cur["meta"].update(extra_meta)
        if self.npz_dir:
            os.makedirs(self.npz_dir, exist_ok=True)
            arrays = {}
            for key, values in cur["arrays"].items():
                if not values:
                    continue
                stack = np.stack(values, axis=0)
                if key == "clamped":
                    arrays[key] = stack.astype(np.int8)
                else:
                    arrays[key] = stack.astype(np.float32)
            data = {
                "step": np.asarray(cur["steps"], dtype=np.int32),
                "action_scale": np.asarray(cur["action_scale"], dtype=np.float32),
                "policy_mode": np.asarray(cur["policy_mode"], dtype=object),
                "tokens_hash": np.asarray([cur.get("tokens_hash", "")], dtype=object),
                "mode": np.asarray([cur.get("mode", "")], dtype=object),
                "metadata_json": np.asarray(
                    [json.dumps(cur["meta"], default=_json_default)],
                    dtype=object,
                ),
            }
            data.update(arrays)
            npz_path = os.path.join(self.npz_dir, f"{episode_id}.npz")
            np.savez(npz_path, **data)
        self._current = None

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        self._current = None


def _action_from_tree(tree, action_dim: int = 7) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return np.zeros((action_dim,), np.float32)
    return np.asarray(leaves[0]).reshape(-1)[:action_dim].astype(np.float32)


def _compose_video_frame(observation: Mapping[str, np.ndarray]) -> np.ndarray:
    primary = np.asarray(observation.get("image_primary"), dtype=np.uint8)
    if primary.ndim != 3:
        primary = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = observation.get("image_wrist")
    frames = [primary]
    if wrist is not None:
        try:
            wrist_arr = np.asarray(wrist, dtype=np.uint8)
            if wrist_arr.ndim == 3 and wrist_arr.shape[0] == primary.shape[0]:
                frames.append(wrist_arr)
        except Exception:
            pass
    if len(frames) == 1:
        return frames[0]
    try:
        return np.concatenate(frames, axis=1)
    except ValueError:
        return primary


def _write_video(frames: list[np.ndarray], fps: float, path: str) -> None:
    if not frames:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit(
            "Recording video requires imageio. Install via 'pip install imageio imageio-ffmpeg'."
        ) from exc

    sanitized = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    imageio.mimwrite(path, sanitized, fps=float(max(1.0, fps)), macro_block_size=None)
    print(f"[video] Saved {len(sanitized)} frames to {path}")


def _materialize_steps(steps_obj):
    if isinstance(steps_obj, Mapping):
        return steps_obj
    if hasattr(steps_obj, "__iter__") and not isinstance(steps_obj, (bytes, str, np.ndarray)):
        samples = []
        for step in steps_obj:
            samples.append(_structured_to_dict(step))
        if not samples:
            return {}
        return _stack_step_samples(samples)
    return steps_obj


def _lookup_nested(tree: dict, key_path: str):
    cur = tree
    for part in key_path.split('/'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first_present(tree: dict, keys):
    for key in keys:
        val = _lookup_nested(tree, key)
        if val is not None:
            return key, val
    return None, None


def _make_first_timestep(env: PandaSimEnv) -> TimeStep:
    env._t = 0
    img_primary, img_wrist = env._render_images()
    return TimeStep(
        dm_env.StepType.FIRST,
        reward=np.float32(0.0),
        discount=np.float32(1.0),
        observation={
            "proprio": env.data.qpos.astype(np.float32).copy(),
            "image_primary": img_primary,
            "image_wrist": img_wrist,
            "timestep": np.int32(env._t),
        },
    )


def _reset_env_to_proprio(env: PandaSimEnv, qpos: np.ndarray) -> TimeStep:
    mujoco.mj_resetData(env.model, env.data)
    env.data.qvel[:] = 0.0
    if qpos.shape[0] == env.model.nq:
        env.data.qpos[:] = qpos.astype(np.float64)
    else:
        env.data.qpos[:] = 0.0
        env.data.qpos[env.arm_qpos_addr] = qpos[: len(env.arm_qpos_addr)].astype(np.float64)
    if getattr(env, "_all_position_act", False):
        env.data.ctrl[env.arm_act_ids] = env.data.qpos[env.arm_qpos_addr]
    mujoco.mj_forward(env.model, env.data)
    return _make_first_timestep(env)


def _extract_episode_arrays(episode: dict):
    flat_keys = [k for k in episode.keys() if isinstance(k, str) and k.startswith("steps/")]
    if flat_keys:
        steps_nested = {}
        for key in flat_keys:
            parts = key.split("/")  # e.g. ["steps", "observation", "proprio"]
            cur = steps_nested
            for part in parts[1:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = episode[key]
        episode = dict(episode)
        episode["steps"] = steps_nested

    steps = _materialize_steps(episode.get("steps"))
    steps = _structured_to_dict(steps)
    if isinstance(steps, np.ndarray):
        if steps.dtype == object:
            try:
                steps = steps.item()
            except Exception:
                pass
        # Slide singleton dimensions
        if getattr(steps, "ndim", 0) == 1 and steps.shape[0] == 1:
            try:
                steps = steps[0]
            except Exception:
                pass
    steps = _structured_to_dict(steps)
    if isinstance(steps, Mapping):
        episode = dict(episode)
        episode["steps"] = steps
    if not isinstance(steps, Mapping):
        available = sorted(str(k) for k in episode.keys())
        details = f"steps type={type(steps)!r}"
        if isinstance(steps, np.ndarray):
            details += f", dtype={steps.dtype!r}, ndim={steps.ndim}, shape={steps.shape}"
        raise ValueError(
            "Episode dictionary does not contain 'steps' (available keys: "
            + ", ".join(available)
            + f") [{details}]"
        )

    proprio_keys = (
        "observation/proprio",
        "proprio",
        "observation/qpos",
        "qpos",
    )
    action_keys = (
        "action",
        "policy/action",
        "actions",
        "observation/action",
    )

    pk, proprio = _first_present(steps, proprio_keys)
    ak, actions = _first_present(steps, action_keys)
    if proprio is None or actions is None:
        raise KeyError(
            "Dataset episode missing proprio or action arrays. "
            f"Checked proprio keys {proprio_keys}, action keys {action_keys}."
        )

    proprio = np.asarray(proprio)
    actions = np.asarray(actions)
    if proprio.ndim != 2:
        raise ValueError(f"Expected proprio rank-2 [T, nq], got shape {proprio.shape} (key '{pk}')")
    if actions.ndim == 1:
        actions = actions[:, None]
    if actions.ndim != 2:
        raise ValueError(f"Expected action rank-2 [T, na], got shape {actions.shape} (key '{ak}')")

    is_last = _lookup_nested(steps, "is_last")
    if is_last is not None:
        is_last = np.asarray(is_last).astype(bool)
        last_indices = np.where(is_last)[0]
        if last_indices.size:
            valid = last_indices[0] + 1
            proprio = proprio[:valid]
            actions = actions[:valid]

    return proprio, actions

class ObservationHistory:
    def __init__(self, meta: dict):
        self.meta = meta
        self.window_size = self._infer_window_size()
        self.storage: dict[tuple[str, ...], deque[np.ndarray]] = {}

    def _infer_window_size(self) -> int:
        for key, (shape, _) in self.meta.items():
            if not key:
                continue
            if key[0] == "timestep" and len(shape) > 1:
                return int(shape[1])
            if key[0] == "observation" and len(shape) > 1:
                return int(shape[1])
        return 1

    def push(self, key: tuple[str, ...], value: np.ndarray) -> None:
        buf = self.storage.setdefault(key, deque(maxlen=self.window_size))
        buf.append(value)

    def aggregate(self, key: tuple[str, ...], allocate_fn):
        arr = allocate_fn(key)
        if arr is None:
            return None
        buf = self.storage.get(key)
        if not buf:
            return arr
        buf_list = list(buf)
        start = arr.shape[1] - len(buf_list)
        for offset, val in enumerate(buf_list):
            slices = (
                slice(None),
                slice(start + offset, start + offset + val.shape[1]),
            ) + tuple(slice(None) for _ in range(2, arr.ndim))
            arr[slices] = val
        return arr


def build_obs_for_octo(
    ts_obs,
    t_idx: int,
    meta: dict,
    history: ObservationHistory,
    img_shapes: dict | None = None,
):
    """Format simulator observations using a rolling history matching example_batch."""

    def _allocate(path):
        info = meta.get(path)
        if info is None:
            return None
        shape, dtype = info
        return np.zeros((1,) + tuple(shape[1:]), dtype=dtype)

    def _step_value(path, value):
        info = meta.get(path)
        if info is None:
            return None
        target_shape, dtype = info
        step_shape = (target_shape[0], 1) + tuple(target_shape[2:])
        arr = np.asarray(value, dtype=dtype)
        arr = np.reshape(arr, step_shape)
        return arr

    img_p = np.asarray(ts_obs["image_primary"], np.uint8)
    img_w = np.asarray(ts_obs["image_wrist"], np.uint8)
    if img_shapes:
        if "image_primary" in img_shapes:
            img_p = _nearest_downsample(img_p, img_shapes["image_primary"])
        if "image_wrist" in img_shapes:
            img_w = _nearest_downsample(img_w, img_shapes["image_wrist"])

    img_p = img_p[None, None, ...]
    img_w = img_w[None, None, ...]
    proprio = np.asarray(ts_obs["proprio"], np.float32)[None, None, ...]
    timestep = np.array([[t_idx]], np.int32)

    task_completed = ts_obs.get("task_completed")
    if task_completed is None:
        info = meta.get(("observation", "task_completed"))
        if info is not None:
            step_shape = (info[0][0], 1) + tuple(info[0][2:])
            task_completed = np.zeros(step_shape, dtype=info[1])
        else:
            task_completed = np.zeros((1, 1, 4), np.float32)
    else:
        task_completed = np.asarray(task_completed, np.float32)
        if task_completed.ndim == 1:
            task_completed = task_completed[None, None, ...]
        elif task_completed.ndim == 2:
            task_completed = task_completed[None, ...]

    entries = {
        ("observation","image_primary"): img_p,
        ("observation","image_wrist"): img_w,
        ("observation","proprio"): proprio,
        ("observation","task_completed"): task_completed,
        ("timestep",): timestep,
    }

    timestep_mask_step = np.ones((1, 1), dtype=np.bool_)
    entries[("timestep_pad_mask",)] = timestep_mask_step
    entries[("observation", "timestep_pad_mask")] = timestep_mask_step

    # --- Mask names can live either under top-level 'pad_mask_dict/*' (legacy)
    # --- or 'observation/pad_mask_dict/*' (new). Accept both.
    mask_names = set()
    for key in meta:
        if len(key) == 2 and key[0] == "pad_mask_dict":
            mask_names.add(key[1])
        if len(key) == 3 and key[0] == "observation" and key[1] == "pad_mask_dict":
            mask_names.add(key[2])
    if not mask_names:
        mask_names = {"image_primary", "image_wrist", "proprio", "timestep"}

    # Write BOTH namespaces so aggregation can find whichever meta declares.
    for name in sorted(mask_names):
        mask_step = np.ones((1, 1), dtype=np.bool_)
        entries[("observation","pad_mask_dict", name)] = mask_step
        entries[("pad_mask_dict", name)] = mask_step

    for key, val in entries.items():
        step_val = _step_value(key, val)
        if step_val is not None:
            history.push(key, step_val)

    flat_result = {}
    for key in meta:
        if key[0] in ("observation", "pad_mask_dict") or key == ("timestep_pad_mask",):
            aggregated = history.aggregate(key, _allocate)
            if aggregated is not None:
                flat_result[key] = aggregated

    # Ensure every meta entry exists (fill zeros if we haven't seen it yet)
    for key in meta:
        if key not in flat_result:
            alloc = _allocate(key)
            if alloc is not None:
                flat_result[key] = alloc

    nested = unflatten_dict(flat_result)

    # Prune any keys the model never saw during finetuning
    allowed = set(meta.keys())
    # always allow the top-level timestep mask for convenience
    allowed.add(("timestep_pad_mask",))
    flattened = flatten_dict(nested)
    for key in list(flattened.keys()):
        if key not in allowed:
            flattened.pop(key)
    nested = unflatten_dict(flattened)

    # Legacy aliases for tokenizers that look for unnamespaced keys.
    obs_dict = nested.get("observation", {})
    for k in ("image_primary", "image_wrist", "proprio", "timestep", "task_completed"):
        if k not in nested and k in obs_dict:
            nested[k] = obs_dict[k]

    # Ensure presence + mirror masks across observation/task namespaces.
    obs_dict = nested.setdefault("observation", {})
    pad_masks_top = nested.setdefault("pad_mask_dict", {})
    obs_pad_masks = obs_dict.setdefault("pad_mask_dict", {})
    task_dict = nested.setdefault("task", {})
    task_pad_masks = task_dict.setdefault("pad_mask_dict", {})

    pad_meta = meta.get(("observation", "timestep_pad_mask")) or meta.get(("timestep_pad_mask",))
    pad_val = nested.get("timestep_pad_mask")
    if pad_val is None:
        pad_val = obs_dict.get("timestep_pad_mask")
    if pad_val is None:
        if pad_meta is not None:
            target_shape, _ = pad_meta
            pad_val = np.ones(target_shape, dtype=np.bool_)
        else:
            pad_val = np.ones((1, history.window_size), dtype=np.bool_)
    else:
        pad_val = np.asarray(pad_val, dtype=np.bool_)
    nested["timestep_pad_mask"] = pad_val
    obs_dict["timestep_pad_mask"] = pad_val
    pad_masks_top["timestep"] = pad_val
    obs_pad_masks["timestep"] = pad_val
    task_pad_masks["timestep"] = pad_val

    mask_names_all = (
        set(pad_masks_top.keys())
        | set(obs_pad_masks.keys())
        | set(task_pad_masks.keys())
    )
    for key in meta:
        if len(key) == 3 and key[0] == "observation" and key[1] == "pad_mask_dict":
            mask_names_all.add(key[2])
        elif len(key) == 2 and key[0] == "pad_mask_dict":
            mask_names_all.add(key[1])
    mask_names_all.update(mask_names)
    mask_names_all.add("timestep")

    for name in mask_names_all:
        if name == "timestep":
            src = pad_val
        else:
            src = (
                pad_masks_top.get(name)
                or obs_pad_masks.get(name)
                or task_pad_masks.get(name)
            )
            if src is None:
                shape_info = (
                    meta.get(("observation", "pad_mask_dict", name))
                    or meta.get(("pad_mask_dict", name))
                )
                if shape_info is not None:
                    target_shape, _ = shape_info
                    src = np.ones(target_shape, dtype=np.bool_)
                else:
                    src = np.ones((1, history.window_size), dtype=np.bool_)
            else:
                src = np.asarray(src, dtype=np.bool_)
        pad_masks_top[name] = src
        obs_pad_masks[name] = src
        task_pad_masks[name] = src
    # Mirror example_batch structure: ensure every declared key exists
    for key, (shape, dtype) in meta.items():
        slot = nested
        for part in key[:-1]:
            slot = slot.setdefault(part, {})
        leaf_key = key[-1]
        if leaf_key not in slot:
            slot[leaf_key] = np.zeros(shape, dtype=dtype)

    return nested

def expected_shapes_for(meta: dict):
    """Infer expected (H, W) for primary and wrist images from metadata."""

    shapes = {}
    for key, name in (
        (("observation","image_primary"), "image_primary"),
        (("observation","image_wrist"), "image_wrist"),
    ):
        info = meta.get(key)
        if info is not None:
            shape, _ = info
            if len(shape) >= 4:
                shapes[name] = (int(shape[2]), int(shape[3]))

    shapes.setdefault("image_primary", (256, 256))
    shapes.setdefault("image_wrist", (128, 128))
    return shapes


def _np_shape(value: Any) -> Optional[tuple]:
    try:
        return tuple(np.asarray(value).shape)
    except Exception:
        return None


def _np_array(value: Any) -> Optional[np.ndarray]:
    try:
        return np.asarray(value)
    except Exception:
        return None


def _debug_print_language(task: Optional[Mapping[str, Any]]) -> None:
    lang = None
    if isinstance(task, Mapping):
        lang = task.get("language_instruction")
    ids = None
    mask = None
    if isinstance(lang, Mapping):
        ids = lang.get("input_ids")
        mask = lang.get("attention_mask")
    ids_arr = _np_array(ids)
    mask_arr = _np_array(mask)
    ids_present = ids_arr is not None
    mask_present = mask_arr is not None
    shape_ids = _np_shape(ids_arr) if ids_present else None
    shape_mask = _np_shape(mask_arr) if mask_present else None
    preview: list[int] = []
    if ids_present:
        try:
            flat = ids_arr.reshape(-1)
            preview = [int(x) for x in flat[:8]]
        except Exception:
            preview = []
    print(
        "[LANG] present ids="
        f"{ids_present} mask={mask_present} "
        f"shape_ids={shape_ids} shape_mask={shape_mask} preview={preview}"
    )


def _debug_print_vision(
    observations: Mapping[str, Any],
    print_norms: bool = False,
) -> None:
    for key, label in (("image_primary", "primary"), ("image_wrist", "wrist")):
        arr = observations.get(key) if isinstance(observations, Mapping) else None
        if arr is None:
            print(f"[VISION] {label}: present=False")
            continue
        arr_np = _np_array(arr)
        shape = _np_shape(arr_np)
        line = f"[VISION] {label}: shape={shape}"
        if print_norms and arr_np is not None:
            try:
                stats_arr = arr_np.astype(np.float32)
                mean = float(stats_arr.mean())
                std = float(stats_arr.std())
                line += f" mean={mean:.4f} std={std:.4f}"
            except Exception:
                pass
        print(line)


def _debug_print_modalities(
    observations: Mapping[str, Any],
    task: Optional[Mapping[str, Any]] = None,
) -> None:
    proprio = observations.get("proprio") if isinstance(observations, Mapping) else None
    proprio_arr = _np_array(proprio)
    proprio_present = proprio_arr is not None
    proprio_shape = _np_shape(proprio_arr)
    if proprio_arr is not None and proprio_arr.size:
        try:
            if proprio_arr.ndim >= 3:
                step_slice = proprio_arr[:, -1, :]
            elif proprio_arr.ndim == 2:
                step_slice = proprio_arr
            else:
                step_slice = proprio_arr.reshape(proprio_arr.shape[0], -1)
            mins = np.round(step_slice.min(axis=0), 3).tolist()
            maxs = np.round(step_slice.max(axis=0), 3).tolist()
        except Exception:
            mins = []
            maxs = []
    else:
        mins = []
        maxs = []
    print(f"[PROP] present={proprio_present} shape={proprio_shape} min={mins} max={maxs}")

    pad_mask_dict = observations.get("pad_mask_dict") if isinstance(observations, Mapping) else None
    if isinstance(pad_mask_dict, Mapping):
        for name, mask in sorted(pad_mask_dict.items()):
            mask_arr = _np_array(mask)
            shape = _np_shape(mask_arr)
            frac = None
            if mask_arr is not None and mask_arr.size:
                try:
                    frac = float(mask_arr.astype(np.float32).mean())
                except Exception:
                    frac = None
            line = f"[MASK] {name}: shape={shape}"
            if frac is not None:
                line += f" frac_ones={frac:.2f}"
                if frac <= 0.0:
                    line += " (all zeros!)"
            print(line)
    timestep_mask = observations.get("timestep_pad_mask") if isinstance(observations, Mapping) else None
    ts_arr = _np_array(timestep_mask)
    if ts_arr is not None:
        frac = None
        try:
            frac = float(ts_arr.astype(np.float32).mean())
        except Exception:
            frac = None
        line = f"[MASK] timestep_pad_mask: shape={_np_shape(ts_arr)}"
        if frac is not None:
            line += f" frac_ones={frac:.2f}"
        print(line)

    if isinstance(task, Mapping):
        task_masks = task.get("pad_mask_dict")
        if isinstance(task_masks, Mapping):
            for name, mask in sorted(task_masks.items()):
                mask_arr = _np_array(mask)
                shape = _np_shape(mask_arr)
                frac = None
                if mask_arr is not None and mask_arr.size:
                    try:
                        frac = float(mask_arr.astype(np.float32).mean())
                    except Exception:
                        frac = None
                line = f"[MASK] task/{name}: shape={shape}"
                if frac is not None:
                    line += f" frac_ones={frac:.2f}"
                    if frac <= 0.0:
                        line += " (all zeros!)"
                print(line)


def _ee_position(env: PandaSimEnv):
    kind, name = env.ee_ref
    if kind == "site":
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return env.data.site_xpos[sid].copy()
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        return env.data.xpos[bid].copy()
    return env.data.site_xpos[0].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["model", "dataset"], default="model",
                    help="Run Octo model inference or replay a logged dataset")
    ap.add_argument("--exp", help="Octo experiment directory OR a parent containing experiment_*/config.json (model mode)")
    ap.add_argument("--ckpt_file", type=str, default=None,
                    help="Path to Flax train_state checkpoint (e.g., tiny_overfit checkpoint_XXXXX). If set, overrides params loaded from --exp.")
    ap.add_argument("--per_exp_steps", type=int, default=None, help="If multiple experiments are found under --exp, run this many steps per experiment (defaults to --max_steps)")
    ap.add_argument("--model_xml", default=os.environ.get("MODEL_XML", ""), help="Path to base MuJoCo scene xml (required for dynamic plant builder)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_language", action="store_true", default=False)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--substeps", type=int, default=80,
                    help="Inner MuJoCo substeps per control tick (matches oracle dataset default=80).")
    ap.add_argument("--kp", type=float, default=100.0,
                    help="Joint-space PD proportional gain (matches oracle dataset default=100).")
    ap.add_argument("--action_scale", type=float, default=10)
    ap.add_argument("--task_mode", choices=["language", "goal", "language_goal"], default="language",
                help="Control conditioning: text only, goal imagery only, or both simultaneously.")
    ap.add_argument("--language_prompt", default="Pick the tomato.",
                    help="Language instruction to feed the policy when --task_mode=language")
    ap.add_argument("--goal_image_primary", default=None,
                    help="Path to a primary goal image (goal mode).")
    ap.add_argument("--goal_image_wrist", default=None,
                    help="Path to a wrist goal image (goal mode).")
    ap.add_argument("--goal_dataset_name", default=None,
                    help="TFDS dataset name containing goal images (goal mode).")
    ap.add_argument("--goal_dataset_dir", default=None,
                    help="TFDS data_dir for the goal-image dataset.")
    ap.add_argument("--goal_dataset_split", default="train",
                    help="TFDS split to use when loading goal images (goal mode).")
    ap.add_argument("--goal_dataset_episode", type=int, default=0,
                    help="Episode index to draw goal images from when using --goal_dataset_name.")
    ap.add_argument("--debug_constant_action", type=float, default=None,
                help="If set, ignore policy and apply a constant delta on joint 1 (radians).")
    ap.add_argument("--mc_dropout_samples", type=int, default=1,
                    help="Number of stochastic policy evaluations with dropout per control step (>=1).")
    ap.add_argument("--strict_eval_hygiene", type=int, choices=[0, 1], default=1,
                    help="Enforce eval-only behavior (disables dropout, freezes adapters).")
    ap.add_argument("--no_amp", type=int, choices=[0, 1], default=0,
                    help="Disable AMP/autocast paths inside the runner when supported.")
    ap.add_argument("--history", type=int, default=None,
                    help="Optional runtime history override; only used for spec cross-checks.")
    ap.add_argument("--action_horizon", type=int, default=None,
                    help="Optional runtime action horizon override; only used for spec cross-checks.")
    ap.add_argument("--record_video", action="store_true",
                    help="Save a side-by-side (primary|wrist) video of each rollout (model mode only).")
    ap.add_argument("--video_dir", default=os.path.join("octo", "outputs", "inference_videos"),
                    help="Directory where recorded videos will be stored (implies --record_video when non-empty).")
    ap.add_argument("--video_fps", type=float, default=None,
                    help="FPS for the saved video (defaults to --fps).")
    ap.add_argument("--debug_trace_dir", default=None,
                    help="If set, write JSONL traces of observations/actions to this directory.")
    ap.add_argument("--debug_trace_every", type=int, default=1,
                    help="Record every Nth control step in the debug trace (requires --debug_trace_dir).")
    ap.add_argument("--debug_norm_probe", action="store_true",
                    help="Print first 20 timesteps of RAW and SCALED action norms and a brief diagnosis; dump input shapes if both tiny.")
    ap.add_argument("--debug_print_lang", type=int, choices=[0, 1], default=0,
                    help="When set to 1, print language token presence, shapes, and first 8 IDs at step 0.")
    ap.add_argument("--print_image_norms", type=int, choices=[0, 1], default=0,
                    help="When set to 1, print per-stream image shapes and post-transform mean/std at step 0.")
    ap.add_argument("--debug_modality_presence", type=int, choices=[0, 1], default=0,
                    help="When set to 1, print proprio/vision presence plus modality masks (warn if all zeros) at step 0.")
    ap.add_argument("--force_task_masks", type=int, choices=[0, 1], default=1,
                    help="When 1, overwrite task modality masks to ones for active modalities before inference.")
    ap.add_argument("--assert_nonzero_inputs", type=int, choices=[0, 1], default=1,
                    help="When 1, assert that post-transform vision/proprio inputs are non-zero; when 0, warn once.")
    ap.add_argument("--debug_minmax_inputs", type=int, choices=[0, 1], default=0,
                    help="When 1, print step-0 min/max stats for post-transform vision and proprio inputs.")
    ap.add_argument("--dataset_name", help="TFDS builder name for dataset replay (dataset mode)")
    ap.add_argument("--dataset_split", default="train", help="TFDS split to replay (dataset mode)")
    ap.add_argument("--dataset_dir", help="TFDS data_dir containing the dataset (dataset mode)")
    ap.add_argument("--dataset_episodes", type=int, default=5, help="Number of dataset episodes to replay (dataset mode)")
    ap.add_argument("--dataset_action_scale", type=float, default=0.05, help="Action scale used when the dataset was recorded (dataset mode)")
    ap.add_argument("--dataset_shuffle", action="store_true", help="Shuffle dataset episodes before replay")
    ap.add_argument("--dataset_control", choices=["pd", "direct"], default="direct",
                    help="Replay using PD action tracking or direct proprio injection")
    ap.add_argument("--headless", action="store_true",
                    help="Disable viewer/rendering (forces zero images, no GUI).")
    ap.add_argument("--debug_ckpt_inspect", action="store_true",
                    help="Print deep structure of restored checkpoint/params.")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    global DEBUG_CKPT_INSPECT
    DEBUG_CKPT_INSPECT = bool(args.debug_ckpt_inspect)

    args.strict_eval_hygiene = bool(int(args.strict_eval_hygiene))
    args.no_amp = bool(int(args.no_amp))
    args.force_task_masks = bool(int(args.force_task_masks))
    args.assert_nonzero_inputs = bool(int(args.assert_nonzero_inputs))
    args.debug_minmax_inputs = bool(int(args.debug_minmax_inputs))

    if args.ckpt_file and args.mode != "model":
        raise SystemExit("--ckpt_file is only supported in model mode")
    if args.ckpt_file:
        args.ckpt_file = os.path.expanduser(args.ckpt_file)

    trace_logger = TraceLogger(args.debug_trace_dir, every=args.debug_trace_every)

    if args.mc_dropout_samples < 1:
        raise SystemExit("--mc_dropout_samples must be >= 1")
    if args.strict_eval_hygiene and args.mc_dropout_samples > 1:
        raise SystemExit(
            "Strict eval hygiene requires deterministic inference; set --mc_dropout_samples=1."
        )
    if args.video_fps is not None and args.video_fps <= 0:
        raise SystemExit("--video_fps must be positive")

    headless = bool(args.headless)

    record_video = bool(args.record_video and not headless)
    if record_video and args.mode != "model":
        print("[video] Recording is only supported in model mode; ignoring request.")
        record_video = False
    if args.record_video and headless:
        print("[video] Disabled because --headless was requested.")
    video_dir = os.path.abspath(args.video_dir) if record_video else None
    video_fps = float(args.video_fps if args.video_fps is not None else args.fps)

    if args.mode == "model":
        if not args.exp:
            raise SystemExit("--exp is required in model mode")
        if "://" in args.exp:
            exp_dirs = [args.exp]
            print(f"[resolve] Using remote/pretrained identifier: {args.exp}")
        else:
            exp_dirs = resolve_exp_dirs(args.exp)
        steps_per = int(args.per_exp_steps or args.max_steps)
    else:
        if not args.dataset_name or not args.dataset_dir:
            raise SystemExit("--dataset_name and --dataset_dir are required in dataset mode")
        exp_dirs = []
        steps_per = None

    # Always build dynamic plant scene
    if _build_and_load_scene is None:
        raise SystemExit(
            "Dynamic scene builder not found. Ensure octo/record_dataset/helpers.py is importable."
        )
    if not args.model_xml:
        raise SystemExit("--model_xml must point to your base scene (e.g. scene.xml)")

    print("[scene] Regenerating dynamic tomato plant scene")
    sim_model, sim_data = _build_and_load_scene(args.model_xml)

    # --- NEW: build mapping/env after model+data exist (for BOTH branches) ---
    arm_act_ids, arm_qpos_addr, arm_gain_type = build_arm_mapping_from_model(
        sim_model, prefer_position=True
    )
    arm_dof_idx = build_arm_dof_indices(sim_model, arm_act_ids)
    gripper_idx = find_gripper_actuator(sim_model)
    ee_ref = auto_ee_ref(sim_model, sim_data)

    goal_target = None
    if args.mode == "model":
        try:
            targets = _find_top_targets(sim_model, sim_data, k=1, s=0.66)
            if targets:
                goal_target = targets[0]
                print(f"[target] Selected {goal_target[0]} at {goal_target[1]}")
        except Exception as exc:
            print(f"[target] Warning: unable to determine grasp target ({exc})")

    env_action_scale = args.action_scale if args.mode == "model" else args.dataset_action_scale

    env = PandaSimEnv(
        sim_model, sim_data,
        arm_act_ids, arm_qpos_addr, ee_ref,
        substeps=args.substeps, gripper_idx=gripper_idx, arm_dof_idx=arm_dof_idx,
        kp=args.kp, kd=None, action_scale=env_action_scale,
        arm_gain_type=arm_gain_type,
        capture_images=not headless,
    )

    # (optional: tiny debug to confirm position-mode detection)
    print(f"[actuators] position_mode={getattr(env, '_all_position_act', False)} "
        f"gain_types={arm_gain_type.tolist()}")


    period = 1.0 / max(1e-6, float(args.fps))
    video_run_root = None
    video_counter = 0
    if record_video:
        video_run_root = os.path.join(video_dir, f"local_{int(time.time())}")
        os.makedirs(video_run_root, exist_ok=True)
        print(f"[video] Recording enabled → {video_run_root}")

    class _HeadlessViewer:
        def is_running(self):
            return True
        def sync(self):
            pass
        def close(self):
            pass

    viewer_ctx = viewer.launch_passive(sim_model, sim_data) if not headless else nullcontext(_HeadlessViewer())

    with viewer_ctx as v:
        if args.mode == "model":
            for i, exp_dir in enumerate(exp_dirs, 1):
                print(f"[load] ({i}/{len(exp_dirs)}) Octo from: {exp_dir}")
                model = _load_model_from_exp_dir(exp_dir, args.ckpt_file)
                history, action_horizon, action_dim = _extract_model_spec(model)
                print(f"[SPEC] history={history} horizon={action_horizon} action_dim={action_dim}")
                if args.history is not None and args.history != history:
                    print(
                        f"[SPEC][warn] runtime --history={args.history} "
                        f"does not match model history={history}"
                    )
                if args.action_horizon is not None and args.action_horizon != action_horizon:
                    print(
                        f"[SPEC][warn] runtime --action_horizon={args.action_horizon} "
                        f"does not match model action_horizon={action_horizon}"
                    )
                trace_logger.start_run(exp_dir)
                trace_logger.log_model_template(model)

                example_meta = build_example_meta(model.example_batch)
                img_shapes = expected_shapes_for(example_meta)
                history = ObservationHistory(example_meta)
                trace_logger.set_example_meta(example_meta)
                print(f"[shapes] primary={img_shapes['image_primary']} wrist={img_shapes['image_wrist']}")
                goal_pos = goal_target[1] if goal_target is not None else None
                gripper_cmd = GRIPPER_OPEN_CMD
                closing_started = False
                mc_samples = max(1, int(args.mc_dropout_samples))
                use_dropout = mc_samples > 1
                if use_dropout:
                    print(f"[mc-dropout] Using {mc_samples} stochastic forward passes per control step")

                dropout_modules = _count_dropout_hints(getattr(model, "config", {}))
                torch_objs = tuple(
                    obj
                    for obj in (
                        getattr(model, "torch_model", None),
                        getattr(model, "policy", None),
                        getattr(model, "module", None),
                        model,
                    )
                    if obj is not None
                )
                trainable_params = _summarize_torch_trainables(torch_objs, freeze=args.strict_eval_hygiene)
                amp_state = "off"

                action_stats = (
                    _find_action_statistics(model.dataset_statistics)
                    if model.dataset_statistics is not None
                    else None
                )
                if action_stats is None:
                    raise SystemExit(
                        "Loaded OctoModel is missing 'action' statistics required for unnormalization."
                    )
                trace_logger.log_action_statistics(action_stats)

                goal_primary_img = None
                goal_wrist_img = None
                goal_task = None
                base_task = None
                fallback_warned = False
                goal_language_text = None
                language_prompt = args.language_prompt

                use_goal = args.task_mode in ("goal", "language_goal")
                use_language = args.task_mode in ("language", "language_goal")

                if use_goal:
                    goal_source = None
                    if args.goal_image_primary or args.goal_image_wrist:
                        try:
                            if args.goal_image_primary:
                                goal_primary_img = _read_image_file(args.goal_image_primary)
                                goal_source = args.goal_image_primary
                            if args.goal_image_wrist:
                                goal_wrist_img = _read_image_file(args.goal_image_wrist)
                                goal_source = goal_source or args.goal_image_wrist
                            print(f"[goal] Loaded goal image(s) from file(s).")
                        except Exception as exc:
                            print(f"[goal] Failed to load goal image files: {exc}")
                            goal_primary_img = goal_wrist_img = None
                    elif args.goal_dataset_name and args.goal_dataset_dir:
                        try:
                            goal_primary_img, goal_wrist_img, goal_language_text = _load_goal_images_from_dataset_episode(
                                args.goal_dataset_name,
                                args.goal_dataset_dir,
                                args.goal_dataset_split,
                                int(args.goal_dataset_episode),
                            )
                            goal_source = f"{args.goal_dataset_name}/{args.goal_dataset_split}@{args.goal_dataset_episode}"
                            print(f"[goal] Loaded goal image(s) from dataset episode {goal_source}.")
                        except Exception as exc:
                            print(f"[goal] Failed to load goal images from dataset: {exc}")
                            goal_primary_img = goal_wrist_img = None
                            goal_language_text = None

                    if goal_language_text is not None:
                        language_prompt = goal_language_text

                    if goal_primary_img is not None or goal_wrist_img is not None:
                        goal_payload, missing_keys = _prepare_goal_payload(
                            model, goal_primary_img, goal_wrist_img
                        )
                        if missing_keys:
                            print(
                                "[goal] Warning: missing imagery for keys "
                                + ", ".join(missing_keys)
                                + "; filling zeros."
                            )
                        if goal_language_text is not None:
                            language_prompt = goal_language_text
                            print(f"[goal] Using language prompt from dataset: '{language_prompt}'")
                        if use_language:
                            goal_task = model.create_tasks(goals=goal_payload, texts=[language_prompt])
                        else:
                            goal_task = model.create_tasks(goals=goal_payload)
                    else:
                        print(
                            "[goal] Warning: no explicit goal image provided; "
                            "falling back to using the first observation frame."
                        )

                if use_language:
                    effective_prompt = language_prompt
                    base_task = model.create_tasks(texts=[effective_prompt])
                if goal_task is not None:
                    trace_logger.log_task("task_base", goal_task)
                elif base_task is not None:
                    trace_logger.log_task("task_base", base_task)
                task_logged = False

                policy = supply_rng(
                    partial(
                        model.sample_actions,
                        unnormalization_statistics=action_stats,
                    ),
                    seed=args.seed,
                )
                ts = env.reset()
                env._last_gripper_cmd = gripper_cmd
                video_frames = []
                video_path = None
                if record_video:
                    video_counter += 1
                    exp_tag = os.path.basename(exp_dir.rstrip(os.sep)) or f"exp_{video_counter:03d}"
                    safe_tag = exp_tag.replace(os.sep, "_").replace(" ", "_")
                    video_path = os.path.join(video_run_root, f"{video_counter:03d}_{safe_tag}.mp4")
                    video_frames.append(_compose_video_frame(ts.observation))
                if use_goal:
                    if goal_task is not None:
                        task = goal_task
                    else:
                        fallback_primary = ts.observation.get("goal_image_primary")
                        if fallback_primary is None:
                            fallback_primary = ts.observation.get("image_primary")
                        fallback_wrist = ts.observation.get("goal_image_wrist")
                        if fallback_wrist is None:
                            fallback_wrist = ts.observation.get("image_wrist")
                        if not fallback_warned:
                            print("[goal] Using initial observation frame as goal image fallback.")
                            fallback_warned = True
                        fallback_payload, _ = _prepare_goal_payload(
                            model, fallback_primary, fallback_wrist
                        )
                        if use_language:
                            fallback_task = model.create_tasks(goals=fallback_payload, texts=[language_prompt])
                        else:
                            fallback_task = model.create_tasks(goals=fallback_payload)
                        task = fallback_task
                else:
                    task = base_task
                if not task_logged:
                    trace_logger.log_task("task_active", task)
                    task_logged = True

                t_last = time.perf_counter()
                steps_executed = 0
                scaled_norms = []
                debug_norm_probe_enabled = bool(args.debug_norm_probe)
                debug_norms_raw: list[float] = []
                debug_norms_scaled: list[float] = []
                debug_last_inputs: Optional[Mapping[str, Any]] = None
                input_warning_emitted = False
                dropout_note = "all.training=False" if not use_dropout else "training=True"
                no_grad_ctx = torch.no_grad() if torch is not None else nullcontext()
                with no_grad_ctx:
                    no_grad_state = (torch is None) or (not torch.is_grad_enabled())
                    print(
                        f"[EVAL] eval=True no_grad={no_grad_state} amp={amp_state} "
                        f"trainable_params={trainable_params} dropout_modules={dropout_modules} "
                        f"{dropout_note}"
                    )
                    for step in range(steps_per):
                        if not v.is_running():
                            break
                        prev_ts = ts
                        obs_for_octo = build_obs_for_octo(
                            prev_ts.observation,
                            t_idx=step,
                            meta=example_meta,
                            history=history,
                            img_shapes=img_shapes,
                        )
                        if step == 0:
                            if args.debug_print_lang:
                                _debug_print_language(task)
                            if args.print_image_norms:
                                _debug_print_vision(obs_for_octo, print_norms=True)
                            if args.debug_modality_presence:
                                _debug_print_modalities(obs_for_octo, task)
                        obs_dict = obs_for_octo.get("observation") if isinstance(obs_for_octo, Mapping) else None
                        pad_dict_obs = obs_for_octo.get("pad_mask_dict") if isinstance(obs_for_octo, Mapping) else None

                        batch_size = 1
                        if isinstance(obs_dict, Mapping):
                            for candidate in ("proprio", "image_primary", "timestep"):
                                candidate_arr = obs_dict.get(candidate)
                                if candidate_arr is None:
                                    continue
                                try:
                                    batch_size = int(np.asarray(candidate_arr).shape[0])
                                    break
                                except Exception:
                                    continue

                        def _latest_step(arr):
                            if arr is None:
                                return None
                            try:
                                arr_np = np.asarray(arr)
                            except Exception:
                                return None
                            if arr_np.size == 0:
                                return None
                            if arr_np.ndim >= 2:
                                arr_np = arr_np[:, -1]
                            return arr_np

                        primary_last = _latest_step(obs_dict.get("image_primary") if isinstance(obs_dict, Mapping) else None)
                        wrist_last = _latest_step(obs_dict.get("image_wrist") if isinstance(obs_dict, Mapping) else None)
                        proprio_last = _latest_step(obs_dict.get("proprio") if isinstance(obs_dict, Mapping) else None)

                        zero_tol = 1e-6
                        issues = []
                        primary_mean = primary_std = primary_min = primary_max = None
                        if primary_last is not None:
                            primary_stats = primary_last.astype(np.float32)
                            primary_mean = float(primary_stats.mean())
                            primary_std = float(primary_stats.std())
                            primary_min = float(primary_stats.min())
                            primary_max = float(primary_stats.max())
                            if primary_std <= zero_tol:
                                issues.append(f"image_primary post-transform std={primary_std:.6e} (step {step})")

                        wrist_mean = wrist_std = wrist_min = wrist_max = None
                        if wrist_last is not None:
                            wrist_stats = wrist_last.astype(np.float32)
                            wrist_mean = float(wrist_stats.mean())
                            wrist_std = float(wrist_stats.std())
                            wrist_min = float(wrist_stats.min())
                            wrist_max = float(wrist_stats.max())
                            if wrist_std <= zero_tol:
                                issues.append(f"image_wrist post-transform std={wrist_std:.6e} (step {step})")

                        proprio_min = proprio_max = None
                        if proprio_last is not None:
                            proprio_stats = proprio_last.astype(np.float32)
                            proprio_min = float(proprio_stats.min())
                            proprio_max = float(proprio_stats.max())
                            if np.all(np.abs(proprio_stats) <= zero_tol):
                                issues.append(f"proprio post-transform values near zero (step {step})")

                        if issues:
                            if args.assert_nonzero_inputs:
                                raise AssertionError("[inputs] " + "; ".join(issues))
                            if not input_warning_emitted:
                                for msg in issues:
                                    print(f"[warn][inputs] {msg}")
                                input_warning_emitted = True

                        mask_targets: list[str] = []
                        if primary_last is not None:
                            mask_targets.append("image_primary")
                        if wrist_last is not None:
                            mask_targets.append("image_wrist")
                        mask_targets.extend(["proprio", "timestep"])

                        if args.force_task_masks and isinstance(task, Mapping):
                            if not isinstance(task, dict):
                                task = dict(task)
                            pad_mask_dict = task.get("pad_mask_dict")
                            if not isinstance(pad_mask_dict, dict):
                                pad_mask_dict = dict(pad_mask_dict) if isinstance(pad_mask_dict, Mapping) else {}
                            for name in mask_targets:
                                existing_mask = pad_mask_dict.get(name)
                                if existing_mask is not None:
                                    existing_arr = np.asarray(existing_mask)
                                    ones = np.ones_like(existing_arr, dtype=existing_arr.dtype if existing_arr.dtype != object else bool)
                                else:
                                    fallback_mask = pad_dict_obs.get(name) if isinstance(pad_dict_obs, Mapping) else None
                                    if fallback_mask is not None:
                                        fallback_arr = np.asarray(fallback_mask)
                                        ones = np.ones_like(fallback_arr, dtype=fallback_arr.dtype if fallback_arr.dtype != object else bool)
                                    else:
                                        ones = np.ones((batch_size,), dtype=bool)
                                pad_mask_dict[name] = ones
                            task["pad_mask_dict"] = pad_mask_dict

                        def _mask_flag(value):
                            if value is None:
                                return "N/A"
                            try:
                                arr = np.asarray(value)
                            except Exception:
                                return "N/A"
                            if arr.size == 0:
                                return "0"
                            return "1" if np.all(arr != 0) else "0"

                        task_pad_masks = task.get("pad_mask_dict") if isinstance(task, Mapping) else None
                        primary_mask_flag = _mask_flag(task_pad_masks.get("image_primary") if isinstance(task_pad_masks, Mapping) else None)
                        wrist_mask_flag = _mask_flag(task_pad_masks.get("image_wrist") if isinstance(task_pad_masks, Mapping) else None)
                        proprio_mask_flag = _mask_flag(task_pad_masks.get("proprio") if isinstance(task_pad_masks, Mapping) else None)
                        timestep_mask_flag = _mask_flag(task_pad_masks.get("timestep") if isinstance(task_pad_masks, Mapping) else None)
                        language_mask_val = None
                        if isinstance(task_pad_masks, Mapping):
                            language_mask_val = task_pad_masks.get("language_instruction")
                        if language_mask_val is None and isinstance(task, Mapping):
                            lang_entry = task.get("language_instruction")
                            if isinstance(lang_entry, Mapping):
                                language_mask_val = lang_entry.get("attention_mask")
                        language_mask_flag = _mask_flag(language_mask_val)

                        def _format_mean_std(mean, std):
                            if mean is None or std is None:
                                return "N/A"
                            return f"<{mean:.4f},{std:.4f}>"

                        def _format_min_max(val_min, val_max):
                            if val_min is None or val_max is None:
                                return "N/A"
                            return f"<{val_min:.4f},{val_max:.4f}>"

                        if step == 0 and args.debug_minmax_inputs:
                            print(
                                "[INPUTS][primary]"
                                f" mean={primary_mean:.4f} std={primary_std:.4f} min={primary_min:.4f} max={primary_max:.4f}"
                                if primary_mean is not None
                                else "[INPUTS][primary] N/A"
                            )
                            print(
                                "[INPUTS][wrist]"
                                f" mean={wrist_mean:.4f} std={wrist_std:.4f} min={wrist_min:.4f} max={wrist_max:.4f}"
                                if wrist_mean is not None
                                else "[INPUTS][wrist] N/A"
                            )
                            print(
                                "[INPUTS][proprio]"
                                f" min={proprio_min:.4f} max={proprio_max:.4f}"
                                if proprio_min is not None
                                else "[INPUTS][proprio] N/A"
                            )

                        any_debug_flag = (
                            bool(args.debug_minmax_inputs)
                            or bool(args.debug_modality_presence)
                            or bool(args.print_image_norms)
                            or bool(args.debug_print_lang)
                            or bool(args.debug_norm_probe)
                        )
                        if step == 0 and any_debug_flag:
                            print(
                                "[INPUTS] primary(mean,std)="
                                f"{_format_mean_std(primary_mean, primary_std)} "
                                "wrist(mean,std)="
                                f"{_format_mean_std(wrist_mean, wrist_std)} "
                                "proprio[min,max]="
                                f"{_format_min_max(proprio_min, proprio_max)}"
                            )
                            print(
                                "[TASKMASK] primary="
                                f"{primary_mask_flag} wrist={wrist_mask_flag} "
                                f"proprio={proprio_mask_flag} timestep={timestep_mask_flag} "
                                f"language={language_mask_flag}"
                            )
                        if goal_pos is not None and not closing_started:
                            ee_pos = _ee_position(env)
                            dist = float(np.linalg.norm(goal_pos - ee_pos))
                            if dist <= GRIPPER_CLOSE_DIST:
                                closing_started = True
                        target_grip = GRIPPER_CLOSE_CMD if closing_started else GRIPPER_OPEN_CMD
                        if target_grip < gripper_cmd:
                            gripper_cmd = max(target_grip, gripper_cmd - GRIPPER_RAMP_STEP)
                        else:
                            gripper_cmd = min(target_grip, gripper_cmd + GRIPPER_RAMP_STEP)
                        env._last_gripper_cmd = gripper_cmd

                        policy_mode = "policy"
                        act_tree = None
                        mc_samples_arr = None
                        if args.debug_constant_action is not None:
                            a = np.zeros((7,), np.float32)
                            a[0] = float(args.debug_constant_action)
                            env._last_action_std = None
                            policy_mode = "constant"
                        elif mc_samples == 1:
                            act_tree = policy(obs_for_octo, task, train=use_dropout)
                            a = _action_from_tree(act_tree)
                            env._last_action_std = None
                            policy_mode = "policy_train" if use_dropout else "policy_eval"
                        else:
                            samples = [
                                _action_from_tree(policy(obs_for_octo, task, train=True))
                                for _ in range(mc_samples)
                            ]
                            stacked = np.stack(samples, axis=0)
                            a = stacked.mean(axis=0).astype(np.float32)
                            env._last_action_std = stacked.std(axis=0).astype(np.float32)
                            mc_samples_arr = stacked
                            policy_mode = "mc_dropout"

                        trace_logger.log_step(
                            step_idx=step,
                            timestep=prev_ts,
                            obs_for_model=obs_for_octo,
                            action_tree=act_tree,
                            action_vector=a,
                            gripper_cmd=gripper_cmd,
                            history=history,
                            mc_samples=mc_samples_arr,
                            policy_mode=policy_mode,
                        )

                        arm_addr = env.arm_qpos_addr[:7]
                        q_prev = env.data.qpos[arm_addr].copy()
                        commanded_delta = env.action_scale * np.asarray(a[:7], dtype=np.float32)
                        if debug_norm_probe_enabled:
                            raw_norms = _vec_norms(a)
                            scaled_norms_step = _vec_norms(commanded_delta)
                            debug_norms_raw.append(raw_norms[0] if raw_norms else 0.0)
                            debug_norms_scaled.append(scaled_norms_step[0] if scaled_norms_step else 0.0)
                            debug_last_inputs = obs_for_octo
                        scaled_norms.append(float(np.linalg.norm(commanded_delta)))
                        ts = env.step(a)
                        q_post = env.data.qpos[arm_addr].copy()
                        actual_delta = q_post - q_prev
                        pd_target = getattr(env, "_last_q_target", None)
                        _log_joint_motion_event(
                            trace_logger,
                            "model",
                            step,
                            commanded=commanded_delta,
                            actual=actual_delta,
                            q_before=q_prev,
                            q_after=q_post,
                            raw_action=a,
                            misc={
                                "exp_dir": exp_dir,
                                "policy_mode": policy_mode,
                                "action_scale": env.action_scale,
                                "pd_target": pd_target,
                            },
                        )
                        steps_executed += 1
                        if record_video:
                            video_frames.append(_compose_video_frame(ts.observation))
                        now = time.perf_counter()
                        dt = now - t_last
                        if dt < period:
                            time.sleep(max(0.0, period - dt))
                        t_last = now
                        v.sync()
                if debug_norm_probe_enabled:
                    n = min(20, len(debug_norms_raw))
                    first20_raw = debug_norms_raw[:n]
                    first20_scaled = debug_norms_scaled[:n]
                    print("\n[debug_norm_probe] First 20 timesteps (first-horizon-step norms):")
                    print(" step |  raw_norm  |  scaled_norm ")
                    if n == 0:
                        print("  (no steps captured)")
                    for i in range(n):
                        print(f"{i:5d} | {first20_raw[i]:9.4f} | {first20_scaled[i]:11.4f}")
                    med_raw = _median(first20_raw)
                    med_scaled = _median(first20_scaled)
                    print(f"\n[debug_norm_probe] median_raw={med_raw:.4f}  median_scaled={med_scaled:.4f}")
                    if med_raw >= 0.5 and med_scaled < 0.1:
                        print("[diagnosis] SCALE_MISMATCH: raw looks healthy but scaled is tiny. Check --action_scale and remove any extra divide.")
                    elif med_raw < 0.1 and med_scaled < 0.1:
                        print("[diagnosis] BOTH_TINY: inputs or cadence likely wrong. Dumping inference input shapes and spec...")
                        if debug_last_inputs is not None:
                            try:
                                def _shape_of(value: Any) -> tuple:
                                    if hasattr(value, "shape"):
                                        try:
                                            return tuple(value.shape)
                                        except Exception:
                                            pass
                                    try:
                                        arr = np.asarray(value)
                                        return tuple(arr.shape)
                                    except Exception:
                                        return ()

                                probe_dict: dict[str, Any] = {}
                                for key, value in debug_last_inputs.items():
                                    if isinstance(value, Mapping):
                                        probe_dict[key] = {
                                            sub_key: _shape_of(sub_val)
                                            for sub_key, sub_val in value.items()
                                        }
                                    else:
                                        probe_dict[key] = _shape_of(value)
                                print("[inputs] keys & shapes:", probe_dict)
                                hist = None
                                try:
                                    hist = probe_dict["observation"]["image_primary"][1]
                                except Exception:
                                    pass
                                print(f"[inputs] inferred_history_window={hist} (expect 1)")
                                try:
                                    ip = probe_dict["observation"]["image_primary"]
                                    print(f"[inputs] image_primary shape={ip} (expect ...x256x256x3)")
                                except Exception:
                                    pass
                                try:
                                    iw = probe_dict["observation"]["image_wrist"]
                                    print(f"[inputs] image_wrist   shape={iw} (expect ...x128x128x3)")
                                except Exception:
                                    pass
                                has_lang = False
                                try:
                                    lang_dict = debug_last_inputs.get("task", {}).get("language_instruction", {})
                                    if isinstance(lang_dict, Mapping):
                                        has_lang = "input_ids" in lang_dict and "attention_mask" in lang_dict
                                except Exception:
                                    pass
                                print(f"[inputs] language_tokens_present={has_lang}")
                            except Exception as exc:
                                print("[inputs] failed to dump shapes:", exc)
                        else:
                            print("[diagnosis] No inference inputs captured for inspection.")
                    try:
                        if args.debug_trace_dir:
                            os.makedirs(args.debug_trace_dir, exist_ok=True)
                            outp = os.path.join(args.debug_trace_dir, "debug_norm_probe.txt")
                            with open(outp, "w", encoding="utf-8") as fp:
                                fp.write("step,raw_norm,scaled_norm\n")
                                for idx in range(n):
                                    fp.write(f"{idx},{first20_raw[idx]:.6f},{first20_scaled[idx]:.6f}\n")
                            print(f"[debug_norm_probe] wrote {outp}")
                    except Exception as exc:
                        print("[debug_norm_probe] file write skipped:", exc)
                if scaled_norms:
                    norms_arr = np.asarray(scaled_norms, dtype=np.float32)
                    mean_norm = float(norms_arr.mean())
                    median_norm = float(np.median(norms_arr))
                    near_zero_pct = float((norms_arr < 1e-3).mean() * 100.0)
                    if args.ckpt_file:
                        tag_source = os.path.basename(args.ckpt_file.rstrip(os.sep))
                    else:
                        tag_source = os.path.basename(exp_dir.rstrip(os.sep))
                    metric_tag = _sanitize_name(tag_source)
                    print(
                        f"[{metric_tag}] scaled_action_norms: "
                        f"mean={mean_norm:.3f}, median={median_norm:.3f}, near_zero_rate={near_zero_pct:.1f}%"
                    )
                else:
                    print("[metrics] No control steps executed; skipping action norm summary.")
                trace_logger.log_event("rollout_complete", {"steps": steps_executed, "exp_dir": exp_dir})
                if record_video and video_path:
                    _write_video(video_frames, video_fps, video_path)
                print(f"[done] steps={steps_executed} for {exp_dir}")
                trace_logger.close_run()
        else:
            ds_iter = load_rlds_dataset(
                args.dataset_name,
                data_dir=args.dataset_dir,
                split=args.dataset_split,
                shuffle=args.dataset_shuffle,
                seed=args.seed,
            )
            print(f"[dataset] Replaying '{args.dataset_name}' split '{args.dataset_split}' from {args.dataset_dir}")
            arm_addr = env.arm_qpos_addr[:7]
            dataset_run_started = False
            if trace_logger.enabled:
                run_name = f"dataset_{args.dataset_name}_{args.dataset_split}"
                trace_logger.start_run(run_name)
                trace_logger.log_event(
                    "dataset_replay_begin",
                    {
                        "dataset_name": args.dataset_name,
                        "dataset_split": args.dataset_split,
                        "episodes_requested": int(args.dataset_episodes),
                        "control_mode": args.dataset_control,
                        "action_scale": float(env.action_scale),
                    },
                )
                dataset_run_started = True
            episodes_replayed = 0
            for epi_idx, episode in enumerate(ds_iter, 1):
                if epi_idx > args.dataset_episodes:
                    break
                if not v.is_running():
                    break
                episode = _structured_to_dict(episode)
                try:
                    proprio, actions = _extract_episode_arrays(episode)
                except Exception as exc:
                    print(f"[dataset] Skipping episode {epi_idx}: {exc}")
                    if dataset_run_started:
                        trace_logger.log_event(
                            "dataset_episode_skip",
                            {"episode_index": int(epi_idx), "reason": str(exc)},
                        )
                    continue

                if dataset_run_started:
                    trace_logger.log_event(
                        "dataset_episode_start",
                        {
                            "episode_index": int(epi_idx),
                            "proprio_steps": int(proprio.shape[0]),
                            "action_steps": int(actions.shape[0]) if actions is not None else 0,
                        },
                    )

                start_q = proprio[0]
                ts = _reset_env_to_proprio(env, start_q)
                v.sync()

                if args.dataset_control == "direct":
                    traj_len = proprio.shape[0]
                    steps_executed = 0
                    t_last = time.perf_counter()
                    qdim = min(len(env.arm_qpos_addr), proprio.shape[1])
                    for step in range(min(traj_len, args.max_steps)):
                        if not v.is_running():
                            break
                        q_prev = env.data.qpos[arm_addr].copy()
                        q_target = np.asarray(proprio[step], dtype=np.float32)
                        env.data.qvel[:] = 0.0
                        env.data.ctrl[:] = 0.0
                        env.data.qpos[env.arm_qpos_addr[:qdim]] = q_target[:qdim]
                        mujoco.mj_forward(env.model, env.data)
                        q_post = env.data.qpos[arm_addr].copy()
                        commanded_delta = q_target[: arm_addr.shape[0]] - q_prev
                        actual_delta = q_post - q_prev
                        _log_joint_motion_event(
                            trace_logger,
                            "dataset_direct",
                            step,
                            commanded=commanded_delta,
                            actual=actual_delta,
                            q_before=q_prev,
                            q_after=q_post,
                            misc={
                                "episode_index": epi_idx,
                                "action_scale": float(env.action_scale),
                            },
                        )
                        env._t = step
                        steps_executed += 1
                        now = time.perf_counter()
                        dt = now - t_last
                        if dt < period:
                            time.sleep(max(0.0, period - dt))
                        t_last = now
                        v.sync()
                    print(f"[dataset] Episode {epi_idx} replayed with {steps_executed} steps (direct)")
                else:
                    max_steps = min(actions.shape[0], args.max_steps)
                    warn_extra = True
                    steps_executed = 0
                    t_last = time.perf_counter()
                    for step in range(max_steps):
                        if not v.is_running():
                            break
                        q_prev = env.data.qpos[arm_addr].copy()
                        a = np.asarray(actions[step]).reshape(-1)
                        if a.size < 7:
                            print(f"[dataset] Episode {epi_idx}: action has {a.size} dims (<7); stopping episode")
                            if dataset_run_started:
                                trace_logger.log_event(
                                    "dataset_episode_truncate",
                                    {
                                        "episode_index": int(epi_idx),
                                        "reason": "short_action",
                                        "action_dim": int(a.size),
                                        "step_index": int(step),
                                    },
                                )
                            break
                        if a.size > 7 and warn_extra:
                            print(f"[dataset] Episode {epi_idx}: ignoring extra action dims ({a.size} → 7)")
                            warn_extra = False
                        a = a[:7].astype(np.float32)
                        ts = env.step(a)
                        q_post = env.data.qpos[arm_addr].copy()
                        commanded_delta = env.action_scale * a
                        actual_delta = q_post - q_prev
                        _log_joint_motion_event(
                            trace_logger,
                            "dataset_pd",
                            step,
                            commanded=commanded_delta,
                            actual=actual_delta,
                            q_before=q_prev,
                            q_after=q_post,
                            raw_action=a,
                            misc={
                                "episode_index": epi_idx,
                                "action_scale": float(env.action_scale),
                            },
                        )
                        steps_executed += 1
                        now = time.perf_counter()
                        dt = now - t_last
                        if dt < period:
                            time.sleep(max(0.0, period - dt))
                        t_last = now
                        v.sync()
                    print(f"[dataset] Episode {epi_idx} replayed with {steps_executed} steps (pd)")
                if not v.is_running():
                    break
                if dataset_run_started:
                    trace_logger.log_event(
                        "dataset_episode_end",
                        {
                            "episode_index": int(epi_idx),
                            "steps_replayed": int(steps_executed),
                            "control_mode": args.dataset_control,
                        },
                    )
                episodes_replayed = epi_idx

            if dataset_run_started:
                trace_logger.log_event(
                    "dataset_replay_end",
                    {
                        "episodes_replayed": int(episodes_replayed),
                        "control_mode": args.dataset_control,
                    },
                )
                trace_logger.close_run()

    print("[all done]")


if __name__ == "__main__":
    main()

# Tiny-overfit sanity (expect non-zero image std, non-zero proprio, masks=1.00)
# python /home/myrtheiw/octo_ws/tools/verify_phase1.py \
#   --exp /home/myrtheiw/octo_ws/octo/outputs/tiny_overfit/tiny_overfit_20251024_143550 \
#   --num_steps 1 --assert_lang 1 --assert_vision 1 --assert_proprio 1 --print_image_norms 1

# Full fine-tune sanity
# python /home/myrtheiw/octo_ws/tools/verify_phase1.py \
#   --exp /home/myrtheiw/octo_ws/octo/outputs/octo_finetune/experiment_20251008_194217 \
#   --num_steps 1 --assert_lang 1 --assert_vision 1 --assert_proprio 1 --print_image_norms 1
