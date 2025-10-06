
# ======================== run_inference_sim.py ========================
#!/usr/bin/env python3
import os, argparse, time
import numpy as np
from collections.abc import Mapping
import dm_env
from dm_env import TimeStep

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax, jax.numpy as jnp
from collections import deque
from functools import partial
import mujoco
from mujoco import viewer

from octo.model.octo_model import OctoModel
from flax.traverse_util import flatten_dict, unflatten_dict

from sim_env import (
    PandaSimEnv, build_arm_mapping_from_model, build_arm_dof_indices,
    find_gripper_actuator, auto_ee_ref,
)

GRIPPER_OPEN_CMD = 1.0
GRIPPER_CLOSE_CMD = 0.0
GRIPPER_RAMP_STEP = 0.05
GRIPPER_CLOSE_DIST = 0.006

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
    """Cache (shape, dtype) metadata for the example batch."""

    meta = {}
    for path, value in flatten_dict(example_batch).items():
        if isinstance(value, dict):
            continue
        try:
            shape = tuple(value.shape)
            dtype = np.dtype(value.dtype)
        except AttributeError:
            arr = np.asarray(value)
            shape, dtype = arr.shape, arr.dtype
        meta[path] = (shape, dtype)
    return meta


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
            if key[:2] == ("observation", "timestep"):
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
        ("observation", "image_primary"): img_p,
        ("observation", "image_wrist"): img_w,
        ("observation", "proprio"): proprio,
        ("observation", "task_completed"): task_completed,
        ("observation", "timestep"): timestep,
    }

    timestep_mask_step = np.ones((1, 1), bool)
    entries[("observation", "timestep_pad_mask")] = timestep_mask_step

    mask_names = {
        key[2]
        for key in meta
        if len(key) == 3 and key[:2] == ("observation", "pad_mask_dict")
    }
    if not mask_names:
        mask_names = {"image_primary", "image_wrist", "proprio", "timestep"}

    for name in mask_names:
        mask_step = np.ones((1, 1), bool)
        entries[("observation", "pad_mask_dict", name)] = mask_step
        entries[("pad_mask_dict", name)] = mask_step

    entries[("timestep_pad_mask",)] = timestep_mask_step

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

    nested = unflatten_dict(flat_result)
    # ensure legacy aliases exist even if not saved in example batch
    if "timestep_pad_mask" not in nested:
        obs = nested.get("observation", {})
        if isinstance(obs, dict) and "timestep_pad_mask" in obs:
            nested["timestep_pad_mask"] = obs["timestep_pad_mask"]
    pad_dict = nested.get("pad_mask_dict")
    obs_pad = nested.get("observation", {}).get("pad_mask_dict")
    if pad_dict is None and isinstance(obs_pad, dict):
        nested["pad_mask_dict"] = obs_pad

    obs = nested.get("observation", {})
    for k in ("image_primary", "image_wrist", "proprio", "timestep", "task_completed"):
        if k not in nested and isinstance(obs, dict) and k in obs:
            nested[k] = obs[k]
    return nested

def expected_shapes_for(meta: dict):
    """Infer expected (H, W) for primary and wrist images from metadata."""

    shapes = {}
    for key, name in (
        (("observation", "image_primary"), "image_primary"),
        (("observation", "image_wrist"), "image_wrist"),
    ):
        info = meta.get(key)
        if info is not None:
            shape, _ = info
            if len(shape) >= 4:
                shapes[name] = (int(shape[2]), int(shape[3]))

    shapes.setdefault("image_primary", (256, 256))
    shapes.setdefault("image_wrist", (128, 128))
    return shapes


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
    ap.add_argument("--per_exp_steps", type=int, default=None, help="If multiple experiments are found under --exp, run this many steps per experiment (defaults to --max_steps)")
    ap.add_argument("--model_xml", default=os.environ.get("MODEL_XML", ""), help="Path to base MuJoCo scene xml (required for dynamic plant builder)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_language", action="store_true", default=False)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--substeps", type=int, default=20)
    ap.add_argument("--kp", type=float, default=120.0)
    ap.add_argument("--action_scale", type=float, default=0.05)
    ap.add_argument("--task_mode", choices=["language", "goal"], default="language",
                help="Use language prompt or a blank goal image task (matches many finetunes).")
    ap.add_argument("--language_prompt", default="Pick the tomato.",
                    help="Language instruction to feed the policy when --task_mode=language")
    ap.add_argument("--debug_constant_action", type=float, default=None,
                help="If set, ignore policy and apply a constant delta on joint 1 (radians).")
    ap.add_argument("--mc_dropout_samples", type=int, default=1,
                    help="Number of stochastic policy evaluations with dropout per control step (>=1).")
    ap.add_argument("--record_video", action="store_true",
                    help="Save a side-by-side (primary|wrist) video of each rollout (model mode only).")
    ap.add_argument("--video_dir", default=os.path.join("octo", "outputs", "inference_videos"),
                    help="Directory where recorded videos will be stored (implies --record_video when non-empty).")
    ap.add_argument("--video_fps", type=float, default=None,
                    help="FPS for the saved video (defaults to --fps).")
    ap.add_argument("--dataset_name", help="TFDS builder name for dataset replay (dataset mode)")
    ap.add_argument("--dataset_split", default="train", help="TFDS split to replay (dataset mode)")
    ap.add_argument("--dataset_dir", help="TFDS data_dir containing the dataset (dataset mode)")
    ap.add_argument("--dataset_episodes", type=int, default=5, help="Number of dataset episodes to replay (dataset mode)")
    ap.add_argument("--dataset_action_scale", type=float, default=0.05, help="Action scale used when the dataset was recorded (dataset mode)")
    ap.add_argument("--dataset_shuffle", action="store_true", help="Shuffle dataset episodes before replay")
    ap.add_argument("--dataset_control", choices=["pd", "direct"], default="direct",
                    help="Replay using PD action tracking or direct proprio injection")

    args = ap.parse_args()

    if args.mc_dropout_samples < 1:
        raise SystemExit("--mc_dropout_samples must be >= 1")
    if args.video_fps is not None and args.video_fps <= 0:
        raise SystemExit("--video_fps must be positive")

    record_video = bool(args.record_video)
    if record_video and args.mode != "model":
        print("[video] Recording is only supported in model mode; ignoring request.")
        record_video = False
    video_dir = os.path.abspath(args.video_dir) if record_video else None
    video_fps = float(args.video_fps if args.video_fps is not None else args.fps)

    if args.mode == "model":
        if not args.exp:
            raise SystemExit("--exp is required in model mode")
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

    with viewer.launch_passive(sim_model, sim_data) as v:
        if args.mode == "model":
            for i, exp_dir in enumerate(exp_dirs, 1):
                print(f"[load] ({i}/{len(exp_dirs)}) Octo from: {exp_dir}")
                model = OctoModel.load_pretrained(exp_dir)
                example_meta = build_example_meta(model.example_batch)
                img_shapes = expected_shapes_for(example_meta)
                history = ObservationHistory(example_meta)
                print(f"[shapes] primary={img_shapes['image_primary']} wrist={img_shapes['image_wrist']}")
                goal_pos = goal_target[1] if goal_target is not None else None
                gripper_cmd = GRIPPER_OPEN_CMD
                closing_started = False
                mc_samples = max(1, int(args.mc_dropout_samples))
                use_dropout = mc_samples > 1
                if use_dropout:
                    print(f"[mc-dropout] Using {mc_samples} stochastic forward passes per control step")
                if args.task_mode == "language":
                    base_task = model.create_tasks(texts=[args.language_prompt])
                else:
                    try:
                        base_task = model.create_tasks(goal_images=np.zeros((1, 256, 256, 3), np.uint8))
                    except TypeError as exc:
                        raise SystemExit(
                            "Loaded OctoModel does not accept goal_images conditioning. "
                            "Rerun with --task_mode=language or finetune with goal-image support."
                        ) from exc

                policy = supply_rng(
                    partial(model.sample_actions,
                            unnormalization_statistics=model.dataset_statistics["action"]),
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
                if args.task_mode == "goal":
                    curr = ts.observation["image_primary"]
                    goal_img = curr[None, ...]
                    task = model.create_tasks(goal_images=goal_img)
                else:
                    task = base_task

                t_last = time.perf_counter()
                for step in range(steps_per):
                    if not v.is_running():
                        break
                    obs_for_octo = build_obs_for_octo(
                        ts.observation,
                        t_idx=step,
                        meta=example_meta,
                        history=history,
                        img_shapes=img_shapes,
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
                    if args.debug_constant_action is not None:
                        a = np.zeros((7,), np.float32)
                        a[0] = float(args.debug_constant_action)
                        env._last_action_std = None
                    elif mc_samples == 1:
                        act_tree = policy(obs_for_octo, task, train=use_dropout)
                        a = _action_from_tree(act_tree)
                        env._last_action_std = None
                    else:
                        samples = [
                            _action_from_tree(policy(obs_for_octo, task, train=True))
                            for _ in range(mc_samples)
                        ]
                        stacked = np.stack(samples, axis=0)
                        a = stacked.mean(axis=0).astype(np.float32)
                        env._last_action_std = stacked.std(axis=0).astype(np.float32)
                    ts = env.step(a)
                    if record_video:
                        video_frames.append(_compose_video_frame(ts.observation))
                    now = time.perf_counter()
                    dt = now - t_last
                    if dt < period:
                        time.sleep(max(0.0, period - dt))
                    t_last = now
                    v.sync()
                if record_video and video_path:
                    _write_video(video_frames, video_fps, video_path)
                print(f"[done] steps={step+1} for {exp_dir}")
        else:
            ds_iter = load_rlds_dataset(
                args.dataset_name,
                data_dir=args.dataset_dir,
                split=args.dataset_split,
                shuffle=args.dataset_shuffle,
                seed=args.seed,
            )
            print(f"[dataset] Replaying '{args.dataset_name}' split '{args.dataset_split}' from {args.dataset_dir}")
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
                    continue

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
                        q_target = proprio[step]
                        env.data.qvel[:] = 0.0
                        env.data.ctrl[:] = 0.0
                        env.data.qpos[env.arm_qpos_addr[:qdim]] = q_target[:qdim]
                        mujoco.mj_forward(env.model, env.data)
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
                        a = np.asarray(actions[step]).reshape(-1)
                        if a.size < 7:
                            print(f"[dataset] Episode {epi_idx}: action has {a.size} dims (<7); stopping episode")
                            break
                        if a.size > 7 and warn_extra:
                            print(f"[dataset] Episode {epi_idx}: ignoring extra action dims ({a.size} → 7)")
                            warn_extra = False
                        a = a[:7].astype(np.float32)
                        ts = env.step(a)
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

    print("[all done]")


if __name__ == "__main__":
    main()
