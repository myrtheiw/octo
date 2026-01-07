import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "/home/myrtheiw/tfds_out/tomato_rlds/0.0.40/tomato_rlds-train.tfrecord-00000-of-00090"
NUM_EPISODES_TO_SAMPLE = 5
MAX_TIMESTEPS_TO_PLOT = 200
TFDS_SPLIT = "train"  # split name when reading TFDS/RLDS directories
PLOT_OUTPUT_ROOT = Path("/home/myrtheiw/octo_ws/octo/outputs/plots")

# Optional action limits; set to arrays of shape (action_dim,) to enable checks.
ACTION_MIN = None
ACTION_MAX = None


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    """
    Load dataset from an .npz file or a TFDS/RLDS directory/TFRecord shard.

    For .npz: expects "actions" (num_episodes, T, action_dim) and optionally "states"/"obs".
    For TFDS/RLDS: expects a dataset directory (or shard inside it) with "steps" features
    containing "action" and optional observation["proprio"].
    """
    path_obj = Path(path)
    if path_obj.suffix == ".npz":
        return _load_npz_dataset(path_obj)
    return _load_rlds_dataset(path_obj)


def _load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    """
    Load dataset from an .npz file.

    Expected keys:
        - "actions": shape (num_episodes, T, action_dim)
        - "states" or "obs": shape (num_episodes, T, state_dim) (optional)

    Returns:
        dict with keys "actions" and optionally "states".
    """
    data = np.load(path, allow_pickle=True)
    data_dict = dict(data)

    actions = data_dict.get("actions")
    if actions is None:
        raise KeyError("Dataset must contain an 'actions' array.")

    if actions.ndim != 3:
        raise ValueError(
            f"'actions' must be 3-D (num_episodes, T, action_dim); got shape {actions.shape}"
        )

    states = data_dict.get("states", data_dict.get("obs"))
    if states is not None and states.shape[:2] != actions.shape[:2]:
        raise ValueError(
            f"'states/obs' leading dims must match actions: {states.shape[:2]} vs {actions.shape[:2]}"
        )

    result = {"actions": actions}
    if states is not None:
        result["states"] = states
    return result


def _resolve_tfds_dir(path: Path) -> Path:
    """Find the TFDS dataset directory that contains dataset_info.json."""
    candidates = [path]
    if path.is_file():
        candidates.append(path.parent)
    candidates.append(path.parent.parent)
    for cand in candidates:
        if cand is not None and (cand / "dataset_info.json").exists():
            return cand
    raise FileNotFoundError(
        f"Could not find dataset_info.json near '{path}'. "
        "Point DATA_PATH to the TFDS version directory or a shard within it."
    )


def _load_rlds_dataset(path: Path) -> Dict[str, np.ndarray]:
    """
    Load an RLDS/TFDS dataset from a directory (or shard inside it).

    Expects episodic structure with a "steps" dataset, where each step has:
      - "action"
      - optional "observation" dict with "proprio"
    """
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow Datasets is required to read TFRecord RLDS datasets. "
            "Install with `pip install tensorflow tensorflow-datasets` (or tf-nightly)."
        ) from exc

    # Find the TFDS dataset root (directory that has dataset_info.json)
    dataset_dir = _resolve_tfds_dir(path)

    builder = tfds.builder_from_directory(str(dataset_dir))
    if TFDS_SPLIT not in builder.info.splits:
        raise ValueError(
            f"Split '{TFDS_SPLIT}' not found. Available splits: {list(builder.info.splits.keys())}"
        )

    # Episodic RLDS dataset: each element is an "episode" with a "steps" Dataset
    ds = builder.as_dataset(split=TFDS_SPLIT, read_config=tfds.ReadConfig(shuffle_seed=0))

    actions_list: List[np.ndarray] = []
    states_list: List[Optional[np.ndarray]] = []

    # Iterate over episodes
    for episode_idx, episode in enumerate(ds):
        if "steps" not in episode:
            raise ValueError(f"Episode {episode_idx} missing 'steps' field.")

        steps_ds = episode["steps"]  # this is a tf.data.Dataset (IterableDataset)

        episode_actions: List[np.ndarray] = []
        episode_proprio: List[np.ndarray] = []

        # Now iterate over steps inside this episode
        for step in tfds.as_numpy(steps_ds):
            if "action" not in step:
                raise ValueError(f"Step in episode {episode_idx} missing 'action' field.")

            # Action per step: shape (action_dim,)
            episode_actions.append(np.asarray(step["action"]))

            # Optional proprio in observation
            obs = step.get("observation")
            if isinstance(obs, dict) and "proprio" in obs:
                episode_proprio.append(np.asarray(obs["proprio"]))

        if not episode_actions:
            raise ValueError(f"Episode {episode_idx} has no steps/actions.")

        # Stack actions into (T, action_dim)
        action_arr = np.stack(episode_actions, axis=0)
        actions_list.append(action_arr)

        # Stack proprio if present for all steps
        if episode_proprio and len(episode_proprio) == len(episode_actions):
            proprio_arr = np.stack(episode_proprio, axis=0)
            states_list.append(proprio_arr)
        else:
            states_list.append(None)

    if not actions_list:
        raise ValueError("No episodes found in TFDS dataset.")

    # Allow varying episode lengths by trimming everything to the minimum length.
    episode_lengths = [arr.shape[0] for arr in actions_list]
    min_len = min(episode_lengths)

    if len(set(episode_lengths)) != 1:
        print(
            f"Episode lengths vary in TFDS dataset ({sorted(set(episode_lengths))}). "
            f"Trimming all episodes to length {min_len} for analysis."
        )

    # Trim all episodes to the same length and stack
    actions = np.stack([arr[:min_len] for arr in actions_list], axis=0)  # (num_episodes, min_len, action_dim)

    states = None
    if all(s is not None for s in states_list):
        states = np.stack([s[:min_len] for s in states_list if s is not None], axis=0)
        if states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"States/obs leading dims must match actions: {states.shape[:2]} vs {actions.shape[:2]}"
            )

    result = {"actions": actions}
    if states is not None:
        result["states"] = states
    return result

    states = None
    if all(s is not None for s in states_list):
        states = np.stack([s for s in states_list if s is not None], axis=0)
        if states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"States/obs leading dims must match actions: {states.shape[:2]} vs {actions.shape[:2]}"
            )

    result = {"actions": actions}
    if states is not None:
        result["states"] = states
    return result

def compute_action_stats(actions: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-dimension min, max, mean, std over all episodes and timesteps."""
    flat = actions.reshape(-1, actions.shape[-1])
    return {
        "min": flat.min(axis=0),
        "max": flat.max(axis=0),
        "mean": flat.mean(axis=0),
        "std": flat.std(axis=0),
    }


def compute_delta_stats(actions: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Compute per-dimension stats for action deltas a[t+1] - a[t]."""
    if actions.shape[1] < 2:
        return None
    deltas = actions[:, 1:, :] - actions[:, :-1, :]
    flat = deltas.reshape(-1, deltas.shape[-1])
    return {
        "min": flat.min(axis=0),
        "max": flat.max(axis=0),
        "mean": flat.mean(axis=0),
        "std": flat.std(axis=0),
    }


def print_stats_table(name: str, stats: Dict[str, np.ndarray]) -> None:
    """Pretty-print per-dimension stats."""
    print(f"\n{name} (per dimension):")
    header = f"{'dim':>5} {'min':>12} {'max':>12} {'mean':>12} {'std':>12}"
    print(header)
    for i in range(len(stats["min"])):
        print(
            f"{i:5d} {stats['min'][i]:12.6f} {stats['max'][i]:12.6f} "
            f"{stats['mean'][i]:12.6f} {stats['std'][i]:12.6f}"
        )


def sample_episodes(
    actions: np.ndarray,
    states: Optional[np.ndarray],
    num_episodes_to_sample: int,
    max_timesteps: int,
) -> List[Tuple[int, int, int, np.ndarray, Optional[np.ndarray]]]:
    """
    Randomly sample episodes and windows.

    Returns a list of tuples:
        (episode_index, t_start, t_end, actions_window, states_window_or_None)
    """
    num_episodes, total_T, _ = actions.shape
    if num_episodes_to_sample > num_episodes:
        raise ValueError(
            f"Requested {num_episodes_to_sample} episodes, but dataset has {num_episodes}."
        )

    sampled_indices = random.sample(range(num_episodes), num_episodes_to_sample)
    windows = []
    for idx in sampled_indices:
        window_len = min(total_T, max_timesteps)
        if window_len <= 0:
            raise ValueError("Time dimension T must be positive.")
        t_start = 0 if total_T == window_len else random.randint(0, total_T - window_len)
        t_end = t_start + window_len
        actions_window = actions[idx, t_start:t_end]
        states_window = None if states is None else states[idx, t_start:t_end]
        windows.append((idx, t_start, t_end, actions_window, states_window))
    return windows


def plot_actions(
    episode_index: int, t_start: int, t_end: int, actions_window: np.ndarray
) -> plt.Figure:
    """Plot actions over a window."""
    action_dim = actions_window.shape[-1]
    timesteps = np.arange(actions_window.shape[0])

    if action_dim <= 6:
        fig, axes = plt.subplots(action_dim, 1, sharex=True, figsize=(8, 2 * action_dim))
        if action_dim == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(timesteps, actions_window[:, i], label=f"action_{i}")
            ax.set_ylabel(f"a[{i}]")
            ax.legend()
        axes[-1].set_xlabel("timestep (window)")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(action_dim):
            ax.plot(timesteps, actions_window[:, i], label=f"a[{i}]")
        ax.set_xlabel("timestep (window)")
        ax.set_ylabel("action value")
        ax.legend(ncol=2, fontsize=8)

    fig.suptitle(f"Episode {episode_index} actions (t={t_start}..{t_end-1})")
    plt.tight_layout()
    return fig


def plot_action_deltas(
    episode_index: int, t_start: int, t_end: int, actions_window: np.ndarray
) -> Optional[plt.Figure]:
    """Plot action deltas over a window."""
    if actions_window.shape[0] < 2:
        print(f"Episode {episode_index}: window too short for deltas; skipping.")
        return

    deltas = actions_window[1:] - actions_window[:-1]
    action_dim = deltas.shape[-1]
    timesteps = np.arange(deltas.shape[0])

    if action_dim <= 6:
        fig, axes = plt.subplots(action_dim, 1, sharex=True, figsize=(8, 2 * action_dim))
        if action_dim == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(timesteps, deltas[:, i], label=f"delta_a[{i}]")
            ax.set_ylabel(f"Δa[{i}]")
            ax.legend()
        axes[-1].set_xlabel("timestep (window)")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(action_dim):
            ax.plot(timesteps, deltas[:, i], label=f"Δa[{i}]")
        ax.set_xlabel("timestep (window)")
        ax.set_ylabel("action delta")
        ax.legend(ncol=2, fontsize=8)

    fig.suptitle(f"Episode {episode_index} action deltas (t={t_start}..{t_end-2})")
    plt.tight_layout()
    return fig


def plot_states(
    episode_index: int, t_start: int, t_end: int, states_window: np.ndarray
) -> plt.Figure:
    """Plot up to the first six state dimensions over a window."""
    state_dim = states_window.shape[-1]
    dims_to_plot = min(6, state_dim)
    timesteps = np.arange(states_window.shape[0])

    fig, axes = plt.subplots(dims_to_plot, 1, sharex=True, figsize=(8, 2 * dims_to_plot))
    if dims_to_plot == 1:
        axes = [axes]

    for i in range(dims_to_plot):
        axes[i].plot(timesteps, states_window[:, i], label=f"state_{i}")
        axes[i].set_ylabel(f"s[{i}]")
        axes[i].legend()

    axes[-1].set_xlabel("timestep (window)")
    fig.suptitle(f"Episode {episode_index} states (t={t_start}..{t_end-1})")
    plt.tight_layout()
    return fig


def check_joint_limits(actions: np.ndarray) -> None:
    """Check for action limit violations if ACTION_MIN/MAX are provided."""
    if ACTION_MIN is None or ACTION_MAX is None:
        print("\nJoint-limit check: skipped (ACTION_MIN/MAX not set).")
        return

    action_min = np.asarray(ACTION_MIN)
    action_max = np.asarray(ACTION_MAX)
    if action_min.shape != (actions.shape[-1],) or action_max.shape != (actions.shape[-1],):
        raise ValueError(
            f"ACTION_MIN/MAX must have shape ({actions.shape[-1]},); "
            f"got {action_min.shape} and {action_max.shape}"
        )

    flat = actions.reshape(-1, actions.shape[-1])
    violations_lower = (flat < action_min).sum(axis=0)
    violations_upper = (flat > action_max).sum(axis=0)
    total = flat.shape[0]
    print("\nJoint-limit violations per dimension:")
    print(f"{'dim':>5} {'below_min':>12} {'above_max':>12} {'pct_viol':>12}")
    for i in range(actions.shape[-1]):
        total_viol = violations_lower[i] + violations_upper[i]
        pct = (total_viol / total) * 100.0 if total > 0 else 0.0
        print(f"{i:5d} {violations_lower[i]:12d} {violations_upper[i]:12d} {pct:12.4f}%")


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    actions = dataset["actions"]
    states = dataset.get("states")

    print(f"Loaded actions with shape {actions.shape}")
    if states is not None:
        print(f"Loaded states with shape {states.shape}")

    # Prepare output directory for plots.
    run_dir = PLOT_OUTPUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {run_dir}")

    action_stats = compute_action_stats(actions)
    print_stats_table("Action stats", action_stats)

    delta_stats = compute_delta_stats(actions)
    if delta_stats is None:
        print("\nAction deltas: cannot compute (T < 2).")
    else:
        print_stats_table("Action delta stats", delta_stats)

    check_joint_limits(actions)

    windows = sample_episodes(
        actions, states, num_episodes_to_sample=NUM_EPISODES_TO_SAMPLE, max_timesteps=MAX_TIMESTEPS_TO_PLOT
    )

    for episode_index, t_start, t_end, actions_window, states_window in windows:
        fig_actions = plot_actions(episode_index, t_start, t_end, actions_window)
        fig_actions.savefig(run_dir / f"episode{episode_index}_actions_{t_start}-{t_end-1}.png", bbox_inches="tight")

        fig_deltas = plot_action_deltas(episode_index, t_start, t_end, actions_window)
        if fig_deltas is not None:
            fig_deltas.savefig(run_dir / f"episode{episode_index}_deltas_{t_start}-{t_end-1}.png", bbox_inches="tight")

        if states_window is not None:
            fig_states = plot_states(episode_index, t_start, t_end, states_window)
            fig_states.savefig(run_dir / f"episode{episode_index}_states_{t_start}-{t_end-1}.png", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
