import argparse

import matplotlib.pyplot as plt
import numpy as np

from replay_from_actions import iter_episodes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot raw RLDS/EnvLogger actions over time for a single episode."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to an uncompressed TFRecord shard containing episodes.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to visualize.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="actions_episode.png",
        help="Output PNG path for the action plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    episodes = list(iter_episodes(args.dataset_path))
    if not episodes:
        raise RuntimeError(f"No episodes found in {args.dataset_path}")

    if args.episode < 0 or args.episode >= len(episodes):
        raise IndexError(f"Episode index {args.episode} out of range [0, {len(episodes)-1}]")

    episode_dict, _ = episodes[args.episode]
    actions = np.asarray(episode_dict["action"])
    proprio = np.asarray(episode_dict["proprio"])  # kept for potential future diagnostics

    if actions.ndim != 2:
        raise ValueError(f"Expected 'action' to be 2D [T, Da], got shape {actions.shape}")

    T, Da = actions.shape
    print(f"Episode {args.episode}: T={T}, action_dim={Da}")
    print(f"actions.shape: {actions.shape}")
    print("Action stats (dim j: min, max, mean, std):")
    for j in range(Da):
        dim_vals = actions[:, j]
        print(
            f"j={j}: min={dim_vals.min():.5f}, max={dim_vals.max():.5f}, "
            f"mean={dim_vals.mean():.5f}, std={dim_vals.std():.5f}"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    timesteps = np.arange(T)
    for j in range(Da):
        ax.plot(timesteps, actions[:, j], label=f"a[{j}]")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_title(f"Actions over time (episode {args.episode})")
    ax.set_xlabel("timestep")
    ax.set_ylabel("action value")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved action plot to {args.out}")


if __name__ == "__main__":
    main()
