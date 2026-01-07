"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/02_finetune_new_observation_action.py script.

For installing the ALOHA sim environment, clone: https://github.com/tonyzhaozh/act
Then run:
pip3 install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py

Finally, modify the `sys.path.append` statement below to add the ACT repo to your path.
If you are running this on a head-less server, start a virtual display:
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1

To run this script, run:
    cd examples
    export MUJOCO_GL=egl
    python3 03_eval_finetuned.py --finetuned_path=<path_to_finetuned_aloha_checkpoint>
"""
from functools import partial
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from absl import app, flags, logging
import gym
import jax
import jax.numpy as jnp
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import wandb

sys.path.append("/home/myrtheiw/octo_ws/act")

# keep this to register ALOHA sim env
from envs.aloha_sim_env import AlohaGymEnv  # noqa

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)

def sample_mc_actions(model, obs, task, n_samples=10, base_rng=None):
    if base_rng is None:
        base_rng = jax.random.PRNGKey(np.random.randint(0, 1e6))

    obs_batched = jax.tree_map(lambda x: x[None], obs)
    rngs = jax.random.split(base_rng, n_samples)

    def single_sample(rng_key):
        return model.sample_actions(
            obs_batched,
            task,
            train=True,
            rng=rng_key,
            unnormalization_statistics=model.dataset_statistics["action"]
        )[0]

    actions = jax.lax.map(single_sample, rngs)  # propagate PRNGs for Dropout
    return actions


def make_env(model):
    env = gym.make("aloha-sim-cube-v0")
    env = NormalizeProprio(env, model.dataset_statistics)
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)
    return env


def apply_occlusion(image, fraction=0.3):
    img = np.array(image)
    if img.ndim < 3:
        return image
    h = img.shape[-3]
    w = img.shape[-2]
    occ_h = max(1, int(h * fraction))
    occ_w = max(1, int(w * fraction))
    y0 = (h - occ_h) // 2
    x0 = (w - occ_w) // 2
    if img.ndim == 3:
        img[y0:y0 + occ_h, x0:x0 + occ_w, :] = 0
    else:
        img[..., y0:y0 + occ_h, x0:x0 + occ_w, :] = 0
    return img


def run_condition(model, condition_name, occlude=False, n_episodes=3):
    env = make_env(model)
    episode_returns = []
    episode_uncertainties = []
    saved_snapshot = False

    for i in range(n_episodes):
        obs, info = env.reset()

        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction)

        images = [obs["image_primary"][0]]
        episode_return = 0.0
        step_uncertainties = []
        while len(images) < 400:
            mc_actions = sample_mc_actions(
                model,
                obs,
                task,
                n_samples=10,
                base_rng=jax.random.PRNGKey(np.random.randint(1e6)),
            )
            action_mean = jnp.mean(mc_actions, axis=0)
            action_std = jnp.std(mc_actions, axis=0)
            actions = np.array(action_mean)
            step_uncertainties.append(float(jnp.mean(action_std)))

            obs, reward, done, trunc, info = env.step(actions)
            if occlude:
                obs["image_primary"] = jnp.array(apply_occlusion(obs["image_primary"]))
                if not saved_snapshot:
                    snapshot = np.array(obs["image_primary"])
                    if snapshot.ndim == 4:
                        snapshot = snapshot[0]
                    imageio.imwrite("occlusion_snapshot.png", snapshot)
                    saved_snapshot = True

            images.extend([o["image_primary"][0] for o in info["observations"]])
            episode_return += reward
            if done or trunc:
                break

        avg_uncertainty = float(np.mean(step_uncertainties)) if step_uncertainties else 0.0
        episode_returns.append(episode_return)
        episode_uncertainties.append(avg_uncertainty)
        print(f"{condition_name} Episode return: {episode_return}")

        with open("eval_mc_uncertainty_metrics.txt", "a") as f:
            f.write(f"{condition_name} Episode {i+1}:\n")
            f.write(f"  Return: {episode_return:.2f}\n")
            f.write(f"  Avg Action Std (Uncertainty): {avg_uncertainty:.4f}\n")
            f.write("\n")

        wandb.log({
            f"{condition_name}_rollout_video": wandb.Video(
                np.array(images).transpose(0, 3, 1, 2)[::2]
            ),
            f"{condition_name}_avg_action_std": avg_uncertainty,
        })

    return episode_returns, episode_uncertainties



def main(_):
    # setup wandb for logging
    wandb.init(name="eval_aloha", project="octo")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    normal_returns, normal_uncertainties = run_condition(
        model, "normal", occlude=False, n_episodes=10
    )
    occlusion_returns, occlusion_uncertainties = run_condition(
        model, "occlusion", occlude=True, n_episodes=10
    )

    episodes = np.arange(1, len(normal_uncertainties) + 1)
    plt.figure(figsize=(8, 4))
    plt.scatter(episodes, normal_uncertainties, label="Normal")
    plt.scatter(episodes, occlusion_uncertainties, label="Occluded")
    plt.xlabel("Episode")
    plt.ylabel("Avg Action Std (Uncertainty)")
    plt.title("MC Dropout Uncertainty: Normal vs Occluded")
    plt.xticks(episodes)
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mc_uncertainty_comparison.png")
    plt.close()


if __name__ == "__main__":
    app.run(main)
