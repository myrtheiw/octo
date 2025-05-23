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




def main(_):
    # setup wandb for logging
    wandb.init(name="eval_aloha", project="octo")

    # load finetuned model
    logging.info("Loading finetuned model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path)

    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_primary": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_primary": ...
    #     }
    #   }
    ##################################################################################################################
    env = gym.make("aloha-sim-cube-v0")

    # wrap env to normalize proprio
    env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)

    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )

    # running rollouts
    for i in range(3):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction)

        # run rollout for 400 steps
        images = [obs["image_primary"][0]]
        episode_return = 0.0
        while len(images) < 400:
            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            mc_actions = sample_mc_actions(model, obs, task, n_samples=10, base_rng=jax.random.PRNGKey(np.random.randint(1e6)))
            action_mean = jnp.mean(mc_actions, axis=0)
            action_std = jnp.std(mc_actions, axis=0)  # Optional: for visualization or thresholds
            actions = np.array(action_mean)


            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)
            images.extend([o["image_primary"][0] for o in info["observations"]])
            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")

        # Save MC Dropout metrics to a file
        with open("eval_mc_uncertainty_metrics.txt", "a") as f:
            f.write(f"Episode {i+1}:\n")
            f.write(f"  Return: {episode_return:.2f}\n")
            f.write(f"  Avg Action Std (Uncertainty): {float(jnp.mean(action_std)):.4f}\n")
            f.write(f"  Max Action Std: {float(jnp.max(action_std)):.4f}\n")
            f.write(f"  Min Action Std: {float(jnp.min(action_std)):.4f}\n")
            f.write(f"  Action Std Shape: {action_std.shape}\n")
            f.write("\n")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log({
            "rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2]),
            "avg_action_std": float(jnp.mean(action_std))
        })


if __name__ == "__main__":
    app.run(main)