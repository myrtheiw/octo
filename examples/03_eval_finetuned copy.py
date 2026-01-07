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
    python3 03_eval_finetuned.py --finetuned_path=<path_to_finetuned_aloha_checkpoint>
"""
from functools import partial
import sys
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb
from typing import Optional

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
flags.DEFINE_string(
    "debug_trace_path",
    None,
    "If set, write JSONL traces containing commanded vs. actual joint deltas.",
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _maybe_list(value: Optional[np.ndarray]):
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.tolist()


class ChunkLogger:
    """Write JSONL rows tracking commanded vs actual joint motion."""

    def __init__(self, path: Optional[str]):
        self._fp = open(path, "w", encoding="utf-8") if path else None
        self._chunk_counter = 0

    def log_chunk(
        self,
        *,
        episode_index: int,
        chunk_index: int,
        language_instruction: str,
        policy_chunk: np.ndarray,
        commanded: Optional[np.ndarray],
        actual: Optional[np.ndarray],
        pre_proprio: Optional[np.ndarray],
        post_proprio: Optional[np.ndarray],
        reward: float,
        done: bool,
        trunc: bool,
    ):
        if self._fp is None:
            return
        entry = {
            "episode_index": int(episode_index),
            "chunk_index": int(chunk_index),
            "language_instruction": str(language_instruction),
            "policy_chunk": _maybe_list(policy_chunk),
            "commanded_actions": _maybe_list(commanded),
            "actual_deltas": _maybe_list(actual),
            "pre_proprio": _maybe_list(pre_proprio),
            "post_proprio": _maybe_list(post_proprio),
            "reward": float(reward),
            "done": bool(done),
            "trunc": bool(trunc),
        }
        self._fp.write(json.dumps(entry, default=_json_default) + "\n")
        self._fp.flush()
        self._chunk_counter += 1

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None


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

    trace_logger = ChunkLogger(FLAGS.debug_trace_path)

    try:
        # running rollouts
        for episode_index in range(3):
            obs, info = env.reset()

            # create task specification --> use model utility to create task dict with correct entries
            language_instruction = env.get_task()["language_instruction"]
            task = model.create_tasks(texts=language_instruction)

            # track last proprio vector (latest timestep in the history stack)
            prev_proprio = None
            if "proprio" in obs:
                prev_proprio = np.asarray(obs["proprio"])[-1]

            # run rollout for 400 steps
            images = [obs["image_primary"][0]]
            episode_return = 0.0
            chunk_index = 0
            while len(images) < 400:
                # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
                policy_out = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
                policy_chunk = np.asarray(policy_out[0], dtype=np.float32)

                pre_proprio = prev_proprio

                # step env -- info contains full "chunk" of observations for logging
                # obs only contains observation for final step of chunk
                obs, reward, done, trunc, info = env.step(policy_chunk)
                observations = info.get("observations", []) or []
                images.extend([o["image_primary"][0] for o in observations if "image_primary" in o])
                episode_return += reward

                executed = min(len(observations), policy_chunk.shape[0])
                commanded = policy_chunk[:executed] if executed > 0 else None

                actual_deltas = None
                post_proprio = prev_proprio
                if executed > 0:
                    proprio_seq = []
                    for obs_step in observations[:executed]:
                        proprio = obs_step.get("proprio")
                        if proprio is None:
                            proprio_seq = []
                            break
                        proprio_seq.append(np.asarray(proprio))
                    if proprio_seq:
                        proprio_stack = np.stack(proprio_seq, axis=0)
                        post_proprio = proprio_stack[-1]
                        if pre_proprio is not None:
                            deltas = [proprio_stack[0] - pre_proprio]
                        else:
                            deltas = []
                        if proprio_stack.shape[0] > 1:
                            deltas.extend(proprio_stack[1:] - proprio_stack[:-1])
                        if deltas:
                            actual_deltas = np.stack(deltas, axis=0)
                    elif "proprio" in obs:
                        post_proprio = np.asarray(obs["proprio"])[-1]

                if actual_deltas is not None and commanded is not None:
                    clip_len = min(actual_deltas.shape[0], commanded.shape[0])
                    actual_deltas = actual_deltas[:clip_len]
                    commanded = commanded[:clip_len]

                trace_logger.log_chunk(
                    episode_index=episode_index,
                    chunk_index=chunk_index,
                    language_instruction=language_instruction,
                    policy_chunk=policy_chunk,
                    commanded=commanded,
                    actual=actual_deltas,
                    pre_proprio=pre_proprio,
                    post_proprio=post_proprio,
                    reward=reward,
                    done=done,
                    trunc=trunc,
                )

                prev_proprio = post_proprio
                if prev_proprio is None and "proprio" in obs:
                    prev_proprio = np.asarray(obs["proprio"])[-1]
                chunk_index += 1

                if done or trunc:
                    break

            print(f"Episode return: {episode_return}")

            # log rollout video to wandb -- subsample temporally 2x for faster logging
            wandb.log(
                {
                    "rollout_video": wandb.Video(
                        np.array(images).transpose(0, 3, 1, 2)[::2]
                    )
                }
            )
    finally:
        trace_logger.close()


if __name__ == "__main__":
    app.run(main)
