# save as octo/examples/peek_initial_action.py
import os, sys
import numpy as np
import jax
from functools import partial

# add examples/ to import tomato_env
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import NormalizeProprio, HistoryWrapper, RHCWrapper
from octo.utils.train_callbacks import supply_rng
from envs.tomato_env import DEFAULT_MODEL_PATH, PandaTomatoSimEnv, TomatoGymEnv

CKPT_DIR = "/home/myrtheiw/octo_ws/octo/outputs/checkpoint_20nov/experiment_20251119_155226"
CKPT_STEP = 50000

def main():
    # Headless + offline
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"Loading model from {CKPT_DIR} (step {CKPT_STEP})")
    model = OctoModel.load_pretrained(CKPT_DIR, step=CKPT_STEP)

    # Build env; disable images to avoid renderer
    panda_env = PandaTomatoSimEnv(
        model_xml=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        substeps=40,
        kp=120.0,
        kd=None,
        action_scale=1.0,
        capture_images=False,
    )
    env = TomatoGymEnv(panda_env, max_steps=400)
    env = NormalizeProprio(env, model.dataset_statistics)
    env = HistoryWrapper(env, horizon=2)
    env = RHCWrapper(env, exec_horizon=4)

    # Reset and build task
    obs, _ = env.reset()
    language_instruction = env.get_task()["language_instruction"]
    task = model.create_tasks(texts=language_instruction)

    # Batch obs for model
    obs_b = jax.tree_map(lambda x: x[None], obs)

    # Normalized chunk
    normalized_chunk = np.asarray(
        model.sample_actions(obs_b, task, rng=jax.random.PRNGKey(0))[0],
        dtype=np.float32,
    )

    # Unnormalized joint deltas (rad)
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        )
    )
    unnorm_chunk = np.asarray(policy_fn(obs_b, task)[0], dtype=np.float32)

    print("Language instruction:", language_instruction)
    print("Normalized chunk:\n", normalized_chunk)
    print("Unnormalized joint deltas (rad):\n", unnorm_chunk)

    # Measure executed Δq from MuJoCo qpos around the first exec_horizon chunk
    q_pre = panda_env._env.data.qpos.copy()  # PandaSimEnv is nested inside PandaTomatoSimEnv
    obs2, reward, done, trunc, info2 = env.step(unnorm_chunk)
    q_post = panda_env._env.data.qpos.copy()
    dq_exec = q_post - q_pre
    print("\nExecuted Δq (qpos post - pre):\n", dq_exec)
    print("Reward:", reward, "Done:", done, "Trunc:", trunc)

if __name__ == "__main__":
    main()
