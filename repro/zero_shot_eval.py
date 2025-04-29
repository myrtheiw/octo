from functools import partial
import sys
import gymnasium as gym
import panda_gym
import cv2

from absl import app, flags, logging
import jax
import numpy as np
import wandb

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import RHCWrapper
from octo.utils.train_callbacks import supply_rng

class OctoPandaWrapper(gym.Wrapper):
    def __init__(self, env_name="PandaReach-v3"):
        env = gym.make(env_name, render_mode="rgb_array")
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({
            "image_primary": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
            "image_wrist": gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
        })
        self.action_space = self.env.action_space

        # self.action_space = gym.spaces.Box(
        #     low=-1.0, high=1.0, 
        #     shape=(4,),
        #     dtype=np.float32
        # )

    def reset(self, **kwargs):
        self.obs_raw, info = self.env.reset(**kwargs)
        img = self.env.render()
        obs_dict = self._format_obs(img)
        return obs_dict, info

    def step(self, action):
        self.obs_raw, reward, terminated, truncated, info = self.env.step(action)
        img = self.env.render()

        obs_dict = self._format_obs(img)
        info["observations"] = [obs_dict]
        return obs_dict, reward, terminated, truncated, info

    def _resize_image(self, img, size):
        img = np.array(img).astype(np.uint8)
        return cv2.resize(img, size)

    def _get_proprioception(self):
        return np.concatenate([
            self.obs_raw['observation'],
            self.obs_raw['desired_goal']
        ])

    def _format_obs(self, img):
        image_primary = self._resize_image(img, (256, 256))
        image_wrist = np.zeros((128, 128, 3), dtype=np.uint8)

        image_primary = np.stack([image_primary] * 2)[None]
        image_wrist = np.stack([image_wrist] * 2)[None]
        proprio = np.stack([self._get_proprioception()] * 2)[None]

        return {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "proprio": proprio,
            "timestep": np.array([[0, 1]], dtype=np.int32),
            "timestep_pad_mask": np.array([[1, 1]], dtype=bool),
            "pad_mask_dict/image_primary": np.array([[1, 1]], dtype=bool),
            "pad_mask_dict/image_wrist": np.array([[1, 1]], dtype=bool),
            "pad_mask_dict/timestep": np.array([[1, 1]], dtype=bool),
            "task_completed": np.zeros((1, 2, 4), dtype=np.float32)
        }

    def get_task(self):
        goal_image = self.env.render_goal() if hasattr(self.env, "render_goal") else self.env.render()
        return {
            "language_instruction": "reach the target position",
            "goal": {
                "image_primary": self._resize_image(goal_image, size=(256, 256))
            }
        }

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_path", None, "Path to pretrained Octo checkpoint directory or huggingface path."
)

def main(_):
    wandb.init(name="octo_panda_eval", project="octo")

    logging.info("Loading pretrained Octo model from Hugging Face...")
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

    env = OctoPandaWrapper(env_name="PandaReach-v3")
    env = RHCWrapper(env, exec_horizon=50)

    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics.get("action", None),
        )
    )

    for episode in range(3):
        obs, _ = env.reset()
        task = model.create_tasks(texts=["reach the target position"])

        images = []
        episode_return = 0.0

        for t in range(200):
            actions = policy_fn(obs, task)[0, -1, :3]  # Shape: (4,)
            obs, reward, terminated, truncated, info = env.step(np.array(actions))

            images.append(obs["image_primary"][0, 0])
            episode_return += reward

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} return: {episode_return:.2f}")
        
        

        success = info.get("is_success", 0.0)
        

        wandb.log({
            "epsiode_return": episode_return,
            "episode_length": t +1,
            "success": success,
            "rollout_video": wandb.Video(
                np.array(images).transpose(0, 3, 1, 2)[::2], fps=10
            )
        })

if __name__ == "__main__":
    app.run(main)