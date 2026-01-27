# ============================ tomato_env.py ============================
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import gym
import numpy as np
from PIL import Image

DEBUG_TERM = True

try:
    import dm_env
except Exception: 
    dm_env = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.sim_env import PandaSimEnv, build_arm_dof_indices, build_arm_mapping_from_model, auto_ee_ref
    import mujoco
except Exception:
    PandaSimEnv = None
    mujoco = None

# This matches the scale used in your training dataset
try:
    from record_dataset.oracle_dynamic_norm import JOINT_DELTA_SCALE
except Exception:
    JOINT_DELTA_SCALE = np.full(7, 0.02, dtype=np.float32)

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml",
)

class PandaTomatoSimEnv:
    """Wrapper to bridge PandaSimEnv (dm_env) to a simpler interface."""
    def __init__(self, model_xml, **kwargs):
        self.model_xml = model_xml
        self.kwargs = kwargs
        self._env = self._build_env()
        self.current_instruction = "pick the ripe tomato"
        self.current_goal_primary = None
        self.current_goal_wrist = None
        self._dbg_term_obs_count = 0

    def _build_env(self):
        model = mujoco.MjModel.from_xml_path(self.model_xml)
        data = mujoco.MjData(model)
        arm_act_ids, arm_qpos_addr, arm_gain_type = build_arm_mapping_from_model(model)
        ee_ref = auto_ee_ref(model, data)
        arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)
        
        return PandaSimEnv(
            model=model, data=data,
            arm_act_ids=arm_act_ids, arm_qpos_addr=arm_qpos_addr,
            ee_ref=ee_ref, arm_dof_idx=arm_dof_idx,
            arm_gain_type=arm_gain_type,
            **self.kwargs
        )

    def reset(self):
        ts = self._env.reset()
        return self._timestep_to_obs(ts), {}

    def step(self, action):
        ts = self._env.step(action)
        done = (ts.step_type == dm_env.StepType.LAST)
        info = {
            "is_success": int(ts.observation.get("is_success", 0)),
            "step_type": ts.step_type,
        }
        if DEBUG_TERM and (done or (self._env._t % 20 == 0)):
            print(
                f"[TERM WRAP] t={self._env._t} done={done} step_type={ts.step_type} "
                f"is_success={info['is_success']}",
                flush=True,
            )
        return self._timestep_to_obs(ts), float(ts.reward), done, False, info

    def _timestep_to_obs(self, ts):
        obs = ts.observation
        if DEBUG_TERM:
            has_success = "is_success" in obs
            if self._dbg_term_obs_count < 5 or not has_success:
                self._dbg_term_obs_count += 1
                print(
                    f"[TERM WRAP] dm_env is_success_present={has_success}",
                    flush=True,
                )
        return {
            "image_primary": np.asarray(obs.get("image_primary")),
            "image_wrist": np.asarray(obs.get("image_wrist")),
            "proprio": np.asarray(obs.get("proprio"), dtype=np.float32),
            "is_success": np.asarray(obs.get("is_success", 0), dtype=np.int32),
        }

    def set_goal_pos(self, goal_pos_world):
        self.current_goal_primary = goal_pos_world
        if hasattr(self._env, "set_goal_pos"):
            self._env.set_goal_pos(goal_pos_world)

class TomatoGymEnv(gym.Env):
    """
    Gym Wrapper matching AlohaGymEnv structure.
    """
    def __init__(self, panda_env, max_steps=400):
        self.panda_env = panda_env
        self.max_steps = max_steps
        self._step_counter = 0
        self.observation_space = gym.spaces.Dict({
            "image_primary": gym.spaces.Box(0, 255, (256, 256, 3), np.uint8),
            "image_wrist": gym.spaces.Box(0, 255, (128, 128, 3), np.uint8),
            "proprio": gym.spaces.Box(-np.inf, np.inf, (14,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)
    # --- ADDED: Expose Physics for the Eval Script ---
    @property
    def physics(self):
        """
        Creates a bridge to the internal MuJoCo data so env.unwrapped.physics works.
        """
        class PhysicsBridge:
            def __init__(self, internal_env):
                self.data = internal_env.data
                self.model = internal_env.model
            
            def forward(self):
                import mujoco
                mujoco.mj_forward(self.model, self.data)

        # Access chain: self.panda_env (PandaTomatoSimEnv) -> ._env (PandaSimEnv)
        return PhysicsBridge(self.panda_env._env)
    # -------------------------------------------------
    def reset(self, **kwargs):
        self._step_counter = 0
        obs, info = self.panda_env.reset()
        return self._format_obs(obs), info

    def step(self, action):
        # Flatten action if it comes in as (1, 7) or similar
        action = np.array(action, dtype=np.float32).reshape(-1)

        # --- CHANGED BLOCK STARTS HERE ---
        # Direct-control paths supply absolute joint targets. Otherwise we keep
        # the original scaled delta behavior.
        if getattr(self, "expects_absolute_action", False):
            action = np.asarray(action, dtype=np.float32)
        
        else:
            # Original behavior for training/standard envs
            ACTION_SCALE = 1
            action = action * ACTION_SCALE
            # action = np.clip(action, -0.2, 0.2) 
        # --- CHANGED BLOCK ENDS HERE ---
       # print("expects_absolute_action (panda_env):",
            #getattr(self.panda_env, "expects_absolute_action", False))

        obs, reward, done, trunc, info = self.panda_env.step(action)
        if DEBUG_TERM:
            should_log = done or trunc or (self._step_counter % 20 == 0)
            if should_log:
                print(
                    f"[TERM GYM] t={self._step_counter} done={done} trunc={trunc} "
                    f"is_success={info.get('is_success', None)}",
                    flush=True,
                )
        
        self._step_counter += 1
        if self._step_counter >= self.max_steps:
            trunc = True
            
        return self._format_obs(obs), reward, done, trunc, info

    def _format_obs(self, obs):
        # Resize images to match Octo requirements
        def resize(img, size):
            if img.shape[0] == size[0]*2: return img[::2, ::2] # Fast 2x downsample
            return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))
            
        return {
            "image_primary": resize(obs["image_primary"], (256, 256)),
            "image_wrist": resize(obs["image_wrist"], (128, 128)),
            "proprio": obs["proprio"] # Pass raw proprio (wrappers normalize it later)
        }
    
    def set_goal_pos(self, goal_pos_world):
        if hasattr(self.panda_env, "set_goal_pos"):
            self.panda_env.set_goal_pos(goal_pos_world)
    
    def get_task(self):
        return {"language_instruction": ["pick the ripe tomato"]}
