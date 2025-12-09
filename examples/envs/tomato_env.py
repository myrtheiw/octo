import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import gym
import numpy as np
from PIL import Image

try:
    import dm_env
except Exception:  # pragma: no cover - optional dependency
    dm_env = None

# Make sure we can import utilities under octo/scripts when running from examples/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.sim_env import (
        PandaSimEnv,
        auto_ee_ref,
        build_arm_dof_indices,
        build_arm_mapping_from_model,
    )
    import mujoco
except Exception:
    PandaSimEnv = None  # type: ignore
    mujoco = None

try:
    from record_dataset.oracle_dynamic_norm import JOINT_DELTA_SCALE  # type: ignore
except Exception:
    # Fall back to the dataset collection scale if the collector imports are unreachable.
    JOINT_DELTA_SCALE = np.full(7, 0.02, dtype=np.float32)


DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml",
)


class PandaTomatoSimEnv:
    """
    Minimal MuJoCo Panda simulator for tomato picking.

    This wraps `PandaSimEnv` (see scripts/sim_env.py) so that evaluation can run
    without depending on the ACT ALOHA environment. The actions are interpreted
    as joint deltas in **radians** for the 7 arm joints.
    """

    def __init__(
        self,
        model_xml: str = DEFAULT_MODEL_PATH,
        *,
        substeps: int = 40,
        kp: float = 120.0,
        kd: Optional[float] = None,
        action_scale: float = 1.0,
        primary_cam: Optional[str] = None,
        wrist_cam: Optional[str] = None,
        reset_callback: Optional[Callable[["PandaTomatoSimEnv"], None]] = None,
        task_sampler: Optional[
            Callable[[], Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]
        ] = None,
        capture_images: bool = True,
    ):
        if PandaSimEnv is None or mujoco is None:
            raise ImportError(
                "PandaSimEnv / mujoco not available. Ensure octo/scripts is on the "
                "PYTHONPATH and mujoco is installed."
            )
        self.model_xml = model_xml
        self.substeps = int(substeps)
        self.kp = float(kp)
        self.kd = float(2.0 * np.sqrt(self.kp) if kd is None else kd)
        self.action_scale = float(action_scale)
        self.primary_cam = primary_cam
        self.wrist_cam = wrist_cam
        self.capture_images = bool(capture_images)
        self._reset_callback = reset_callback
        self._task_sampler = task_sampler

        self.current_goal_primary: Optional[np.ndarray] = None
        self.current_goal_wrist: Optional[np.ndarray] = None
        self.current_instruction: str = ""

        self._env = self._build_env()

    def _build_env(self) -> PandaSimEnv:
        model = mujoco.MjModel.from_xml_path(self.model_xml)
        data = mujoco.MjData(model)
        arm_act_ids, arm_qpos_addr, arm_gain_type = build_arm_mapping_from_model(model)
        arm_dof_idx = build_arm_dof_indices(model, arm_act_ids)
        ee_ref = auto_ee_ref(model, data)

        return PandaSimEnv(
            model=model,
            data=data,
            arm_act_ids=arm_act_ids,
            arm_qpos_addr=arm_qpos_addr,
            ee_ref=ee_ref,
            arm_dof_idx=arm_dof_idx,
            substeps=self.substeps,
            kp=self.kp,
            kd=self.kd,
            action_scale=self.action_scale,  # interpret inputs as radian deltas
            primary_cam=self.primary_cam,
            wrist_cam=self.wrist_cam,
            arm_gain_type=arm_gain_type,
            capture_images=self.capture_images,
        )

    def _timestep_to_obs(self, ts: "dm_env.TimeStep") -> Dict[str, np.ndarray]:
        obs = ts.observation
        return {
            "image_primary": np.asarray(obs.get("image_primary")),
            "image_wrist": np.asarray(obs.get("image_wrist")),
            "proprio": np.asarray(obs.get("proprio")),
            "timestep": np.asarray(obs.get("timestep", 0), dtype=np.int32),
        }

    def _update_task_metadata(self, obs: Dict[str, np.ndarray]) -> None:
        if self._task_sampler is not None:
            language, goal_primary, goal_wrist = self._task_sampler()
            self.current_instruction = language
            self.current_goal_primary = goal_primary
            self.current_goal_wrist = goal_wrist
            return

        if not self.current_instruction:
            self.current_instruction = "pick the ripe tomato in front"
        if self.current_goal_primary is None and "image_primary" in obs:
            self.current_goal_primary = obs["image_primary"]
        if self.current_goal_wrist is None and "image_wrist" in obs:
            self.current_goal_wrist = obs["image_wrist"]

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if self._reset_callback is not None:
            self._reset_callback(self)
        ts = self._env.reset()
        obs = self._timestep_to_obs(ts)
        self._update_task_metadata(obs)
        return obs, {}

    def step(
        self, action_rad: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        ts = self._env.step(action_rad)
        obs = self._timestep_to_obs(ts)
        reward = float(ts.reward if hasattr(ts, "reward") else 0.0)
        done = bool(ts.step_type == dm_env.StepType.LAST) if dm_env else False
        trunc = False
        return obs, reward, done, trunc, {}

    def observation_spec(self) -> Dict[str, Any]:
        return self._env.observation_spec()


class TomatoGymEnv(gym.Env):
    """
    Gym wrapper that accepts normalized joint-delta chunks from Octo and
    unnormalizes them with JOINT_DELTA_SCALE before stepping the Panda sim.
    """

    def __init__(
        self,
        panda_env: PandaTomatoSimEnv,
        *,
        max_steps: int = 400,
        joint_delta_scale: Union[float, Sequence[float], np.ndarray] = JOINT_DELTA_SCALE,
        primary_size: Tuple[int, int] = (256, 256),
        wrist_size: Tuple[int, int] = (128, 128),
        action_horizon: int = 4,
    ):
        super().__init__()
        self.panda_env = panda_env
        self.max_steps = int(max_steps)
        self.joint_delta_scale = np.asarray(joint_delta_scale, dtype=np.float32).reshape(
            -1
        )
        if self.joint_delta_scale.shape[0] != 7:
            raise ValueError(
                f"joint_delta_scale must have 7 elements, got {self.joint_delta_scale.shape}"
            )
        self._step_counter = 0
        self.primary_size = tuple(primary_size)
        self.wrist_size = tuple(wrist_size)
        self.action_horizon = int(action_horizon)

        (
            primary_shape,
            wrist_shape,
            proprio_shape,
        ) = self._infer_obs_shapes_from_spec(panda_env)
        primary_shape = self.primary_size + (3,)
        wrist_shape = self.wrist_size + (3,)

        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=primary_shape, dtype=np.uint8
                ),
                "image_wrist": gym.spaces.Box(
                    low=0, high=255, shape=wrist_shape, dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=proprio_shape, dtype=np.float32
                ),
                "timestep": gym.spaces.Box(
                    low=0, high=np.inf, shape=(), dtype=np.int32
                ),
                "task_completed": gym.spaces.Box(
                    low=0, high=1, shape=(), dtype=np.int32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0 * np.ones((7,), dtype=np.float32),
            high=np.ones((7,), dtype=np.float32),
            dtype=np.float32,
        )

    def _infer_obs_shapes_from_spec(
        self, panda_env: PandaTomatoSimEnv
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        primary_shape = (256, 256, 3)
        wrist_shape = (256, 256, 3)
        proprio_shape = (14,)

        if hasattr(panda_env, "observation_spec"):
            try:
                spec = panda_env.observation_spec()
            except Exception:
                spec = None
            if isinstance(spec, dict):
                if "image_primary" in spec:
                    primary_shape = tuple(spec["image_primary"].shape)
                if "image_wrist" in spec:
                    wrist_shape = tuple(spec["image_wrist"].shape)
                if "proprio" in spec:
                    proprio_shape = tuple(spec["proprio"].shape)

        if hasattr(panda_env, "observation_space"):
            space = panda_env.observation_space
            if isinstance(space, gym.spaces.Dict):
                if "image_primary" in space.spaces:
                    primary_shape = space.spaces["image_primary"].shape
                if "image_wrist" in space.spaces:
                    wrist_shape = space.spaces["image_wrist"].shape
                if "proprio" in space.spaces:
                    proprio_shape = space.spaces["proprio"].shape

        return primary_shape, wrist_shape, proprio_shape

    def _format_obs(self, raw_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        obs["image_primary"] = np.asarray(
            raw_obs.get(
                "image_primary",
                np.zeros(self.observation_space["image_primary"].shape, dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        if obs["image_primary"].shape[:2] != self.primary_size:
            obs["image_primary"] = self._resize_image(
                obs["image_primary"], self.primary_size
            )
        obs["image_wrist"] = np.asarray(
            raw_obs.get(
                "image_wrist",
                np.zeros(self.observation_space["image_wrist"].shape, dtype=np.uint8),
            ),
            dtype=np.uint8,
        )
        if obs["image_wrist"].shape[:2] != self.wrist_size:
            obs["image_wrist"] = self._resize_image(obs["image_wrist"], self.wrist_size)
        obs["proprio"] = np.asarray(
            raw_obs.get(
                "proprio",
                np.zeros(self.observation_space["proprio"].shape, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        obs["timestep"] = np.asarray(raw_obs.get("timestep", 0), dtype=np.int32)
        tc = raw_obs.get("task_completed", None)
        if tc is None:
            tc = np.zeros((self.action_horizon,), dtype=np.int32)
        tc_arr = np.asarray(tc, dtype=np.int32)
        if tc_arr.shape == ():
            tc_arr = np.full((self.action_horizon,), int(tc_arr), dtype=np.int32)
        elif tc_arr.shape != (self.action_horizon,):
            tc_arr = np.resize(tc_arr, (self.action_horizon,))
        obs["task_completed"] = tc_arr
        return obs

    def _resize_image(self, image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        pil = Image.fromarray(image)
        resized = pil.resize((w, h), resample=Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self._step_counter = 0
        raw_obs, info = self.panda_env.reset()
        obs = self._format_obs(raw_obs)
        return obs, info or {}

    def _step_once(
        self, action_rad: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        result = self.panda_env.step(action_rad)
        if len(result) == 5:
            obs, reward, done, trunc, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            trunc = False
        else:
            raise ValueError(
                "panda_env.step must return (obs, reward, done, trunc, info) or (obs, reward, done, info)"
            )
        self._step_counter += 1
        if self._step_counter >= self.max_steps:
            trunc = True
        return self._format_obs(obs), float(reward), bool(done), bool(trunc), info or {}

    def step(
        self, action_chunk: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        actions = np.asarray(action_chunk, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None]
        if actions.shape[1] != 7:
            raise ValueError(
                f"Expected action dimension of 7, got {actions.shape} after reshape"
            )

        total_reward = 0.0
        observations = []
        done = False
        trunc = False
        info: Dict[str, Any] = {}
        obs = None

        for a_norm in actions:
            delta_q = a_norm * self.joint_delta_scale
            obs, reward, done, trunc, step_info = self._step_once(delta_q)
            observations.append(obs)
            total_reward += reward
            info = step_info or {}
            if done or trunc:
                break

        info.setdefault("observations", observations)
        info.setdefault("rewards", total_reward)
        if obs is None:
            obs = self.reset()[0]
        return obs, total_reward, done, trunc, info



    def get_task(self) -> Dict[str, Any]:
        goal: Dict[str, Any] = {}
        if getattr(self.panda_env, "current_goal_primary", None) is not None:
            goal["image_primary"] = self.panda_env.current_goal_primary
        if getattr(self.panda_env, "current_goal_wrist", None) is not None:
            goal["image_wrist"] = self.panda_env.current_goal_wrist

        return {
            "language_instruction": getattr(
                self.panda_env, "current_instruction", ""
            ),
            "goal": goal,
        }
