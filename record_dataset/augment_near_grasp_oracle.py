#!/usr/bin/env python3
"""
augment_near_grasp_oracle.py
----------------------------
Augment an existing TFDS RLDS dataset with fresh near-grasp oracle rollouts.

Run:
  python augment_near_grasp_oracle.py --out_version 0.0.41 --num_new_episodes 10

How it starts from late trajectory points:
- Samples episodes from the source split and iterates `episode["steps"].as_numpy_iterator()`.
- Selects a late timestep (via --start_frac or --tail_window).
- Uses the step's `observation["proprio"]` first 7 dims as the new START_JOINTS.
- Calls oracle_dynamic_norm.run_oracle_once to plan/execute from that state.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import mujoco
import envlogger
from envlogger.backends import tfds_backend_writer

oracle = None


def _import_oracle(live_render: bool):
    if live_render:
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        from octo.record_dataset import oracle_dynamic_norm as _oracle  # noqa: E402
        return _oracle
    except ModuleNotFoundError:
        script_dir = Path(__file__).resolve().parent
        local_oracle = script_dir / "oracle_dynamic_norm.py"
        if not local_oracle.exists():
            raise
        sys_path = str(script_dir)
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        import oracle_dynamic_norm as _oracle  # type: ignore  # noqa: E402
        return _oracle


def _maybe_fullscreen_viewer(v) -> None:
    try:
        import glfw
    except Exception as exc:
        print(f"[VIEWER] glfw import failed; fullscreen skipped: {exc}")
        return
    window = getattr(v, "_window", None)
    if window is None:
        return
    monitor = glfw.get_primary_monitor()
    if monitor is None:
        return
    mode = glfw.get_video_mode(monitor)
    if mode is None:
        return
    try:
        glfw.set_window_pos(window, 0, 0)
        glfw.set_window_size(window, mode.size.width, mode.size.height)
        glfw.maximize_window(window)
        print("[VIEWER] fullscreen enabled")
    except Exception as exc:
        print(f"[VIEWER] fullscreen failed: {exc}")


def _patch_viewer_fullscreen() -> None:
    orig_launch = oracle.viewer.launch_passive

    def _launch_passive_fullscreen(*args, **kwargs):
        ctx = orig_launch(*args, **kwargs)

        class _Ctx:
            def __enter__(self_inner):
                v = ctx.__enter__()
                _maybe_fullscreen_viewer(v)
                return v

            def __exit__(self_inner, exc_type, exc, tb):
                return ctx.__exit__(exc_type, exc, tb)

        return _Ctx()

    oracle.viewer.launch_passive = _launch_passive_fullscreen


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment tomato_rlds with near-grasp oracle rollouts."
    )
    parser.add_argument("--tfds_root", default="/home/myrtheiw/tfds_out")
    parser.add_argument("--dataset_name", default="tomato_rlds")
    parser.add_argument("--in_version", default="0.0.40")
    parser.add_argument("--out_version", required=True)
    parser.add_argument("--num_new_episodes", type=int, default=50)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--start_frac", type=float, default=0.9)
    parser.add_argument("--tail_window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--live_render", action="store_true")
    parser.add_argument(
        "--output_split",
        default=None,
        help="Split to write new episodes to (default: same as source_split).",
    )
    return parser.parse_args()


def _copy_version_dir(in_version_dir: str, out_version_dir: str) -> None:
    if not os.path.isdir(in_version_dir):
        raise FileNotFoundError(f"Input version dir not found: {in_version_dir}")
    if os.path.exists(out_version_dir):
        raise FileExistsError(f"Output version already exists: {out_version_dir}")
    shutil.copytree(in_version_dir, out_version_dir)


def _select_late_index(
    length: int, start_frac: float, tail_window: int | None, rng: np.random.Generator
) -> int:
    if length <= 0:
        raise ValueError("Episode has no steps")
    if tail_window is not None:
        if tail_window <= 0:
            raise ValueError("tail_window must be > 0")
        start = max(0, length - tail_window)
    else:
        start = int(np.floor(float(start_frac) * length))
    start = int(np.clip(start, 0, length - 1))
    return int(rng.integers(start, length))


def _extract_proprio_steps(episode) -> np.ndarray:
    proprio_list = []
    steps = episode["steps"].as_numpy_iterator()
    for step in steps:
        obs = step.get("observation", None)
        if obs is None or "proprio" not in obs:
            raise ValueError("Episode step missing observation['proprio']")
        proprio_list.append(np.asarray(obs["proprio"], dtype=np.float32))
    if not proprio_list:
        raise ValueError("Episode has no proprio steps")
    return np.stack(proprio_list, axis=0)


def _build_env_and_config(model, data):
    ee_ref = ("site", "tcp") if oracle._site_exists(model, "tcp") else ("body", "hand")
    obstacles_all = oracle._collect_plant_obstacles(model)
    arm_act_ids, arm_qpos_addr = oracle.build_arm_mapping_from_model(
        model, prefer_position=True
    )
    arm_dof_idx = oracle.build_arm_dof_indices(model, arm_act_ids)

    if oracle.AUTO_WIDEN_LIMITS:
        for i, dof in enumerate(arm_dof_idx):
            jid = int(model.dof_jntid[dof])
            model.jnt_limited[jid] = 1
            lo, hi = oracle.TYPICAL_FRANKA_LIMITS[i]
            model.jnt_range[jid][0] = lo
            model.jnt_range[jid][1] = hi

    gripper_idx = oracle.find_gripper_actuator(model)

    ds_config = tfds.rlds.rlds_base.DatasetConfig(
        version=tfds.core.Version(oracle.DATASET_VERSION),
        name=oracle.DATASET_NAME,
        observation_info=tfds.features.FeaturesDict(
            {
                "proprio": tfds.features.Tensor(shape=(model.nq,), dtype=np.float32),
                "language_instruction": tfds.features.Text(),
                "image_primary": tfds.features.Image(
                    shape=(oracle.IMG_H, oracle.IMG_W, 3), encoding_format="jpeg"
                ),
                "image_wrist": tfds.features.Image(
                    shape=(oracle.IMG_H, oracle.IMG_W, 3), encoding_format="jpeg"
                ),
                "goal_image_primary": tfds.features.Image(
                    shape=(oracle.IMG_H, oracle.IMG_W, 3), encoding_format="jpeg"
                ),
                "goal_image_wrist": tfds.features.Image(
                    shape=(oracle.IMG_H, oracle.IMG_W, 3), encoding_format="jpeg"
                ),
            }
        ),
        action_info=tfds.features.Tensor(shape=(7,), dtype=np.float32),
        reward_info=tf.float32,
        discount_info=tf.float32,
    )

    desired_dt = float(oracle.TARGET_ACTION_DT)
    timestep = float(model.opt.timestep)
    substeps_calc = max(1, int(round(desired_dt / max(timestep, 1e-6))))

    base_env = oracle.PandaOracleEnv(
        model,
        data,
        arm_act_ids,
        arm_qpos_addr,
        ee_ref,
        language_instruction="Pick the specified tomato by name.",
        substeps=substeps_calc,
        gripper_idx=gripper_idx,
        arm_dof_idx=arm_dof_idx,
        kp=oracle.PD_KP,
        kd=oracle.PD_KD,
        action_scale=1.0,
    )

    return base_env, ee_ref, obstacles_all, arm_dof_idx, arm_qpos_addr, ds_config


def _choose_target(top_targets: list, rng: np.random.Generator):
    if not top_targets:
        return None
    return top_targets[int(rng.integers(0, len(top_targets)))]


def _front_back_sets(top_targets: list) -> Tuple[set, set]:
    if not top_targets:
        return set(), set()
    ordered = sorted(
        ((name, float(np.linalg.norm(grasp[:2]))) for name, grasp, *_ in top_targets),
        key=lambda t: t[1],
    )
    mid_idx = max(1, len(ordered) // 2)
    front_targets = {name for name, _ in ordered[:mid_idx]}
    back_targets = {name for name, _ in ordered[mid_idx:]}
    return front_targets, back_targets


def _source_episode_iter(
    tfds_root: str, dataset_name: str, in_version: str, source_split: str, seed: int
) -> Iterable:
    builder_dir = os.path.join(tfds_root, dataset_name, in_version)
    builder = tfds.builder_from_directory(builder_dir)
    if source_split not in builder.info.splits:
        raise ValueError(
            f"Split '{source_split}' not found in {builder.info.splits.keys()}"
        )
    num_examples = int(builder.info.splits[source_split].num_examples or 0)
    ds = builder.as_dataset(split=source_split, shuffle_files=True)
    shuffle_buf = min(max(10, num_examples), 1000)
    ds = ds.shuffle(
        buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True
    ).repeat()
    return iter(ds)


def main() -> None:
    args = _parse_args()
    global oracle
    oracle = _import_oracle(args.live_render)
    if args.live_render:
        _patch_viewer_fullscreen()
    rng = np.random.default_rng(args.seed)

    dataset_root = os.path.join(args.tfds_root, args.dataset_name)
    in_version_dir = os.path.join(dataset_root, args.in_version)
    out_version_dir = os.path.join(dataset_root, args.out_version)

    _copy_version_dir(in_version_dir, out_version_dir)
    print(f"[COPY] {in_version_dir} -> {out_version_dir}")

    oracle.TFDS_ROOT_DIR = args.tfds_root
    oracle.DATASET_NAME = args.dataset_name
    oracle.DATASET_VERSION = args.out_version
    oracle.EPISODES_TOTAL = int(args.num_new_episodes)
    oracle.LIVE_RENDER = bool(args.live_render)
    oracle.LIVE_RENDER_QC = bool(args.live_render)

    episode_iter = _source_episode_iter(
        args.tfds_root, args.dataset_name, args.in_version, args.source_split, args.seed
    )
    output_split = args.output_split or args.source_split

    action_delta_stats = oracle.ActionDeltaStatistics(action_dim=7)

    episodes_done = 0
    attempts = 0
    qc_failures = 0
    max_attempts = max(args.num_new_episodes * 5, args.num_new_episodes + 5)

    while episodes_done < args.num_new_episodes and attempts < max_attempts:
        attempts += 1
        try:
            if oracle.USE_DYNAMIC_PLANT:
                model, data = oracle._build_and_load_scene(oracle.MODEL_PATH)
            else:
                model = mujoco.MjModel.from_xml_path(oracle.MODEL_PATH)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)

            base_env, ee_ref, obstacles_all, arm_dof_idx, arm_qpos_addr, ds_config = (
                _build_env_and_config(model, data)
            )

            top = oracle._find_top_targets(model, data, k=oracle.EPISODES_PER_PLANT, s=0.66)
            front_targets, back_targets = _front_back_sets(top)
            target = _choose_target(top, rng)
            if target is None:
                print("[WARN] No targets found on plant; skipping.")
                continue
            name, goal_pos, _approach_xy, truss_name = target

            if name in front_targets or not back_targets:
                base_env.set_language_instruction("Pick the top tomato in front.")
            else:
                base_env.set_language_instruction("Pick the top tomato in the back.")

            base_env.set_goal_images(None, None)

            episode = next(episode_iter)
            proprio_steps = _extract_proprio_steps(episode)
            late_idx = _select_late_index(
                len(proprio_steps), args.start_frac, args.tail_window, rng
            )
            start_q = np.asarray(proprio_steps[late_idx], dtype=np.float32).reshape(-1)
            if start_q.shape[0] < 7:
                raise ValueError(
                    f"proprio has {start_q.shape[0]} dims, expected >=7"
                )
            start_q = start_q[:7]
            oracle.START_JOINTS = start_q.copy()

            print(
                f"[START] epi={episodes_done} source_idx={late_idx}/{len(proprio_steps)} "
                f"q0={np.array2string(start_q, precision=3)} target={name}"
            )

            err_final, waypoints, goal_frame_idx = oracle.run_oracle_once(
                env=None,
                base_env=base_env,
                model=model,
                data=data,
                arm_dof_idx=arm_dof_idx,
                arm_qpos_addr=arm_qpos_addr,
                ee_ref=ee_ref,
                goal_pos=goal_pos,
                obstacles=obstacles_all,
                goal_body_name=truss_name,
                waypoints=None,
                dry_run=True,
            )
            if err_final > oracle.MAX_FINAL_ERR:
                qc_failures += 1
                print(
                    f"[QC] Skip (err={err_final:.4f} > {oracle.MAX_FINAL_ERR})"
                )
                continue

            goal_primary, goal_wrist = oracle._render_goal_images_at_idx(
                base_env=base_env,
                model=model,
                data=data,
                arm_qpos_addr=arm_qpos_addr,
                waypoints=waypoints,
                goal_idx=goal_frame_idx,
            )
            base_env.set_goal_images(goal_primary, goal_wrist)

            with envlogger.EnvLogger(
                base_env,
                backend=tfds_backend_writer.TFDSBackendWriter(
                    data_directory=args.tfds_root,
                    split_name=output_split,
                    max_episodes_per_file=8,
                    ds_config=ds_config,
                ),
                metadata={
                    "language_instruction": base_env.lang,
                    "action_type": "joint_delta",
                    "action_scale": base_env.action_scale,
                    "action_dt_sec": float(oracle.TARGET_ACTION_DT),
                },
            ) as env:
                oracle.run_oracle_once(
                    env=env,
                    base_env=base_env,
                    model=model,
                    data=data,
                    arm_dof_idx=arm_dof_idx,
                    arm_qpos_addr=arm_qpos_addr,
                    ee_ref=ee_ref,
                    goal_pos=goal_pos,
                    obstacles=obstacles_all,
                    goal_body_name=truss_name,
                    waypoints=waypoints,
                    dry_run=False,
                    goal_frame_idx=goal_frame_idx,
                    action_delta_stats=action_delta_stats,
                )

            episodes_done += 1
            print(
                f"[PROGRESS] Episodes saved: {episodes_done}/{args.num_new_episodes}"
            )

            oracle.move_stray_shards_into_version_dir(args.tfds_root, ds_config)
            if oracle.ENABLE_TFRECORD_HARVEST:
                oracle.harvest_any_tfrecords(
                    args.tfds_root, out_version_dir, args.dataset_name, output_split
                )
            oracle._post_write_repair(out_version_dir, args.dataset_name)
            oracle._relocate_dataset_metadata(
                args.tfds_root, out_version_dir, args.out_version
            )
            oracle._ensure_min_train_split(out_version_dir, args.dataset_name)
            oracle.repair_tfds_splits(out_version_dir, args.dataset_name)
        except Exception as exc:
            print(f"[ERROR] attempt={attempts}: {exc}")

    if episodes_done < args.num_new_episodes:
        print(
            f"[WARN] Only wrote {episodes_done}/{args.num_new_episodes} episodes after "
            f"{attempts} attempts"
        )
    if qc_failures:
        print(f"[QC] Skipped {qc_failures} episodes due to MAX_FINAL_ERR={oracle.MAX_FINAL_ERR}")

    stats_path = action_delta_stats.write_dataset_statistics(
        out_version_dir,
        args.dataset_name,
        args.out_version,
        num_trajectories=episodes_done,
    )
    print(f"[ACTION_NORM] saved dataset_statistics to {stats_path}")
    print(f"[DONE] TFDS version written: {out_version_dir}")


if __name__ == "__main__":
    main()
