import os
import cv2
import numpy as np
import tensorflow as tf
import mujoco
from ikpy.chain import Chain
from tqdm import trange

# Setup paths for model and robot description
urdf_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/panda.urdf"
model_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/scene.xml"

# Initialize kinematics chain (Franka Panda robot) – used for potential IK or trajectory planning
panda_chain = Chain.from_urdf_file(
    urdf_path,
    base_elements=["panda_link0"],
    last_link_vector=[0, 0, 0.11],
    active_links_mask=[False, True, True, True, True, True, True, True, False, False, False, False]
)

# Load the MuJoCo model and create data and renderer
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=256, width=256)

# Map camera names to IDs for rendering
camera_name_to_id = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i): i
    for i in range(model.ncam)
}
third_person_cam_id = camera_name_to_id["third_person"]
wrist_cam_id = camera_name_to_id["wrist_camera"]

GRIPPER_IDX = 7  # Index of the gripper actuator in control vector (Franka Panda has 7 arm joints + 1 gripper)

# Define trajectories: each entry is (waypoints array, language instruction)
trajectories = {
    "traj1": (
        np.array([
            [0.0, -0.4,  0.0,  0.0,   0.0,  0.0,  0.9, 0.04],
            [0.0, -0.85, 0.0, -2.0,   0.0,  1.6,  0.9, 0.04],
            [0.0, -0.85, 0.0, -2.25,  0.0,  1.75, 0.9, 0.0],
        ], dtype=np.float32),
        "Pick the tomato truss on top, closest to you",
    ),
    "traj2": (
        np.array([
            [0.0, -1.15, 0.0, -2.6,   0.0, 1.45, 0.0, 0.0],
            [0.0, -1.3,  0.0, -2.85,  0.0, 2.15, 0.8, 0.0],
            [0.0, -1.3,  0.0, -3.1,  -0.05, 2.35, 0.8, 0.0],
        ], dtype=np.float32),
        "Pick the tomato truss on the bottom, closest to you",
    ),
}

TOTAL_STEPS = 300
EPISODE_COUNT = 100
DATASET_DIR = "/home/myrtheiw/octo_ws/octo/record_dataset"  # base directory for dataset

def interpolate_waypoints(waypoints, step, total_steps):
    """Linearly interpolate between waypoints for the given step in the trajectory."""
    phase_duration = total_steps // (len(waypoints) - 1)
    phase = min(step // phase_duration, len(waypoints) - 2)
    t = (step % phase_duration) / phase_duration
    return (1 - t) * waypoints[phase] + t * waypoints[phase + 1]

def run_trajectory(name, waypoints, instruction, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_dataset.tfrecord")

    if not instruction:
        raise ValueError(f"No language instruction provided for trajectory '{name}'")

    instruction_bytes = instruction.encode()

    with tf.io.TFRecordWriter(output_path) as writer:
        for ep in trange(EPISODE_COUNT, desc=f"Recording {name}"):
            mujoco.mj_resetData(model, data)

            # Define context (episode metadata)
            context = tf.train.Features(feature={
                "episode_metadata.language_instruction": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[instruction_bytes]))
            })

            feature_lists = tf.train.FeatureLists(feature_list={
                "state": tf.train.FeatureList(),
                "action": tf.train.FeatureList(),
                "reward": tf.train.FeatureList(),
                "is_terminal": tf.train.FeatureList(),
                "is_first": tf.train.FeatureList(),
                "image_0": tf.train.FeatureList(),
                "image_1": tf.train.FeatureList(),
            })

            for step in range(TOTAL_STEPS):
                target = interpolate_waypoints(waypoints, step, TOTAL_STEPS)
                q_target, g_target = target[:7], target[7]
                q_current = data.qpos[:7]
                q_vel = data.qvel[:7]

                data.ctrl[:7] = 100 * (q_target - q_current) - 5 * q_vel
                data.ctrl[GRIPPER_IDX] = g_target
                action_snapshot = data.ctrl.copy()

                mujoco.mj_step(model, data)

                renderer.cam_id = third_person_cam_id
                renderer.update_scene(data)
                third_img = renderer.render()
                _, img0_jpeg = cv2.imencode(".jpg", cv2.cvtColor(third_img, cv2.COLOR_RGB2BGR))

                renderer.cam_id = wrist_cam_id
                renderer.update_scene(data)
                wrist_img = renderer.render()
                _, img1_jpeg = cv2.imencode(".jpg", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

                feature_lists.feature_list["state"].feature.add(
                    float_list=tf.train.FloatList(value=data.qpos[:8].copy()))
                feature_lists.feature_list["action"].feature.add(
                    float_list=tf.train.FloatList(value=action_snapshot))
                feature_lists.feature_list["reward"].feature.add(
                    int64_list=tf.train.Int64List(value=[0]))
                feature_lists.feature_list["is_terminal"].feature.add(
                    int64_list=tf.train.Int64List(value=[int(step == TOTAL_STEPS - 1)]))
                feature_lists.feature_list["is_first"].feature.add(
                    int64_list=tf.train.Int64List(value=[int(step == 0)]))
                feature_lists.feature_list["image_0"].feature.add(
                    bytes_list=tf.train.BytesList(value=[img0_jpeg.tobytes()]))
                feature_lists.feature_list["image_1"].feature.add(
                    bytes_list=tf.train.BytesList(value=[img1_jpeg.tobytes()]))

            sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
            writer.write(sequence_example.SerializeToString())

    print(f"✅ Saved {EPISODE_COUNT} episodes for {name} to {output_path}")


if __name__ == "__main__":
    # Use a 'dataset' subdirectory for output, as expected by TFDS builder
    dataset_output_dir = os.path.join(DATASET_DIR, "dataset")
    os.makedirs(dataset_output_dir, exist_ok=True)
    for traj_name, (waypoints, instruction) in trajectories.items():
        run_trajectory(traj_name, waypoints, instruction, dataset_output_dir)
