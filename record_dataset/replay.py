import mujoco
import tensorflow as tf
import numpy as np
import imageio
import time


# Load model
model_path = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Set up renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Load dataset
def parse_example(example_proto):
    feature_description = {
        'state': tf.io.VarLenFeature(tf.float32),
        'action': tf.io.VarLenFeature(tf.float32),
        'reward': tf.io.FixedLenFeature([], tf.int64),
        'is_terminal': tf.io.FixedLenFeature([], tf.int64),
        'is_first': tf.io.FixedLenFeature([], tf.int64),
        'language_command': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

dataset_path = "/home/myrtheiw/octo_ws/octo/record_dataset/dataset/tomato_dataset.tfrecord"
raw_dataset = tf.data.TFRecordDataset(dataset_path)
parsed_dataset = raw_dataset.map(parse_example)

# Record frames
frames = []

for parsed_record in parsed_dataset:
    state = tf.sparse.to_dense(parsed_record['state']).numpy()
    action = tf.sparse.to_dense(parsed_record['action']).numpy()

    data.qpos[:len(state)] = state
    data.ctrl[:model.nu] = action
    mujoco.mj_forward(model, data)

    renderer.update_scene(data)
    rgb_frame = renderer.render()
    frames.append(rgb_frame)

    time.sleep(0.02)

# Save as video
imageio.mimsave("replay_video.mp4", frames, fps=50)

print("Replay and recording completed.")
