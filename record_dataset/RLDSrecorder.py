import mujoco
import numpy as np
import rlds
import cv2
import time

# filepath: /home/myrtheiw/octo_ws/record_rlds_demo.py

# Initialize MuJoCo simulation
model_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/tomato_plant_v10.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Set up cameras
def setup_cameras(model):
    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    third_person_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person_camera")
    return wrist_camera_id, third_person_camera_id

wrist_camera_id, third_person_camera_id = setup_cameras(model)

# RLDS demonstration recording setup
demonstrations = []
current_episode = []
metadata = {"task": "manual_control_tomato_plant"}

# Placeholder for manual control (keyboard or joystick)
def manual_control(data):
    # Example: Move the robot's end-effector in the x-direction
    data.ctrl[:] = np.random.uniform(-1, 1, size=data.ctrl.shape)  # Replace with actual control logic

# Render camera views
def render_camera(model, data, camera_id):
    width, height = 640, 480
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjv_updateScene(model, data, mujoco.MjvScene(), mujoco.MjvOption(), mujoco.MjvCamera())
    mujoco.mjr_renderCamera(camera_id, rgb_buffer)
    return rgb_buffer

# Simulation loop
def simulate_and_record():
    global current_episode
    sim_time = 0
    step_size = model.opt.timestep

    while sim_time < 10:  # Simulate for 10 seconds
        mujoco.mj_step(model, data)
        manual_control(data)

        # Record observations
        wrist_view = render_camera(model, data, wrist_camera_id)
        third_person_view = render_camera(model, data, third_person_camera_id)
        observation = {
            "state": data.qpos.copy(),
            "action": data.ctrl.copy(),
            "reward": 0,  # Placeholder reward
            "done": False,
            "camera_views": {
                "wrist": wrist_view,
                "third_person": third_person_view,
            },
        }
        current_episode.append(observation)

        # Display camera views (optional)
        cv2.imshow("Wrist Camera", wrist_view)
        cv2.imshow("Third-Person Camera", third_person_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sim_time += step_size

    # End episode
    current_episode[-1]["done"] = True
    demonstrations.append(current_episode)
    current_episode = []

# Run simulation and record demonstrations
simulate_and_record()

# Save RLDS demonstrations
dataset = rlds.Dataset(demonstrations, metadata)
rlds.save(dataset, "/home/myrtheiw/octo_ws/rlds_demonstrations.tfrecord")

print("Demonstrations saved successfully!")