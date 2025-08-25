import mujoco
import numpy as np
import rlds
import cv2
import pygame
import time

# Initialize MuJoCo simulation
model_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Initialize pygame for mouse control
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Mouse Control")

# Set up cameras
def setup_cameras(model):
    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    third_person_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "third_person")
    return wrist_camera_id, third_person_camera_id

wrist_camera_id, third_person_camera_id = setup_cameras(model)

# RLDS demonstration recording setup
demonstrations = []
current_episode = []
metadata = {"task": "mouse_control_tomato_plant"}

# Manual control using mouse
def mouse_control(data):
    mouse_pos = pygame.mouse.get_pos()
    ctrl = np.zeros(data.ctrl.shape)

    # Map mouse position to control actions
    # Assuming the screen size is 400x300
    ctrl[0] = (mouse_pos[0] - 200) / 100.0  # Map X-axis to control[0]
    ctrl[1] = (mouse_pos[1] - 150) / 100.0  # Map Y-axis to control[1]

    # Apply control
    data.ctrl[:] = ctrl

# Render camera views
def render_camera(model, data, camera_id):
    width, height = 640, 480
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)

    # Initialize scene, option, and camera
    scn = mujoco.MjvScene(model, maxgeom=1000)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.fixedcamid = camera_id
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # Update the scene
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

    # Render the scene
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scn, con)

    # Read pixels from the rendered scene
    mujoco.mjr_readPixels(rgb_buffer, None, mujoco.MjrRect(0, 0, width, height), con)

    return rgb_buffer

# Simulation loop
def simulate_and_record():
    global current_episode
    sim_time = 0
    step_size = model.opt.timestep

    while sim_time < 10:  # Simulate for 10 seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        mujoco.mj_step(model, data)
        mouse_control(data)

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
pygame.quit()