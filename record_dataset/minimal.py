import mujoco
import numpy as np
import os

# os.environ["MUJOCO_GL"] = "egl"  # Use software rendering

model_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

scn = mujoco.MjvScene(model, maxgeom=1000)
opt = mujoco.MjvOption()
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

width, height = 640, 480
rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
depth_buffer = np.zeros((height, width), dtype=np.float32)

mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
viewport = mujoco.MjrRect(0, 0, width, height)
mujoco.mjr_render(viewport, scn, con)
mujoco.mjr_readPixels(rgb_buffer, depth_buffer, viewport, con)

print("Rendering successful!")