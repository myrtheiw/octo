# mujococameracheck.py

import mujoco
import numpy as np
import cv2

MODEL_PATH = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"


m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

# List cameras so we know the names/ids
cam_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
print("Cameras in model:", cam_names)

# Render third-person
r_tp = mujoco.Renderer(m, height=256, width=256)
r_tp.update_scene(d, camera="third_person_cam")   # <-- pass the name (or id)
img_tp = r_tp.render()
cv2.imwrite("render_third_person.jpg", cv2.cvtColor(img_tp, cv2.COLOR_RGB2BGR))

# Render wrist/gripper
r_wr = mujoco.Renderer(m, height=256, width=256)
r_wr.update_scene(d, camera="gripper_cam")        # <-- pass the name (or id)
img_wr = r_wr.render()
cv2.imwrite("render_wrist_camera.jpg", cv2.cvtColor(img_wr, cv2.COLOR_RGB2BGR))
print("✅ Saved both camera views.")
