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
print("✅ Saved both camera views.")# mujococameracheck_all.py
import os
import re
import cv2
import mujoco
import numpy as np

MODEL_PATH = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"

# Load model & data
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

# List cameras
cam_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
print("Cameras in model:", cam_names)

# Render settings (HxW)
H, W = 480, 640

# Create output dir
out_dir = "renders"
os.makedirs(out_dir, exist_ok=True)

# Reusable renderer
renderer = mujoco.Renderer(m, height=H, width=W)

def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

# Render each camera
for cam_name in cam_names:
    try:
        renderer.update_scene(d, camera=cam_name)
        img = renderer.render()  # RGB
        fname = os.path.join(out_dir, f"render_{sanitize(cam_name)}_{W}x{H}.jpg")
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved {fname}")
    except Exception as e:
        print(f"⚠️  Skipped camera '{cam_name}' due to error: {e}")

print("Done.")

