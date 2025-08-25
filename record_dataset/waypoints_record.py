import mujoco
import mujoco.viewer
import numpy as np
import sys
import termios
import tty
import time
import os

# Load model
model_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Teleop params
joint_idx = 0
step_size = 0.05
waypoints = []

# Launch viewer
viewer = mujoco.viewer.launch_passive(model, data)
time.sleep(0.01)
viewer.sync()

print("\n--- Teleop tuning helper ---")
print("Controls:")
print("  [a] decrease current joint")
print("  [d] increase current joint")
print("  [w] select next joint")
print("  [s] select previous joint")
print("  [q] close gripper")
print("  [e] open gripper")
print("  [z] save current qpos as a waypoint")
print("  [Enter] finish and save all waypoints")
print("Currently selected joint: 0")

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Main loop
done = False
while not done and viewer.is_running():
    viewer.sync()
    ch = getch()

    if ch == 'a':
        data.qpos[joint_idx] -= step_size
        print(f"Joint {joint_idx} decreased: {data.qpos[joint_idx]:.3f}")
    elif ch == 'd':
        data.qpos[joint_idx] += step_size
        print(f"Joint {joint_idx} increased: {data.qpos[joint_idx]:.3f}")
    elif ch == 'w':
        joint_idx = (joint_idx + 1) % 7
        print(f"Selected joint: {joint_idx}")
    elif ch == 's':
        joint_idx = (joint_idx - 1) % 7
        print(f"Selected joint: {joint_idx}")
    elif ch == 'q':
        data.qpos[7] = max(data.qpos[7] - 0.01, 0.0)
        print(f"Gripper closed: {data.qpos[7]:.3f}")
    elif ch == 'e':
        data.qpos[7] = min(data.qpos[7] + 0.01, 0.04)
        print(f"Gripper opened: {data.qpos[7]:.3f}")
    elif ch == 'z':
        pose = data.qpos[:8].copy()
        waypoints.append(pose)
        print(f"✅ Waypoint saved: {np.array2string(pose, precision=4)}")
    elif ch == '\r' or ch == '\n':
        done = True
    else:
        print("Unknown key. Use a/d/w/s/q/e/z/Enter.")

    mujoco.mj_forward(model, data)

# Save waypoints
# Save waypoints to text file (append mode)
save_path = "/home/myrtheiw/octo_ws/octo/record_dataset/teleop_waypoints.txt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "a") as f:
    for pose in waypoints:
        line = ", ".join(f"{x:.6f}" for x in pose)
        f.write(line + "\n")

print(f"\n✅ Appended {len(waypoints)} waypoint(s) to {save_path}")


# Keep viewer open
while viewer.is_running():
    viewer.sync()
