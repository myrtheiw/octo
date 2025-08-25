import mujoco
import mujoco.viewer
import numpy as np

# Load model
model_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Teleop params
joint_idx = 0  # Which joint you are controlling
step_size = 0.05  # How much to increment/decrement

# Launch viewer
viewer = mujoco.viewer.launch_passive(model, data)
import time
time.sleep(0.01)  # Small pause to allow viewer to refresh
viewer.sync()


print("\n--- Teleop tuning helper ---")
print("Use keys to move joints:")
print("  [a] decrease current joint")
print("  [d] increase current joint")
print("  [w] select next joint")
print("  [s] select previous joint")
print("  [Enter] print qpos[:7] and exit")
print("Currently selected joint: 0")

# Simple keyboard read (cross-platform safe)
import sys
import termios
import tty

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
    # Sync viewer
    viewer.sync()

    # Read key
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
    elif ch == '\r' or ch == '\n':
        # Enter pressed → print qpos and exit
        done = True
    else:
        print("Unknown key. Use a/d/w/s/Enter.")

    # Update model
    mujoco.mj_forward(model, data)

# Print final qpos
print("\n--- Final qpos[:7] ---")
print(np.array2string(data.qpos[:7], precision=4, separator=', '))
print("Copy this to target_qpos_arm in your oracle_recordings.py.")

# Wait for user to close viewer
print("Viewer will stay open. Close it manually when done.")
while viewer.is_running():
    viewer.sync()
