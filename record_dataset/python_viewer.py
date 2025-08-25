import mujoco
import mujoco_viewer
import numpy as np
import sys
import termios
import tty
import time
import select

# === Load model ===
model_path = "/home/myrtheiw/octo_ws/mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# === Launch viewer with UI menus/sliders enabled ===
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=False)

# === Teleop config ===
joint_idx = 0              # start with joint 0
step_size = 0.05           # how much to move joint per key
max_joint_idx = 9          # 7 joints + 2 fingers

print("""
TELEOP CONTROLS:
  w → next joint
  s → previous joint
  a → decrease joint angle
  d → increase joint angle
  q → quit and print qpos[:7]
Currently selected joint: 0
""")

# === Simple getch for keypress reading ===
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# === Main loop ===
try:
    while True:
        # Advance kinematics (no physics step)
        mujoco.mj_forward(model, data)
        viewer.render()

        # Check for key press (non-blocking)
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            ch = getch()
            if ch == 'a':
                data.qpos[joint_idx] -= step_size
                print(f"Joint {joint_idx} ↓ → {data.qpos[joint_idx]:.3f}")
            elif ch == 'd':
                data.qpos[joint_idx] += step_size
                print(f"Joint {joint_idx} ↑ → {data.qpos[joint_idx]:.3f}")
            elif ch == 'w':
                joint_idx = (joint_idx + 1) % max_joint_idx
                print(f"→ Selected joint {joint_idx}")
            elif ch == 's':
                joint_idx = (joint_idx - 1) % max_joint_idx
                print(f"← Selected joint {joint_idx}")
            elif ch == 'q':
                print("Quitting teleop...")
                break
            else:
                print("Unknown key. Use w/s/a/d/q.")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nInterrupted.")
except Exception as e:
    print(f"Error: {e}")

# === After exit: print joint positions ===
print("\n--- Final qpos[:7] ---")
print(np.array2string(data.qpos[:7], precision=4, separator=', '))
print("Copy this to target_qpos_arm in your oracle_recordings.py.")
