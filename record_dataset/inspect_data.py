import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

# Default paths matching your oracle script
DEFAULT_TFDS_DIR = "/home/myrtheiw/tfds_out"
DATASET_NAME = "tomato_rlds"
# Try to find the latest version automatically, or specify "0.0.41"
DATASET_VERSION = "0.0.41"

def main():
    parser = argparse.ArgumentParser(description="Replay RLDS dataset to inspect saved data.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_TFDS_DIR, help="Root TFDS output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to inspect (train/val)")
    parser.add_argument("--version", type=str, default=DATASET_VERSION, help="Dataset version string")
    parser.add_argument("--wait", type=int, default=100, help="Wait time in ms per frame (0 = wait for keypress)")
    args = parser.parse_args()

    # Construct full path to the specific version
    full_path = os.path.join(args.data_dir, DATASET_NAME, args.version)
    
    print(f"[-] Looking for dataset at: {full_path}")
    if not os.path.exists(full_path):
        print(f"[!] Error: Directory not found. Please check paths.")
        return

    # Load the dataset using the builder from directory
    try:
        builder = tfds.builder_from_directory(full_path)
    except Exception as e:
        print(f"[!] Failed to load builder: {e}")
        print("    Ensure dataset_info.json exists in the directory.")
        return

    print(f"[-] Loading split: {args.split}")
    ds = builder.as_dataset(split=args.split)

    print("[-] Starting Replay. Press 'q' to quit, 'space' to pause/resume.")
    
    for ep_idx, episode in enumerate(ds):
        steps = list(episode['steps'])
        print(f"\n=== Episode {ep_idx} (Length: {len(steps)}) ===")
        
        # Extract Language Instruction (usually constant per episode)
        lang = steps[0]['observation']['language_instruction'].numpy().decode('utf-8')
        print(f"    Instruction: {lang}")

        for step_idx, step in enumerate(steps):
            # 1. Decode Images
            # TFDS loads as RGB, OpenCV expects BGR
            img_main = step['observation']['image_primary'].numpy()
            img_main = cv2.cvtColor(img_main, cv2.COLOR_RGB2BGR)
            
            img_wrist = step['observation']['image_wrist'].numpy()
            img_wrist = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
            
            # 2. Decode Data
            proprio = step['observation']['proprio'].numpy()
            action = step['action'].numpy()
            reward = step['reward'].numpy()
            is_terminal = step['is_terminal'].numpy()

            # 3. Print stats to console
            # Detect suspicious "Teleport" Actions (if action > 0.5 rads approx 30 deg, likely a bug)
            max_act = np.max(np.abs(action))
            status_tag = "[OK]"
            if max_act > 0.5: 
                status_tag = "[!!! EXPLOSIVE !!!]"
            
            print(f"    Step {step_idx:03d} | Action (max Δ): {max_act:.4f} {status_tag} | Reward: {reward}")

            # 4. Visualization Overlay
            # Combine images side-by-side
            h_main, w_main = img_main.shape[:2]
            h_wrist, w_wrist = img_wrist.shape[:2]
            
            # Create a canvas
            canvas = np.zeros((max(h_main, h_wrist), w_main + w_wrist, 3), dtype=np.uint8)
            canvas[:h_main, :w_main] = img_main
            canvas[:h_wrist, w_main:w_main+w_wrist] = img_wrist

            # Add Text Overlay
            cv2.putText(canvas, f"Ep {ep_idx} Step {step_idx}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, f"Action Max: {max_act:.4f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(canvas, f"Lang: {lang[:30]}...", (10, h_main - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Dataset Inspector", canvas)

            # Keyboard controls
            key = cv2.waitKey(args.wait) & 0xFF
            if key == ord('q'):
                print("[-] Quitting...")
                return
            elif key == ord(' '):
                cv2.waitKey(0) # Pause until key press

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()