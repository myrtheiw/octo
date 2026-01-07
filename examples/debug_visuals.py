import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from pathlib import Path

# --- Imports from your codebase ---
# Ensure these match your file structure
from envs.tomato_env import PandaTomatoSimEnv, TomatoGymEnv
from scripts import sim_env as sim_env_module
from record_dataset.helpers import build_and_load_scene, scene_dir_from_model_path

# 1. SETUP: Define Paths & Config
DATA_DIR = '/home/myrtheiw/tfds_out'
XML_PATH = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene.xml"

# Force the EXACT start joints we found earlier
CORRECT_START_JOINTS = np.array([
    5.5947731e-19, -4.9482849e-01, 1.0067819e-18, -1.5171977e+00,
    -4.1786848e-20, 1.4901806e+00, 4.3187124e-21
], dtype=float)

# Inject into sim_env module to ensure it's used
sim_env_module.START_JOINTS = CORRECT_START_JOINTS

def get_dataset_image():
    print(f"Loading dataset from {DATA_DIR}...")
    ds = tfds.load('tomato_rlds', split='train', data_dir=DATA_DIR, shuffle_files=False)
    for episode in ds.take(1):
        steps = list(episode['steps'])
        # Get the first image (256x256)
        img = steps[0]['observation']['image_primary'].numpy()
        
        # Apply the SAME slicing transform the model uses
        # (This simulates what the model sees during training)
        # If your model expects 256x256, keep as is. 
        # If it expects 128x128, apply the slice:
        # img = img[::2, ::2] 
        return img

def get_sim_image():
    print("Setting up Simulation...")
    # Regenerate scene to match 03_eval logic
    build_and_load_scene(XML_PATH)
    scene_dir = Path(scene_dir_from_model_path(XML_PATH))
    dynamic_xml_path = str(scene_dir / "scene_dynamic.xml")
    
    print(f"Loading XML: {dynamic_xml_path}")
    
    panda_env = PandaTomatoSimEnv(
        model_xml=dynamic_xml_path,
        substeps=40,
        kp=100.0, kd=20.0
    )
    env = TomatoGymEnv(panda_env)
    
    # Reset and capture
    obs, _ = env.reset()
    img = obs['image_primary'] # Should be 256x256 or 128x128 depending on tomato_env config
    return img

def main():
    # 1. Get Images
    ds_img = get_dataset_image()
    sim_img = get_sim_image()
    
    print(f"Dataset Image Shape: {ds_img.shape}")
    print(f"Sim Image Shape:     {sim_img.shape}")
    
    # 2. Check for Sizing Mismatch
    if ds_img.shape != sim_img.shape:
        print("!! SHAPE MISMATCH DETECTED !!")
        print("Resize/Slicing logic in tomato_env.py is likely different from dataset loader.")
        # Resize for visualization
        ds_img = np.array(Image.fromarray(ds_img).resize((256, 256)))
        sim_img = np.array(Image.fromarray(sim_img).resize((256, 256)))

    # 3. Create Comparison
    # Concatenate side-by-side
    combined = np.concatenate([ds_img, sim_img], axis=1)
    
    # 4. Save
    save_path = "visual_debug_comparison.png"
    Image.fromarray(combined).save(save_path)
    print(f"\nSAVED COMPARISON TO: {save_path}")
    print("---------------------------------------------------")
    print("LEFT:  Dataset (What the model expects)")
    print("RIGHT: Simulation (What the model sees now)")
    print("---------------------------------------------------")
    print("CHECK FOR:")
    print("1. Camera Angle: Is the plant centered exactly the same?")
    print("2. Distance: Does the robot look closer/further?")
    print("3. Lighting: Is the shadow in the same place?")
    print("4. Sharpness: Is one blurry and the other pixelated? (Slicing vs Resizing)")

if __name__ == "__main__":
    main()