import numpy as np
import jax
import tensorflow_datasets as tfds
from PIL import Image
from octo.model.octo_model import OctoModel

# --- SETUP ---
CHECKPOINT_PATH = "/home/myrtheiw/octo_ws/octo/outputs/checkpoint1dec/experiment_20251201_161752"
DATASET_PATH = '/home/myrtheiw/tfds_out'
# Import your env (ensure fixed START_JOINTS are active in sim_env.py)
from envs.tomato_env import PandaTomatoSimEnv, TomatoGymEnv
from scripts import sim_env as sim_env_module
import numpy as np
# Force correct start joints
sim_env_module.START_JOINTS = np.array([
    5.5947731e-19, -4.9482849e-01, 1.0067819e-18, -1.5171977e+00,
    -4.1786848e-20, 1.4901806e+00, 4.3187124e-21
], dtype=float)

def main():
    # 1. Load Model
    print("Loading Model...")
    model = OctoModel.load_pretrained(CHECKPOINT_PATH, step=25000)
    
    # 2. Get Dataset Observation (The "Gold Standard")
    print("Loading Dataset Step...")
    ds = tfds.load('tomato_rlds', split='train', data_dir=DATASET_PATH, shuffle_files=False)
    for ep in tfds.as_numpy(ds.take(1)):
        step = list(ep['steps'])[0]
        # Get raw image (256x256)
        ds_img = step['observation']['image_primary'] 
        # Get raw proprio
        ds_prop = step['observation']['proprio']
        break
        
    # 3. Get Live Env Observation (The "Suspect")
    print("Loading Sim Step...")
    # Setup env (simplified)
    xml_path = "/home/myrtheiw/octo_ws/mujoco_playground/mujoco_playground/external_deps/mujoco_menagerie/franka_emika_panda/scene_dynamic.xml"
    panda_env = PandaTomatoSimEnv(model_xml=xml_path, substeps=40, kp=100.0, kd=20.0)
    env = TomatoGymEnv(panda_env)
    obs, _ = env.reset()
    sim_img = obs['image_primary']
    
    # 4. Force Proprio Match (Eliminate Physics Variable)
    # We overwrite the sim proprio with the dataset proprio to isolate Vision
    sim_prop_matched = ds_prop.copy()

    # 5. Prepare Batch for Model
    # We construct a batch of 2: [Dataset_Obs, Sim_Obs_with_DS_Proprio]
    
    # Helper to norm proprio
    stats = model.dataset_statistics["proprio"]
    mean, std = stats["mean"], stats["std"]
    norm_prop = (ds_prop - mean) / np.maximum(std, 1e-6)
    
    # Create History Window (Horizon=2, duplicating start step)
    # Image: (2, 2, 256, 256, 3)
    batch_images = np.stack([
        np.stack([ds_img, ds_img]),   # Batch 0: Dataset
        np.stack([sim_img, sim_img])  # Batch 1: Sim
    ])
    
    # Proprio: (2, 2, 9) - Both use PERFECT dataset proprio
    batch_proprio = np.stack([
        np.stack([norm_prop, norm_prop]),
        np.stack([norm_prop, norm_prop])
    ])
    
    # Timestep/Masks
    batch_ts = np.zeros((2, 2), dtype=np.int32)
    batch_mask = np.ones((2, 2), dtype=bool)
    
    # Construct Dict
    obs_batch = {
        "image_primary": batch_images,
        "proprio": batch_proprio,
        "timestep": batch_ts,
        # --- FIX: ADD timestep_pad_mask ---
        "timestep_pad_mask": batch_mask,  # Shape (2, 2), True means valid
        # ----------------------------------
        "pad_mask_dict": {
            "image_primary": batch_mask, "proprio": batch_mask, "timestep": batch_mask
        }
    }
    
    # Dummy Task
    task = model.create_tasks(texts=["pick tomato", "pick tomato"])
    
    # 6. Run Model
    print("Running Inference...")
    actions = model.sample_actions(obs_batch, task, rng=jax.random.PRNGKey(0))
    actions = np.array(actions) # (2, Horizon, 7)
    
    act_ds = actions[0, 0]
    act_sim = actions[1, 0]
    
    print("\n--- RESULTS ---")
    print(f"Dataset Action (Should be small): {act_ds}")
    print(f"Sim Action     (The Jerk?):     {act_sim}")
    
    diff = np.linalg.norm(act_ds - act_sim)
    print(f"\nAction Difference (L2 Norm): {diff:.4f}")
    
    # 7. Analyze Image Difference
    img_diff = np.abs(ds_img.astype(float) - sim_img.astype(float))
    mae = np.mean(img_diff)
    print(f"Image Mean Absolute Error (MAE): {mae:.4f} (pixel levels)")
    
    if diff > 0.1:
        print("\n>>> CONCLUSION: The policy sees the images as DIFFERENT.")
        if mae < 5.0:
            print("    The pixel difference is small, but 'Adversarial'.")
        else:
            print("    The pixel difference is large. Check Camera Pose.")
            
        # Save Difference Image
        Image.fromarray(img_diff.astype(np.uint8)).save("debug_diff_heatmap.png")
        print("    Saved 'debug_diff_heatmap.png'. Brighter pixels = Mismatch.")

if __name__ == "__main__":
    main()