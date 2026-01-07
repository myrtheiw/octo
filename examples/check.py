import numpy as np
import os
# Adjust path to your checkpoint
CHECKPOINT_PATH = "/home/myrtheiw/octo_ws/octo/outputs/checkpoint1dec/experiment_20251201_161752"

# Use Octo's loading mechanism (or just load the JSON if available)
# Assuming standard Octo structure, we can load the dataset_statistics.json directly if present,
# or load the model to get them.
from octo.model.octo_model import OctoModel

try:
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = OctoModel.load_pretrained(CHECKPOINT_PATH, step=25000)
    
    stats = model.dataset_statistics["proprio"]
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    
    print("\n=== PROPRIO STATISTICS IN CHECKPOINT ===")
    print(f"Shape: {mean.shape}")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    
    # Calculate what the start pose *should* be to get a norm of 0 (i.e. the mean)
    print("\n=== EXPECTED MEAN POSE (Unnormalized) ===")
    print(mean)
    
    # Calculate what our current start pose (-0.4948, etc.) normalizes to
    # (Using the values from your logs)
    current_start = np.array([
        5.59e-19, -0.4948, 1.00e-18, -1.517, -4.18e-20, 1.49, 4.31e-21, 0.0, 0.0
    ])
    
    norm_val = (current_start - mean) / std
    print("\n=== NORMALIZED START POSE (Calculated) ===")
    print(norm_val)
    print(f"Norm: {np.linalg.norm(norm_val)}")

except Exception as e:
    print(f"Error: {e}")