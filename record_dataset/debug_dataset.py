import tensorflow_datasets as tfds
import numpy as np
import cv2

def save_and_inspect_images(image_0, image_1, index):
    cv2.imwrite(f"debug_image_0_{index}.jpg", cv2.cvtColor(image_0, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"debug_image_1_{index}.jpg", cv2.COLOR_RGB2BGR)
    print(f"Saved debug_image_0_{index}.jpg and debug_image_1_{index}.jpg")

def main():
    print("Loading dataset using TFDS registry...")
    ds = tfds.load("test_dataset_builder", split="train", data_dir="/home/myrtheiw/octo_ws/octo/record_dataset/dataset")

    print("Dataset loaded. Displaying and checking first 3 examples:\n")
    for i, example in enumerate(tfds.as_numpy(ds)):

        print(example.keys())
        print("episode_metadata content:", example['episode_metadata'])

        lang_instr = example['episode_metadata']['language_instruction']
        print(f"\nExample {i}: Language instruction -> {lang_instr}")

        steps = example['steps']
        print(f"Example {i} steps type: {type(steps)}, length: {len(steps)}")
        print(f"Example {i} steps value: {steps}")

        if isinstance(steps, dict):
            step_count = len(steps["action"])
            for j in range(min(step_count, 3)):  # only show first 3 steps
                step = {k: steps[k][j] for k in steps}
                print(f"Step {j} keys: {step.keys()}")
                save_and_inspect_images(step["image_0"], step["image_1"], f"{i}_{j}")



        else:
            print(f"No valid steps in example {i}")
        
        if i == 2:
            break

if __name__ == "__main__":
    main()
