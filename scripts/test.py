import numpy as np
import tensorflow_datasets as tfds

builder = tfds.builder_from_directory("/home/myrtheiw/tfds_out/tomato_rlds/0.0.25")
ds = builder.as_dataset(split="train")
for episode in tfds.as_numpy(ds.take(1)):
    actions = episode["steps"]["action"]
    magnitudes = np.linalg.norm(actions, axis=1)
    print("non-zero indices:", np.where(magnitudes > 1e-3)[0][:20])
    print("total non-zero steps:", np.sum(magnitudes > 1e-3))
    break
