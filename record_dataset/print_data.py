import tensorflow as tf
import os

data_dir = '/home/myrtheiw/octo_ws/octo/record_dataset/dataset'

for filename in os.listdir(data_dir):
    if filename.endswith(".tfrecord"):
        print(f"\nInspecting: {filename}")
        path = os.path.join(data_dir, filename)
        for raw_record in tf.data.TFRecordDataset(path).take(1):
            try:
                seq_ex = tf.train.SequenceExample()
                seq_ex.ParseFromString(raw_record.numpy())
                print("Context keys:", seq_ex.context.feature.keys())
            except Exception as e:
                print("Could not parse:", e)
