import tensorflow as tf

dataset_path = "/home/myrtheiw/octo_ws/octo/dataset/blokje.tfrecord"
raw_dataset = tf.data.TFRecordDataset(dataset_path)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    for key in example.features.feature:
        print(key, example.features.feature[key])