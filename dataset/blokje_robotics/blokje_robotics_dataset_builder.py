import tensorflow_datasets as tfds
import numpy as np
import os
import tensorflow as tf
import cv2  # for PNG decode

class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": {
                        "proprio": tfds.features.Tensor(shape=(50,), dtype=np.float32),  # <-- set to your true size
                        "agentview": tfds.features.Image(shape=(256, 256, 3)),
                        "eye_in_hand": tfds.features.Image(shape=(256, 256, 3)),
                    },
                    "action": tfds.features.Tensor(shape=(None,), dtype=np.float32, encoding='zlib'),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    #"success": tfds.features.Scalar(dtype=np.int32),
                }),
                "language_instruction": tfds.features.Text(),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        tfrec_dir = "/home/myrtheiw/record/dataset/tfrecords"  # must match writer
        tfrec_files = tf.io.gfile.glob(os.path.join(tfrec_dir, "episode_*.tfrecord*"))
        return {"train": self._generate_examples(tfrec_files)}

    def _decode_png(self, png_bytes):
        buf = np.frombuffer(png_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # HxWx3 BGR
        rgb = bgr[:, :, ::-1]
        return rgb  # uint8

    def _generate_examples(self, tfrec_files):
        eid = 0
        for path in sorted(tfrec_files):
            for raw in tf.data.TFRecordDataset(path, compression_type="GZIP").as_numpy_iterator():
                seq = tf.train.SequenceExample.FromString(raw)

                # Context
                language_instruction = seq.context.feature["language_instruction"].bytes_list.value[0].decode("utf-8")
                success_episode = int(seq.context.feature["success"].int64_list.value[0])

                # Feature lists (length T)
                fl = seq.feature_lists.feature_list
                T = len(fl["reward"].feature)

                steps = []
                for t in range(T):
                    proprio = np.array(fl["observation/proprio"].feature[t].float_list.value, dtype=np.float32)
                    agent_png = fl["observation/agentview_png"].feature[t].bytes_list.value[0]
                    eih_png = fl["observation/eye_in_hand_png"].feature[t].bytes_list.value[0]
                    action = np.array(fl["action"].feature[t].float_list.value, dtype=np.float32)
                    reward = float(fl["reward"].feature[t].float_list.value[0])
                    is_terminal = bool(fl["is_terminal"].feature[t].int64_list.value[0])
                    is_first = bool(fl["is_first"].feature[t].int64_list.value[0])
                    is_last = bool(fl["is_last"].feature[t].int64_list.value[0])

                    step_dict = {
                        "observation": {
                            "proprio": proprio,
                            "agentview": self._decode_png(agent_png),
                            "eye_in_hand": self._decode_png(eih_png),
                        },
                        "action": action,
                        "reward": np.float32(reward),
                        "is_terminal": np.bool_(is_terminal),
                        "is_first": np.bool_(is_first),
                        "is_last": np.bool_(is_last),
                        "success": np.int32(success_episode),
                    }
                    steps.append(step_dict)

                yield str(eid), {
                    "steps": steps,
                    "language_instruction": language_instruction,
                }
                eid += 1
