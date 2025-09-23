import os, tensorflow as tf, numpy as np

vdir  = "/home/myrtheiw/tfds_out/tomato_rlds/0.0.7"
shard = os.path.join(vdir, "tomato_rlds-train.tfrecord-00000-of-00001")

def peek(ds, max_n=1):
    it = iter(ds)
    out = []
    for _ in range(max_n):
        try: out.append(next(it).numpy())
        except StopIteration: break
    return out

def dump_example(buf):
    try:
        ex = tf.train.Example.FromString(buf)
        f = ex.features.feature
        got = False
        print("== Example features ==")
        for k, v in f.items():
            n_bytes  = len(v.bytes_list.value)
            n_float  = len(v.float_list.value)
            n_int64  = len(v.int64_list.value)
            if n_bytes or n_float or n_int64:
                got = True
                print(f"{k:45s} bytes:{n_bytes:3d}  float:{n_float:3d}  int64:{n_int64:3d}")
        return got
    except Exception:
        return False

def dump_seqexample(buf):
    try:
        sex = tf.train.SequenceExample.FromString(buf)
        fl = sex.feature_lists.feature_list
        got = False
        print("== SequenceExample feature_lists ==")
        for k, flist in fl.items():
            n_bytes = sum(len(feat.bytes_list.value)  for feat in flist.feature)
            n_float = sum(len(feat.float_list.value)  for feat in flist.feature)
            n_int64 = sum(len(feat.int64_list.value) for feat in flist.feature)
            if n_bytes or n_float or n_int64:
                got = True
                print(f"{k:45s} bytes:{n_bytes:3d}  float:{n_float:3d}  int64:{n_int64:3d}")
        return got
    except Exception:
        return False

def try_one(compression=None, label="(raw)"):
    print(f"\n---- Reading {label} ----")
    ds = tf.data.TFRecordDataset(shard, compression_type=compression) if compression else tf.data.TFRecordDataset(shard)
    bufs = peek(ds, 1)
    if not bufs:
        print("No records")
        return
    buf = bufs[0]
    if not dump_example(buf):
        if not dump_seqexample(buf):
            print("Could not decode as Example or SequenceExample.")

try_one(None, "uncompressed")
try_one("GZIP", "GZIP")
try_one("ZLIB", "ZLIB")
