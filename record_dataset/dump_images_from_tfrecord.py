#!/usr/bin/env python3
import os, argparse, numpy as np, imageio.v2 as imageio, tensorflow as tf

def as_numpy_bytes_list(feat, key):
    return list(feat[key].bytes_list.value) if key in feat and feat[key].bytes_list.value else None

def as_numpy_int(feat, key):
    return np.array(feat[key].int64_list.value, dtype=np.int64) if key in feat else None

def maybe_decode_frame(blob, fallback_shape=None):
    """Return HxWx3 uint8 image from bytes or raw array bytes."""
    try:
        img = imageio.imread(blob)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img
    except Exception:
        if fallback_shape is not None:
            arr = np.frombuffer(blob, dtype=np.uint8)
            expect = np.prod(fallback_shape)
            if arr.size == expect:
                return arr.reshape(*fallback_shape)
        raise

def sample_indices(T, k=20):
    """Evenly spaced indices for sampling T frames down to k."""
    k = max(1, min(k, T))
    return np.linspace(0, T - 1, num=k, dtype=int)

def dump_episode_images(example, out_dir, episode_idx, streams=("image_primary", "image_wrist"),
                        H=256, W=256, n_samples=20):
    feat = example.features.feature
    is_first = as_numpy_int(feat, "steps/is_first")
    T = int(is_first.size) if is_first is not None else None

    os.makedirs(out_dir, exist_ok=True)

    for s in streams:
        key = f"steps/observation/{s}"
        blobs = as_numpy_bytes_list(feat, key)
        if blobs is None:
            for alt in (f"{key}/encoded", f"{key}:0", f"{key}/bytes"):
                blobs = as_numpy_bytes_list(feat, alt)
                if blobs is not None:
                    key = alt
                    break
        if blobs is None:
            print(f"[warn] stream '{s}' not found; skipping.")
            continue

        if T is None:
            T = len(blobs)
        idxs = sample_indices(len(blobs), k=n_samples)

        stream_dir = os.path.join(out_dir, f"ep{episode_idx:04d}_{s}")
        os.makedirs(stream_dir, exist_ok=True)

        for out_i, t in enumerate(idxs):
            try:
                img = maybe_decode_frame(blobs[t], fallback_shape=(H, W, 3))
                imageio.imwrite(os.path.join(stream_dir, f"{out_i:05d}.png"), img)
            except Exception as e:
                print(f"[warn] failed to decode step {t} for '{s}': {e}")

        print(f"[ok] episode {episode_idx}: wrote {len(idxs)} sampled frames to {stream_dir}")

    # Episode-level goal images
    for gk in ("goal_image_primary", "goal_image_wrist"):
        key = f"steps/observation/{gk}"
        blobs = as_numpy_bytes_list(feat, key)
        if blobs is None:
            for alt in (f"{key}/encoded", f"{key}:0", f"{key}/bytes"):
                blobs = as_numpy_bytes_list(feat, alt)
                if blobs is not None:
                    key = alt
                    break
        if blobs:
            try:
                img = maybe_decode_frame(blobs[0], fallback_shape=(H, W, 3))
                imageio.imwrite(os.path.join(out_dir, f"ep{episode_idx:04d}_{gk}.png"), img)
                print(f"[ok] wrote {gk}")
            except Exception as e:
                print(f"[warn] failed to decode {gk}: {e}")

def iter_examples(shard_path):
    ds = tf.data.TFRecordDataset(shard_path)
    for raw in ds:
        yield tf.train.Example.FromString(raw.numpy())

def main():
    ap = argparse.ArgumentParser("Dump sampled images from a TFRecord shard (no MuJoCo).")
    ap.add_argument("--dataset_path", required=True, help="Path to a TFRecord shard (.tfrecord-xxxxx-of-xxxxx)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--episode", type=int, default=None, help="If set, dump only this episode index (0-based within shard)")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--streams", type=str, default="image_primary,image_wrist",
                    help="Comma-separated observation image keys under steps/observation/")
    ap.add_argument("--n_samples", type=int, default=20, help="Number of frames to sample per episode")
    args = ap.parse_args()

    streams = tuple(s.strip() for s in args.streams.split(",") if s.strip())

    for epi, ex in enumerate(iter_examples(args.dataset_path)):
        if args.episode is None or epi == args.episode:
            dump_episode_images(ex, args.out_dir, epi, streams, args.height, args.width, args.n_samples)
            if args.episode is not None:
                break

if __name__ == "__main__":
    main()
