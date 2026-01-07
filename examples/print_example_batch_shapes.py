import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from absl import app, flags, logging
import flax
import jax

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path",
    None,
    "Path to finetuned Octo checkpoint directory.",
)
flags.DEFINE_integer(
    "finetuned_step",
    None,
    "Checkpoint step to load. If None, load the latest in finetuned_path.",
)


def _shape_tree(tree):
    return jax.tree_map(lambda x: getattr(x, "shape", None), tree)


def main(_):
    if FLAGS.finetuned_path is None:
        raise ValueError("--finetuned_path is required")

    logging.info("Loading finetuned model from %s", FLAGS.finetuned_path)
    model = OctoModel.load_pretrained(FLAGS.finetuned_path, step=FLAGS.finetuned_step)
    eb = model.example_batch

    obs = eb.get("observation", {})
    task = eb.get("task", {})

    print("\n=== Observation shapes ===")
    print(flax.core.pretty_repr(_shape_tree(obs)))

    if "pad_mask_dict" in obs:
        print("\n--- pad_mask_dict shapes ---")
        print(flax.core.pretty_repr(_shape_tree(obs["pad_mask_dict"])))

    if "timestep_pad_mask" in obs:
        print("\n--- timestep_pad_mask shape ---")
        print(obs["timestep_pad_mask"].shape)

    if "task_completed" in obs:
        print("\n--- task_completed shape ---")
        print(obs["task_completed"].shape)

    print("\n=== Task shapes ===")
    print(flax.core.pretty_repr(_shape_tree(task)))

    print("\nDone.")


if __name__ == "__main__":
    app.run(main)
