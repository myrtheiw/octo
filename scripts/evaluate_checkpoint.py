#!/usr/bin/env python3
"""Offline evaluation for finetuned Octo checkpoints."""

import argparse
import json
import os
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from ml_collections import ConfigDict

from octo.model.octo_model import OctoModel
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import ValidationCallback
from octo.utils.train_utils import (
    TrainState,
    create_optimizer,
    process_text,
)


def _to_config(obj: Any) -> Any:
    """Recursively convert dicts to ConfigDict instances."""

    if isinstance(obj, dict):
        return ConfigDict({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _load_config(exp_dir: str) -> ConfigDict:
    config_path = os.path.join(exp_dir, "finetune_config.json")
    with tf.io.gfile.GFile(config_path, "r") as fh:
        raw = json.load(fh)
    return _to_config(raw)


def _instantiate_text_processor(model_config):
    spec = model_config.get("text_processor")
    if spec is None:
        return None
    return ModuleSpec.instantiate(spec)()


def _process_batch_factory(text_processor):
    def _process(batch):
        batch = process_text(batch, text_processor)
        batch.pop("dataset_name", None)
        return batch

    return _process


def _resolve_modes(modality: str) -> list[str]:
    if modality == "image_conditioned":
        return ["image_conditioned"]
    if modality == "text_conditioned":
        return ["text_conditioned"]
    if modality == "multimodal":
        return ["image_conditioned", "text_conditioned"]
    return ["base"]


def _tree_to_python(tree):
    def convert(x):
        arr = np.asarray(x)
        if arr.shape == ():
            return float(arr)
        return arr.tolist()

    return jax.tree_map(convert, tree)


def _evaluate_with_model(
    config: ConfigDict,
    model: OctoModel,
    override_batches: int | None,
    data_dir_override: str | None,
    quiet: bool,
    output: str | None,
    mc_dropout_samples: int = 1,
) -> dict:
    cfg = ConfigDict(config)

    if override_batches is not None:
        cfg.val_kwargs["num_val_batches"] = override_batches

    if data_dir_override is not None:
        cfg.dataset_kwargs["data_dir"] = data_dir_override

    text_processor = _instantiate_text_processor(model.config)
    process_batch = _process_batch_factory(text_processor)

    def _base_loss_fn(params, batch, rng, train):
        bound = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    if mc_dropout_samples <= 1:

        def loss_fn(params, batch, rng, train=True):  # noqa: D401
            """Single deterministic evaluation (dropout disabled)."""

            return _base_loss_fn(params, batch, rng, train=False)

    else:

        def loss_fn(params, batch, rng, train=True):  # noqa: D401
            """Average metrics across multiple dropout samples."""

            subkeys = jax.random.split(rng, mc_dropout_samples)
            losses = []
            metrics = []
            for sub in subkeys:
                loss, metric = _base_loss_fn(params, batch, sub, train=True)
                losses.append(loss)
                metrics.append(metric)
            stacked_loss = jnp.stack(losses)
            mean_loss = jnp.mean(stacked_loss, axis=0)
            mean_metrics = jax.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
                *metrics,
            )
            return mean_loss, mean_metrics

    tx, _, _ = create_optimizer(
        model.params,
        **cfg.optimizer.to_dict(),
    )
    rng = jax.random.PRNGKey(int(cfg.seed))
    train_state = TrainState.create(rng=rng, model=model, tx=tx)

    modes = _resolve_modes(cfg.modality)
    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=[cfg.dataset_kwargs],
        dataset_kwargs=cfg,
        modes_to_evaluate=modes,
        **cfg.val_kwargs.to_dict(),
    )

    metrics = val_callback(train_state, step=0)
    metrics = _tree_to_python(metrics)

    if not quiet:
        print(json.dumps(metrics, indent=2))
    if output:
        with tf.io.gfile.GFile(output, "w") as fh:
            json.dump(metrics, fh, indent=2)

    return metrics


def evaluate(
    exp_dir: str,
    override_batches: int | None,
    output: str | None,
    data_dir_override: str | None,
    quiet: bool = False,
    mc_dropout_samples: int = 1,
) -> dict:
    exp_dir = os.path.abspath(exp_dir)
    config = _load_config(exp_dir)
    model = OctoModel.load_pretrained(exp_dir)
    return _evaluate_with_model(
        config,
        model,
        override_batches,
        data_dir_override,
        quiet,
        output,
        mc_dropout_samples,
    )


def evaluate_pretrained(
    exp_dir: str,
    pretrained_path: str,
    override_batches: int | None,
    data_dir_override: str | None,
    quiet: bool = False,
    mc_dropout_samples: int = 1,
) -> dict:
    config = _load_config(exp_dir)
    model = OctoModel.load_pretrained(pretrained_path)
    return _evaluate_with_model(
        config,
        model,
        override_batches,
        data_dir_override,
        quiet,
        output=None,
        mc_dropout_samples=mc_dropout_samples,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Octo checkpoint on its validation split.")
    parser.add_argument("--exp", required=True, help="Experiment directory containing the checkpoint")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--num_val_batches",
        type=int,
        default=None,
        help="Override number of validation batches",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Override dataset data_dir from config (useful when checkpoints were saved on another machine)",
    )
    parser.add_argument(
        "--mc_dropout_samples",
        type=int,
        default=1,
        help="Number of dropout-enabled forward passes to average per batch (>=1).",
    )
    args = parser.parse_args()

    if args.mc_dropout_samples < 1:
        raise SystemExit("--mc_dropout_samples must be >= 1")

    evaluate(
        args.exp,
        args.num_val_batches,
        args.output,
        args.data_dir,
        quiet=False,
        mc_dropout_samples=args.mc_dropout_samples,
    )


if __name__ == "__main__":
    main()
