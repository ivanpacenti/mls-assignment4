"""Part 1: Structured and unstructured pruning (original assignment signatures)."""
from __future__ import annotations
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import copy
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if os.getenv("PART1_CPU_ONLY") == "1":
    tf.config.set_visible_devices([], "GPU")


def _eligible(model):
    """Prunable kernels, including all nested backbone convolutional kernels."""
    variables = []
    for layer in model._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D,
                              tf.keras.layers.Dense)):
            variables.extend(
                weight for weight in layer.weights
                if "kernel" in weight.name and len(weight.shape) in (2, 4)
            )
    return variables


def _fresh(model):
    replica = tf.keras.models.clone_model(model)
    replica.set_weights(model.get_weights())
    replica.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-5),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return replica


def _mask_at_sparsity(weights, target, old_masks=None):
    """Monotonic global magnitude selection; zeroed weights never revive."""
    n = sum(int(np.prod(w.shape)) for w in weights)
    target_count = min(n, round(n * target))
    masks = ([np.ones(w.shape, dtype=np.float32) for w in weights] if old_masks is None
             else [m.copy() for m in old_masks])
    already = n - sum(int(m.sum()) for m in masks)
    extra = max(0, target_count - already)
    if extra:
        candidates = []
        for w, m in zip(weights, masks):
            arr = np.abs(w.numpy()).ravel()
            active = np.flatnonzero(m.ravel() > 0)
            candidates.append(arr[active])
        combined = np.concatenate(candidates)
        if extra >= len(combined):
            return [np.zeros_like(m) for m in masks]
        threshold_ids = np.argpartition(combined, extra - 1)[:extra]
        selected = np.zeros(len(combined), dtype=bool)
        selected[threshold_ids] = True
        start = 0
        for i, m in enumerate(masks):
            flat = m.ravel()
            active = np.flatnonzero(flat)
            flat[active[selected[start:start + len(active)]]] = 0
            start += len(active)
    return masks


class _MaskCallback(tf.keras.callbacks.Callback):
    def __init__(self, owner, target, schedule):
        super().__init__()
        self.owner, self.target, self.schedule = owner, target, schedule
        self.masks = None
        self.epoch_sparsity = []

    def on_train_begin(self, logs=None):
        self.weights = _eligible(self.model)
        self.masks = [np.ones(v.shape, dtype=np.float32) for v in self.weights]

    def on_epoch_begin(self, epoch, logs=None):
        total_epochs = self.params["epochs"]
        pruning_epochs = max(1, total_epochs - 5)
        progress = min(1.0, (epoch + 1) / pruning_epochs)
        if self.schedule == "constant":
            desired = self.target
        elif self.schedule == "polynomial":
            desired = self.target * (1 - (1 - progress) ** 3)
        elif self.schedule == "gradual":
            desired = self.target * progress
        else:
            raise ValueError("schedule must be polynomial, constant, or gradual")
        self.masks = _mask_at_sparsity(self.weights, desired, self.masks)
        self._enforce()
        self.epoch_sparsity.append(self.owner._sparsity(self.weights))

    def _enforce(self):
        for v, mask in zip(self.weights, self.masks):
            v.assign(v * tf.convert_to_tensor(mask, dtype=v.dtype))

    def on_train_batch_end(self, batch, logs=None):
        self._enforce()


class PruningComparator:
    def __init__(self, base_model_path):
        self.base_model = tf.keras.models.load_model(base_model_path)
        self.pruning_results = {}
        from sys import path as sys_path
        if str(ROOT) not in sys_path:
            sys_path.insert(0, str(ROOT))
        from baseline_model import prepare_compression_datasets
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, _ = prepare_compression_datasets()
        self.epochs = 15
        self.batch_size = 64
        self.sample_count = 45000
        self.val_count = 5000
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "checkpoints").mkdir(exist_ok=True)

    @staticmethod
    def _sparsity(weights):
        count = sum(int(tf.size(w)) for w in weights)
        return float(sum(int(tf.reduce_sum(tf.cast(tf.equal(w, 0), tf.int32))) for w in weights) / count) if count else 0.

    def _evaluate(self, model, size=None):
        n = size or self.val_count
        return float(model.evaluate(self.x_val[:n], self.y_val[:n],
                                    batch_size=128, verbose=0)[1])

    def _fit(self, model, callback=None, epochs=None):
        callbacks = [callback] if callback is not None else []
        start = time.perf_counter()
        history = model.fit(self.x_train[:self.sample_count].astype(np.float32),
                            self.y_train[:self.sample_count],
                            validation_data=(self.x_val[:self.val_count], self.y_val[:self.val_count]),
                            batch_size=self.batch_size, epochs=epochs or self.epochs,
                            callbacks=callbacks, verbose=2)
        return history.history, time.perf_counter() - start

    @staticmethod
    def _layer_analysis(model):
        """Measure sparsity for all prunable layer types."""
        out = {}

        eligible_types = (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.Dense,
        )

        for layer in model._flatten_layers(
                include_self=False,
                recursive=True,
        ):
            if not isinstance(layer, eligible_types):
                continue

            for kernel in layer.weights:
                if "kernel" not in kernel.name:
                    continue

                if len(kernel.shape) not in (2, 4):
                    continue

                weights = kernel.numpy()
                total = int(weights.size)
                nonzero = int(np.count_nonzero(weights))

                out[layer.name] = {
                    "type": type(layer).__name__,
                    "sparsity": float(1 - nonzero / total),
                    "nonzero": nonzero,
                    "total": total,
                }

        return out

    def magnitude_based_pruning(self, target_sparsity=0.7, pruning_schedule='polynomial'):
        if not 0 <= target_sparsity < 1:
            raise ValueError("target_sparsity must be in [0, 1)")
        if pruning_schedule not in ("polynomial", "constant", "gradual"):
            raise ValueError("Unsupported pruning schedule")
        model = _fresh(self.base_model)
        callback = _MaskCallback(self, target_sparsity, pruning_schedule)
        history, seconds = self._fit(model, callback)
        analysis = self._layer_analysis(model)
        result = {"model": model, "final_accuracy": self._evaluate(model),
                  "sparsity_achieved": self._sparsity(_eligible(model)),
                  "layer_sparsity_analysis": analysis,
                  "training_history": {**history, "sparsity": callback.epoch_sparsity,
                                       "seconds": seconds}}
        self.pruning_results[f"magnitude_{pruning_schedule}"] = result
        return result

    def structured_pruning(self, target_reduction=0.5, importance_metric='l1_norm'):
        """Physically shrink compression_conv, its BN and downstream Dense classifier.

        EfficientNet's internal residual branches remain intact. The removable adapter
        channels give a safe and real structural reduction of the deployment graph.
        """
        if not 0 < target_reduction < 1:
            raise ValueError("target_reduction must be in (0, 1)")
        if importance_metric not in ('l1_norm', 'l2_norm', 'gradient', 'taylor'):
            raise ValueError("Unsupported importance metric")
        baseline = _fresh(self.base_model)
        conv = baseline.get_layer("compression_conv")
        bn = baseline.get_layer("compression_bn")
        dense = baseline.get_layer("classifier")
        kernel = conv.get_weights()[0]
        old_filters = kernel.shape[-1]
        keep_count = max(1, round(old_filters * (1 - target_reduction)))
        if importance_metric == 'l1_norm':
            importance = np.sum(np.abs(kernel), axis=(0, 1, 2))
        elif importance_metric == 'l2_norm':
            importance = np.sqrt(np.sum(kernel ** 2, axis=(0, 1, 2)))
        else:
            with tf.GradientTape() as tape:
                x = tf.cast(self.x_train[:32], tf.float32)
                pred = baseline(x, training=False)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    self.y_train[:32], pred))
            grad = tape.gradient(loss, conv.trainable_weights[0])
            if grad is None:
                raise RuntimeError("Could not calculate adapter gradients")
            grad = grad.numpy()
            importance = np.sum(np.abs(grad if importance_metric == 'gradient'
                                       else grad * kernel), axis=(0, 1, 2))
        keep = np.sort(np.argsort(importance)[-keep_count:])
        backbone = next(layer for layer in baseline.layers if isinstance(layer, tf.keras.Model))
        # The backbone instance belongs exclusively to this cloned model.
        inputs = tf.keras.Input(shape=self.base_model.input_shape[1:])
        resize = baseline.get_layer("resize")
        x = tf.keras.layers.Resizing(resize.height, resize.width, name="resize")(inputs)
        x = backbone(x, training=False)
        x = tf.keras.layers.Conv2D(keep_count, 1, use_bias=False,
                                   name="compression_conv")(x)
        x = tf.keras.layers.BatchNormalization(name="compression_bn")(x)
        x = tf.keras.layers.Activation("relu", name="compression_relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="compression_pool")(x)
        x = tf.keras.layers.Dropout(
            baseline.get_layer("compression_dropout").rate, name="compression_dropout"
        )(x)
        outputs = tf.keras.layers.Dense(dense.units, activation="softmax", name="classifier")(x)
        compact = tf.keras.Model(inputs, outputs)
        compact.get_layer("compression_conv").set_weights([kernel[..., keep]])
        compact.get_layer("compression_bn").set_weights([a[keep] for a in bn.get_weights()])
        old_dense_weights = dense.get_weights()
        compact.get_layer("classifier").set_weights([old_dense_weights[0][keep, :], old_dense_weights[1]])
        compact.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-5),
                        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history, seconds = self._fit(compact)
        original_params = self.base_model.count_params()
        compact_params = compact.count_params()
        result = {"model": compact,
                  "filters_removed_per_layer": {"compression_conv": int(old_filters - keep_count)},
                  "architecture_changes": {"compression_conv": [int(old_filters), int(keep_count)],
                                           "classifier_input": [int(old_filters), int(keep_count)],
                                           "original_parameters": int(original_params),
                                           "remaining_parameters": int(compact_params),
                                           "importance_metric": importance_metric,
                                           "selected_channels": keep.tolist(),
                                           "history": history, "seconds": seconds},
                  "final_accuracy": self._evaluate(compact),
                  "model_size_reduction": float(1 - compact_params / original_params)}
        self.pruning_results["structured"] = result
        return result

    def gradual_vs_oneshot_pruning(self, target_sparsity=0.5):
        gradual = self.magnitude_based_pruning(target_sparsity, pruning_schedule="gradual")
        oneshot = self.magnitude_based_pruning(target_sparsity, pruning_schedule="constant")
        gh, oh = gradual["training_history"], oneshot["training_history"]
        comparison = {
            "gradual_pruning": {"final_accuracy": gradual["final_accuracy"],
                                "training_seconds": gh["seconds"], "history": gh},
            "oneshot_pruning": {"final_accuracy": oneshot["final_accuracy"],
                                "training_seconds": oh["seconds"], "history": oh},
            "convergence_analysis": {"gradual_val_accuracy": gh["val_accuracy"],
                                     "oneshot_val_accuracy": oh["val_accuracy"]},
            "stability_metrics": {"gradual_val_loss_std": float(np.std(gh["val_loss"])),
                                  "oneshot_val_loss_std": float(np.std(oh["val_loss"]))}}
        self.pruning_results["gradual_vs_oneshot"] = comparison
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(gh["val_accuracy"], label="Gradual")
        ax.plot(oh["val_accuracy"], label="One-shot")
        ax.set(xlabel="Epoch", ylabel="Validation accuracy", title="Pruning convergence")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "convergence.png", dpi=160)
        plt.close(fig)
        return comparison

    def analyze_pruning_sensitivity(self):
        levels = [0.25, 0.5, 0.7]
        base_acc = self._evaluate(self.base_model, size=500)
        # Select the adapter and classifier, plus representative backbone Conv2D layers.
        layers = [l for l in self.base_model._flatten_layers(include_self=False, recursive=True)
                  if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))]
        # Avoid hundreds of full forward-pass evaluations; samples across the backbone.
        selected = layers[::max(1, len(layers) // 8)][:8]
        for important_name in ("compression_conv", "classifier"):
            found = next((l for l in layers if l.name == important_name), None)
            if found is not None and all(l.name != important_name for l in selected):
                selected.append(found)
        results = {}
        for layer in selected:
            drops = []
            for level in levels:
                model = _fresh(self.base_model)
                match = next(l for l in model._flatten_layers(include_self=False, recursive=True)
                             if l.name == layer.name and isinstance(l, type(layer)))
                kernel = next((w for w in match.weights if "kernel" in w.name and len(w.shape) in (2, 4)), None)
                if kernel is None:
                    drops.append(float("nan"))
                    continue
                mask = _mask_at_sparsity([kernel], level)[0]
                kernel.assign(kernel * mask)
                drops.append(float(base_acc - self._evaluate(model, size=500)))
            results[layer.name] = drops
        matrix = np.array(list(results.values()))
        fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(results))))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(results)), list(results))
        ax.set_xticks(range(len(levels)), [str(x) for x in levels])
        ax.set(xlabel="Layer sparsity", title="Accuracy drop (baseline minus pruned)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "sensitivity_heatmap.png", dpi=160)
        plt.close(fig)
        self.pruning_results["sensitivity"] = {"levels": levels, "accuracy_drop": results,
                                               "baseline_accuracy": base_acc}
        return self.pruning_results["sensitivity"]

    def lottery_ticket_hypothesis_test(self):
        init_path = ROOT / "checkpoints" / "baseline_initial.weights.h5"
        if not init_path.exists():
            raise FileNotFoundError(
                f"Initial weights required: {init_path}"
            )

        initial = _fresh(self.base_model)
        initial.load_weights(init_path)

        trained = _fresh(self.base_model)
        trained_vars = _eligible(trained)
        initial_vars = _eligible(initial)

        if len(trained_vars) != len(initial_vars):
            raise RuntimeError("Model architectures do not match")

        masks = [
            np.ones(v.shape, dtype=np.float32)
            for v in trained_vars
        ]

        rng = np.random.default_rng(42)
        rounds = []

        class FixedMask(tf.keras.callbacks.Callback):
            def __init__(self, masks):
                super().__init__()
                self.masks = masks

            def _apply(self):
                for var, mask in zip(_eligible(self.model), self.masks):
                    var.assign(var * mask)

            def on_train_begin(self, logs=None):
                self._apply()

            def on_train_batch_end(self, batch, logs=None):
                self._apply()

            def on_epoch_end(self, epoch, logs=None):
                self._apply()

        def train_ticket(ticket_masks):
            ticket = _fresh(initial)

            variables = _eligible(ticket)
            for var, mask in zip(variables, ticket_masks):
                var.assign(var * mask)

            history, seconds = self._fit(
                ticket,
                callback=FixedMask(ticket_masks),
                epochs=3,
            )

            accuracy = self._evaluate(ticket)
            sparsity = self._sparsity(_eligible(ticket))

            return ticket, {
                "accuracy": accuracy,
                "sparsity": sparsity,
                "training_seconds": seconds,
                "validation_accuracy_history": history["val_accuracy"],
                "validation_loss_history": history["val_loss"],
            }

        def make_random_masks(reference_masks):
            random_masks = []

            for reference in reference_masks:
                flat = np.zeros(reference.size, dtype=np.float32)
                keep_count = int(np.count_nonzero(reference))

                if keep_count:
                    indices = rng.choice(
                        reference.size,
                        size=keep_count,
                        replace=False,
                    )
                    flat[indices] = 1.0

                random_masks.append(flat.reshape(reference.shape))

            return random_masks

        for iteration, target_sparsity in enumerate(
                [0.20, 0.36, 0.488, 0.5904],
                start=1,
        ):
            print(
                f"\nLTH iteration {iteration}/4 "
                f"(target sparsity: {target_sparsity:.2%})",
                flush=True,
            )

            masks = _mask_at_sparsity(
                trained_vars,
                target_sparsity,
                masks,
            )

            print("Training magnitude ticket...", flush=True)
            ticket, ticket_result = train_ticket(masks)

            print("Training random-mask control...", flush=True)
            random_masks = make_random_masks(masks)
            _, random_result = train_ticket(random_masks)

            rounds.append({
                "iteration": iteration,
                "target_sparsity": target_sparsity,
                "magnitude_ticket": ticket_result,
                "random_mask_control": random_result,
                "accuracy_difference_pp": (
                                                  ticket_result["accuracy"]
                                                  - random_result["accuracy"]
                                          ) * 100,
            })

            trained_vars = _eligible(ticket)

            print(
                f"Magnitude: {ticket_result['accuracy']:.2%} | "
                f"Random: {random_result['accuracy']:.2%}",
                flush=True,
            )

        result = {
            "baseline_accuracy": self._evaluate(self.base_model),
            "training_epochs_per_model": 3,
            "random_seed": 42,
            "rounds": rounds,
        }

        self.pruning_results["lottery_ticket"] = result
        return result


# Entry point for Part 1. Run this file directly.
def _save(name, result):
    def convert(value):
        if isinstance(value, tf.keras.Model):
            path = ROOT / "checkpoints" / f"{name}.keras"
            value.save(path)
            return str(path)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Unable to serialize {type(value)}")
    with (ROOT / "results" / f"{name}.json").open("w") as f:
        json.dump(result, f, default=convert, indent=2)


def main():
    """Run one experiment or the full Part 1 from this file."""
    import argparse

    parser = argparse.ArgumentParser(description="CIFAR-100 Part 1 pruning experiments")
    parser.add_argument(
        "--experiment",
        choices=("magnitude", "structured", "comparison", "sensitivity", "lottery", "all"),
        default="all",
    )
    args = parser.parse_args()
    base_path = ROOT / "baseline_model.keras"
    if not base_path.exists():
        raise FileNotFoundError(f"Train the baseline first: {base_path}")
    comparator = PruningComparator(str(base_path))
    experiments = {
        "magnitude": lambda: comparator.magnitude_based_pruning(
    target_sparsity=0.5,
    pruning_schedule="polynomial",
),
        "structured": comparator.structured_pruning,
        "comparison": comparator.gradual_vs_oneshot_pruning,
        "sensitivity": comparator.analyze_pruning_sensitivity,
        "lottery": comparator.lottery_ticket_hypothesis_test,
    }
    selected = experiments if args.experiment == "all" else {
        args.experiment: experiments[args.experiment]
    }
    for name, experiment in selected.items():
        print(f"Running {name} experiment...", flush=True)
        result = experiment()
        _save(name, result)
        print(f"Saved {name} results.", flush=True)
    print("Part 1 complete. See part1/results/ and part1/checkpoints/.")


if __name__ == "__main__":
    main()
