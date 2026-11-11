
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "part1"))
sys.path.insert(0, str(PROJECT / "part3"))
sys.path.insert(0, str(PROJECT / "part4"))

from baseline_model import prepare_compression_datasets
from temperature_optimization import distillation_loss
from pruning_quantization import (
    convert_dynamic,
    evaluate_tflite,
    benchmark_latency,
)

ROOT = PROJECT / "part4"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
CHECKPOINTS = ROOT / "checkpoints" / "combined"

for directory in (RESULTS, MODELS, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

SOURCE = (
    PROJECT / "part3/checkpoints/T_8_extended.keras"
)
TEACHER_CACHE = (
    PROJECT / "part3/cache/teacher_probs_12000.npy"
)

OUTPUT = RESULTS / "combined_compression.json"

SEED = 42
TRAIN_SAMPLES = 12000
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TEMPERATURE = 8.0
ALPHA = 0.7
SPARSITIES = (0.25, 0.50)


def accuracy(model, x, y):
    predictions = model.predict(
        x, batch_size=64, verbose=0
    )
    return float(np.mean(
        np.argmax(predictions, axis=1) == y
    ))


def get_kernel_variables(model):
    return [
        variable
        for variable in model.trainable_variables
        if "kernel" in variable.name.lower()
        and len(variable.shape) in (2, 4)
    ]


def apply_pruning(model, sparsity):
    variables = get_kernel_variables(model)

    if not variables:
        raise RuntimeError("No kernel variables found")

    weights = [v.numpy() for v in variables]
    magnitudes = np.concatenate([
        np.abs(w).reshape(-1) for w in weights
    ])

    count = int(round(magnitudes.size * sparsity))

    if count <= 0:
        threshold = -1.0
    else:
        threshold = float(
            np.partition(magnitudes, count - 1)[count - 1]
        )

    masks = []

    for variable, original in zip(variables, weights):
        mask = (
            np.abs(original) > threshold
        ).astype(np.float32)

        variable.assign(original * mask)
        masks.append(tf.constant(mask))

    total = sum(int(tf.size(v)) for v in variables)
    nonzero = sum(
        int(tf.math.count_nonzero(v))
        for v in variables
    )

    achieved = 1.0 - nonzero / total

    return variables, masks, achieved


def enforce_masks(variables, masks):
    for variable, mask in zip(variables, masks):
        variable.assign(variable * mask)


def fine_tune(model, variables, masks, dataset, x_val, y_val):
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    )

    @tf.function
    def train_step(images, labels, teacher_probs):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)

            loss = distillation_loss(
                labels,
                teacher_probs,
                logits,
                tf.constant(TEMPERATURE, tf.float32),
                tf.constant(ALPHA, tf.float32),
            )

        gradients = tape.gradient(
            loss, model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

        enforce_masks(variables, masks)

        return loss

    history = []

    for epoch in range(EPOCHS):
        losses = []

        for images, labels, teacher_probs in dataset:
            loss = train_step(
                images, labels, teacher_probs
            )
            losses.append(float(loss.numpy()))

        val_accuracy = accuracy(
            model, x_val[:1000], y_val[:1000]
        )

        item = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "validation_accuracy_1000": val_accuracy,
        }

        history.append(item)

        print(
            f"epoch={epoch + 1}/{EPOCHS} "
            f"loss={item['loss']:.4f} "
            f"val_acc={val_accuracy:.2%}",
            flush=True,
        )

    return history


def run_experiment(sparsity, x_train, y_train,
                   teacher_probs, x_val, y_val):
    name = f"kd_prune_{int(sparsity * 100)}"
    checkpoint = CHECKPOINTS / f"{name}.keras"
    tflite_path = MODELS / f"{name}_dynamic.tflite"

    model = tf.keras.models.load_model(
        SOURCE, compile=False
    )

    variables, masks, achieved = apply_pruning(
        model, sparsity
    )

    before_accuracy = accuracy(model, x_val, y_val)

    print(
        f"\n{name}: "
        f"achieved_sparsity={achieved:.2%} "
        f"accuracy_before_finetune={before_accuracy:.2%}",
        flush=True,
    )

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (x_train, y_train, teacher_probs)
        )
        .shuffle(
            TRAIN_SAMPLES,
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    start = time.perf_counter()

    history = fine_tune(
        model, variables, masks,
        dataset, x_val, y_val
    )

    training_seconds = time.perf_counter() - start

    # Confirm sparsity is preserved after fine-tuning.
    total = sum(int(tf.size(v)) for v in variables)
    zeros = sum(
        int(tf.size(v)) - int(tf.math.count_nonzero(v))
        for v in variables
    )
    final_sparsity = zeros / total

    after_accuracy = accuracy(model, x_val, y_val)

    model.save(checkpoint)

    del model

    convert_dynamic(checkpoint, tflite_path)

    evaluation = evaluate_tflite(
        tflite_path, x_val, y_val
    )

    latency = benchmark_latency(
        tflite_path, x_val
    )

    result = {
        "target_sparsity": sparsity,
        "achieved_sparsity": achieved,
        "final_kernel_sparsity": final_sparsity,
        "accuracy_before_finetuning": before_accuracy,
        "keras_accuracy_after_finetuning": after_accuracy,
        "tflite_accuracy": evaluation["accuracy"],
        "size_bytes": tflite_path.stat().st_size,
        "size_mib": (
            tflite_path.stat().st_size / 1024 ** 2
        ),
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint),
        "tflite_path": str(tflite_path),
        "history": history,
        **latency,
    }

    print(
        f"\n{name} RESULTS\n"
        f"Before fine-tuning: {before_accuracy:.2%}\n"
        f"After fine-tuning: {after_accuracy:.2%}\n"
        f"TFLite accuracy: {result['tflite_accuracy']:.2%}\n"
        f"Final kernel sparsity: {final_sparsity:.2%}\n"
        f"TFLite size: {result['size_mib']:.3f} MiB\n"
        f"Median latency: "
        f"{result['median_latency_ms']:.3f} ms",
        flush=True,
    )

    return result


def main():
    if os.environ.get("PART4_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(SEED)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    if not TEACHER_CACHE.exists():
        raise FileNotFoundError(TEACHER_CACHE)

    x_train, y_train, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_train = np.asarray(
        x_train[:TRAIN_SAMPLES], dtype=np.float32
    )
    y_train = np.asarray(
        y_train[:TRAIN_SAMPLES]
    ).reshape(-1)

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    teacher_probs = np.load(TEACHER_CACHE)

    if teacher_probs.shape != (TRAIN_SAMPLES, 100):
        raise ValueError("Teacher cache shape mismatch")

    config = {
        "source": str(SOURCE),
        "train_samples": TRAIN_SAMPLES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "temperature": TEMPERATURE,
        "alpha": ALPHA,
        "sparsities": list(SPARSITIES),
        "seed": SEED,
    }

    results = {
        "configuration": config,
        "experiments": {},
    }

    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text())

        if previous.get("configuration") != config:
            raise RuntimeError(
                "Existing results have different settings"
            )

        results = previous

    for sparsity in SPARSITIES:
        name = f"kd_prune_{int(sparsity * 100)}"

        if name in results["experiments"]:
            print(f"Skipping completed: {name}")
            continue

        result = run_experiment(
            sparsity,
            x_train, y_train, teacher_probs,
            x_val, y_val,
        )

        results["experiments"][name] = result
        OUTPUT.write_text(json.dumps(results, indent=2))

    print("\n=== THREE-TECHNIQUE COMPARISON ===")
    print("Reference KD T=8 + Dynamic TFLite: 37.02%")

    for name, result in results["experiments"].items():
        print(
            f"{name:16s} "
            f"accuracy={result['tflite_accuracy']:.2%} "
            f"sparsity={result['final_kernel_sparsity']:.2%} "
            f"size={result['size_mib']:.3f}MiB "
            f"latency={result['median_latency_ms']:.3f}ms"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
