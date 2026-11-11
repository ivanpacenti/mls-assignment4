
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from temperature_optimization import (
    PROJECT,
    RESULTS,
    CHECKPOINTS,
    CACHE,
    create_student,
    distillation_loss,
    evaluate,
    prepare_compression_datasets,
    ALPHA,
    SEED,
    TRAIN_SAMPLES,
)

EXTRA_EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 3e-4

TRIALS = {
    "scratch": None,
    "T_1": 1.0,
    "T_2": 2.0,
    "T_4": 4.0,
    "T_8": 8.0,
}


def continue_trial(name, temperature, dataset, x_val, y_val):
    source = CHECKPOINTS / f"{name}.keras"
    destination = CHECKPOINTS / f"{name}_extended.keras"

    if not source.exists():
        raise FileNotFoundError(source)

    model = tf.keras.models.load_model(source, compile=False)

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    )

    alpha = 0.0 if temperature is None else ALPHA
    t = 1.0 if temperature is None else temperature

    @tf.function
    def train_step(images, labels, teacher_probs):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)

            loss = distillation_loss(
                labels,
                teacher_probs,
                logits,
                tf.constant(t, dtype=tf.float32),
                tf.constant(alpha, dtype=tf.float32),
            )

        gradients = tape.gradient(
            loss, model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )

        return loss

    best_accuracy = -1.0
    history = []
    start = time.perf_counter()

    for epoch in range(EXTRA_EPOCHS):
        losses = []

        for images, labels, teacher_probs in dataset:
            loss = train_step(images, labels, teacher_probs)
            losses.append(float(loss.numpy()))

        accuracy = evaluate(
            model, x_val[:1000], y_val[:1000]
        )

        history.append({
            "additional_epoch": epoch + 1,
            "total_epoch": epoch + 5,
            "training_loss": float(np.mean(losses)),
            "validation_accuracy_1000": accuracy,
        })

        print(
            f"{name:10s} "
            f"epoch={epoch + 5}/16 "
            f"loss={np.mean(losses):.4f} "
            f"val_acc={accuracy:.2%}",
            flush=True,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save(destination)

    best_model = tf.keras.models.load_model(
        destination, compile=False
    )

    full_accuracy = evaluate(best_model, x_val, y_val)

    print(
        f"{name} extended full accuracy: {full_accuracy:.2%}"
    )

    return {
        "temperature": temperature,
        "additional_epochs": EXTRA_EPOCHS,
        "best_validation_accuracy_1000": best_accuracy,
        "full_validation_accuracy": full_accuracy,
        "additional_training_seconds": (
            time.perf_counter() - start
        ),
        "checkpoint": str(destination),
        "history": history,
    }


def main():
    if os.environ.get("PART3_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(SEED)

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

    teacher_probs = np.load(
        CACHE / f"teacher_probs_{TRAIN_SAMPLES}.npy"
    )

    assert teacher_probs.shape == (
        TRAIN_SAMPLES, 100
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

    results_path = RESULTS / "temperature_extended.json"

    results = {
        "original_epochs": 4,
        "additional_epochs": EXTRA_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_samples": TRAIN_SAMPLES,
        "trials": {},
    }

    if results_path.exists():
        previous = json.loads(results_path.read_text())

        for key in (
            "original_epochs",
            "additional_epochs",
            "learning_rate",
            "train_samples",
        ):
            if previous.get(key) != results[key]:
                raise RuntimeError(
                    "Existing results use different settings"
                )

        results = previous

    for name, temperature in TRIALS.items():
        if name in results["trials"]:
            print(f"Skipping completed trial: {name}")
            continue

        print(f"\nContinuing {name}", flush=True)

        results["trials"][name] = continue_trial(
            name, temperature, dataset, x_val, y_val
        )

        results_path.write_text(
            json.dumps(results, indent=2)
        )

    print("\n=== EXTENDED TEMPERATURE RESULTS ===")

    for name, result in results["trials"].items():
        print(
            f"{name:10s} "
            f"accuracy={result['full_validation_accuracy']:.2%}"
        )

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
