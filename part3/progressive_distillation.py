
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part3"))

from baseline_model import prepare_compression_datasets
from temperature_optimization import (
    create_student,
    distillation_loss,
    evaluate,
)

PART3 = ROOT / "part3"
CHECKPOINTS = PART3 / "checkpoints" / "progressive"
CACHE = PART3 / "cache"
RESULTS = PART3 / "results"

for path in (CHECKPOINTS, CACHE, RESULTS):
    path.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_SAMPLES = 12000
EPOCHS = 12
BATCH_SIZE = 64
TEMPERATURE = 4.0
ALPHA = 0.7
LEARNING_RATE = 3e-4

RESULT_PATH = RESULTS / "progressive_distillation.json"


def create_intermediate(filters, dense_units, name):
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    for channels in filters:
        for _ in range(2):
            x = tf.keras.layers.Conv2D(
                channels,
                3,
                padding="same",
                use_bias=False,
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        dense_units, activation="relu"
    )(x)

    outputs = tf.keras.layers.Dense(100)(x)
    return tf.keras.Model(inputs, outputs, name=name)


def make_model(kind):
    if kind == "large":
        return create_intermediate(
            (64, 128, 256), 256, "intermediate_large"
        )

    if kind == "medium":
        return create_intermediate(
            (48, 96, 192), 192, "intermediate_medium"
        )

    if kind == "final":
        return create_student()

    raise ValueError(f"Unknown model kind: {kind}")


def make_dataset(x_train, y_train, teacher_probs):
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, teacher_probs)
    )

    dataset = dataset.shuffle(
        len(x_train),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    return dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )


def train_stage(
    name,
    kind,
    teacher_probs,
    x_train,
    y_train,
    x_val,
    y_val,
):
    checkpoint = CHECKPOINTS / f"{name}.keras"

    tf.keras.utils.set_random_seed(SEED)
    model = make_model(kind)

    parameters = int(model.count_params())

    print(
        f"\nStarting {name}: {parameters:,} parameters",
        flush=True,
    )

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    )

    dataset = make_dataset(
        x_train, y_train, teacher_probs
    )

    @tf.function
    def train_step(images, labels, soft_targets):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)

            loss = distillation_loss(
                labels,
                soft_targets,
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

        return loss

    history = []
    best_accuracy = -1.0
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        losses = []

        for images, labels, soft_targets in dataset:
            loss = train_step(
                images, labels, soft_targets
            )
            losses.append(float(loss.numpy()))

        val_accuracy = evaluate(
            model,
            x_val[:1000],
            y_val[:1000],
        )

        mean_loss = float(np.mean(losses))

        history.append({
            "epoch": epoch + 1,
            "loss": mean_loss,
            "validation_accuracy_1000": val_accuracy,
        })

        print(
            f"{name:20s} "
            f"epoch={epoch + 1:02d}/{EPOCHS} "
            f"loss={mean_loss:.4f} "
            f"val_acc={val_accuracy:.2%}",
            flush=True,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save(checkpoint)

    training_seconds = time.perf_counter() - start

    best_model = tf.keras.models.load_model(
        checkpoint, compile=False
    )

    full_accuracy = evaluate(
        best_model, x_val, y_val
    )

    print(
        f"{name} full validation: {full_accuracy:.2%}",
        flush=True,
    )

    return {
        "kind": kind,
        "parameters": parameters,
        "best_validation_accuracy_1000": best_accuracy,
        "full_validation_accuracy": full_accuracy,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint),
        "history": history,
    }


def predict_teacher(model_path, x_train, cache_name):
    cache_path = CACHE / cache_name

    if cache_path.exists():
        predictions = np.load(cache_path)

        if predictions.shape != (len(x_train), 100):
            raise ValueError(
                f"Unexpected cache shape: {cache_path}"
            )

        print(f"Loaded predictions: {cache_path}")
        return predictions

    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    logits = model.predict(
        x_train, batch_size=64, verbose=1
    )

    # Intermediate students produce logits.
    probabilities = tf.nn.softmax(
        logits, axis=-1
    ).numpy().astype(np.float32)

    np.save(cache_path, probabilities)

    print(f"Saved predictions: {cache_path}")

    return probabilities


def main():
    if os.environ.get("PART3_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(SEED)

    x_train, y_train, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_train = np.asarray(
        x_train[:TRAIN_SAMPLES],
        dtype=np.float32,
    )

    y_train = np.asarray(
        y_train[:TRAIN_SAMPLES]
    ).reshape(-1)

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    # Reuse teacher outputs from temperature optimization.
    teacher_cache = (
        CACHE / f"teacher_probs_{TRAIN_SAMPLES}.npy"
    )

    if not teacher_cache.exists():
        raise FileNotFoundError(
            f"Teacher cache not found: {teacher_cache}"
        )

    teacher_probs = np.load(teacher_cache)

    if teacher_probs.shape != (TRAIN_SAMPLES, 100):
        raise ValueError("Unexpected teacher cache shape")

    results = {
        "configuration": {
            "seed": SEED,
            "train_samples": TRAIN_SAMPLES,
            "validation_samples": len(x_val),
            "epochs_per_stage": EPOCHS,
            "batch_size": BATCH_SIZE,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
            "learning_rate": LEARNING_RATE,
        },
        "stages": {},
    }

    if RESULT_PATH.exists():
        previous = json.loads(RESULT_PATH.read_text())

        if previous.get("configuration") != results["configuration"]:
            raise RuntimeError(
                "Existing results use different settings"
            )

        results = previous

    def run_stage(name, kind, targets):
        if name in results["stages"]:
            checkpoint = Path(
                results["stages"][name]["checkpoint"]
            )

            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

            print(f"Skipping completed stage: {name}")
            return

        results["stages"][name] = train_stage(
            name,
            kind,
            targets,
            x_train,
            y_train,
            x_val,
            y_val,
        )

        RESULT_PATH.write_text(
            json.dumps(results, indent=2)
        )

    # Direct teacher -> final student.
    run_stage(
        "direct_final",
        "final",
        teacher_probs,
    )

    # Progressive teacher -> large -> medium -> final.
    run_stage(
        "progressive_large",
        "large",
        teacher_probs,
    )

    large_probs = predict_teacher(
        CHECKPOINTS / "progressive_large.keras",
        x_train,
        "progressive_large_probs.npy",
    )

    run_stage(
        "progressive_medium",
        "medium",
        large_probs,
    )

    medium_probs = predict_teacher(
        CHECKPOINTS / "progressive_medium.keras",
        x_train,
        "progressive_medium_probs.npy",
    )

    run_stage(
        "progressive_final",
        "final",
        medium_probs,
    )

    direct = results["stages"]["direct_final"]
    progressive = results["stages"]["progressive_final"]

    print("\n=== PROGRESSIVE DISTILLATION COMPARISON ===")
    print(
        f"Direct student: "
        f"{direct['full_validation_accuracy']:.2%}"
    )
    print(
        f"Progressive student: "
        f"{progressive['full_validation_accuracy']:.2%}"
    )

    improvement = (
        progressive["full_validation_accuracy"]
        - direct["full_validation_accuracy"]
    ) * 100

    print(f"Progressive improvement: {improvement:+.2f} pp")

    print("\nIntermediate models:")

    for name in (
        "progressive_large",
        "progressive_medium",
    ):
        stage = results["stages"][name]

        print(
            f"{name}: "
            f"{stage['parameters']:,} parameters, "
            f"{stage['full_validation_accuracy']:.2%}"
        )

    results["comparison"] = {
        "direct_accuracy": direct["full_validation_accuracy"],
        "progressive_accuracy": progressive[
            "full_validation_accuracy"
        ],
        "improvement_pp": improvement,
        "direct_training_seconds": direct["training_seconds"],
        "progressive_total_training_seconds": sum(
            results["stages"][name]["training_seconds"]
            for name in (
                "progressive_large",
                "progressive_medium",
                "progressive_final",
            )
        ),
    }

    RESULT_PATH.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
