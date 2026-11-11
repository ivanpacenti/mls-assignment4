
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

from baseline_model import prepare_compression_datasets


ROOT = PROJECT / "part3"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
CACHE = ROOT / "cache"

for directory in (RESULTS, CHECKPOINTS, CACHE):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_SAMPLES = 12000
EPOCHS = 4
BATCH_SIZE = 64
ALPHA = 0.7
TEMPERATURES = [1.0, 2.0, 4.0, 8.0]


def create_student():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    outputs = tf.keras.layers.Dense(100)(x)

    return tf.keras.Model(inputs, outputs, name="student_cnn")


def load_teacher_probabilities(x_train):
    cache_path = CACHE / f"teacher_probs_{len(x_train)}.npy"

    if cache_path.exists():
        probabilities = np.load(cache_path)
        if probabilities.shape == (len(x_train), 100):
            print("Loaded cached teacher predictions")
            return probabilities

    teacher = tf.keras.models.load_model(
        PROJECT / "part1/baseline_model.keras",
        compile=False,
    )
    teacher.trainable = False

    probabilities = teacher.predict(
        x_train, batch_size=32, verbose=1
    ).astype(np.float32)

    if not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-3
    ):
        raise ValueError(
            "Teacher output is not normalized probabilities"
        )

    np.save(cache_path, probabilities)

    del teacher
    tf.keras.backend.clear_session()

    print("Teacher predictions cached")
    return probabilities


def distillation_loss(labels, teacher_probs, student_logits,
                      temperature, alpha):
    labels = tf.cast(labels, tf.int32)

    hard_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, student_logits, from_logits=True
        )
    )

    teacher_log_probs = tf.math.log(
        tf.clip_by_value(teacher_probs, 1e-7, 1.0)
    )

    teacher_soft = tf.nn.softmax(
        teacher_log_probs / temperature, axis=-1
    )

    student_log_soft = tf.nn.log_softmax(
        student_logits / temperature, axis=-1
    )

    soft_loss = tf.reduce_mean(
        tf.reduce_sum(
            teacher_soft * (
                tf.math.log(
                    tf.clip_by_value(teacher_soft, 1e-7, 1.0)
                ) - student_log_soft
            ),
            axis=-1,
        )
    )

    total = (
        (1.0 - alpha) * hard_loss
        + alpha * (temperature ** 2) * soft_loss
    )

    return total


def evaluate(model, images, labels):
    predictions = model.predict(
        images, batch_size=64, verbose=0
    )
    predicted_classes = np.argmax(predictions, axis=1)
    return float(np.mean(predicted_classes == labels))


def train_trial(
    name,
    temperature,
    initial_weights,
    images,
    labels,
    teacher_probs,
    x_val,
    y_val,
):
    tf.keras.utils.set_random_seed(SEED)

    model = create_student()
    model.set_weights(initial_weights)

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=1e-3
    )

    alpha = 0.0 if temperature is None else ALPHA
    temperature_value = (
        1.0 if temperature is None else temperature
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels, teacher_probs)
    )
    dataset = dataset.shuffle(
        min(len(images), 12000),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )

    @tf.function
    def train_step(batch_x, batch_y, batch_teacher):
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)

            loss = distillation_loss(
                batch_y,
                batch_teacher,
                logits,
                tf.constant(temperature_value, tf.float32),
                tf.constant(alpha, tf.float32),
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
    checkpoint = CHECKPOINTS / f"{name}.keras"

    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        losses = []

        for batch_x, batch_y, batch_teacher in dataset:
            loss = train_step(
                batch_x, batch_y, batch_teacher
            )
            losses.append(float(loss.numpy()))

        val_accuracy = evaluate(
            model, x_val[:1000], y_val[:1000]
        )

        mean_loss = float(np.mean(losses))

        history.append({
            "epoch": epoch + 1,
            "training_loss": mean_loss,
            "validation_accuracy_1000": val_accuracy,
        })

        print(
            f"{name:15s} "
            f"epoch={epoch + 1}/{EPOCHS} "
            f"loss={mean_loss:.4f} "
            f"val_acc={val_accuracy:.2%}",
            flush=True,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save(checkpoint)

    training_seconds = time.perf_counter() - start_time

    best_model = tf.keras.models.load_model(
        checkpoint, compile=False
    )

    full_accuracy = evaluate(best_model, x_val, y_val)

    result = {
        "temperature": temperature,
        "alpha": alpha,
        "student_parameters": int(best_model.count_params()),
        "best_validation_accuracy_1000": best_accuracy,
        "full_validation_accuracy": full_accuracy,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint),
        "history": history,
    }

    print(
        f"{name} full validation: {full_accuracy:.2%}",
        flush=True,
    )

    del best_model, model, optimizer

    return result


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

    teacher_probs = load_teacher_probabilities(x_train)

    tf.keras.utils.set_random_seed(SEED)
    initial_student = create_student()

    initial_weights = [
        weight.copy() for weight in initial_student.get_weights()
    ]

    print(
        "Student parameters:",
        initial_student.count_params()
    )

    del initial_student

    trials = [("scratch", None)] + [
        (f"T_{int(t)}", t) for t in TEMPERATURES
    ]

    results_path = RESULTS / "temperature_optimization.json"

    results = {
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": len(x_val),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "alpha": ALPHA,
        "trials": {},
    }

    if results_path.exists():
        existing = json.loads(results_path.read_text())

        settings = (
            "train_samples", "validation_samples",
            "epochs", "batch_size", "seed", "alpha"
        )

        if any(existing.get(k) != results[k] for k in settings):
            raise RuntimeError(
                "Existing results have different settings. "
                "Archive them before changing hyperparameters."
            )

        results = existing

    for name, temperature in trials:
        if name in results["trials"]:
            print(f"Skipping completed trial: {name}")
            continue

        print(f"\nStarting trial: {name}", flush=True)

        result = train_trial(
            name,
            temperature,
            initial_weights,
            x_train,
            y_train,
            teacher_probs,
            x_val,
            y_val,
        )

        results["trials"][name] = result

        results_path.write_text(
            json.dumps(results, indent=2)
        )

    print("\n=== TEMPERATURE COMPARISON ===")

    for name, trial in results["trials"].items():
        print(
            f"{name:12s} "
            f"accuracy={trial['full_validation_accuracy']:.2%} "
            f"params={trial['student_parameters']:,} "
            f"time={trial['training_seconds']:.1f}s"
        )

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
