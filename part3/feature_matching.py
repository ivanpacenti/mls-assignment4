
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

from baseline_model import prepare_compression_datasets
from temperature_optimization import (
    create_student,
    distillation_loss,
    evaluate,
)

ROOT = PROJECT / "part3"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints" / "feature_matching"
CACHE = ROOT / "cache"

for directory in (RESULTS, CHECKPOINTS, CACHE):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_SAMPLES = 12000
EPOCHS = 12
BATCH_SIZE = 64
TEMPERATURE = 4.0
ALPHA = 0.7
LEARNING_RATE = 3e-4

GAMMAS = [0.1, 1.0]
TEACHER_LAYER = "compression_relu"

RESULT_PATH = RESULTS / "feature_matching.json"


def create_student_extractor():
    student = create_student()

    pooling_layers = [
        layer for layer in student.layers
        if isinstance(layer, tf.keras.layers.MaxPooling2D)
    ]

    if len(pooling_layers) != 3:
        raise RuntimeError(
            "Expected three pooling layers in student"
        )

    feature_layer = pooling_layers[-1]

    feature_model = tf.keras.Model(
        student.input,
        [student.output, feature_layer.output],
    )

    return student, feature_model


def prepare_teacher_features(x_train):
    cache_path = CACHE / (
        f"teacher_features_{TEACHER_LAYER}_"
        f"{len(x_train)}.npy"
    )

    if cache_path.exists():
        features = np.load(cache_path)
        if (
            features.ndim != 4
            or features.shape[0] != len(x_train)
        ):
            raise ValueError(
                "Cached teacher feature shape mismatch"
            )

        print(
            f"Loaded teacher features: {features.shape}",
            flush=True,
        )
        return features

    teacher = tf.keras.models.load_model(
        PROJECT / "part1/baseline_model.keras",
        compile=False,
    )
    teacher.trainable = False

    try:
        layer = teacher.get_layer(TEACHER_LAYER)
    except ValueError as exc:
        raise RuntimeError(
            f"Teacher layer {TEACHER_LAYER!r} not found"
        ) from exc

    extractor = tf.keras.Model(
        teacher.input,
        layer.output,
    )

    batches = []

    print(
        f"Extracting teacher features from "
        f"{TEACHER_LAYER}...",
        flush=True,
    )

    for start in range(0, len(x_train), BATCH_SIZE):
        images = x_train[start:start + BATCH_SIZE]

        features = extractor(images, training=False)
        features = tf.cast(features, tf.float32)

        # Normalize channels at each spatial position.
        features = tf.nn.l2_normalize(
            features, axis=-1, epsilon=1e-8
        )

        # Store float16 to reduce cache size.
        batches.append(
            features.numpy().astype(np.float16)
        )

    output = np.concatenate(batches, axis=0)

    np.save(cache_path, output)

    print(
        f"Cached teacher features: {output.shape}",
        flush=True,
    )

    del teacher, extractor

    return output


def feature_matching_loss(student_features, teacher_features):
    student_features = tf.cast(
        student_features, tf.float32
    )
    teacher_features = tf.cast(
        teacher_features, tf.float32
    )

    target_size = tf.shape(student_features)[1:3]

    teacher_features = tf.image.resize(
        teacher_features,
        target_size,
        method="bilinear",
    )

    # Compare normalized feature vectors at each location.
    student_normalized = tf.nn.l2_normalize(
        student_features,
        axis=-1,
        epsilon=1e-8,
    )
    teacher_normalized = tf.nn.l2_normalize(
        teacher_features,
        axis=-1,
        epsilon=1e-8,
    )

    squared_distance = tf.reduce_sum(
        tf.square(
            student_normalized - teacher_normalized
        ),
        axis=-1,
    )

    return tf.reduce_mean(squared_distance)


def train_trial(
    gamma,
    x_train,
    y_train,
    teacher_probs,
    teacher_features,
    x_val,
    y_val,
):
    tf.keras.utils.set_random_seed(SEED)

    name = f"feature_gamma_{str(gamma).replace('.', '_')}"

    model_path = CHECKPOINTS / f"{name}.keras"
    adapter_path = CHECKPOINTS / f"{name}_adapter.npz"

    student, feature_model = create_student_extractor()

    student_shape = feature_model.output_shape[1]
    teacher_channels = teacher_features.shape[-1]

    adapter = tf.keras.layers.Conv2D(
        filters=teacher_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="feature_projection",
    )

    # Build projection weights before creating the optimizer.
    probe = feature_model(
        tf.convert_to_tensor(x_train[:1]),
        training=False,
    )[1]

    adapter(probe)

    print(
        f"\n{name}",
        f"\nStudent features: {student_shape}",
        f"\nTeacher features: {teacher_features.shape[1:]}",
        f"\nStudent parameters: {student.count_params():,}",
        f"\nAdapter parameters: {adapter.count_params():,}",
        flush=True,
    )

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    )

    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                x_train,
                y_train,
                teacher_probs,
                teacher_features,
            )
        )
        .shuffle(
            TRAIN_SAMPLES,
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    variables = (
        student.trainable_variables
        + adapter.trainable_variables
    )

    @tf.function
    def train_step(images, labels, probs, target_features):
        with tf.GradientTape() as tape:
            logits, student_features = feature_model(
                images,
                training=True,
            )

            projected_features = adapter(
                student_features,
                training=True,
            )

            kd = distillation_loss(
                labels,
                probs,
                logits,
                tf.constant(TEMPERATURE, tf.float32),
                tf.constant(ALPHA, tf.float32),
            )

            fm = feature_matching_loss(
                projected_features,
                target_features,
            )

            total = kd + tf.constant(
                gamma, tf.float32
            ) * fm

        gradients = tape.gradient(total, variables)

        optimizer.apply_gradients(
            zip(gradients, variables)
        )

        return total, kd, fm

    best_accuracy = -1.0
    history = []
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        total_losses = []
        kd_losses = []
        fm_losses = []

        for images, labels, probs, features in dataset:
            total, kd, fm = train_step(
                images, labels, probs, features
            )

            total_losses.append(float(total.numpy()))
            kd_losses.append(float(kd.numpy()))
            fm_losses.append(float(fm.numpy()))

        validation_accuracy = evaluate(
            student,
            x_val[:1000],
            y_val[:1000],
        )

        metrics = {
            "epoch": epoch + 1,
            "total_loss": float(np.mean(total_losses)),
            "kd_loss": float(np.mean(kd_losses)),
            "feature_loss": float(np.mean(fm_losses)),
            "validation_accuracy_1000": (
                validation_accuracy
            ),
        }

        history.append(metrics)

        print(
            f"{name:22s} "
            f"epoch={epoch + 1:02d}/{EPOCHS} "
            f"KD={metrics['kd_loss']:.4f} "
            f"FM={metrics['feature_loss']:.4f} "
            f"acc={validation_accuracy:.2%}",
            flush=True,
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            student.save(model_path)
            np.savez_compressed(
                adapter_path,
                *adapter.get_weights(),
            )

    training_seconds = time.perf_counter() - start_time

    best_student = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    full_accuracy = evaluate(
        best_student,
        x_val,
        y_val,
    )

    print(
        f"{name} full validation: {full_accuracy:.2%}",
        flush=True,
    )

    return {
        "gamma": gamma,
        "student_parameters": int(
            best_student.count_params()
        ),
        "adapter_parameters": int(
            adapter.count_params()
        ),
        "best_validation_accuracy_1000": (
            best_accuracy
        ),
        "full_validation_accuracy": full_accuracy,
        "training_seconds": training_seconds,
        "student_checkpoint": str(model_path),
        "adapter_checkpoint": str(adapter_path),
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
        x_train[:TRAIN_SAMPLES],
        dtype=np.float32,
    )
    y_train = np.asarray(
        y_train[:TRAIN_SAMPLES]
    ).reshape(-1)

    x_val = np.asarray(
        x_val,
        dtype=np.float32,
    )
    y_val = np.asarray(y_val).reshape(-1)

    probability_path = (
        CACHE / f"teacher_probs_{TRAIN_SAMPLES}.npy"
    )

    if not probability_path.exists():
        raise FileNotFoundError(probability_path)

    teacher_probs = np.load(probability_path)

    if teacher_probs.shape != (TRAIN_SAMPLES, 100):
        raise ValueError(
            "Unexpected teacher probability shape"
        )

    teacher_features = prepare_teacher_features(
        x_train
    )

    config = {
        "seed": SEED,
        "train_samples": TRAIN_SAMPLES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "alpha": ALPHA,
        "learning_rate": LEARNING_RATE,
        "gammas": GAMMAS,
        "teacher_layer": TEACHER_LAYER,
    }

    results = {
        "configuration": config,
        "trials": {},
    }

    if RESULT_PATH.exists():
        previous = json.loads(RESULT_PATH.read_text())

        if previous.get("configuration") != config:
            raise RuntimeError(
                "Existing results use different settings"
            )

        results = previous

    for gamma in GAMMAS:
        name = (
            f"feature_gamma_"
            f"{str(gamma).replace('.', '_')}"
        )

        if name in results["trials"]:
            trial = results["trials"][name]
            if not (
                Path(trial["student_checkpoint"]).exists()
                and Path(trial["adapter_checkpoint"]).exists()
            ):
                raise FileNotFoundError(
                    f"Missing checkpoint for {name}"
                )

            print(f"Skipping completed trial: {name}")
            continue

        results["trials"][name] = train_trial(
            gamma,
            x_train,
            y_train,
            teacher_probs,
            teacher_features,
            x_val,
            y_val,
        )

        RESULT_PATH.write_text(
            json.dumps(results, indent=2)
        )

    direct_checkpoint = (
        ROOT / "checkpoints"
        / "progressive"
        / "direct_final.keras"
    )

    if not direct_checkpoint.exists():
        raise FileNotFoundError(direct_checkpoint)

    direct_model = tf.keras.models.load_model(
        direct_checkpoint,
        compile=False,
    )

    direct_accuracy = evaluate(
        direct_model,
        x_val,
        y_val,
    )

    results["baseline"] = {
        "name": "vanilla_direct_kd",
        "accuracy": direct_accuracy,
        "checkpoint": str(direct_checkpoint),
    }

    print("\n=== FEATURE MATCHING COMPARISON ===")
    print(f"Vanilla KD: {direct_accuracy:.2%}")

    for name, trial in results["trials"].items():
        difference = 100 * (
            trial["full_validation_accuracy"]
            - direct_accuracy
        )

        print(
            f"{name}: "
            f"{trial['full_validation_accuracy']:.2%} "
            f"({difference:+.2f} pp)"
        )

    RESULT_PATH.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
