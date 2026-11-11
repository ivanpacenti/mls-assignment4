
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
CHECKPOINTS = ROOT / "checkpoints" / "attention"
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

BETAS = (1.0, 10.0)

RESULT_PATH = RESULTS / "attention_transfer.json"


def attention_map(features):
    """Channel-aggregated, L2-normalized spatial attention."""
    features = tf.cast(features, tf.float32)

    attention = tf.reduce_sum(
        tf.square(features), axis=-1
    )

    batch_size = tf.shape(attention)[0]
    flattened = tf.reshape(attention, (batch_size, -1))

    return tf.nn.l2_normalize(
        flattened, axis=1, epsilon=1e-8
    )


def attention_loss(student_features, teacher_attention):
    student_attention = attention_map(student_features)

    # Each sample contributes one squared L2 distance.
    return tf.reduce_mean(
        tf.reduce_sum(
            tf.square(
                student_attention - teacher_attention
            ),
            axis=1,
        )
    )


def build_student_feature_model():
    student = create_student()

    pooling_layers = [
        layer
        for layer in student.layers
        if isinstance(layer, tf.keras.layers.MaxPooling2D)
    ]

    if len(pooling_layers) < 2:
        raise ValueError(
            "Expected at least two student pooling layers"
        )

    feature_layer = pooling_layers[1]

    feature_model = tf.keras.Model(
        inputs=student.input,
        outputs=[student.output, feature_layer.output],
    )

    return student, feature_model


def find_teacher_feature_layer(teacher):
    """Find a late spatial feature map in the teacher graph."""
    candidates = []

    for layer in teacher.layers:
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue

        if shape.rank != 4:
            continue

        height, width = shape[1], shape[2]

        if height is None or width is None:
            continue

        if int(height) > 1 and int(width) > 1:
            candidates.append(layer)

    if not candidates:
        raise RuntimeError(
            "No suitable teacher spatial feature layer found"
        )

    return candidates[-1]


def prepare_teacher_attention(x_train, target_shape):
    teacher_path = PROJECT / "part1/baseline_model.keras"

    teacher = tf.keras.models.load_model(
        teacher_path, compile=False
    )
    teacher.trainable = False

    feature_layer = find_teacher_feature_layer(teacher)

    print(
        "Teacher attention layer:",
        feature_layer.name,
        "shape:",
        feature_layer.output.shape,
        flush=True,
    )

    cache_path = CACHE / (
        f"teacher_attention_"
        f"{feature_layer.name}_"
        f"{len(x_train)}_"
        f"{target_shape[0]}x{target_shape[1]}.npy"
    )

    expected_shape = (
        len(x_train),
        target_shape[0] * target_shape[1],
    )

    if cache_path.exists():
        cached = np.load(cache_path)

        if cached.shape != expected_shape:
            raise RuntimeError(
                "Cached attention shape does not match"
            )

        print("Loaded cached teacher attention")
        return cached

    extractor = tf.keras.Model(
        inputs=teacher.input,
        outputs=feature_layer.output,
    )

    attention_batches = []

    for start in range(0, len(x_train), 32):
        batch = x_train[start:start + 32]

        features = extractor(
            batch, training=False
        )

        # Resize spatial features before computing attention.
        features = tf.image.resize(
            features,
            size=target_shape,
            method="bilinear",
        )

        normalized = attention_map(features)

        attention_batches.append(
            normalized.numpy().astype(np.float32)
        )

    teacher_attention = np.concatenate(
        attention_batches, axis=0
    )

    assert teacher_attention.shape == expected_shape

    np.save(cache_path, teacher_attention)
    print("Cached teacher attention:", cache_path)

    return teacher_attention


def train_trial(
    beta,
    x_train,
    y_train,
    teacher_probs,
    teacher_attention,
    x_val,
    y_val,
):
    tf.keras.utils.set_random_seed(SEED)

    name = f"attention_beta_{str(beta).replace('.', '_')}"
    checkpoint = CHECKPOINTS / f"{name}.keras"

    student, feature_model = build_student_feature_model()

    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            x_train,
            y_train,
            teacher_probs,
            teacher_attention,
        )
    )

    dataset = (
        dataset.shuffle(
            len(x_train),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    @tf.function
    def train_step(images, labels, probs, target_attention):
        with tf.GradientTape() as tape:
            logits, student_features = feature_model(
                images, training=True
            )

            kd_loss = distillation_loss(
                labels,
                probs,
                logits,
                tf.constant(TEMPERATURE, tf.float32),
                tf.constant(ALPHA, tf.float32),
            )

            at_loss = attention_loss(
                student_features,
                target_attention,
            )

            total_loss = kd_loss + beta * at_loss

        gradients = tape.gradient(
            total_loss, student.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradients, student.trainable_variables)
        )

        return total_loss, kd_loss, at_loss

    history = []
    best_accuracy = -1.0
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        total_losses = []
        kd_losses = []
        attention_losses = []

        for images, labels, probs, maps in dataset:
            total, kd, at = train_step(
                images, labels, probs, maps
            )

            total_losses.append(float(total.numpy()))
            kd_losses.append(float(kd.numpy()))
            attention_losses.append(float(at.numpy()))

        val_accuracy = evaluate(
            student, x_val[:1000], y_val[:1000]
        )

        metrics = {
            "epoch": epoch + 1,
            "total_loss": float(np.mean(total_losses)),
            "kd_loss": float(np.mean(kd_losses)),
            "attention_loss": float(
                np.mean(attention_losses)
            ),
            "validation_accuracy_1000": val_accuracy,
        }

        history.append(metrics)

        print(
            f"{name:22s} "
            f"epoch={epoch + 1:02d}/{EPOCHS} "
            f"loss={metrics['total_loss']:.4f} "
            f"KD={metrics['kd_loss']:.4f} "
            f"AT={metrics['attention_loss']:.4f} "
            f"acc={val_accuracy:.2%}",
            flush=True,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            student.save(checkpoint)

    training_seconds = time.perf_counter() - start_time

    best_model = tf.keras.models.load_model(
        checkpoint, compile=False
    )

    full_accuracy = evaluate(
        best_model, x_val, y_val
    )

    print(
        f"\n{name} full validation: {full_accuracy:.2%}",
        flush=True,
    )

    return {
        "beta": beta,
        "parameters": int(best_model.count_params()),
        "best_validation_accuracy_1000": best_accuracy,
        "full_validation_accuracy": full_accuracy,
        "training_seconds": training_seconds,
        "checkpoint": str(checkpoint),
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

    if teacher_probs.shape != (TRAIN_SAMPLES, 100):
        raise ValueError("Unexpected teacher probability shape")

    # Find the spatial size of the student's second pooling layer.
    _, probe_model = build_student_feature_model()

    student_feature_shape = probe_model.output_shape[1]

    target_height = int(student_feature_shape[1])
    target_width = int(student_feature_shape[2])

    del probe_model

    print(
        f"Student attention resolution: "
        f"{target_height}x{target_width}"
    )

    teacher_attention = prepare_teacher_attention(
        x_train,
        (target_height, target_width),
    )

    config = {
        "seed": SEED,
        "train_samples": TRAIN_SAMPLES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "alpha": ALPHA,
        "learning_rate": LEARNING_RATE,
        "betas": list(BETAS),
        "attention_resolution": [
            target_height, target_width
        ],
    }

    results = {
        "configuration": config,
        "trials": {},
    }

    if RESULT_PATH.exists():
        previous = json.loads(RESULT_PATH.read_text())

        if previous.get("configuration") != config:
            raise RuntimeError(
                "Existing attention results use different settings"
            )

        results = previous

    for beta in BETAS:
        name = f"attention_beta_{str(beta).replace('.', '_')}"

        if name in results["trials"]:
            print(f"Skipping completed trial: {name}")
            continue

        print(f"\nTraining {name}", flush=True)

        results["trials"][name] = train_trial(
            beta,
            x_train,
            y_train,
            teacher_probs,
            teacher_attention,
            x_val,
            y_val,
        )

        RESULT_PATH.write_text(json.dumps(results, indent=2))

    # The previously trained direct-KD student is the control.
    direct_path = (
        CHECKPOINTS.parent
        / "progressive"
        / "direct_final.keras"
    )

    if not direct_path.exists():
        raise FileNotFoundError(direct_path)

    direct_model = tf.keras.models.load_model(
        direct_path, compile=False
    )

    direct_accuracy = evaluate(
        direct_model, x_val, y_val
    )

    results["baseline"] = {
        "name": "vanilla_direct_kd",
        "full_validation_accuracy": direct_accuracy,
        "checkpoint": str(direct_path),
    }

    print("\n=== ATTENTION TRANSFER COMPARISON ===")
    print(f"Vanilla KD: {direct_accuracy:.2%}")

    for name, result in results["trials"].items():
        improvement = (
            result["full_validation_accuracy"]
            - direct_accuracy
        ) * 100

        print(
            f"{name}: "
            f"{result['full_validation_accuracy']:.2%} "
            f"({improvement:+.2f} pp)"
        )

    RESULT_PATH.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
