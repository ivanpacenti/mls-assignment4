
import os
import sys
import json
import re
from collections import defaultdict
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path("part1").resolve()))
from baseline_model import prepare_compression_datasets


MODEL_PATH = "part1/baseline_model.keras"
RESULTS_PATH = Path("part2/results/mixed_bit_analysis.json")

SEED = 42
SENSITIVITY_BATCHES = 3
SENSITIVITY_BATCH_SIZE = 8
EVALUATION_SAMPLES = 200


def group_name(variable):
    name = variable.name.lower()

    match = re.search(r"block([1-7])[a-z]", name)
    if match:
        return f"block{match.group(1)}"

    if "stem_" in name:
        return "stem"

    if "top_" in name:
        return "top"

    if "compression_" in name or "classifier" in name:
        return "head"

    return "other"


def quantize_weights(values, bits):
    """
    Symmetric per-tensor weight fake quantization.

    At 2 bits this scheme produces three signed levels:
    -scale, 0, +scale. It is not binary quantization.
    """
    qmax = (2 ** (bits - 1)) - 1

    maximum = np.max(np.abs(values))

    if maximum == 0:
        return values.copy()

    scale = maximum / qmax

    quantized = np.clip(
        np.round(values / scale),
        -qmax,
        qmax,
    )

    return (quantized * scale).astype(values.dtype)


def evaluate(model, images, labels):
    predictions = model.predict(
        images,
        batch_size=32,
        verbose=0,
    )

    return float(np.mean(
        np.argmax(predictions, axis=1) == labels
    ))


def calculate_sensitivity(model, images, labels):
    variables = [
        variable
        for variable in model.trainable_weights
        if "kernel" in variable.name.lower()
    ]

    if not variables:
        raise RuntimeError("No trainable kernels found")

    scores = defaultdict(float)
    counts = defaultdict(int)

    for i in range(SENSITIVITY_BATCHES):
        start = i * SENSITIVITY_BATCH_SIZE
        stop = start + SENSITIVITY_BATCH_SIZE

        batch_x = tf.convert_to_tensor(images[start:stop])
        batch_y = tf.convert_to_tensor(labels[start:stop])

        with tf.GradientTape() as tape:
            output = model(batch_x, training=False)

            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    batch_y, output
                )
            )

        gradients = tape.gradient(loss, variables)

        for variable, gradient in zip(variables, gradients):
            if gradient is None:
                continue

            group = group_name(variable)

            contribution = tf.reduce_sum(
                tf.abs(variable * gradient)
            ).numpy()

            scores[group] += float(contribution)
            counts[group] += int(np.prod(variable.shape))

        print(f"Sensitivity batch {i + 1}/{SENSITIVITY_BATCHES}")

    # Normalize by tensor element count and number of batches.
    return {
        group: score / counts[group]
        for group, score in scores.items()
        if counts[group] > 0
    }


def assign_mixed_bits(sensitivity):
    ranked = sorted(
        sensitivity,
        key=sensitivity.get,
        reverse=True,
    )

    configuration = {}

    for position, group in enumerate(ranked):
        fraction = position / len(ranked)

        if fraction < 0.34:
            configuration[group] = 8
        elif fraction < 0.67:
            configuration[group] = 4
        else:
            configuration[group] = 2

    return configuration


def main():
    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1)

    x_val = np.asarray(
        x_val[:EVALUATION_SAMPLES],
        dtype=np.float32,
    )
    y_val = np.asarray(
        y_val[:EVALUATION_SAMPLES]
    ).reshape(-1)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    kernels = [
        variable
        for variable in model.weights
        if "kernel" in variable.name.lower()
    ]

    original_weights = [
        variable.numpy().copy()
        for variable in kernels
    ]

    baseline_accuracy = evaluate(model, x_val, y_val)

    print(f"\nFP32 accuracy: {baseline_accuracy:.2%}")

    sensitivity = calculate_sensitivity(
        model,
        x_train,
        y_train,
    )

    print("\n=== LAYER GROUP SENSITIVITY ===")
    for group, score in sorted(
        sensitivity.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{group:12s}: {score:.8e}")

    mixed_configuration = assign_mixed_bits(sensitivity)

    print("\n=== MIXED BIT ASSIGNMENT ===")
    for group, bits in mixed_configuration.items():
        print(f"{group:12s}: {bits} bits")

    configurations = {
        "uniform_8bit": {
            group: 8 for group in sensitivity
        },
        "uniform_4bit": {
            group: 4 for group in sensitivity
        },
        "uniform_2bit": {
            group: 2 for group in sensitivity
        },
        "mixed_precision": mixed_configuration,
    }

    results = {
        "evaluation_samples": EVALUATION_SAMPLES,
        "baseline_accuracy": baseline_accuracy,
        "sensitivity": sensitivity,
        "bit_assignment": mixed_configuration,
        "experiments": {},
    }

    original_weight_bits = sum(
        weight.size * 32
        for weight in original_weights
    )

    try:
        for name, configuration in configurations.items():
            print(f"\nTesting {name}...")

            simulated_bits = 0

            for variable, original in zip(
                kernels, original_weights
            ):
                group = group_name(variable)
                bits = configuration.get(group, 8)

                variable.assign(
                    quantize_weights(original, bits)
                )

                simulated_bits += original.size * bits

            accuracy = evaluate(model, x_val, y_val)

            results["experiments"][name] = {
                "accuracy": accuracy,
                "accuracy_drop_pp": (
                    baseline_accuracy - accuracy
                ) * 100,
                "theoretical_kernel_compression": (
                    original_weight_bits / simulated_bits
                ),
                "bit_configuration": configuration,
            }

            print(f"Accuracy: {accuracy:.2%}")
            print(
                "Theoretical kernel compression:",
                f"{original_weight_bits / simulated_bits:.2f}x"
            )

            # Restore FP32 weights before the next experiment.
            for variable, original in zip(
                kernels, original_weights
            ):
                variable.assign(original)

            RESULTS_PATH.write_text(
                json.dumps(results, indent=2)
            )

    finally:
        for variable, original in zip(
            kernels, original_weights
        ):
            variable.assign(original)

    print("\n=== FINAL RESULTS ===")

    for name, result in results["experiments"].items():
        print(
            f"{name:18s} "
            f"accuracy={result['accuracy']:.2%} "
            f"kernel_compression="
            f"{result['theoretical_kernel_compression']:.2f}x"
        )

    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
