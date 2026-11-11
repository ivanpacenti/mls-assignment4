
import os
import sys
import json
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path("part1").resolve()))

from baseline_model import prepare_compression_datasets
from mixed_bit_analysis import (
    group_name,
    quantize_weights,
    evaluate,
)

MODEL_PATH = "part1/baseline_model.keras"
OUTPUT = Path("part2/results/mixed_bit_refinement.json")

# Progressive combinations based on measured layer sensitivity.
CONFIGURATIONS = {
    "uniform_8": [],
    "top_4": ["top"],
    "top_head_4": ["top", "head"],
    "top_head_block7_4": ["top", "head", "block7"],
    "top_head_block7_block6_4": [
        "top", "head", "block7", "block6"
    ],
    "all_4": "all",
}

PREVIEW_SAMPLES = 200
FULL_SAMPLES = 5000


def main():
    tf.config.set_visible_devices([], "GPU")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

    _, _, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_val = np.asarray(
        x_val[:FULL_SAMPLES], dtype=np.float32
    )
    y_val = np.asarray(
        y_val[:FULL_SAMPLES]
    ).reshape(-1)

    kernels = [
        variable for variable in model.weights
        if "kernel" in variable.name.lower()
    ]

    originals = [
        variable.numpy().copy()
        for variable in kernels
    ]

    groups = [group_name(v) for v in kernels]

    baseline = evaluate(
        model,
        x_val[:PREVIEW_SAMPLES],
        y_val[:PREVIEW_SAMPLES],
    )

    total_fp32_bits = sum(
        w.size * 32 for w in originals
    )

    results = {
        "preview_samples": PREVIEW_SAMPLES,
        "baseline_preview_accuracy": baseline,
        "experiments": {},
    }

    try:
        for name, selected in CONFIGURATIONS.items():
            print(f"\nTesting {name}")

            total_quantized_bits = 0
            assignment = {}

            for variable, original, group in zip(
                kernels, originals, groups
            ):
                use_4bit = (
                    selected == "all"
                    or group in selected
                )

                bits = 4 if use_4bit else 8
                assignment[group] = bits

                variable.assign(
                    quantize_weights(original, bits)
                )

                total_quantized_bits += original.size * bits

            accuracy = evaluate(
                model,
                x_val[:PREVIEW_SAMPLES],
                y_val[:PREVIEW_SAMPLES],
            )

            compression = (
                total_fp32_bits / total_quantized_bits
            )

            results["experiments"][name] = {
                "preview_accuracy": accuracy,
                "accuracy_drop_pp": (
                    baseline - accuracy
                ) * 100,
                "theoretical_kernel_compression": compression,
                "bit_assignment": assignment,
            }

            print(f"Accuracy: {accuracy:.2%}")
            print(
                f"Accuracy drop: "
                f"{(baseline - accuracy) * 100:.2f} pp"
            )
            print(
                f"Theoretical kernel compression: "
                f"{compression:.2f}x"
            )

            for variable, original in zip(kernels, originals):
                variable.assign(original)

            OUTPUT.write_text(
                json.dumps(results, indent=2)
            )

    finally:
        for variable, original in zip(kernels, originals):
            variable.assign(original)

    # Select the best compression that preserves accuracy
    # within two percentage points on the preview subset.
    valid = [
        (name, result)
        for name, result in results["experiments"].items()
        if result["accuracy_drop_pp"] <= 2.0 + 1e-8
    ]

    if not valid:
        print("No configuration meets the accuracy budget.")
        return

    best_name, best_result = max(
        valid,
        key=lambda item: (
            item[1]["theoretical_kernel_compression"],
            item[1]["preview_accuracy"],
        ),
    )

    print(f"\nSelected configuration: {best_name}")

    print("\n=== FULL VALIDATION COMPARISON ===")

    for name, configuration in results["experiments"].items():
        try:
            for variable, original, group in zip(
                kernels, originals, groups
            ):
                bits = configuration["bit_assignment"][group]
                variable.assign(
                    quantize_weights(original, bits)
                )

            accuracy = evaluate(model, x_val, y_val)

            configuration["full_validation_accuracy"] = accuracy

            print(
                f"{name:28s} "
                f"accuracy={accuracy:.2%}"
            )

        finally:
            for variable, original in zip(kernels, originals):
                variable.assign(original)

    OUTPUT.write_text(json.dumps(results, indent=2))

    try:
        for variable, original, group in zip(
            kernels, originals, groups
        ):
            bits = best_result["bit_assignment"][group]
            variable.assign(quantize_weights(original, bits))

        full_accuracy = evaluate(model, x_val, y_val)

        print(f"Full validation accuracy: {full_accuracy:.2%}")

        results["selected_configuration"] = best_name
        results["selected_full_validation_accuracy"] = (
            full_accuracy
        )
        results["full_validation_samples"] = len(x_val)

    finally:
        for variable, original in zip(kernels, originals):
            variable.assign(original)

    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
