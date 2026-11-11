
import os
import sys
import json
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path("part1").resolve()))

from baseline_model import prepare_compression_datasets
from mixed_bit_analysis import group_name, quantize_weights, evaluate

OUTPUT = Path("part2/results/binary_quantization.json")

CONFIGURATIONS = {
    "uniform_8": [],
    "binary_top": ["top"],
    "binary_head": ["head"],
    "binary_top_head": ["top", "head"],
    "binary_all": "all",
}


def binarize_weights(weights):
    alpha = np.mean(np.abs(weights))

    signs = np.where(weights >= 0, 1.0, -1.0)

    return (alpha * signs).astype(weights.dtype)


def main():
    tf.config.set_visible_devices([], "GPU")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(
        "part1/baseline_model.keras",
        compile=False,
    )

    _, _, x_val, y_val, _, _, _ = prepare_compression_datasets()

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    kernels = [
        variable for variable in model.weights
        if "kernel" in variable.name.lower()
    ]

    originals = [variable.numpy().copy() for variable in kernels]
    groups = [group_name(variable) for variable in kernels]

    baseline = evaluate(model, x_val, y_val)
    print(f"FP32 baseline: {baseline:.2%}")

    fp32_bits = sum(w.size * 32 for w in originals)

    results = {
        "baseline_accuracy": baseline,
        "evaluation_samples": len(x_val),
        "experiments": {},
    }

    try:
        for name, selected in CONFIGURATIONS.items():
            total_bits = 0

            try:
                for variable, original, group in zip(
                    kernels, originals, groups
                ):
                    use_binary = (
                        selected == "all" or group in selected
                    )

                    if use_binary:
                        new_weights = binarize_weights(original)
                        bits = 1
                    else:
                        new_weights = quantize_weights(original, 8)
                        bits = 8

                    variable.assign(new_weights)
                    total_bits += original.size * bits

                accuracy = evaluate(model, x_val, y_val)
                compression = fp32_bits / total_bits

                results["experiments"][name] = {
                    "accuracy": accuracy,
                    "accuracy_drop_pp": (baseline - accuracy) * 100,
                    "theoretical_kernel_compression": compression,
                    "binary_groups": (
                        "all" if selected == "all" else selected
                    ),
                }

                print(
                    f"{name:20s} "
                    f"accuracy={accuracy:.2%} "
                    f"compression={compression:.2f}x"
                )

            finally:
                for variable, original in zip(kernels, originals):
                    variable.assign(original)

            OUTPUT.write_text(json.dumps(results, indent=2))

    finally:
        for variable, original in zip(kernels, originals):
            variable.assign(original)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

