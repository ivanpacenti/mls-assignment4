
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

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

OUTPUT = Path("part2/results/layer_quantization_sensitivity.json")
BITS = [8, 4, 2]
SAMPLES = 200


def main():
    tf.config.set_visible_devices([], "GPU")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(
        "part1/baseline_model.keras",
        compile=False,
    )

    _, _, x_val, y_val, _, _, _ = prepare_compression_datasets()

    x_val = np.asarray(x_val[:SAMPLES], dtype=np.float32)
    y_val = np.asarray(y_val[:SAMPLES]).reshape(-1)

    groups = defaultdict(list)

    for variable in model.weights:
        if "kernel" in variable.name.lower():
            groups[group_name(variable)].append(variable)

    original = {
        id(variable): variable.numpy().copy()
        for variables in groups.values()
        for variable in variables
    }

    baseline = evaluate(model, x_val, y_val)
    print(f"Baseline FP32: {baseline:.2%}")

    results = {
        "samples": SAMPLES,
        "baseline_accuracy": baseline,
        "groups": {},
    }

    for group, variables in sorted(groups.items()):
        results["groups"][group] = {}

        for bits in BITS:
            try:
                for variable in variables:
                    variable.assign(
                        quantize_weights(
                            original[id(variable)],
                            bits,
                        )
                    )

                accuracy = evaluate(model, x_val, y_val)

                drop = (baseline - accuracy) * 100

                results["groups"][group][str(bits)] = {
                    "accuracy": accuracy,
                    "accuracy_drop_pp": drop,
                    "parameters": sum(
                        int(np.prod(v.shape))
                        for v in variables
                    ),
                }

                print(
                    f"{group:10s} "
                    f"{bits}-bit | "
                    f"accuracy={accuracy:.2%} | "
                    f"drop={drop:+.2f} pp"
                )

            finally:
                for variable in variables:
                    variable.assign(original[id(variable)])

            OUTPUT.write_text(json.dumps(results, indent=2))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
