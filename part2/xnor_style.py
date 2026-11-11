
import os
import sys
import json
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path("part1").resolve()))
from baseline_model import prepare_compression_datasets


OUTPUT = Path("part2/results/xnor_style.json")


def binary_sign(x):
    return np.where(x >= 0, 1.0, -1.0).astype(np.float32)


def main():
    tf.config.set_visible_devices([], "GPU")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(
        "part1/baseline_model.keras",
        compile=False,
    )

    _, _, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    classifier = model.get_layer("classifier")

    # Extract the classifier input features, leaving
    # the preceding network in FP32.
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=classifier.input,
    )

    features = feature_model.predict(
        x_val, batch_size=32, verbose=0
    ).astype(np.float32)

    kernel, bias = classifier.get_weights()

    # FP32 reference.
    fp32_logits = features @ kernel + bias

    # Binary weights, FP32 activations.
    # Per-output-channel scale for binary weights.
    weight_scale = np.mean(
        np.abs(kernel), axis=0, keepdims=True
    )
    binary_kernel = binary_sign(kernel) * weight_scale

    binary_weight_logits = (
        features @ binary_kernel + bias
    )

    # XNOR-style: binary weights AND binary activations.
    # Sample-wise activation scale.
    activation_scale = np.mean(
        np.abs(features), axis=1, keepdims=True
    )

    binary_features = binary_sign(features)
    binary_weights = binary_sign(kernel)

    # Numerical equivalent of a signed binary dot product.
    # This uses floating-point matmul, NOT XNOR/popcount.
    xnor_logits = (
        (binary_features @ binary_weights)
        * activation_scale
        * weight_scale
        + bias
    )

    experiments = {
        "fp32_classifier": fp32_logits,
        "binary_weights": binary_weight_logits,
        "xnor_style": xnor_logits,
    }

    results = {
        "samples": len(y_val),
        "experiments": {},
        "note": (
            "XNOR-style numerical simulation of the final "
            "Dense classifier, not an optimized XNOR kernel."
        ),
    }

    for name, logits in experiments.items():
        predictions = np.argmax(logits, axis=1)
        accuracy = float(np.mean(predictions == y_val))

        results["experiments"][name] = {
            "accuracy": accuracy,
        }

        print(f"{name:20s}: {accuracy:.2%}")

    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
