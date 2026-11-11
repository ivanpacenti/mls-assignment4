
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from combined_compression import (
    PROJECT,
    ROOT,
    apply_pruning,
)
from pruning_quantization import (
    convert_dynamic,
    evaluate_tflite,
    benchmark_latency,
)

from sys import path
path.insert(0, str(PROJECT / "part1"))
from baseline_model import prepare_compression_datasets


RESULTS = ROOT / "results"
MODELS = ROOT / "models"
CHECKPOINTS = ROOT / "checkpoints" / "ordering"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULTS / "ordering_analysis.json"

FINETUNED_CONTROL = (
    ROOT / "checkpoints/combined/kd_prune_0.keras"
)

SPARSITIES = [0.25, 0.50]


def main():
    if os.environ.get("PART4_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    if not FINETUNED_CONTROL.exists():
        raise FileNotFoundError(FINETUNED_CONTROL)

    previous = json.loads(
        (RESULTS / "combined_compression.json").read_text()
    )

    _, _, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    results = {"experiments": {}}

    if OUTPUT.exists():
        results = json.loads(OUTPUT.read_text())

    for sparsity in SPARSITIES:
        percentage = int(sparsity * 100)
        key = f"prune_{percentage}"

        if key in results["experiments"]:
            print(f"Skipping {key}")
            continue

        print(f"\nEvaluating {key}", flush=True)

        # Order A was already trained and evaluated.
        order_a = previous["experiments"][
            f"kd_prune_{percentage}"
        ]

        # Order B: identical starting student, identical
        # additional training budget, pruning applied last.
        model = tf.keras.models.load_model(
            FINETUNED_CONTROL,
            compile=False,
        )

        _, _, achieved = apply_pruning(
            model, sparsity
        )

        logits = model.predict(
            x_val,
            batch_size=64,
            verbose=0,
        )

        keras_accuracy = float(np.mean(
            np.argmax(logits, axis=1) == y_val
        ))

        checkpoint = (
            CHECKPOINTS
            / f"finetune_then_prune_{percentage}.keras"
        )

        model.save(checkpoint)
        del model

        tflite_path = (
            MODELS
            / f"finetune_then_prune_{percentage}.tflite"
        )

        if not tflite_path.exists():
            convert_dynamic(checkpoint, tflite_path)

        evaluation = evaluate_tflite(
            tflite_path,
            x_val,
            y_val,
        )

        latency = benchmark_latency(
            tflite_path,
            x_val,
        )

        accuracy_a = order_a["tflite_accuracy"]
        accuracy_b = evaluation["accuracy"]

        results["experiments"][key] = {
            "sparsity": sparsity,
            "order_a": {
                "pipeline": (
                    "KD -> pruning -> "
                    "3 epochs KD fine-tuning -> quantization"
                ),
                "accuracy": accuracy_a,
                "size_mib": order_a["size_mib"],
                "median_latency_ms": (
                    order_a["median_latency_ms"]
                ),
            },
            "order_b": {
                "pipeline": (
                    "KD -> 3 epochs KD fine-tuning -> "
                    "pruning -> quantization"
                ),
                "keras_accuracy": keras_accuracy,
                "accuracy": accuracy_b,
                "achieved_sparsity": achieved,
                "size_mib": (
                    tflite_path.stat().st_size / 1024**2
                ),
                "median_latency_ms": (
                    latency["median_latency_ms"]
                ),
                "checkpoint": str(checkpoint),
                "tflite_path": str(tflite_path),
            },
            "order_a_minus_b_pp": (
                accuracy_a - accuracy_b
            ) * 100,
        }

        OUTPUT.write_text(
            json.dumps(results, indent=2)
        )

        print(
            f"Pruning level: {percentage}%\n"
            f"Order A accuracy: {accuracy_a:.2%}\n"
            f"Order B accuracy: {accuracy_b:.2%}\n"
            f"Difference A-B: "
            f"{(accuracy_a - accuracy_b)*100:+.2f} pp",
            flush=True,
        )

    print("\n=== ORDERING COMPARISON ===")

    for key, item in results["experiments"].items():
        print(
            f"{key:12s} "
            f"A={item['order_a']['accuracy']:.2%} "
            f"B={item['order_b']['accuracy']:.2%} "
            f"delta={item['order_a_minus_b_pp']:+.2f}pp"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
