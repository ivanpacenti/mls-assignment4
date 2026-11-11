
"""Evaluate Part 1 models under identical conditions."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PART1_CPU_ONLY"] = "1"

import numpy as np
import tensorflow as tf

from baseline_model import prepare_compression_datasets

ROOT = Path(__file__).resolve().parent

MODELS = {
    "baseline": ROOT / "baseline_model.keras",
    "magnitude_50": ROOT / "checkpoints" / "magnitude_50.keras",
    "structured": ROOT / "checkpoints" / "structured.keras",
}


def benchmark_latency(model, images, runs=30, warmup=5):
    batch = tf.convert_to_tensor(images[:1], dtype=tf.float32)

    for _ in range(warmup):
        model(batch, training=False).numpy()

    measurements = []

    for _ in range(runs):
        start = time.perf_counter()
        model(batch, training=False).numpy()
        measurements.append(
            (time.perf_counter() - start) * 1000
        )

    return {
        "median_ms": statistics.median(measurements),
        "mean_ms": statistics.mean(measurements),
        "std_ms": statistics.stdev(measurements),
    }


def evaluate_model(name, path, x_val, y_val):
    print(f"\nEvaluating {name}...", flush=True)

    model = tf.keras.models.load_model(path, compile=False)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss, accuracy = model.evaluate(
        x_val,
        y_val,
        batch_size=128,
        verbose=0,
    )

    parameters = model.count_params()
    file_bytes = path.stat().st_size

    result = {
        "model": name,
        "validation_accuracy": float(accuracy),
        "validation_loss": float(loss),
        "parameters": int(parameters),
        "file_bytes": int(file_bytes),
        "file_mb": float(file_bytes / (1024**2)),
        "latency": benchmark_latency(model, x_val),
    }

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Parameters: {parameters:,}")
    print(f"File size: {result['file_mb']:.2f} MiB")
    print(
        "Median latency: "
        f"{result['latency']['median_ms']:.2f} ms"
    )

    return result


def main():
    for path in MODELS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    (
        _,
        _,
        x_val,
        y_val,
        _,
        _,
        _,
    ) = prepare_compression_datasets()

    x_val = x_val.astype(np.float32)

    results = {}

    for name, path in MODELS.items():
        results[name] = evaluate_model(
            name, path, x_val, y_val
        )

    baseline = results["baseline"]

    for name, result in results.items():
        result["accuracy_drop_pp"] = (
            baseline["validation_accuracy"]
            - result["validation_accuracy"]
        ) * 100

        result["parameter_reduction"] = (
            1 - result["parameters"] / baseline["parameters"]
        )

        result["file_reduction"] = (
            1 - result["file_bytes"] / baseline["file_bytes"]
        )

        result["file_compression_ratio"] = (
            baseline["file_bytes"] / result["file_bytes"]
        )

    output = ROOT / "results" / "audit.json"
    output.parent.mkdir(exist_ok=True)

    with output.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit saved: {output}")


if __name__ == "__main__":
    main()
