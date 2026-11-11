
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "part1"))
sys.path.insert(0, str(PROJECT / "part4"))

from baseline_model import prepare_compression_datasets
from pruning_quantization import (
    convert_dynamic,
    evaluate_tflite,
    benchmark_latency,
)

ROOT = PROJECT / "part4"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

OUTPUT = RESULTS / "distillation_quantization.json"

# Compare models only within the same training group.
SOURCES = {
    "scratch_16": (
        "16_epochs",
        PROJECT / "part3/checkpoints/scratch_extended.keras",
    ),
    "kd_T4_16": (
        "16_epochs",
        PROJECT / "part3/checkpoints/T_4_extended.keras",
    ),
    "kd_T8_16": (
        "16_epochs",
        PROJECT / "part3/checkpoints/T_8_extended.keras",
    ),
    "direct_kd_12": (
        "12_epochs",
        PROJECT / "part3/checkpoints/progressive/direct_final.keras",
    ),
    "progressive_kd_12": (
        "12_epochs",
        PROJECT / "part3/checkpoints/progressive/progressive_final.keras",
    ),
    "feature_matching_12": (
        "12_epochs",
        PROJECT / (
            "part3/checkpoints/feature_matching/"
            "feature_gamma_0_1.keras"
        ),
    ),
    "attention_12": (
        "12_epochs",
        PROJECT / (
            "part3/checkpoints/attention/"
            "attention_beta_1_0.keras"
        ),
    ),
}


def evaluate_keras(path, images, labels):
    model = tf.keras.models.load_model(
        path, compile=False
    )

    logits = model.predict(
        images, batch_size=64, verbose=0
    )

    accuracy = float(np.mean(
        np.argmax(logits, axis=1) == labels
    ))

    parameters = int(model.count_params())

    del model

    return accuracy, parameters


def main():
    if os.environ.get("PART4_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    for name, (_, path) in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {name}: {path}"
            )

    _, _, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    config = {
        "validation_samples": len(x_val),
        "quantization": "dynamic_range",
        "num_threads": 1,
        "models": {
            name: str(path)
            for name, (_, path) in SOURCES.items()
        },
    }

    results = {
        "configuration": config,
        "models": {},
    }

    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text())

        if previous.get("configuration") != config:
            raise RuntimeError(
                "Existing results use different settings"
            )

        results = previous

    for name, (group, source_path) in SOURCES.items():
        if name in results["models"]:
            saved_model = Path(
                results["models"][name]["tflite_path"]
            )
            if not saved_model.exists():
                raise FileNotFoundError(saved_model)
            print(f"Skipping completed model: {name}")
            continue

        print(f"\n=== {name} ===", flush=True)

        tflite_path = MODELS / f"{name}_dynamic.tflite"

        if not tflite_path.exists():
            convert_dynamic(
                source_path,
                tflite_path,
            )

        fp32_accuracy, parameters = evaluate_keras(
            source_path, x_val, y_val
        )

        evaluation = evaluate_tflite(
            tflite_path, x_val, y_val
        )

        latency = benchmark_latency(
            tflite_path, x_val
        )

        tflite_accuracy = evaluation["accuracy"]

        item = {
            "group": group,
            "checkpoint": str(source_path),
            "tflite_path": str(tflite_path),
            "parameters": parameters,
            "keras_accuracy": fp32_accuracy,
            "tflite_accuracy": tflite_accuracy,
            "quantization_delta_pp": 100 * (
                tflite_accuracy - fp32_accuracy
            ),
            "size_bytes": tflite_path.stat().st_size,
            "size_mib": (
                tflite_path.stat().st_size / 1024 ** 2
            ),
            **latency,
        }

        results["models"][name] = item

        OUTPUT.write_text(
            json.dumps(results, indent=2)
        )

        print(
            f"Keras: {fp32_accuracy:.2%}\n"
            f"TFLite: {tflite_accuracy:.2%}\n"
            f"Quantization delta: "
            f"{item['quantization_delta_pp']:+.2f} pp\n"
            f"Size: {item['size_mib']:.3f} MiB\n"
            f"Latency: "
            f"{item['median_latency_ms']:.3f} ms",
            flush=True,
        )

    print("\n=== DISTILLATION + QUANTIZATION ===")

    for group in ("16_epochs", "12_epochs"):
        print(f"\n{group}")

        for name, item in results["models"].items():
            if item["group"] != group:
                continue

            print(
                f"{name:22s} "
                f"fp32={item['keras_accuracy']:.2%} "
                f"tflite={item['tflite_accuracy']:.2%} "
                f"delta={item['quantization_delta_pp']:+.2f}pp "
                f"size={item['size_mib']:.3f}MiB "
                f"latency={item['median_latency_ms']:.3f}ms"
            )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
