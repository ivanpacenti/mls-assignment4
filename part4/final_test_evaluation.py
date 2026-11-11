
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "part1"))

from baseline_model import prepare_compression_datasets
from pruning_quantization import evaluate_tflite


ROOT = PROJECT / "part4"
OUTPUT = ROOT / "results/final_test_evaluation.json"

MODELS = {
    "baseline_dynamic": (
        ROOT / "models/baseline_dynamic.tflite"
    ),
    "kd_prune_25": (
        ROOT / "models/kd_prune_25_dynamic.tflite"
    ),
    "structured_dynamic": (
        ROOT / "models/structured_dynamic.tflite"
    ),
}


def main():
    if os.environ.get("PART4_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    missing = [
        str(path)
        for path in MODELS.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing TFLite models:\n" + "\n".join(missing)
        )

    (
        _, _,
        x_val, y_val,
        x_test, y_test,
        _,
    ) = prepare_compression_datasets()

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test).reshape(-1)

    if len(x_test) == 0:
        raise RuntimeError("Test dataset is empty")

    # Check that validation and test data are not the same
    # array in content and order. This is only a basic guard,
    # not a proof of sample-level dataset independence.
    if (
        x_val.shape == x_test.shape
        and np.array_equal(x_val, x_test)
        and np.array_equal(y_val, y_test)
    ):
        raise RuntimeError(
            "Validation and test datasets are identical"
        )

    results = {
        "dataset": "CIFAR-100",
        "test_samples": int(len(x_test)),
        "models": {},
    }

    for name, model_path in MODELS.items():
        print(f"\nEvaluating {name}...", flush=True)

        metrics = evaluate_tflite(
            model_path,
            x_test,
            y_test,
        )

        results["models"][name] = {
            "model_path": str(model_path),
            "test_accuracy": metrics["accuracy"],
            "test_samples": metrics["evaluation_samples"],
            "size_bytes": model_path.stat().st_size,
            "size_mib": (
                model_path.stat().st_size / 1024**2
            ),
        }

        OUTPUT.write_text(
            json.dumps(results, indent=2)
        )

        print(
            f"Test accuracy: {metrics['accuracy']:.2%}\n"
            f"Model size: "
            f"{results['models'][name]['size_mib']:.3f} MiB",
            flush=True,
        )

    print("\n=== FINAL TEST RESULTS ===")

    for name, metrics in results["models"].items():
        print(
            f"{name:22s} "
            f"accuracy={metrics['test_accuracy']:.2%} "
            f"size={metrics['size_mib']:.3f}MiB"
        )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
