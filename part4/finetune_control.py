
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
from combined_compression import (
    run_experiment,
    ROOT,
    TRAIN_SAMPLES,
    TEACHER_CACHE,
    SEED,
)


def main():
    if os.environ.get("PART4_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(SEED)

    output = ROOT / "results/unpruned_finetune_control.json"

    if output.exists():
        result = json.loads(output.read_text())
        print("Existing control:")
        print(json.dumps(result, indent=2))
        return

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

    teacher_probs = np.load(TEACHER_CACHE)

    assert teacher_probs.shape == (TRAIN_SAMPLES, 100)

    # Identical pipeline, but no weights are removed.
    result = run_experiment(
        sparsity=0.0,
        x_train=x_train,
        y_train=y_train,
        teacher_probs=teacher_probs,
        x_val=x_val,
        y_val=y_val,
    )

    output.write_text(json.dumps(result, indent=2))

    print("\n=== CONTROL COMPARISON ===")
    print(
        f"Unpruned + 3 epochs + dynamic TFLite: "
        f"{result['tflite_accuracy']:.2%}"
    )
    print("Pruned 25% + 3 epochs + dynamic TFLite: 40.36%")
    print("Pruned 50% + 3 epochs + dynamic TFLite: 40.28%")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
