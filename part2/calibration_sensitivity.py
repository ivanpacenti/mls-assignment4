
import os
import sys
import json
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path("part1").resolve()))
from baseline_model import prepare_compression_datasets

MODEL_PATH = Path("part1/baseline_model.keras")
OUTPUT_DIR = Path("part2/models/calibration")
RESULTS_PATH = Path("part2/results/calibration_sensitivity.json")

CALIBRATION_SIZES = [300, 1000, 2000]
EVALUATION_SAMPLES = 200
SEED = 42


def convert_int8(model, calibration_images, size):
    def representative_dataset():
        for image in calibration_images[:size]:
            yield [image[np.newaxis, ...].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    converted = converter.convert()

    output_path = OUTPUT_DIR / f"ptq_int8_calib_{size}.tflite"
    output_path.write_bytes(converted)

    return output_path


def evaluate_tflite(path, images, labels):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    assert input_info["dtype"] == np.float32
    assert output_info["dtype"] == np.float32

    correct = 0

    for image, label in zip(images, labels):
        interpreter.set_tensor(
            input_info["index"],
            image[np.newaxis, ...].astype(np.float32),
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_info["index"]
        )

        predicted_class = int(np.argmax(prediction[0]))
        correct += int(predicted_class == int(label))

    return correct / len(images)


def main():
    tf.config.set_visible_devices([], "GPU")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    x_train, _, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_train = np.asarray(x_train, dtype=np.float32)
    x_val = np.asarray(
        x_val[:EVALUATION_SAMPLES], dtype=np.float32
    )
    y_val = np.asarray(
        y_val[:EVALUATION_SAMPLES]
    ).reshape(-1)

    # Use nested subsets from a reproducible random permutation
    # of the training set. Never calibrate on validation data.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x_train))
    calibration_images = x_train[
        indices[:max(CALIBRATION_SIZES)]
    ]

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    baseline_predictions = model.predict(
        x_val, batch_size=32, verbose=0
    )
    baseline_accuracy = np.mean(
        np.argmax(baseline_predictions, axis=1) == y_val
    )

    print(f"FP32 Keras baseline: {baseline_accuracy:.2%}")

    results = {
        "seed": SEED,
        "evaluation_samples": EVALUATION_SAMPLES,
        "baseline_accuracy": float(baseline_accuracy),
        "experiments": [],
    }

    for size in CALIBRATION_SIZES:
        print(f"\nCalibrating INT8 with {size} images...")

        path = convert_int8(
            model, calibration_images, size
        )

        accuracy = evaluate_tflite(
            path, x_val, y_val
        )

        model_size = path.stat().st_size / (1024 ** 2)

        result = {
            "calibration_samples": size,
            "accuracy": float(accuracy),
            "model_size_mib": float(model_size),
            "model_path": str(path),
        }

        results["experiments"].append(result)

        print(f"Accuracy: {accuracy:.2%}")
        print(f"Model size: {model_size:.2f} MiB")
        print(f"Accuracy drop: {(baseline_accuracy - accuracy) * 100:.2f} pp")

        RESULTS_PATH.write_text(
            json.dumps(results, indent=2)
        )

    print("\n=== CALIBRATION COMPARISON ===")

    for result in results["experiments"]:
        print(
            f"{result['calibration_samples']:4d} samples | "
            f"accuracy={result['accuracy']:.2%} | "
            f"size={result['model_size_mib']:.2f} MiB"
        )

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
