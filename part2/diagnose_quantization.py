
"""Compare FP32 and INT8 TFLite outputs on identical images."""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "part1"))

from baseline_model import prepare_compression_datasets


def load_interpreter(name):
    path = ROOT / "models" / f"baseline_{name}.tflite"
    interpreter = tf.lite.Interpreter(
        model_path=str(path),
        num_threads=4,
    )
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, image):
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    x = image[np.newaxis].astype(np.float32)
    interpreter.set_tensor(input_info["index"], x)
    interpreter.invoke()

    return interpreter.get_tensor(output_info["index"])[0]


def main():
    _, _, x_val, y_val, _, _, _ = prepare_compression_datasets()

    fp32 = load_interpreter("float32")
    int8 = load_interpreter("int8")

    count = 200
    fp_outputs = []
    int_outputs = []

    for image in x_val[:count]:
        image = image.astype(np.float32)
        fp_outputs.append(predict(fp32, image))
        int_outputs.append(predict(int8, image))

    a = np.asarray(fp_outputs)
    b = np.asarray(int_outputs)

    labels = np.asarray(y_val[:count]).reshape(-1)

    fp_predictions = np.argmax(a, axis=1)
    int_predictions = np.argmax(b, axis=1)

    print(f"FP32 accuracy: {np.mean(fp_predictions == labels):.2%}")
    print(f"INT8 accuracy: {np.mean(int_predictions == labels):.2%}")
    print(f"Prediction agreement: {np.mean(fp_predictions == int_predictions):.2%}")
    print(f"Mean absolute output error: {np.mean(np.abs(a-b)):.6f}")
    print(f"Maximum absolute output error: {np.max(np.abs(a-b)):.6f}")

    fp_confidence = np.max(a, axis=1)
    int_confidence = np.max(b, axis=1)

    print(f"FP32 mean confidence: {np.mean(fp_confidence):.4f}")
    print(f"INT8 mean confidence: {np.mean(int_confidence):.4f}")

    print("\nWorst disagreement examples:")
    differences = np.max(np.abs(a-b), axis=1)
    worst = np.argsort(differences)[-10:][::-1]

    for idx in worst:
        print(
            f"Sample {idx}: "
            f"true={labels[idx]}, "
            f"FP32={fp_predictions[idx]}, "
            f"INT8={int_predictions[idx]}, "
            f"max_diff={differences[idx]:.4f}"
        )


if __name__ == "__main__":
    main()
