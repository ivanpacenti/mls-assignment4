
import os
import sys
from collections import Counter
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path("part1").resolve()))
from baseline_model import prepare_compression_datasets


def main():
    model_path = "part2/models/selective_qat_int8.tflite"

    _, _, x_val, y_val, _, _, _ = prepare_compression_datasets()

    x_val = np.asarray(x_val[:200], dtype=np.float32)
    y_val = np.asarray(y_val[:200]).reshape(-1)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct = 0

    for image, label in zip(x_val, y_val):
        batch = image[np.newaxis, ...]

        interpreter.set_tensor(
            input_details["index"],
            batch,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        prediction = int(np.argmax(output[0]))
        correct += int(prediction == int(label))

    accuracy = correct / len(x_val)

    print(f"\nTFLite QAT accuracy: {accuracy:.2%}")
    print(f"Samples evaluated: {len(x_val)}")

    size_mib = Path(model_path).stat().st_size / (1024 ** 2)
    print(f"Model size: {size_mib:.2f} MiB")

    print("\nInput dtype:", input_details["dtype"])
    print("Output dtype:", output_details["dtype"])

    tensor_types = Counter(
        str(tensor["dtype"])
        for tensor in interpreter.get_tensor_details()
    )

    print("\nTensor types:")
    for dtype, count in tensor_types.items():
        print(f"  {dtype}: {count}")

    operation_types = Counter(
        op["op_name"]
        for op in interpreter._get_ops_details()
    )

    print("\nTop operations:")
    for name, count in operation_types.most_common(15):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
