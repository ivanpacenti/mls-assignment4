
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "part1"))

from baseline_model import prepare_compression_datasets


ROOT = PROJECT / "part4"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RESULT_PATH = RESULTS / "pruning_quantization.json"

BASELINE_PATH = PROJECT / "part1/baseline_model.keras"

# Paths from Part 1 experiment outputs.
SOURCE_MODELS = {
    "baseline": BASELINE_PATH,
    "structured": (
        PROJECT / "part1/checkpoints/structured.keras"
    ),
    "magnitude": (
        PROJECT / "part1/checkpoints/magnitude.keras"
    ),
}

LATENCY_SAMPLES = 200
WARMUP_RUNS = 20
SEED = 42


def convert_dynamic(model_path, output_path):
    print(f"Converting: {model_path.name}", flush=True)

    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    print(
        f"Saved: {output_path}",
        flush=True,
    )

    del model

    return output_path


def get_tflite_model(name, source):
    output = MODELS / f"{name}_dynamic.tflite"

    if output.exists():
        print(f"Reusing: {output}")
        return output

    if not source.exists():
        raise FileNotFoundError(
            f"Required Part 1 checkpoint missing: {source}"
        )

    return convert_dynamic(source, output)


def create_interpreter(model_path):
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
    )

    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            "Expected one input and one output tensor"
        )

    input_info = inputs[0]
    output_info = outputs[0]

    if input_info["dtype"] != np.float32:
        raise RuntimeError(
            f"Unsupported input dtype: {input_info['dtype']}"
        )

    return interpreter, input_info, output_info


def evaluate_tflite(model_path, images, labels):
    interpreter, input_info, output_info = (
        create_interpreter(model_path)
    )

    input_index = input_info["index"]
    output_index = output_info["index"]

    correct = 0

    start = time.perf_counter()

    for image, label in zip(images, labels):
        batch = np.asarray(
            image[np.newaxis, ...],
            dtype=np.float32,
        )

        interpreter.set_tensor(input_index, batch)
        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_index
        )

        predicted_class = int(
            np.argmax(prediction[0])
        )

        correct += predicted_class == int(label)

    evaluation_seconds = (
        time.perf_counter() - start
    )

    return {
        "accuracy": float(correct / len(labels)),
        "evaluation_seconds": evaluation_seconds,
        "evaluation_samples": int(len(labels)),
    }


def benchmark_latency(model_path, images):
    interpreter, input_info, _ = create_interpreter(
        model_path
    )

    input_index = input_info["index"]

    inputs = np.asarray(
        images[:LATENCY_SAMPLES],
        dtype=np.float32,
    )

    # Warm up the interpreter and delegated kernels.
    for index in range(WARMUP_RUNS):
        sample = inputs[index % len(inputs)]

        interpreter.set_tensor(
            input_index,
            sample[np.newaxis, ...],
        )
        interpreter.invoke()

    times = []

    for sample in inputs:
        interpreter.set_tensor(
            input_index,
            sample[np.newaxis, ...],
        )

        start = time.perf_counter()
        interpreter.invoke()
        elapsed = time.perf_counter() - start

        times.append(elapsed * 1000)

    return {
        "median_latency_ms": float(np.median(times)),
        "mean_latency_ms": float(np.mean(times)),
        "p95_latency_ms": float(
            np.percentile(times, 95)
        ),
        "latency_samples": len(times),
        "warmup_runs": WARMUP_RUNS,
        "num_threads": 1,
    }


def main():
    if os.environ.get("PART4_CPU_ONLY") == "1":
        tf.config.set_visible_devices([], "GPU")

    tf.keras.utils.set_random_seed(SEED)

    missing = {
        name: str(path)
        for name, path in SOURCE_MODELS.items()
        if not path.exists()
    }

    if missing:
        raise FileNotFoundError(
            "Missing model checkpoints:\n"
            + json.dumps(missing, indent=2)
        )

    _, _, x_val, y_val, _, _, _ = (
        prepare_compression_datasets()
    )

    x_val = np.asarray(x_val, dtype=np.float32)
    y_val = np.asarray(y_val).reshape(-1)

    if len(x_val) != len(y_val):
        raise ValueError(
            "Image and label counts do not match"
        )

    results = {
        "configuration": {
            "validation_samples": len(x_val),
            "latency_samples": LATENCY_SAMPLES,
            "num_threads": 1,
            "quantization": "dynamic_range",
            "seed": SEED,
        },
        "models": {},
    }

    if RESULT_PATH.exists():
        previous = json.loads(RESULT_PATH.read_text())

        if previous.get("configuration") != results["configuration"]:
            raise RuntimeError(
                "Existing results use different settings"
            )

        results = previous

    # Evaluate in a single run using the same CPU settings.
    for name, source in SOURCE_MODELS.items():
        key = f"{name}_dynamic"

        if key in results["models"]:
            saved_path = Path(
                results["models"][key]["model_path"]
            )
            if not saved_path.exists():
                raise FileNotFoundError(saved_path)
            print(f"Skipping completed model: {key}")
            continue

        print(f"\n=== {key} ===", flush=True)

        model_path = get_tflite_model(
            name, source
        )

        metrics = evaluate_tflite(
            model_path, x_val, y_val
        )

        latency = benchmark_latency(
            model_path, x_val
        )

        results["models"][key] = {
            "source_checkpoint": str(source),
            "model_path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "size_mib": (
                model_path.stat().st_size / 1024 ** 2
            ),
            **metrics,
            **latency,
        }

        RESULT_PATH.write_text(
            json.dumps(results, indent=2)
        )

        item = results["models"][key]

        print(
            f"Accuracy: {item['accuracy']:.2%}\n"
            f"Size: {item['size_mib']:.2f} MiB\n"
            f"Median latency: "
            f"{item['median_latency_ms']:.3f} ms",
            flush=True,
        )

    baseline = results["models"]["baseline_dynamic"]

    print("\n=== COMPRESSION COMPARISON ===")

    for name, item in results["models"].items():
        size_ratio = (
            baseline["size_bytes"] / item["size_bytes"]
        )

        latency_ratio = (
            baseline["median_latency_ms"]
            / item["median_latency_ms"]
        )

        accuracy_delta = (
            item["accuracy"] - baseline["accuracy"]
        ) * 100

        print(
            f"{name:20s} "
            f"accuracy={item['accuracy']:.2%} "
            f"delta={accuracy_delta:+.2f}pp "
            f"size={item['size_mib']:.2f}MiB "
            f"relative_size_compression={size_ratio:.3f}x "
            f"relative_latency_speedup={latency_ratio:.3f}x"
        )

    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
