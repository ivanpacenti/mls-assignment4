
"""Part 2: Quantization experiments on CIFAR-100."""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PART1 = PROJECT_ROOT / "part1"

if os.getenv("PART2_CPU_ONLY", "1") == "1":
    tf.config.set_visible_devices([], "GPU")

sys.path.insert(0, str(PART1))
sys.path.insert(0, str(ROOT))

from baseline_model import prepare_compression_datasets


class QuantizationPipeline:
    def __init__(self, model):
        self.model = model
        self.quantization_results = {}

        self.output_dir = ROOT / "models"
        self.results_dir = ROOT / "results"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
            self.calibration_data,
        ) = prepare_compression_datasets()

    def representative_dataset(self, count=1000):
        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(self.x_train),
            size=min(count, len(self.x_train)),
            replace=False,
        )

        for idx in indices:
            image = self.x_train[idx].astype(np.float32)
            yield [np.expand_dims(image, axis=0)]

    def convert_tflite(self, name, mode):
        converter = tf.lite.TFLiteConverter.from_keras_model(
            self.model
        )

        if mode == "float32":
            pass

        elif mode == "dynamic":
            converter.optimizations = [
                tf.lite.Optimize.DEFAULT
            ]

        elif mode == "int8":
            converter.optimizations = [
                tf.lite.Optimize.DEFAULT
            ]
            converter.representative_dataset = (
                self.representative_dataset
            )
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        start = time.perf_counter()
        converted = converter.convert()
        seconds = time.perf_counter() - start

        path = self.output_dir / f"{name}.tflite"
        path.write_bytes(converted)

        return path, seconds

    @staticmethod
    def evaluate_tflite(path, images, labels):
        interpreter = tf.lite.Interpreter(
            model_path=str(path),
            num_threads=4,
        )

        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]

        # Fixed batch size of 1 simplifies latency measurement.
        interpreter.resize_tensor_input(
            input_info["index"],
            [1, *images.shape[1:]],
        )
        interpreter.allocate_tensors()

        correct = 0
        durations = []

        for image, label in zip(images, labels):
            batch = np.expand_dims(image, axis=0).astype(
                input_info["dtype"]
            )

            interpreter.set_tensor(input_info["index"], batch)

            start = time.perf_counter()
            interpreter.invoke()
            durations.append(
                (time.perf_counter() - start) * 1000
            )

            prediction = interpreter.get_tensor(
                output_info["index"]
            )

            if int(np.argmax(prediction)) == int(np.squeeze(label)):
                correct += 1

        return {
            "accuracy": correct / len(images),
            "latency_median_ms": float(np.median(durations)),
            "latency_mean_ms": float(np.mean(durations)),
            "samples": len(images),
            "input_dtype": str(input_info["dtype"]),
            "output_dtype": str(output_info["dtype"]),
        }

    def run_ptq_baseline(self, evaluation_samples=1000):
        """Compare dynamic and calibrated INT8 PTQ."""
        x = self.x_val[:evaluation_samples].astype(np.float32)
        y = self.y_val[:evaluation_samples]

        baseline_predictions = self.model.predict(
            x, batch_size=64, verbose=0
        )
        baseline_accuracy = float(np.mean(
            np.argmax(baseline_predictions, axis=1)
            == np.asarray(y).reshape(-1)
        ))

        baseline_bytes = (
            PART1 / "baseline_model.keras"
        ).stat().st_size

        results = {
            "baseline": {
                "accuracy": baseline_accuracy,
                "file_bytes": baseline_bytes,
            },
            "quantized": {},
        }

        for mode in ("float32","dynamic", "int8"):
            print(f"Converting {mode}...", flush=True)

            path, conversion_seconds = self.convert_tflite(
                f"baseline_{mode}", mode
            )

            metrics = self.evaluate_tflite(path, x, y)
            size_bytes = path.stat().st_size

            metrics.update({
                "file": str(path),
                "file_bytes": size_bytes,
                "conversion_seconds": conversion_seconds,
                "compression_ratio_vs_keras": (
                    baseline_bytes / size_bytes
                ),
                "accuracy_drop_pp": (
                    baseline_accuracy - metrics["accuracy"]
                ) * 100,
            })

            results["quantized"][mode] = metrics

            print(
                f"{mode}: accuracy={metrics['accuracy']:.2%}, "
                f"size={size_bytes / 1024**2:.2f} MiB, "
                f"ratio={metrics['compression_ratio_vs_keras']:.2f}x",
                flush=True,
            )

        output = self.results_dir / "ptq_baseline.json"
        output.write_text(json.dumps(results, indent=2))

        return results


    # Methods below retain the exact signatures of the assignment starter code.
    # Other part2/*.py files remain the experimental implementations.

    def _run_experiment(self, filename):
        """Run an existing experiment in its own TF process."""
        script = ROOT / filename
        if not script.is_file():
            raise FileNotFoundError(script)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PART2_CPU_ONLY": "1"},
            check=True,
        )

    def _load_result(self, filename):
        path = self.results_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text())

    def mixed_bit_quantization(self, bit_configurations=[8, 4, 2]):
        """Gradient-sensitive mixed-bit weight fake quantization.

        Quantizes in-memory kernels for evaluation, then restores FP32 values.
        Reported compression is theoretical kernel storage, not TFLite size.
        """
        from mixed_bit_analysis import (
            calculate_sensitivity, group_name, quantize_weights, evaluate
        )

        bits = list(dict.fromkeys(int(b) for b in bit_configurations))
        if not bits or any(b < 2 or b > 16 for b in bits):
            raise ValueError("Use bit widths between 2 and 16")
        bits.sort(reverse=True)

        x_val = np.asarray(self.x_val[:200], dtype=np.float32)
        y_val = np.asarray(self.y_val[:200]).reshape(-1)
        kernels = [v for v in self.model.weights if "kernel" in v.name.lower()]
        if not kernels:
            raise RuntimeError("The model has no kernel tensors")
        original = [v.numpy().copy() for v in kernels]
        total_fp32_bits = sum(a.size * 32 for a in original)

        baseline_accuracy = evaluate(self.model, x_val, y_val)
        sensitivity = calculate_sensitivity(
            self.model,
            np.asarray(self.x_train[:24], dtype=np.float32),
            np.asarray(self.y_train[:24]).reshape(-1),
        )
        if not sensitivity:
            raise RuntimeError("Gradient sensitivity returned no groups")

        ranked = sorted(sensitivity, key=sensitivity.get, reverse=True)
        assignments = {
            "uniform_" + str(b): {g: b for g in ranked}
            for b in bits
        }
        assignments["mixed_precision"] = {
            group: bits[min((i * len(bits)) // len(ranked), len(bits) - 1)]
            for i, group in enumerate(ranked)
        }
        if 8 in bits and 4 in bits:
            for groups in (("top",), ("top", "head")):
                assignment = {group: 8 for group in ranked}

                for group in groups:
                    if group in assignment:
                        assignment[group] = 4

                name = "_".join(groups) + "_4"
                assignments[name] = assignment
        experiments = {}
        try:
            for name, assigned in assignments.items():
                storage_bits = 0
                for variable, weight in zip(kernels, original):
                    group = group_name(variable)
                    width = assigned.get(group, bits[0])
                    variable.assign(quantize_weights(weight, width))
                    storage_bits += int(weight.size) * width

                accuracy = evaluate(self.model, x_val, y_val)
                experiments[name] = {
                    "accuracy": accuracy,
                    "accuracy_drop_pp": (baseline_accuracy - accuracy) * 100,
                    "theoretical_kernel_compression": total_fp32_bits / storage_bits,
                    "bit_configuration": assigned,
                }

                for variable, weight in zip(kernels, original):
                    variable.assign(weight)
        finally:
            for variable, weight in zip(kernels, original):
                variable.assign(weight)
        max_accuracy_drop_pp = 2.0

        eligible = {
            name: metrics
            for name, metrics in experiments.items()
            if metrics["accuracy_drop_pp"] <= max_accuracy_drop_pp + 1e-6
        }

        if not eligible:
            raise RuntimeError(
                "No mixed-bit configuration satisfies the accuracy constraint"
            )

        selected_name = max(
            eligible,
            key=lambda name: eligible[name]["theoretical_kernel_compression"],
        )
        result = {
            "method": "symmetric weight fake quantization, no packed deployment kernels",
            "evaluation_samples": len(x_val),
            "baseline_accuracy": baseline_accuracy,
            "sensitivity_analysis": sensitivity,
            "optimal_bit_assignment": assignments[selected_name],
            "selected_configuration": selected_name,
            "selected_metrics": experiments[selected_name],
            "maximum_allowed_accuracy_drop_pp": max_accuracy_drop_pp,
            "mixed_bit_models": experiments,
            "compression_analysis": {
                name: value["theoretical_kernel_compression"]
                for name, value in experiments.items()
            },
        }
        self.quantization_results["mixed_bit"] = result
        (self.results_dir / "mixed_bit_pipeline.json").write_text(
            json.dumps(result, indent=2)
        )
        return result

    def post_training_vs_qat_comparison(self, bit_widths=[8, 4]):
        """Compare deployable 8-bit PTQ with selective-head QAT.

        The QAT subprocess trains its quantized head if a checkpoint is absent;
        otherwise it evaluates a previously trained checkpoint.
        """
        widths = set(int(b) for b in bit_widths)
        if not widths or not widths.issubset({4, 8}):
            raise ValueError("Supported comparison bit widths: 8 and 4")

        ptq = self.run_ptq_baseline(evaluation_samples=200)
        result = {
            "requested_bit_widths": sorted(widths),
            "ptq_results": ptq,
            "qat_results": {},
            "accuracy_comparison": {},
            "training_efficiency": {"qat_training_reused_when_checkpoint_exists": True},
            "calibration_analysis": {},
            "notes": [
                "4-bit experiments are simulated weight quantization, not TFLite INT4.",
                "Selective QAT trains only its head; the backbone remains full precision.",
            ],
        }
        result["accuracy_comparison"]["ptq_dynamic_vs_int8_pp"] = (
            ptq["quantized"]["dynamic"]["accuracy"]
            - ptq["quantized"]["int8"]["accuracy"]
        ) * 100

        if 4 in widths:
            result["ptq_results"]["simulated_4bit"] = (
                self.mixed_bit_quantization([8, 4])["mixed_bit_models"]["uniform_4"]
            )

        if 8 in widths:
            checkpoint = ROOT / "checkpoints/selective_qat_weights.h5"
            qat_file = ROOT / "models/selective_qat_int8.tflite"
            if not checkpoint.is_file() or not qat_file.is_file():
                self._run_experiment("test_selective_qat.py")
            x = np.asarray(self.x_val[:200], dtype=np.float32)
            y = np.asarray(self.y_val[:200]).reshape(-1)
            qat_metrics = self.evaluate_tflite(qat_file, x, y)
            qat_metrics["file_bytes"] = qat_file.stat().st_size
            qat_metrics["checkpoint"] = str(checkpoint)
            result["qat_results"]["selective_qat_int8"] = qat_metrics
            result["accuracy_comparison"]["qat_vs_int8_ptq_pp"] = (
                qat_metrics["accuracy"] - ptq["quantized"]["int8"]["accuracy"]
            ) * 100

        self.quantization_results["ptq_vs_qat"] = result
        (self.results_dir / "ptq_vs_qat_comparison.json").write_text(
            json.dumps(result, indent=2)
        )
        return result

    def extreme_quantization(self):
        """Run existing binary and XNOR numerical simulations plus 4-bit test."""
        from mixed_bit_analysis import quantize_weights, evaluate

        x = np.asarray(self.x_val[:200], dtype=np.float32)
        y = np.asarray(self.y_val[:200]).reshape(-1)
        kernels = [w for w in self.model.weights if "kernel" in w.name.lower()]
        originals = [w.numpy().copy() for w in kernels]
        try:
            for variable, original in zip(kernels, originals):
                variable.assign(quantize_weights(original, 4))
            int4_accuracy = evaluate(self.model, x, y)
        finally:
            for variable, original in zip(kernels, originals):
                variable.assign(original)

        self._run_experiment("binary_quantization.py")
        self._run_experiment("xnor_style.py")
        result = {
            "int4_quantization": {
                "accuracy": int4_accuracy,
                "samples": len(x),
                "type": "weight fake quantization (not packed INT4 / custom kernel)",
            },
            "binary_quantization": self._load_result("binary_quantization.json"),
            "xnor_style": self._load_result("xnor_style.json"),
            "performance_improvements": {
                "status": "not measured",
                "note": "Numerical simulations cannot establish XNOR hardware speedups",
            },
        }
        self.quantization_results["extreme"] = result
        (self.results_dir / "extreme_quantization.json").write_text(
            json.dumps(result, indent=2)
        )
        return result

    def dynamic_quantization_analysis(self):
        """Measure float32, dynamic-range, and calibrated INT8 TFLite models."""
        ptq = self.run_ptq_baseline(evaluation_samples=200)
        result = {
            "float32": ptq["quantized"]["float32"],
            "dynamic": ptq["quantized"]["dynamic"],
            "static_int8": ptq["quantized"]["int8"],
        }
        self.quantization_results["dynamic_vs_static"] = result
        return result

    def quantization_error_analysis(self):
        """Compare corresponding float32 / INT8 operator activation tensors."""
        for name in ("baseline_float32.tflite", "baseline_int8.tflite"):
            if not (self.output_dir / name).exists():
                raise FileNotFoundError(
                    f"Missing {name}. Run dynamic_quantization_analysis() first."
                )
        self._run_experiment("activation_analysis.py")
        result = self._load_result("activation_analysis.json")
        self.quantization_results["quantization_error"] = result
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["ptq", "mixed", "qat", "extreme", "dynamic", "error", "all"],
        default="ptq",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    model_path = PART1 / "baseline_model.keras"
    model = tf.keras.models.load_model(
        model_path, compile=False
    )

    pipeline = QuantizationPipeline(model)

    runners = {
        "ptq": lambda: pipeline.run_ptq_baseline(args.samples),
        "mixed": pipeline.mixed_bit_quantization,
        "qat": pipeline.post_training_vs_qat_comparison,
        "extreme": pipeline.extreme_quantization,
        "dynamic": pipeline.dynamic_quantization_analysis,
        "error": pipeline.quantization_error_analysis,
    }
    if args.experiment == "all":
        for name, runner in runners.items():
            print(f"\n=== {name} ===", flush=True)
            runner()
    else:
        runners[args.experiment]()


if __name__ == "__main__":
    main()
