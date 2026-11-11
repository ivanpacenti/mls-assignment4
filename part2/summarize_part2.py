
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
OUTPUT = RESULTS / "summary"

OUTPUT.mkdir(parents=True, exist_ok=True)

# Results previously measured on all 5,000 validation images.
# Recorded here because the QAT evaluation script did not save
# a dedicated JSON file.
QAT_FULL_ACCURACY = 0.5272

BASELINE_ACCURACY = 0.7948


def load_json(filename):
    path = RESULTS / filename

    if not path.exists():
        print(f"Missing optional file: {path}")
        return None

    return json.loads(path.read_text())


def save_csv(filename, rows):
    if not rows:
        return

    path = OUTPUT / filename
    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def save_bar_chart(filename, title, labels, values, ylabel):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(range(len(labels)), values)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    path = OUTPUT / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"Saved: {path}")


def summarize_tflite():
    ptq = load_json("ptq_baseline.json")
    calibration = load_json("calibration_sensitivity.json")

    if not ptq:
        return []

    rows = []

    for name, data in ptq["quantized"].items():
        rows.append({
            "method": f"TFLite {name}",
            "accuracy": data["accuracy"],
            "size_mib": data["file_bytes"] / (1024 ** 2),
            "latency_median_ms": data.get("latency_median_ms"),
            "representation": "exported_tflite",
        })

    if calibration:
        for experiment in calibration["experiments"]:
            rows.append({
                "method": (
                    "PTQ INT8 calibration "
                    f"{experiment['calibration_samples']}"
                ),
                # Full-validation results measured separately.
                "accuracy": {
                    300: 0.5558,
                    1000: 0.5408,
                    2000: 0.5422,
                }.get(experiment["calibration_samples"]),
                "size_mib": experiment["model_size_mib"],
                "latency_median_ms": None,
                "representation": "exported_tflite",
            })

    qat_path = MODELS / "selective_qat_int8.tflite"

    if qat_path.exists():
        rows.append({
            "method": "Selective QAT INT8",
            "accuracy": QAT_FULL_ACCURACY,
            "size_mib": qat_path.stat().st_size / (1024 ** 2),
            "latency_median_ms": None,
            "representation": "exported_tflite",
        })

    save_csv("tflite_comparison.csv", rows)

    accuracy_rows = [
        row for row in rows if row["accuracy"] is not None
    ]

    save_bar_chart(
        "tflite_accuracy.png",
        "TFLite Quantization Accuracy on CIFAR-100",
        [r["method"] for r in accuracy_rows],
        [100 * r["accuracy"] for r in accuracy_rows],
        "Validation accuracy (%)",
    )

    return rows


def summarize_mixed_bit():
    data = load_json("mixed_bit_refinement.json")

    if not data:
        return []

    rows = []

    for name, result in data["experiments"].items():
        accuracy = result.get("full_validation_accuracy")

        rows.append({
            "configuration": name,
            "accuracy": accuracy,
            "accuracy_drop_pp": (
                (BASELINE_ACCURACY - accuracy) * 100
                if accuracy is not None else None
            ),
            "theoretical_kernel_compression": result[
                "theoretical_kernel_compression"
            ],
            "representation": "simulated_weights_fp32_storage",
        })

    save_csv("mixed_bit_comparison.csv", rows)

    valid = [r for r in rows if r["accuracy"] is not None]

    save_bar_chart(
        "mixed_bit_accuracy.png",
        "Mixed-Bit Quantization Accuracy",
        [r["configuration"] for r in valid],
        [r["accuracy"] * 100 for r in valid],
        "Validation accuracy (%)",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for row in valid:
        x = row["theoretical_kernel_compression"]
        y = row["accuracy"] * 100

        ax.scatter(x, y, s=60)
        ax.annotate(
            row["configuration"],
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Theoretical kernel compression ratio")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Mixed-Bit Accuracy vs Kernel Compression")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "mixed_bit_tradeoff.png",
        dpi=180,
    )
    plt.close(fig)

    return rows


def summarize_sensitivity():
    data = load_json("layer_quantization_sensitivity.json")

    if not data:
        return []

    rows = []

    for group, measurements in data["groups"].items():
        for bits, measurement in measurements.items():
            rows.append({
                "group": group,
                "bits": int(bits),
                "accuracy": measurement["accuracy"],
                "accuracy_drop_pp": measurement["accuracy_drop_pp"],
                "parameters": measurement["parameters"],
            })

    save_csv("layer_sensitivity.csv", rows)

    int4 = sorted(
        [r for r in rows if r["bits"] == 4],
        key=lambda r: r["accuracy_drop_pp"],
    )

    save_bar_chart(
        "int4_layer_sensitivity.png",
        "INT4 Sensitivity by Layer Group",
        [r["group"] for r in int4],
        [r["accuracy_drop_pp"] for r in int4],
        "Accuracy loss (percentage points)",
    )

    return rows


def summarize_binary():
    binary = load_json("binary_quantization.json")
    xnor = load_json("xnor_style.json")

    rows = []

    if binary:
        for name, experiment in binary["experiments"].items():
            rows.append({
                "experiment": name,
                "accuracy": experiment["accuracy"],
                "kernel_compression": experiment[
                    "theoretical_kernel_compression"
                ],
                "type": "binary_weight_simulation",
            })

    if xnor:
        for name, experiment in xnor["experiments"].items():
            rows.append({
                "experiment": f"xnor_{name}",
                "accuracy": experiment["accuracy"],
                "kernel_compression": None,
                "type": "classifier_only_simulation",
            })

    save_csv("binary_xnor_comparison.csv", rows)

    if rows:
        save_bar_chart(
            "binary_xnor_accuracy.png",
            "Binary and XNOR-Style Accuracy",
            [r["experiment"] for r in rows],
            [100 * r["accuracy"] for r in rows],
            "Validation accuracy (%)",
        )

    return rows


def write_report(tflite, mixed, sensitivity, binary):
    text = [
        "# Part 2: Advanced Quantization Techniques",
        "",
        "## Experimental setup",
        "",
        "- Model: EfficientNet-B0",
        "- Dataset: CIFAR-100",
        "- Validation samples: 5,000",
        "- FP32 baseline accuracy: 79.48%",
        "- Deployment format: TensorFlow Lite for PTQ/QAT",
        "- Mixed-bit, binary, and XNOR-style results: numerical simulations",
        "",
        "## TFLite quantization",
        "",
        "| Method | Accuracy | Size (MiB) | Median latency (ms) |",
        "|---|---:|---:|---:|",
    ]

    for row in tflite:
        accuracy = (
            f"{100 * row['accuracy']:.2f}%"
            if row["accuracy"] is not None else "N/A"
        )

        latency = (
            f"{row['latency_median_ms']:.2f}"
            if row["latency_median_ms"] is not None else "N/A"
        )

        text.append(
            f"| {row['method']} | {accuracy} | "
            f"{row['size_mib']:.2f} | {latency} |"
        )

    text.extend([
        "",
        "## Mixed-bit quantization",
        "",
        "| Configuration | Accuracy | Theoretical kernel compression |",
        "|---|---:|---:|",
    ])

    for row in mixed:
        accuracy = (
            f"{row['accuracy'] * 100:.2f}%"
            if row["accuracy"] is not None else "N/A"
        )

        text.append(
            f"| {row['configuration']} | {accuracy} | "
            f"{row['theoretical_kernel_compression']:.2f}x |"
        )

    text.extend([
        "",
        "## Binary and XNOR-style quantization",
        "",
        "| Experiment | Accuracy |",
        "|---|---:|",
    ])

    for row in binary:
        text.append(
            f"| {row['experiment']} | "
            f"{row['accuracy'] * 100:.2f}% |"
        )

    text.extend([
        "",
        "## Findings",
        "",
        "- Dynamic-range TFLite quantization preserves accuracy "
        "better than the tested full INT8 configurations.",
        "- Representative dataset selection materially affects "
        "INT8 PTQ accuracy.",
        "- Selective QAT of the classification head does not "
        "recover the accuracy loss from quantizing the backbone.",
        "- Mixed 8/4-bit weight quantization can preserve "
        "accuracy when sensitive groups retain 8-bit precision.",
        "- Binary activation quantization severely degrades "
        "the tested classifier without retraining.",
        "",
        "## Limitations",
        "",
        "- Mixed-bit compression ratios are theoretical estimates "
        "for kernel weights only.",
        "- The simulated low-bit weights remain stored as FP32.",
        "- Binary and XNOR-style simulations use floating-point "
        "operations rather than optimized custom kernels.",
        "- The XNOR-style test covers only the final classifier.",
        "- INT4 and binary experiments do not establish "
        "real deployment speedups.",
        "- Some results were selected using validation accuracy; "
        "they are not independent held-out test estimates.",
        "- Latency is only reported where it was actually measured.",
        "",
        "## Figures",
        "",
        "![TFLite accuracy](tflite_accuracy.png)",
        "",
        "![Mixed-bit tradeoff](mixed_bit_tradeoff.png)",
        "",
        "![INT4 sensitivity](int4_layer_sensitivity.png)",
        "",
        "![Binary and XNOR](binary_xnor_accuracy.png)",
    ])

    path = OUTPUT / "part2_summary.md"
    path.write_text("\n".join(text))

    print(f"Saved: {path}")


def main():
    tflite = summarize_tflite()
    mixed = summarize_mixed_bit()
    sensitivity = summarize_sensitivity()
    binary = summarize_binary()

    write_report(tflite, mixed, sensitivity, binary)

    print("\nPART 2 SUMMARY COMPLETE")


if __name__ == "__main__":
    main()
