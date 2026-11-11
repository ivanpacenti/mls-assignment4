
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "summary"
OUTPUT.mkdir(parents=True, exist_ok=True)

REFERENCE = RESULTS / "pruning_quantization.json"
DISTILLATION = RESULTS / "distillation_quantization.json"
COMBINED = RESULTS / "combined_compression.json"
CONTROL = RESULTS / "unpruned_finetune_control.json"
ORDERING = RESULTS / "ordering_analysis.json"


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def collect():
    pruning = load(REFERENCE)
    distillation = load(DISTILLATION)
    combined = load(COMBINED)
    control = load(CONTROL)
    ordering = load(ORDERING)

    rows = []

    def add(name, category, accuracy, size, latency,
            sparsity=0.0, parameters=None):
        rows.append({
            "name": name,
            "category": category,
            "accuracy": float(accuracy),
            "size_mib": float(size),
            "latency_ms": float(latency),
            "kernel_sparsity": float(sparsity),
            "parameters": parameters,
        })

    for name, item in pruning["models"].items():
        add(
            name,
            "pruning_quantization",
            item["accuracy"],
            item["size_mib"],
            item["median_latency_ms"],
        )

    for name, item in distillation["models"].items():
        add(
            name,
            "distillation_quantization",
            item["tflite_accuracy"],
            item["size_mib"],
            item["median_latency_ms"],
            parameters=item["parameters"],
        )

    add(
        "kd_prune_0",
        "finetune_control",
        control["tflite_accuracy"],
        control["size_mib"],
        control["median_latency_ms"],
        parameters=317764,
    )

    for name, item in combined["experiments"].items():
        add(
            name,
            "combined_compression",
            item["tflite_accuracy"],
            item["size_mib"],
            item["median_latency_ms"],
            sparsity=item["final_kernel_sparsity"],
            parameters=317764,
        )

    for name, item in ordering["experiments"].items():
        alternative = item["order_b"]

        add(
            f"{name}_finetune_first",
            "ordering_finetune_first",
            alternative["accuracy"],
            alternative["size_mib"],
            alternative["median_latency_ms"],
            sparsity=alternative["achieved_sparsity"],
            parameters=317764,
        )

    return rows, ordering



def pareto_frontier(rows, size_tolerance_mib=0.01):
    """
    Practical Pareto frontier.

    File sizes within the same 0.01 MiB bucket are treated
    as equivalent. Within each bucket, retain the model
    with the highest accuracy.

    Objectives:
      - maximize validation accuracy
      - minimize actual TFLite file size
    """
    buckets = {}

    for item in rows:
        size_bucket = round(
            item["size_mib"] / size_tolerance_mib
        )

        current = buckets.get(size_bucket)

        if (
            current is None
            or item["accuracy"] > current["accuracy"]
        ):
            buckets[size_bucket] = item

    candidates = list(buckets.values())

    frontier = []

    for candidate in candidates:
        dominated = any(
            other is not candidate
            and other["accuracy"] >= candidate["accuracy"]
            and other["size_mib"]
                <= candidate["size_mib"]
            and (
                other["accuracy"] > candidate["accuracy"]
                or other["size_mib"]
                    < candidate["size_mib"]
            )
            for other in candidates
        )

        if not dominated:
            frontier.append(candidate)

    return sorted(
        frontier,
        key=lambda item: item["size_mib"],
    )



def plot_pareto(rows, frontier):
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = sorted(set(
        item["category"] for item in rows
    ))

    for group in groups:
        subset = [
            item for item in rows
            if item["category"] == group
        ]

        ax.scatter(
            [item["size_mib"] for item in subset],
            [item["accuracy"] * 100 for item in subset],
            label=group.replace("_", " "),
            s=45,
            alpha=0.75,
        )

    for item in frontier:
        ax.annotate(
            item["name"],
            (
                item["size_mib"],
                item["accuracy"] * 100,
            ),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("TFLite model size (MiB)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Compression Pareto Frontier")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "pareto_frontier.png", dpi=180
    )
    plt.close(fig)


def plot_ordering(ordering):
    labels = []
    order_a = []
    order_b = []

    for key, item in ordering["experiments"].items():
        labels.append(key)
        order_a.append(
            item["order_a"]["accuracy"] * 100
        )
        order_b.append(
            item["order_b"]["accuracy"] * 100
        )

    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        [x - 0.2 for x in positions],
        order_a,
        width=0.4,
        label="Prune then fine-tune",
    )
    ax.bar(
        [x + 0.2 for x in positions],
        order_b,
        width=0.4,
        label="Fine-tune then prune",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("TFLite accuracy (%)")
    ax.set_title("Compression Ordering Sensitivity")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "ordering_sensitivity.png",
        dpi=180,
    )
    plt.close(fig)


def plot_sparsity(rows):
    selected = {
        item["name"]: item
        for item in rows
        if item["name"] in (
            "kd_prune_0",
            "kd_prune_25",
            "kd_prune_50",
        )
    }

    names = [
        "kd_prune_0",
        "kd_prune_25",
        "kd_prune_50",
    ]

    if not all(name in selected for name in names):
        raise ValueError(
            "Missing pruning control experiments"
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        [0, 25, 50],
        [
            selected[name]["accuracy"] * 100
            for name in names
        ],
        marker="o",
    )

    ax.set_xlabel("Kernel sparsity (%)")
    ax.set_ylabel("TFLite accuracy (%)")
    ax.set_title("Sparsity vs Accuracy after Fine-tuning")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "sparsity_accuracy.png",
        dpi=180,
    )
    plt.close(fig)


def save_results(rows, frontier, ordering):
    fields = [
        "name", "category", "accuracy",
        "size_mib", "latency_ms",
        "kernel_sparsity", "parameters",
    ]

    with (OUTPUT / "all_compression_results.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT / "pareto_frontier.json").write_text(
        json.dumps(frontier, indent=2)
    )

    lines = [
        "# Part 4: Compression Technique Interaction Analysis",
        "",
        "## Experimental setup",
        "",
        "- Dataset: CIFAR-100",
        "- Validation samples: 5,000",
        "- Deployment format: TensorFlow Lite",
        "- Quantization: dynamic range",
        "- Latency: single-thread CPU, batch size 1",
        "- Platform: Apple Silicon",
        "",
        "## Results",
        "",
        "| Model | Accuracy | Size (MiB) | "
        "Latency (ms) | Kernel sparsity |",
        "|---|---:|---:|---:|---:|",
    ]

    for item in sorted(
        rows,
        key=lambda row: row["accuracy"],
        reverse=True,
    ):
        lines.append(
            f"| {item['name']} "
            f"| {item['accuracy'] * 100:.2f}% "
            f"| {item['size_mib']:.3f} "
            f"| {item['latency_ms']:.3f} "
            f"| {item['kernel_sparsity'] * 100:.0f}% |"
        )

    lines += [
        "",
        "## Pareto frontier: accuracy vs actual file size",
        "",
        "| Model | Accuracy | Size (MiB) |",
        "|---|---:|---:|",
    ]

    for item in frontier:
        lines.append(
            f"| {item['name']} "
            f"| {item['accuracy'] * 100:.2f}% "
            f"| {item['size_mib']:.3f} |"
        )

    lines += [
        "",
        "## Ordering sensitivity",
        "",
        "| Sparsity | Prune then fine-tune | "
        "Fine-tune then prune | Difference |",
        "|---|---:|---:|---:|",
    ]

    for item in ordering["experiments"].values():
        lines.append(
            f"| {item['sparsity'] * 100:.0f}% "
            f"| {item['order_a']['accuracy'] * 100:.2f}% "
            f"| {item['order_b']['accuracy'] * 100:.2f}% "
            f"| {item['order_a_minus_b_pp']:+.2f} pp |"
        )

    lines += [
        "",
        "## Main observations",
        "",
        "- Structured pruning followed by dynamic quantization "
        "improved accuracy and reduced model size compared "
        "with the quantized EfficientNet baseline.",
        "- Student model quantization preserved the main "
        "accuracy differences between distillation methods.",
        "- Fine-tuning accounted for most of the accuracy "
        "recovery after student pruning.",
        "- Applying pruning after fine-tuning without a "
        "recovery stage caused major accuracy degradation, "
        "especially at 50% sparsity.",
        "- Unstructured sparsity did not reduce actual "
        "TFLite model size in this export configuration.",
        "",
        "## Limitations",
        "",
        "- Results use a single random seed.",
        "- Validation accuracy is not independent test accuracy.",
        "- Different model families have different training budgets.",
        "- Latency varies across runs and hardware environments.",
        "- The ordering comparison also changes whether "
        "recovery training follows pruning.",
        "- Dynamic-range quantization is not equivalent "
        "to fully integer-only inference.",
        "",
        "## Figures",
        "",
        "![Pareto frontier](pareto_frontier.png)",
        "",
        "![Ordering sensitivity](ordering_sensitivity.png)",
        "",
        "![Sparsity accuracy](sparsity_accuracy.png)",
    ]

    (OUTPUT / "part4_summary.md").write_text(
        "\n".join(lines)
    )


def main():
    rows, ordering = collect()

    frontier = pareto_frontier(rows)

    plot_pareto(rows, frontier)
    plot_ordering(ordering)
    plot_sparsity(rows)
    save_results(rows, frontier, ordering)

    print("\nPART 4 SUMMARY COMPLETE")
    print(f"Models evaluated: {len(rows)}")
    print(f"Pareto frontier entries: {len(frontier)}")

    print("\nPareto-optimal models:")
    for item in frontier:
        print(
            f"  {item['name']:30s} "
            f"accuracy={item['accuracy']:.2%} "
            f"size={item['size_mib']:.3f}MiB"
        )

    print(f"\nOutput: {OUTPUT}")

    for file in sorted(OUTPUT.iterdir()):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()
