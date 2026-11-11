
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


def read(name):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def save_csv(filename, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(
        key for row in rows for key in row
    ))
    with (OUTPUT / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows):
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [r["method"] for r in rows]
    accuracies = [r["accuracy"] * 100 for r in rows]

    ax.bar(range(len(rows)), accuracies)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0, max(accuracies) * 1.15)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Knowledge Distillation Comparison")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT / "distillation_comparison.png", dpi=180)
    plt.close(fig)


def plot_temperature(initial, extended):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name in ("scratch", "T_1", "T_2", "T_4", "T_8"):
        history = []

        if name in initial["trials"]:
            history += [
                (
                    item["epoch"],
                    item["validation_accuracy_1000"] * 100,
                )
                for item in initial["trials"][name]["history"]
            ]

        if name in extended["trials"]:
            history += [
                (
                    item["total_epoch"],
                    item["validation_accuracy_1000"] * 100,
                )
                for item in extended["trials"][name]["history"]
            ]

        if history:
            ax.plot(
                [x for x, _ in history],
                [y for _, y in history],
                label=name,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy on selection subset (%)")
    ax.set_title("Temperature Optimization: Convergence")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT / "temperature_convergence.png", dpi=180)
    plt.close(fig)


def plot_loss(experiment, filename, title, loss_key):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, trial in experiment["trials"].items():
        history = trial["history"]
        ax.plot(
            [item["epoch"] for item in history],
            [item[loss_key] for item in history],
            label=name,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_key)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=180)
    plt.close(fig)


def main():
    initial = read("temperature_optimization.json")
    extended = read("temperature_extended.json")
    progressive = read("progressive_distillation.json")
    attention = read("attention_transfer.json")
    features = read("feature_matching.json")

    rows = []

    def add(method, accuracy, category, parameters=317764):
        rows.append({
            "method": method,
            "accuracy": float(accuracy),
            "parameters": parameters,
            "category": category,
        })

    for name, result in extended["trials"].items():
        add(
            f"Temperature {name}",
            result["full_validation_accuracy"],
            "temperature_search",
        )

    for name, result in progressive["stages"].items():
        add(
            name,
            result["full_validation_accuracy"],
            "progressive_distillation",
            result["parameters"],
        )

    for name, result in attention["trials"].items():
        add(
            name,
            result["full_validation_accuracy"],
            "attention_transfer",
            result["parameters"],
        )

    for name, result in features["trials"].items():
        add(
            name,
            result["full_validation_accuracy"],
            "feature_matching",
            result["student_parameters"],
        )

    save_csv("all_distillation_results.csv", rows)
    plot_accuracy(rows)
    plot_temperature(initial, extended)

    plot_loss(
        attention,
        "attention_loss.png",
        "Attention Loss Convergence",
        "attention_loss",
    )

    plot_loss(
        features,
        "feature_matching_loss.png",
        "Feature Matching Loss Convergence",
        "feature_loss",
    )

    lines = [
        "# Part 3: Knowledge Distillation Optimization",
        "",
        "## Configuration",
        "",
        "- Dataset: CIFAR-100",
        "- Teacher: EfficientNet-B0",
        "- Baseline teacher accuracy: 79.48%",
        "- Final student parameters: 317,764",
        "- Training subset: 12,000 images",
        "- Full validation: 5,000 images",
        "",
        "## Results",
        "",
        "| Method | Accuracy | Parameters |",
        "|---|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['method']} | "
            f"{row['accuracy'] * 100:.2f}% | "
            f"{row['parameters']:,} |"
        )

    lines += [
        "",
        "## Findings",
        "",
        "- Temperature selection depends on training duration.",
        "- Progressive distillation improved the final student "
        "by 0.74 percentage points against its direct-KD control.",
        "- Attention Transfer reduced classification accuracy "
        "for both tested coefficients.",
        "- Feature Matching achieved a 0.62 percentage-point "
        "improvement with gamma=0.1.",
        "",
        "## Methodological limitations",
        "",
        "- The temperature sweep and later comparisons use "
        "different training schedules.",
        "- The progressive approach requires two additional "
        "intermediate training stages.",
        "- Results are from a single random seed.",
        "- Checkpoint selection used a 1,000-image validation subset.",
        "- The 5,000-image validation accuracy is not an "
        "independent test-set measurement.",
        "- Training-time adapters are excluded from the final "
        "student parameter count.",
        "",
        "## Visualizations",
        "",
        "![Accuracy](distillation_comparison.png)",
        "",
        "![Temperature](temperature_convergence.png)",
        "",
        "![Attention loss](attention_loss.png)",
        "",
        "![Feature loss](feature_matching_loss.png)",
    ]

    (OUTPUT / "part3_summary.md").write_text(
        "\n".join(lines)
    )

    print("\nPART 3 SUMMARY COMPLETE")
    print(f"Saved files in: {OUTPUT}")

    for file in sorted(OUTPUT.iterdir()):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()
