# MLS Assignment 4: Part 1

Root-level `uv` project, with all Part 1 code and outputs under `part1/`.

## Setup on Apple Silicon

```bash
uv python install 3.11
uv sync
uv run python -m py_compile part1/baseline_model.py part1/part1_pruning.py
```

## Train baseline

```bash
uv run python part1/baseline_model.py
```

- Two-stage ImageNet-pretrained EfficientNet-B0, trained on CIFAR-100.
- Resizes CIFAR-100 to 128x128 and uses augmentation during training.
- Saves `part1/checkpoints/best_head.keras`, `best_finetune.keras`, phase-specific automatic backups, and `baseline_initial.weights.h5`.
- Saves the best model selected using validation performance as `part1/baseline_model.keras` and prints **held-out test accuracy**.
- Subsequent runs reuse the final model; interrupted epochs can resume via BackupAndRestore.
- Dataset and ImageNet downloads are cached by Keras; they are not re-downloaded every run.
- **Target >75% is not guaranteed and must be checked from the actual test accuracy.**

## Pruning experiments

```bash
uv run python part1/part1_pruning.py --experiment magnitude
uv run python part1/part1_pruning.py --experiment structured
uv run python part1/part1_pruning.py --experiment comparison
uv run python part1/part1_pruning.py --experiment sensitivity
uv run python part1/part1_pruning.py --experiment lottery
```

`--experiment all` runs all five sequentially. Files go to `part1/results/` and `part1/checkpoints/`.

For a quick exploratory pass, use e.g. `PART1_EPOCHS=1 PART1_TRAIN_SAMPLES=1024 PART1_VAL_SAMPLES=256 uv run python part1/part1_pruning.py --experiment magnitude`. Do not report quick-run metrics as full-run results.

If TensorFlow Metal triggers a native crash specifically during pruning, retry with `PART1_CPU_ONLY=1 uv run python part1/part1_pruning.py --experiment magnitude`. The baseline script uses Metal by default.

## Methodological limits and measurements

- Unstructured pruning zeroes individual Conv2D / DepthwiseConv2D / Dense weights via a persistent mask, but a dense Keras `.keras` file does not become 10x smaller solely because weights contain zeros. Additional sparse storage or quantization is needed to reduce the serialized size.
- Structured pruning *physically* removes channels only from the `compression_conv` adapter and the dependent batch-norm / classifier; it does **not** reduce internal EfficientNet residual blocks. The requested 50% channel reduction refers to that adapter, not to the entire network. `model_size_reduction` reports the true whole-model parameter reduction.
- Pruning fine-tuning defaults to **5 epochs on 10,000 training images and 2,000 validation images** for practicality. Increase the environmental limits for final experiments; the baseline trains on the full 45,000 training split.
- Lottery-ticket iterations use a short re-training budget and should be discussed as an exploratory experiment, not conclusive proof of the Lottery Ticket Hypothesis.
- The two stages and pruning experiments have not been executed on the user's Mac by this packaging environment; only static checks were run.
