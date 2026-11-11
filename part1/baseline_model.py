"""CIFAR-100 baseline shared with the Part 1 pruning experiments.

Run from the repository root:
    uv run python part1/baseline_model.py
"""

import os

# Set before importing TensorFlow: TFMOT 0.8 requires legacy tf.keras.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

PART_DIR = Path(__file__).resolve().parent
MODEL_PATH = PART_DIR / "baseline_model.keras"
CHECKPOINT_DIR = PART_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def prepare_compression_datasets():
    """Return train, validation, test and 1000 calibration images.

    EfficientNet-B0 includes input rescaling: retain CIFAR-100 pixels in [0, 255].
    """
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
        label_mode="fine"
    )
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x))
    validation_indices = indices[:5000]
    training_indices = indices[5000:]

    return (
        x[training_indices],
        y[training_indices].ravel(),
        x[validation_indices],
        y[validation_indices].ravel(),
        x_test,
        y_test.ravel(),
        x[training_indices[:1000]],
    )


def create_baseline_model(input_shape=(32, 32, 3), num_classes=100):
    """Build an ImageNet-pretrained EfficientNet-B0 with a prunable head."""
    tf.keras.utils.set_random_seed(SEED)

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Resizing(128, 128, name="resize")(inputs)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3),
    )
    backbone._name = "backbone"
    backbone.trainable = False
    x = backbone(x, training=False)

    x = tf.keras.layers.Conv2D(
        256, kernel_size=1, use_bias=False, name="compression_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="compression_bn")(x)
    x = tf.keras.layers.Activation("relu", name="compression_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="compression_pool")(x)
    x = tf.keras.layers.Dropout(0.4, name="compression_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classifier"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="cifar100_efficientnet_b0")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _best_logged_accuracy(csv_path):
    """Keep ModelCheckpoint from overwriting a better pre-interruption model."""
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(newline="") as handle:
            values = [
                float(row["val_accuracy"])
                for row in csv.DictReader(handle)
                if row.get("val_accuracy")
            ]
        return max(values) if values else None
    except (OSError, ValueError, KeyError):
        return None


def _callbacks(stage):
    best_path = CHECKPOINT_DIR / f"best_{stage}.keras"
    history_path = CHECKPOINT_DIR / f"{stage}_history.csv"
    return [
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(CHECKPOINT_DIR / f"{stage}_backup"),
            save_freq="epoch",
            delete_checkpoint=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            initial_value_threshold=_best_logged_accuracy(history_path),
        ),
        tf.keras.callbacks.CSVLogger(str(history_path), append=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=3,
            min_lr=1e-7,
        ),
    ]


def train_baseline_model():
    """Train head, fine-tune backbone, and retain the best validation model.

    An interrupted stage resumes at the last completed epoch. Finished stages
    are skipped on subsequent runs. The test set is only used for evaluation.
    """
    if MODEL_PATH.exists():
        print(f"Loading existing baseline: {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

    tf.keras.utils.set_random_seed(SEED)
    x_train, y_train, x_val, y_val, x_test, y_test, _ = (
        prepare_compression_datasets()
    )
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomRotation(0.03),
    ])
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=SEED)
        .batch(64)
        .map(
            lambda images, labels: (
                augmentation(tf.cast(images, tf.float32), training=True),
                labels,
            ),
            num_parallel_calls=2,
        )
        .prefetch(2)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(128)
        .prefetch(2)
    )

    head_done = CHECKPOINT_DIR / "head_complete.flag"
    head_best = CHECKPOINT_DIR / "best_head.keras"
    initial_weights = CHECKPOINT_DIR / "baseline_initial.weights.h5"

    if head_done.exists():
        if not head_best.exists():
            raise FileNotFoundError(f"Missing completed head checkpoint: {head_best}")
        print("Head-training stage already complete; loading checkpoint.")
        model = tf.keras.models.load_model(head_best)
    else:
        model = create_baseline_model()
        if not initial_weights.exists():
            model.save_weights(initial_weights)
        print("Stage 1: training classification head (up to 25 epochs).")
        model.fit(
            train_ds, validation_data=val_ds, epochs=25,
            callbacks=_callbacks("head"),
        )
        # ModelCheckpoint holds the best validation model, not just the last epoch.
        if not head_best.exists():
            raise RuntimeError("Head training ended without a best-model checkpoint")
        model = tf.keras.models.load_model(head_best)
        head_done.touch()

    fine_done = CHECKPOINT_DIR / "finetune_complete.flag"
    fine_best = CHECKPOINT_DIR / "best_finetune.keras"

    if fine_done.exists():
        if not fine_best.exists():
            raise FileNotFoundError(f"Missing completed fine-tuning checkpoint: {fine_best}")
        print("Fine-tuning stage already complete; loading checkpoint.")
    else:
        backbone = model.get_layer("backbone")
        backbone.trainable = True
        # BatchNorm stays in inference mode for small-batch fine-tuning.
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("Stage 2: fine-tuning backbone (up to 35 epochs).")
        model.fit(
            train_ds, validation_data=val_ds, epochs=35,
            callbacks=_callbacks("finetune"),
        )
        if not fine_best.exists():
            raise RuntimeError("Fine-tuning ended without a best-model checkpoint")
        fine_done.touch()

    # Select globally using the same validation set, without test-set tuning.
    head_accuracy = _best_logged_accuracy(CHECKPOINT_DIR / "head_history.csv")
    fine_accuracy = _best_logged_accuracy(CHECKPOINT_DIR / "finetune_history.csv")
    if head_accuracy is None or fine_accuracy is None:
        # Covers external checkpoint files with missing CSV logs.
        head_model = tf.keras.models.load_model(head_best)
        fine_model = tf.keras.models.load_model(fine_best)
        head_accuracy = head_model.evaluate(val_ds, verbose=0)[1]
        fine_accuracy = fine_model.evaluate(val_ds, verbose=0)[1]

    chosen = fine_best if fine_accuracy >= head_accuracy else head_best
    print(f"Best head validation accuracy: {head_accuracy:.4f}")
    print(f"Best fine-tune validation accuracy: {fine_accuracy:.4f}")
    print(f"Selected: {chosen.name}")
    model = tf.keras.models.load_model(chosen)
    model.save(MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, batch_size=128, verbose=0
    )
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4%}")
    print(f"Baseline saved: {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_baseline_model()
