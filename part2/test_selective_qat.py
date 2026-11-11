import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

sys.path.insert(0, str(Path("part1").resolve()))
from baseline_model import prepare_compression_datasets

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "GPU")

model = tf.keras.models.load_model(
    "part1/baseline_model.keras",
    compile=False,
)

backbone = model.get_layer("backbone")

head_names = [
    "compression_conv",
    "compression_bn",
    "compression_relu",
    "compression_pool",
    "compression_dropout",
    "classifier",
]

quantizable = {"compression_conv", "classifier"}

head_layers = []

for name in head_names:
    original = model.get_layer(name)

    cloned = original.__class__.from_config(
        original.get_config()
    )

    if name in quantizable:
        cloned = tfmot.quantization.keras.quantize_annotate_layer(
            cloned
        )

    head_layers.append(cloned)

annotated_head = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=backbone.output_shape[1:]
        ),
        *head_layers,
    ],
    name="annotated_head",
)

# Build layers and copy the original trained weights.
for name in head_names:
    original = model.get_layer(name)

    for layer in annotated_head.layers:
        inner = getattr(layer, "layer", layer)

        if inner.name == name:
            inner.set_weights(original.get_weights())
            break
import numpy as np

x = np.random.default_rng(42).normal(
    size=(2, *backbone.output_shape[1:])
).astype(np.float32)

original_head = tf.keras.Model(
    inputs=model.get_layer("compression_conv").input,
    outputs=model.output,
)

a = original_head(x, training=False).numpy()
b = annotated_head(x, training=False).numpy()

print("Pre-QAT max output difference:", np.max(np.abs(a - b)))
print("Pre-QAT prediction agreement:",
      np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))
qat_head = tfmot.quantization.keras.quantize_apply(
    annotated_head
)
# Initialize all QAT layers before restoring pretrained weights.
sample = tf.zeros(
    (1, *backbone.output_shape[1:]),
    dtype=tf.float32,
)

_ = qat_head(sample, training=False)
# Restore pretrained weights after quantize_apply().
# QuantizeWrapper contains the original layer in .layer.



# Restore original weights into the QAT model.
# QuantizeWrapperV2 can expose kernel variables at wrapper level.

source_weights = {
    variable.name.split(":")[0]: variable.numpy()
    for name in head_names
    for variable in model.get_layer(name).weights
}

restored_names = set()

for variable in qat_head.weights:
    key = variable.name.split(":")[0]

    if key not in source_weights:
        continue

    source = source_weights[key]

    if tuple(variable.shape) != tuple(source.shape):
        raise ValueError(
            f"Shape mismatch for {key}: "
            f"{variable.shape} != {source.shape}"
        )

    variable.assign(source)
    restored_names.add(key)

missing = set(source_weights) - restored_names

if missing:
    raise RuntimeError(
        f"Original weights not restored: {sorted(missing)}"
    )

for variable in qat_head.weights:
    key = variable.name.split(":")[0]

    if key in source_weights:
        np.testing.assert_allclose(
            variable.numpy(),
            source_weights[key],
            rtol=1e-6,
            atol=1e-7,
        )

print(f"Restored and verified {len(restored_names)} weight tensors")


for layer in qat_head.layers:
    inner = getattr(layer, "layer", layer)

    if inner.name in ("compression_conv", "classifier"):
        original = model.get_layer(inner.name)

        differences = [
            float(np.max(np.abs(a - b)))
            for a, b in zip(
                original.get_weights(),
                inner.get_weights(),
            )
        ]

        print(
            f"{inner.name}: "
            f"max weight differences={differences}"
        )
# Connect the pretrained backbone to the QAT head.
backbone.trainable = False
for layer in qat_head.layers:
    inner = getattr(layer, "layer", layer)

    if isinstance(inner, tf.keras.layers.BatchNormalization):
        inner.trainable = False
inputs = model.input
features = backbone(model.get_layer("resize")(inputs), training=False)
outputs = qat_head(features)

qat_model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="efficientnet_selective_qat",
)

# Calibrate QAT quantizers without optimizer updates.

x_train, _, x_val, y_val, _, _, _ = prepare_compression_datasets()

def show_quantizer_ranges():
    for variable in qat_head.weights:
        name = variable.name

        if name.endswith("_min:0") or name.endswith("_max:0"):
            values = variable.numpy()
            print(
                name,
                "min=", float(np.min(values)),
                "max=", float(np.max(values)),
            )

print("\nQuantizer ranges BEFORE calibration:")
show_quantizer_ranges()

# This updates quantizer statistics but does not run
# gradient descent or optimizer steps.
for start in range(0, 200, 16):
    batch = x_train[start:start + 16].astype(np.float32)
    _ = qat_model(batch, training=True)

print("\nQuantizer ranges AFTER calibration:")
show_quantizer_ranges()

# Evaluate with training=False, disabling Dropout.



sys.path.insert(0, str(Path("part1").resolve()))

_, _, x_val, y_val, _, _, _ = prepare_compression_datasets()

x = x_val[:200].astype(np.float32)
y = np.asarray(y_val[:200]).reshape(-1)

baseline_output = model.predict(x, batch_size=32, verbose=0)
qat_output = qat_model.predict(x, batch_size=32, verbose=0)

baseline_acc = np.mean(np.argmax(baseline_output, axis=1) == y)
qat_acc = np.mean(np.argmax(qat_output, axis=1) == y)

print(f"Baseline accuracy: {baseline_acc:.2%}")
print(f"Selective QAT initial accuracy: {qat_acc:.2%}")
print(f"Accuracy difference: {(baseline_acc - qat_acc) * 100:.2f} pp")

if qat_acc < baseline_acc - 0.10:
    raise RuntimeError(
        "Unexpected accuracy degradation before QAT training."
    )

print("SELECTIVE_QAT_OK")
print("QAT head parameters:", qat_head.count_params())

for layer in qat_head.layers:
    if "QuantizeWrapper" in type(layer).__name__:
        print("Quantized:", layer.name)

sample = tf.zeros(
    (1, *backbone.output_shape[1:]),
    dtype=tf.float32,
)

output = qat_head(sample, training=False)

# QAT fine-tuning: frozen backbone, trainable quantized head.

x_train, y_train, _, _, _, _, _ = prepare_compression_datasets()

y_train = np.asarray(y_train).reshape(-1)
y_val = np.asarray(y_val).reshape(-1)

qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    ),
    metrics=["accuracy"],
)

# Reuse an existing checkpoint, otherwise perform real QAT fine-tuning.
checkpoint_path = Path("part2/checkpoints/selective_qat_weights.h5")
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

if checkpoint_path.is_file():
    qat_model.load_weights(str(checkpoint_path))
    print("Existing QAT checkpoint loaded successfully")
else:
    print("No QAT checkpoint found; training selective quantized head.")
    history = qat_model.fit(
        x_train[:10000].astype(np.float32),
        y_train[:10000],
        batch_size=32,
        epochs=2,
        validation_data=(
            x_val[:1000].astype(np.float32),
            y_val[:1000],
        ),
        verbose=2,
    )
    qat_model.save_weights(str(checkpoint_path))
    print("New QAT checkpoint saved:", checkpoint_path)

print("QAT model ready for validation")
results = qat_model.evaluate(
    x_val.astype(np.float32),
    y_val,
    batch_size=32,
    verbose=0,
)

print(
    f"QAT validation accuracy: {results[1]:.2%}"
)

# Save trained weights for later TFLite conversion.
os.makedirs("part2/checkpoints", exist_ok=True)

qat_model.save_weights(
    "part2/checkpoints/selective_qat_weights.h5"
)

# Export the trained QAT model to TFLite.

from pathlib import Path

output_path = Path("part2/models/selective_qat_int8.tflite")
output_path.parent.mkdir(parents=True, exist_ok=True)

def representative_dataset():
    for i in range(300):
        yield [
            x_train[i:i + 1].astype(np.float32)
        ]

converter = tf.lite.TFLiteConverter.from_keras_model(
    qat_model
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

converted = converter.convert()
output_path.write_bytes(converted)

print("TFLite QAT model saved:", output_path)
print(
    "TFLite model size:",
    f"{output_path.stat().st_size / (1024 ** 2):.2f} MiB"
)

print("QAT weights saved.")

print("Output shape:", output.shape)
