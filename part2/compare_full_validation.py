
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np

sys.path.insert(0, str(Path("part1").resolve()))
from baseline_model import prepare_compression_datasets
from calibration_sensitivity import evaluate_tflite

models = {
    "FP32": "part2/models/baseline_float32.tflite",
    "Dynamic": "part2/models/baseline_dynamic.tflite",
    "PTQ original": "part2/models/baseline_int8.tflite",
    "PTQ 300": "part2/models/calibration/ptq_int8_calib_300.tflite",
    "PTQ 1000": "part2/models/calibration/ptq_int8_calib_1000.tflite",
    "PTQ 2000": "part2/models/calibration/ptq_int8_calib_2000.tflite",
    "Selective QAT": "part2/models/selective_qat_int8.tflite",
}

_, _, x_val, y_val, _, _, _ = prepare_compression_datasets()
x_val = np.asarray(x_val, dtype=np.float32)
y_val = np.asarray(y_val).reshape(-1)

for name, path in models.items():
    accuracy = evaluate_tflite(
        Path(path), x_val, y_val
    )
    print(f"{name:15s}: {accuracy:.2%}")
