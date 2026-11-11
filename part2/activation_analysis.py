
"""Diagnose TFLite activation quantization errors."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "part1"))

from baseline_model import prepare_compression_datasets


def load_model(name):
    interpreter = tf.lite.Interpreter(
        model_path=str(ROOT / "models" / f"baseline_{name}.tflite"),
        num_threads=1,
        experimental_preserve_all_tensors=True,
        experimental_delegates=[],
    )
    interpreter.allocate_tensors()
    return interpreter


def dequantize(values, info):
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(np.float32)

    params = info["quantization_parameters"]
    scales = params["scales"]
    zeros = params["zero_points"]

    if len(scales) != 1:
        return None

    scale = float(scales[0])
    zero = float(zeros[0])

    if scale <= 0:
        return None

    return (values.astype(np.float32) - zero) * scale


def match_operators(reference, quantized):
    ops_a = reference._get_ops_details()
    ops_b = quantized._get_ops_details()

    names_a = [op["op_name"] for op in ops_a]
    names_b = [op["op_name"] for op in ops_b]

    matcher = SequenceMatcher(
        None, names_a, names_b, autojunk=False
    )

    matches = []

    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            i = block.a + offset
            j = block.b + offset

            if not len(ops_a[i]["outputs"]) == 1:
                continue
            if not len(ops_b[j]["outputs"]) == 1:
                continue

            matches.append((
                i, j,
                int(ops_a[i]["outputs"][0]),
                int(ops_b[j]["outputs"][0]),
            ))

    print(
        f"Operator alignment: {len(matches)} "
        f"of {len(ops_a)} FP32 operators"
    )

    return matches


def main():
    _, _, x_val, _, _, _, _ = prepare_compression_datasets()

    fp = load_model("float32")
    q = load_model("int8")

    matches = match_operators(fp, q)

    fp_info = {
        t["index"]: t for t in fp.get_tensor_details()
    }
    q_info = {
        t["index"]: t for t in q.get_tensor_details()
    }

    statistics = defaultdict(list)
    metadata = {}

    samples = 5

    for sample in range(samples):
        image = x_val[sample:sample + 1].astype(np.float32)

        for interpreter in (fp, q):
            input_info = interpreter.get_input_details()[0]
            interpreter.set_tensor(input_info["index"], image)
            interpreter.invoke()

        for i, j, fp_idx, q_idx in matches:
            try:
                a = fp.get_tensor(fp_idx)
                b_raw = q.get_tensor(q_idx)
            except (ValueError, RuntimeError):
                continue

            if a.shape != b_raw.shape:
                continue

            if a.dtype != np.float32:
                continue
            if b_raw.dtype != np.int8:
                continue

            b = dequantize(b_raw, q_info[q_idx])
            if b is None:
                continue

            a = a.astype(np.float32)

            error = a - b

            relative_error = (
                np.linalg.norm(error.ravel())
                / (np.linalg.norm(a.ravel()) + 1e-8)
            )

            mae = float(np.mean(np.abs(error)))

            denominator = (
                np.linalg.norm(a.ravel())
                * np.linalg.norm(b.ravel())
            )

            cosine = float(
                np.dot(a.ravel(), b.ravel())
                / (denominator + 1e-8)
            )

            saturation = float(np.mean(
                (b_raw == -128) | (b_raw == 127)
            ))

            key = f"{i}:{j}"

            statistics[key].append({
                "relative_error": float(relative_error),
                "mae": mae,
                "cosine_similarity": cosine,
                "saturation_fraction": saturation,
            })

            metadata[key] = {
                "fp32_op_index": i,
                "int8_op_index": j,
                "operation": fp._get_ops_details()[i]["op_name"],
                "fp32_tensor": fp_info[fp_idx]["name"],
                "int8_tensor": q_info[q_idx]["name"],
                "shape": list(a.shape),
            }

        print(f"Processed image {sample + 1}/{samples}")

    results = []

    for key, records in statistics.items():
        item = metadata[key].copy()

        for metric in (
            "relative_error",
            "mae",
            "cosine_similarity",
            "saturation_fraction",
        ):
            item[metric] = float(np.mean([
                record[metric] for record in records
            ]))

        item["samples_compared"] = len(records)
        results.append(item)

    results.sort(key=lambda x: x["fp32_op_index"])

    print("\nFirst 15 comparable operations:")
    for item in results[:15]:
        print(
            f"{item['fp32_op_index']:3d} "
            f"{item['operation']:18s} "
            f"relative_error={item['relative_error']:.4f} "
            f"cosine={item['cosine_similarity']:.4f}"
        )

    print("\nTop 15 largest relative errors:")
    for item in sorted(
        results,
        key=lambda x: x["relative_error"],
        reverse=True,
    )[:15]:
        print(
            f"{item['fp32_op_index']:3d} "
            f"{item['operation']:18s} "
            f"relative_error={item['relative_error']:.4f} "
            f"cosine={item['cosine_similarity']:.4f} "
            f"saturation={item['saturation_fraction']:.4f}"
        )

    output = ROOT / "results" / "activation_analysis.json"
    output.parent.mkdir(exist_ok=True)

    output.write_text(json.dumps({
        "samples": samples,
        "matched_operations": len(matches),
        "compared_operations": len(results),
        "alignment_method": "heuristic_opcode_sequence",
        "results": results,
    }, indent=2))

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
