#!/usr/bin/env python3
"""Export fine-tuned FAST and QUALITY checkpoints as AniScale Core ML models."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from aniscale_models import build_model, checkpoint_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "ios" / "Runner" / "Models"


def load_variant(path: Path) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise RuntimeError(f"Fine-tuned AniScale checkpoint has no config: {path}")
    config = dict(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return model.eval(), config


def export(path: Path, output_directory: Path, expected_name: str) -> dict[str, object]:
    model, config = load_variant(path)
    coreml = dict(config["coreml"])
    output_name = str(coreml["output_name"])
    if output_name != expected_name:
        raise RuntimeError(
            f"{path} declares {output_name}; expected the existing iOS resource {expected_name}"
        )
    input_size = int(coreml["input_size"])
    example = torch.zeros(1, 3, input_size, input_size)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=True)
        torch_output = model(example).numpy()
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    converted.author = "AniScale fine-tune; architecture and initialization from Real-ESRGAN"
    converted.license = "BSD-3-Clause"
    converted.short_description = (
        f"{config['name']}: faithful 4x video restoration with temporal sequence fine-tuning"
    )
    converted.user_defined_metadata["aniscale_variant"] = str(config["name"])
    converted.user_defined_metadata["aniscale_architecture"] = str(config["architecture"])
    converted.user_defined_metadata["aniscale_temporal_training"] = "motion-aligned residual"
    output = output_directory / output_name
    if output.exists():
        shutil.rmtree(output)
    converted.save(output)

    comparison: dict[str, object] = {"available": False}
    try:
        coreml_output = converted.predict({"input": example.numpy()})["output"]
        difference = np.abs(coreml_output - torch_output)
        comparison = {
            "available": True,
            "mean_absolute_error": float(difference.mean()),
            "maximum_absolute_error": float(difference.max()),
        }
    except Exception as error:  # Prediction is only available on supported macOS hosts.
        comparison["reason"] = str(error)
    return {
        "checkpoint": str(path),
        "output": str(output),
        "architecture": config["architecture"],
        "input_size": input_size,
        "conversion_check": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "training" / "reports" / "coreml_export.json"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "fast": export(
            args.fast,
            args.output,
            "AniScale_turbo_animevideo_266_fp16.mlpackage",
        ),
        "quality": export(
            args.quality,
            args.output,
            "RealESRGAN_anime_6B_266_fp16.mlpackage",
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

