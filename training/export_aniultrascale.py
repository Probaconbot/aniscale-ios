#!/usr/bin/env python3
"""Export a trained AniUltraScale checkpoint for iOS and Android.

This exporter intentionally refuses architecture-only or randomly initialized
weights. A mobile build may only bundle a checkpoint written by the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aniultrascale_model import AniUltraScale, build_aniultrascale


class CenterFrameExport(nn.Module):
    def __init__(self, model: AniUltraScale) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return self.model(frames, controls).clamp(0, 1)


def load_trained(path: Path) -> tuple[AniUltraScale, dict[str, object], dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format") != "aniultrascale-v2"
        or checkpoint.get("trained") is not True
        or not checkpoint.get("dataset_manifest_sha256")
    ):
        raise RuntimeError(
            "Refusing to export: this is not a trained, dataset-traceable AniUltraScale checkpoint"
        )
    config = dict(checkpoint["config"])
    model = build_aniultrascale(config)
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    model.eval()
    model.switch_to_deploy()
    return model, config, checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(
    wrapper: CenterFrameExport,
    frames: torch.Tensor,
    controls: torch.Tensor,
    output: Path,
) -> None:
    torch.onnx.export(
        wrapper,
        (frames, controls),
        output,
        opset_version=18,
        input_names=["frames", "controls"],
        output_names=["output"],
        dynamic_axes=None,
    )


def export_coreml(
    wrapper: CenterFrameExport,
    frames: torch.Tensor,
    controls: torch.Tensor,
    output: Path,
    variant: str,
) -> None:
    try:
        import coremltools as ct
    except ImportError as error:
        raise RuntimeError("coremltools is required for --coreml") from error
    traced = torch.jit.trace(wrapper, (frames, controls), strict=True)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="frames", shape=frames.shape, dtype=np.float32),
            ct.TensorType(name="controls", shape=controls.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    converted.author = "AniScale"
    converted.license = "AniScale model weights; architecture includes MIT-licensed influences"
    converted.short_description = f"AniUltraScale {variant}: five-frame temporal 2x restoration"
    converted.user_defined_metadata["aniscale_engine"] = "AniUltraScale"
    converted.user_defined_metadata["aniscale_variant"] = variant
    converted.user_defined_metadata["aniscale_scale"] = "2"
    if output.exists():
        shutil.rmtree(output)
    converted.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--coreml", action="store_true")
    args = parser.parse_args()
    if not args.onnx and not args.coreml:
        raise RuntimeError("Choose at least one export target: --onnx and/or --coreml")
    model, config, checkpoint = load_trained(args.checkpoint)
    deployment = dict(config["deployment"])
    frames = torch.zeros(
        1,
        int(deployment["frames"]),
        3,
        int(deployment["input_height"]),
        int(deployment["input_width"]),
    )
    controls = torch.tensor([[1.0, 0.82]], dtype=torch.float32)
    wrapper = CenterFrameExport(model).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    variant = str(config["name"]).split()[-1].lower()
    artifacts: dict[str, object] = {}
    if args.onnx:
        onnx_path = args.output / f"aniultrascale_{variant}_2x.onnx"
        export_onnx(wrapper, frames, controls, onnx_path)
        artifacts["onnx"] = {"path": str(onnx_path), "sha256": sha256(onnx_path)}
    if args.coreml:
        coreml_path = args.output / f"AniUltraScale_{variant}_2x.mlpackage"
        export_coreml(wrapper, frames, controls, coreml_path, variant)
        artifacts["coreml"] = {"path": str(coreml_path)}
    report = {
        "format": "aniultrascale-mobile-export-v2",
        "trained": True,
        "variant": variant,
        "scale": 2,
        "frames": int(deployment["frames"]),
        "input_height": int(deployment["input_height"]),
        "input_width": int(deployment["input_width"]),
        "checkpoint_sha256": sha256(args.checkpoint),
        "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
        "training_iteration": int(checkpoint["iteration"]),
        "validation_loss": float(checkpoint["validation_loss"]),
        "artifacts": artifacts,
    }
    (args.output / f"aniultrascale_{variant}_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
