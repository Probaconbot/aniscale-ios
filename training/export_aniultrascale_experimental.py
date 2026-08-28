#!/usr/bin/env python3
"""Export the explicitly untrained AniUltraScale experimental runtime.

This exists only because the developer requested architecture testing before
training. The generated manifest and model metadata always state trained=false.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aniultrascale_model import build_aniultrascale


ROOT = Path(__file__).resolve().parent


class ExperimentalCenterFrame(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return self.model(frames, controls).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--coreml", action="store_true")
    args = parser.parse_args()
    if not args.onnx and not args.coreml:
        raise RuntimeError("Choose --onnx and/or --coreml")

    config = json.loads(
        (ROOT / "configs" / "aniultrascale_fast.json").read_text(encoding="utf-8")
    )
    deployment = dict(config["deployment"])
    torch.manual_seed(20260828)
    model = build_aniultrascale(config).eval()
    model.switch_to_deploy()
    wrapper = ExperimentalCenterFrame(model).eval()
    frames = torch.zeros(
        1,
        int(deployment["frames"]),
        3,
        int(deployment["input_height"]),
        int(deployment["input_width"]),
    )
    controls = torch.tensor([[1.0, 0.82]], dtype=torch.float32)
    args.output.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    if args.onnx:
        onnx_path = args.output / "AniUltraScale_experimental_2x.onnx"
        torch.onnx.export(
            wrapper,
            (frames, controls),
            onnx_path,
            opset_version=18,
            input_names=["frames", "controls"],
            output_names=["output"],
            dynamic_axes=None,
        )
        artifacts["onnx"] = onnx_path.name

    if args.coreml:
        import coremltools as ct

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
        converted.short_description = "UNTRAINED AniUltraScale architecture preview"
        converted.user_defined_metadata["aniscale_engine"] = "AniUltraScale Experimental"
        converted.user_defined_metadata["trained"] = "false"
        coreml_path = args.output / "AniUltraScale_experimental_2x.mlpackage"
        if coreml_path.exists():
            shutil.rmtree(coreml_path)
        converted.save(coreml_path)
        artifacts["coreml"] = coreml_path.name

    manifest = {
        "format": "aniultrascale-experimental-v1",
        "trained": False,
        "warning": "Random initialization; noisy or corrupted output is expected.",
        "seed": 20260828,
        "architecture": config["architecture"],
        "artifacts": artifacts,
    }
    (args.output / "AniUltraScale_experimental_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
