"""Convert the official Apache-2.0 SPAN checkpoints to fixed-tile Core ML.

The app tiles video frames, so fixed 256px inputs compile more reliably for
the Neural Engine than a fully dynamic graph while still supporting arbitrary
video sizes in the Swift streaming pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import torch
from spandrel import ModelLoader


def convert(checkpoint: Path, output: Path, scale: int) -> None:
    descriptor = ModelLoader().load_from_file(checkpoint)
    if int(descriptor.scale) != scale:
        raise ValueError(
            f"Expected a {scale}x SPAN checkpoint, got {descriptor.scale}x"
        )

    model = descriptor.model.eval()
    example = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)

    package = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=example.shape)],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    package.author = "SPAN authors; mobile conversion by AniScale"
    package.license = "Apache-2.0"
    package.short_description = (
        f"SuperUltra {scale}x SPAN tile model for offline video upscaling"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    package.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x2", type=Path, required=True)
    parser.add_argument("--x4", type=Path, required=True)
    parser.add_argument("--anime-x2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.x2, args.output / "SuperUltra_span_x2_fp16.mlpackage", 2)
    convert(args.x4, args.output / "SuperUltra_span_x4_fp16.mlpackage", 4)
    convert(
        args.anime_x2,
        args.output / "SuperUltra_span_anime_x2_fp16.mlpackage",
        2,
    )


if __name__ == "__main__":
    main()
