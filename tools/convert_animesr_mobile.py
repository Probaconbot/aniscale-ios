"""Export the official AnimeSR_v2 recurrent cell for Android and iOS.

AnimeSR is Copyright (C) 2022 THL A29 Limited and licensed under Apache-2.0.
This mobile wrapper preserves its real recurrent ABI:

    previous/current/next LR frames + previous SR + hidden state
      -> current 4x SR frame + next hidden state

The app owns chronological state and resets it at detected scene cuts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import torch
from torch import nn
from torch.nn import functional as functional


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv2(self.relu(self.conv1(value)))


class MultiScaleCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_channels = 9 + 3 * 4 * 4 + 64
        self.conv_s1_first = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv_s2_first = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv_s4_first = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.body_s1_first = nn.ModuleList([ResidualBlockNoBN() for _ in range(5)])
        self.body_s2_first = nn.ModuleList([ResidualBlockNoBN() for _ in range(3)])
        self.body_s4_first = nn.ModuleList([ResidualBlockNoBN() for _ in range(2)])
        self.upsample_x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 224, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(224, 112, 3, 1, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale1 = self.conv_s1_first(value)
        scale2 = self.conv_s2_first(scale1)
        scale4 = self.conv_s4_first(scale2)
        has_scale2 = False
        has_scale4 = False
        for index in range(5):
            scale1 = self.body_s1_first[index](
                scale1
                + (self.upsample_x2(scale2) if has_scale2 else 0)
                + (self.upsample_x4(scale4) if has_scale4 else 0)
            )
            if index >= 2:
                scale2 = self.body_s2_first[index - 2](
                    scale2 + (self.upsample_x2(scale4) if has_scale4 else 0)
                )
                has_scale2 = True
            if index >= 3:
                scale4 = self.body_s4_first[index - 3](scale4)
                has_scale4 = True
        return self.fusion(
            torch.cat(
                (scale1, self.upsample_x2(scale2), self.upsample_x4(scale4)),
                dim=1,
            )
        )


class AnimeSRCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent_cell = MultiScaleCell()
        self.lrelu = nn.LeakyReLU(0.1)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(
        self,
        frames: torch.Tensor,
        feedback: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = frames[:, 3:6]
        input_value = torch.cat((frames, functional.pixel_unshuffle(feedback, 4), state), dim=1)
        output = self.recurrent_cell(input_value)
        enhanced = self.pixel_shuffle(output[:, :48]) + functional.interpolate(
            current,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        next_state = self.lrelu(output[:, 48:])
        return enhanced, next_state


def load_model(checkpoint: Path) -> AnimeSRCell:
    model = AnimeSRCell()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(loaded, strict=True)
    return model.eval()


def export(checkpoint: Path, output: Path) -> None:
    model = load_model(checkpoint)
    frames = torch.zeros(1, 9, 180, 320)
    feedback = torch.zeros(1, 3, 720, 1280)
    state = torch.zeros(1, 64, 180, 320)

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "AniUltraAnime_v2_recurrent.onnx"
    torch.onnx.export(
        model,
        (frames, feedback, state),
        onnx_path,
        input_names=["frames", "feedback", "state"],
        output_names=["enhanced", "next_state"],
        dynamic_axes={
            "frames": {2: "height", 3: "width"},
            "feedback": {2: "height4", 3: "width4"},
            "state": {2: "height", 3: "width"},
            "enhanced": {2: "height4", 3: "width4"},
            "next_state": {2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    traced = torch.jit.trace(model, (frames, feedback, state), strict=False)
    height = ct.RangeDim(lower_bound=32, upper_bound=320, default=180)
    width = ct.RangeDim(lower_bound=32, upper_bound=320, default=320)
    height4 = ct.RangeDim(lower_bound=128, upper_bound=1280, default=720)
    width4 = ct.RangeDim(lower_bound=128, upper_bound=1280, default=1280)
    package = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="frames", shape=(1, 9, height, width)),
            ct.TensorType(name="feedback", shape=(1, 3, height4, width4)),
            ct.TensorType(name="state", shape=(1, 64, height, width)),
        ],
        outputs=[
            ct.TensorType(name="enhanced"),
            ct.TensorType(name="next_state"),
        ],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    package.author = "Tencent ARC Lab; mobile conversion by AniScale"
    package.license = "Apache-2.0"
    package.short_description = "AnimeSR_v2 recurrent 4x anime video super-resolution cell"
    package.save(output / "AniUltraAnime_v2_recurrent.mlpackage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.checkpoint, arguments.output)


if __name__ == "__main__":
    main()
