"""Architectures used by AniScale's existing Real-ESRGAN model family."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + growth * 2, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + growth * 3, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + growth * 4, features, 3, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value1 = F.leaky_relu(self.conv1(value), 0.2)
        value2 = F.leaky_relu(self.conv2(torch.cat((value, value1), 1)), 0.2)
        value3 = F.leaky_relu(
            self.conv3(torch.cat((value, value1, value2), 1)), 0.2
        )
        value4 = F.leaky_relu(
            self.conv4(torch.cat((value, value1, value2, value3), 1)), 0.2
        )
        value5 = self.conv5(torch.cat((value, value1, value2, value3, value4), 1))
        return value + value5 * 0.2


class RRDB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.rdb3(self.rdb2(self.rdb1(value)))
        return value + output * 0.2


class RRDBNet(nn.Module):
    def __init__(self, blocks: int = 6) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB() for _ in range(blocks)))
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.conv_first(value)
        features = features + self.conv_body(self.body(features))
        features = F.leaky_relu(
            self.conv_up1(F.interpolate(features, scale_factor=2, mode="nearest")),
            0.2,
        )
        features = F.leaky_relu(
            self.conv_up2(F.interpolate(features, scale_factor=2, mode="nearest")),
            0.2,
        )
        return self.conv_last(F.leaky_relu(self.conv_hr(features), 0.2))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_conv: int = 16, upscale: int = 4) -> None:
        super().__init__()
        body: list[nn.Module] = [
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.PReLU(num_parameters=64),
        ]
        for _ in range(num_conv):
            body.extend(
                [nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(num_parameters=64)]
            )
        body.append(nn.Conv2d(64, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)
        self.upscale = upscale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.upsampler(self.body(value))
        base = F.interpolate(value, scale_factor=self.upscale, mode="nearest")
        return residual + base


def build_model(config: Mapping[str, object]) -> nn.Module:
    architecture = str(config["architecture"])
    blocks = int(config["num_blocks"])
    scale = int(config.get("scale", 4))
    if scale != 4:
        raise ValueError("AniScale's current model family expects native 4x output")
    if architecture == "srvgg":
        return SRVGGNetCompact(num_conv=blocks, upscale=scale)
    if architecture == "rrdb":
        return RRDBNet(blocks=blocks)
    raise ValueError(f"Unsupported existing AniScale architecture: {architecture}")


def checkpoint_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a state dictionary")
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain params or params_ema")
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        result[key.removeprefix("module.")] = value
    return result


def load_existing_weights(model: nn.Module, path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)

