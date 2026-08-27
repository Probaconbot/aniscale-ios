"""Convert the official Real-ESRGAN Fusion and Render weights for iOS.

The generated model has a fixed 266x266 input: a 256px image tile plus
10px reflected padding. AniScale stitches tiles on-device in Swift.
"""

from pathlib import Path
from urllib.request import urlretrieve
import hashlib

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ios" / "Runner" / "Models"
MODELS = (
    {
        "name": "Fusion",
        "weights": MODEL_DIR / "RealESRGAN_x4plus_anime_6B.pth",
        "output": MODEL_DIR / "RealESRGAN_anime_6B_266_fp16.mlpackage",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ),
        "sha256": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        "blocks": 6,
        "description": "4x anime and stylized 3D super-resolution",
        "architecture": "rrdb",
    },
    {
        "name": "Render",
        "weights": MODEL_DIR / "RealESRGAN_x4plus.pth",
        "output": MODEL_DIR / "RealESRGAN_render_x4plus_266_fp16.mlpackage",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        ),
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "blocks": 23,
        "description": "4x general and 3D render super-resolution",
        "architecture": "rrdb",
    },
    {
        "name": "Turbo",
        "weights": MODEL_DIR / "realesr-animevideov3.pth",
        "output": MODEL_DIR / "AniScale_turbo_animevideo_266_fp16.mlpackage",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.5.0/realesr-animevideov3.pth"
        ),
        "sha256": "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        "blocks": 16,
        "description": "4x compact anime video super-resolution",
        "architecture": "srvgg",
    },
)


class ResidualDenseBlock(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + growth * 2, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + growth * 3, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + growth * 4, features, 3, 1, 1)

    def forward(self, value):
        value1 = functional.leaky_relu(self.conv1(value), 0.2)
        value2 = functional.leaky_relu(self.conv2(torch.cat((value, value1), 1)), 0.2)
        value3 = functional.leaky_relu(
            self.conv3(torch.cat((value, value1, value2), 1)), 0.2
        )
        value4 = functional.leaky_relu(
            self.conv4(torch.cat((value, value1, value2, value3), 1)), 0.2
        )
        value5 = self.conv5(torch.cat((value, value1, value2, value3, value4), 1))
        return value5 * 0.2 + value


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, value):
        output = self.rdb1(value)
        output = self.rdb2(output)
        output = self.rdb3(output)
        return output * 0.2 + value


class RRDBNet(nn.Module):
    def __init__(self, blocks: int):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualInResidualDenseBlock() for _ in range(blocks)]
        )
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, value):
        features = self.conv_first(value)
        body = self.conv_body(self.body(features))
        features = features + body
        features = functional.leaky_relu(
            self.conv_up1(functional.interpolate(features, scale_factor=2, mode="nearest")),
            0.2,
        )
        features = functional.leaky_relu(
            self.conv_up2(functional.interpolate(features, scale_factor=2, mode="nearest")),
            0.2,
        )
        return self.conv_last(functional.leaky_relu(self.conv_hr(features), 0.2))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_conv: int = 16, upscale: int = 4):
        super().__init__()
        body = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(num_parameters=64)]
        for _ in range(num_conv):
            body.extend(
                [nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(num_parameters=64)]
            )
        body.append(nn.Conv2d(64, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)
        self.upscale = upscale

    def forward(self, value):
        output = self.upsampler(self.body(value))
        base = functional.interpolate(value, scale_factor=self.upscale, mode="nearest")
        return output + base


def convert_model(spec):
    output = spec["output"]
    weights = spec["weights"]
    if output.exists():
        print(f"Core ML model already exists: {output}")
        return
    if not weights.exists():
        print(f"Downloading official Real-ESRGAN {spec['name']} weights...")
        urlretrieve(spec["url"], weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        weights.unlink(missing_ok=True)
        raise RuntimeError(f"Real-ESRGAN weights checksum mismatch: {digest}")

    network = (
        SRVGGNetCompact(spec["blocks"])
        if spec["architecture"] == "srvgg"
        else RRDBNet(spec["blocks"])
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint.get("params_ema", checkpoint), strict=True)
    network.eval()

    example = torch.zeros(1, 3, 266, 266)
    with torch.no_grad():
        traced = torch.jit.trace(network, example)

    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    model.author = "Xintao Wang et al.; iOS conversion for AniScale"
    model.license = "BSD-3-Clause"
    model.short_description = spec["description"]
    model.save(output)
    print(f"Saved {output}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        convert_model(spec)


if __name__ == "__main__":
    main()
