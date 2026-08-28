"""AniUltraScale: mobile-friendly 2x recurrent video super-resolution.

The implementation is original AniScale code. Its propagation and deploy-time
reparameterisation are informed by the MIT-licensed NanoVSR project, while the
separate fidelity/detail controls implement a general published idea without
copying PiSA-SR source code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UltraControls:
    fidelity: float
    detail: float


MODE_CONTROLS = {
    "subtle": UltraControls(1.00, 0.35),
    "detailed": UltraControls(1.00, 0.82),
    "creative": UltraControls(0.86, 1.20),
}


class RepBlock(nn.Module):
    """3x3 + 1x1 + identity while training; one 3x3 convolution at export."""

    def __init__(self, channels: int, deploy: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        else:
            self.dense = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.pointwise = nn.Sequential(
                nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.identity = nn.BatchNorm2d(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.activation(self.reparam(value))
        return self.activation(
            self.dense(value) + self.pointwise(value) + self.identity(value)
        )

    @staticmethod
    def _fuse(branch: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
        convolution = branch[0]
        batch_norm = branch[1]
        assert isinstance(convolution, nn.Conv2d)
        assert isinstance(batch_norm, nn.BatchNorm2d)
        standard_deviation = torch.sqrt(batch_norm.running_var + batch_norm.eps)
        multiplier = (batch_norm.weight / standard_deviation).reshape(-1, 1, 1, 1)
        kernel = convolution.weight * multiplier
        bias = batch_norm.bias - batch_norm.running_mean * batch_norm.weight / standard_deviation
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        dense_kernel, dense_bias = self._fuse(self.dense)
        point_kernel, point_bias = self._fuse(self.pointwise)
        point_kernel = F.pad(point_kernel, (1, 1, 1, 1))
        standard_deviation = torch.sqrt(self.identity.running_var + self.identity.eps)
        multiplier = self.identity.weight / standard_deviation
        identity_kernel = torch.zeros_like(dense_kernel)
        indices = torch.arange(self.channels, device=dense_kernel.device)
        identity_kernel[indices, indices, 1, 1] = multiplier
        identity_bias = (
            self.identity.bias
            - self.identity.running_mean * self.identity.weight / standard_deviation
        )
        reparam = nn.Conv2d(self.channels, self.channels, 3, 1, 1, bias=True)
        reparam.weight.data.copy_(dense_kernel + point_kernel + identity_kernel)
        reparam.bias.data.copy_(dense_bias + point_bias + identity_bias)
        self.reparam = reparam
        del self.dense
        del self.pointwise
        del self.identity
        self.deploy = True


class PropagationStack(nn.Module):
    def __init__(self, channels: int, blocks: int, deploy: bool = False) -> None:
        super().__init__()
        self.input = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.body = nn.Sequential(*(RepBlock(channels, deploy) for _ in range(blocks)))

    def forward(self, current: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.body(self.input(torch.cat((current, state), dim=1)))


class ResidualHead(nn.Module):
    def __init__(self, channels: int, blocks: int, scale: int) -> None:
        super().__init__()
        self.body = nn.Sequential(*(RepBlock(channels) for _ in range(blocks)))
        self.output = nn.Sequential(
            nn.Conv2d(channels, 3 * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(value))


class AniUltraScale(nn.Module):
    """Bidirectional recurrent 2x VSR with independently mixed residual heads.

    Input is B,T,3,H,W. ``forward_sequence`` reconstructs every frame for
    temporal supervision; ``forward`` returns the center frame for five-frame
    mobile windows.
    """

    def __init__(
        self,
        channels: int = 32,
        temporal_blocks: int = 8,
        reconstruction_blocks: int = 4,
        scale: int = 2,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("AniUltraScale's first shipping architecture is native 2x")
        self.channels = channels
        self.temporal_blocks = temporal_blocks
        self.reconstruction_blocks = reconstruction_blocks
        self.scale = scale
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            RepBlock(channels, deploy),
        )
        self.forward_propagation = PropagationStack(channels, temporal_blocks, deploy)
        self.backward_propagation = PropagationStack(channels, temporal_blocks, deploy)
        self.structure_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            RepBlock(channels, deploy),
        )
        self.base_head = ResidualHead(channels, reconstruction_blocks, scale)
        self.fidelity_head = ResidualHead(channels, max(1, reconstruction_blocks // 2), scale)
        self.detail_head = ResidualHead(channels, reconstruction_blocks, scale)

    @staticmethod
    def controls_for_mode(
        mode: str,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if mode not in MODE_CONTROLS:
            raise ValueError(f"Unknown AniUltraScale mode: {mode}")
        selected = MODE_CONTROLS[mode]
        return torch.tensor(
            [[selected.fidelity, selected.detail]], device=device, dtype=dtype
        ).expand(batch, -1)

    def _propagate(
        self, features: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch, frames, channels, height, width = features.shape
        state = features.new_zeros((batch, channels, height, width))
        forwards: list[torch.Tensor] = []
        for position in range(frames):
            state = self.forward_propagation(features[:, position], state)
            forwards.append(state)
        state = features.new_zeros((batch, channels, height, width))
        backwards: list[torch.Tensor] = [state] * frames
        for position in range(frames - 1, -1, -1):
            state = self.backward_propagation(features[:, position], state)
            backwards[position] = state
        return forwards, backwards

    def forward_sequence_components(
        self,
        frames: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("AniUltraScale expects B,T,3,H,W video tensors")
        batch, count, channels, height, width = frames.shape
        if count < 3:
            raise ValueError("AniUltraScale requires at least three adjacent frames")
        flat = frames.reshape(batch * count, channels, height, width)
        encoded = self.encoder(flat).reshape(batch, count, self.channels, height, width)
        forwards, backwards = self._propagate(encoded)
        if controls is None:
            controls = self.controls_for_mode("detailed", batch, frames.device, frames.dtype)
        if controls.shape != (batch, 2):
            raise ValueError("controls must have shape B,2 (fidelity, detail)")
        fidelity_weight = controls[:, 0].reshape(batch, 1, 1, 1)
        detail_weight = controls[:, 1].reshape(batch, 1, 1, 1)
        outputs = []
        bases = []
        fidelity_residuals = []
        detail_residuals = []
        for position in range(count):
            structure = self.structure_fusion(
                torch.cat((encoded[:, position], forwards[position], backwards[position]), dim=1)
            )
            interpolated = F.interpolate(
                frames[:, position], scale_factor=self.scale, mode="bilinear", align_corners=False
            )
            base = interpolated + self.base_head(structure)
            fidelity = self.fidelity_head(structure)
            detail = self.detail_head(structure)
            output = base + fidelity * fidelity_weight + detail * detail_weight
            outputs.append(output)
            bases.append(base)
            fidelity_residuals.append(fidelity)
            detail_residuals.append(detail)
        return {
            "output": torch.stack(outputs, dim=1),
            "base": torch.stack(bases, dim=1),
            "fidelity_residual": torch.stack(fidelity_residuals, dim=1),
            "detail_residual": torch.stack(detail_residuals, dim=1),
        }

    def forward_sequence(
        self, frames: torch.Tensor, controls: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_sequence_components(frames, controls)["output"]

    def forward(
        self, frames: torch.Tensor, controls: torch.Tensor | None = None
    ) -> torch.Tensor:
        sequence = self.forward_sequence(frames, controls)
        return sequence[:, sequence.shape[1] // 2]

    def switch_to_deploy(self) -> None:
        for module in self.modules():
            if isinstance(module, RepBlock):
                module.switch_to_deploy()


def build_aniultrascale(config: dict[str, object]) -> AniUltraScale:
    model_config = dict(config["model"])
    return AniUltraScale(
        channels=int(model_config["channels"]),
        temporal_blocks=int(model_config["temporal_blocks"]),
        reconstruction_blocks=int(model_config["reconstruction_blocks"]),
        scale=int(config["scale"]),
    )
