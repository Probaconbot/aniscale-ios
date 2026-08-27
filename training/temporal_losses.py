"""Faithful spatial and motion-aligned temporal losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


def charbonnier(value: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon).mean()


def sobel_edges(value: torch.Tensor) -> torch.Tensor:
    kernel_x = value.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    kernel_x = kernel_x.repeat(value.shape[1], 1, 1, 1)
    kernel_y = kernel_y.repeat(value.shape[1], 1, 1, 1)
    x = F.conv2d(value, kernel_x, padding=1, groups=value.shape[1])
    y = F.conv2d(value, kernel_y, padding=1, groups=value.shape[1])
    return torch.sqrt(x.square() + y.square() + 1e-6)


def warp_previous(previous: torch.Tensor, backward_flow: torch.Tensor, scale: int) -> torch.Tensor:
    height, width = previous.shape[-2:]
    flow = F.interpolate(
        backward_flow,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ) * float(scale)
    y, x = torch.meshgrid(
        torch.arange(height, device=previous.device, dtype=previous.dtype),
        torch.arange(width, device=previous.device, dtype=previous.dtype),
        indexing="ij",
    )
    grid_x = 2.0 * (x.unsqueeze(0) + flow[:, 0]) / max(width - 1, 1) - 1.0
    grid_y = 2.0 * (y.unsqueeze(0) + flow[:, 1]) / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(previous, grid, mode="bilinear", padding_mode="border", align_corners=True)


class PerceptualFeatures(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:18].eval()
        for parameter in features.parameters():
            parameter.requires_grad_(False)
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.features((value - self.mean) / self.std)


class VideoRestorationLoss(nn.Module):
    def __init__(self, weights: dict[str, float], scale: int = 4) -> None:
        super().__init__()
        self.weights = weights
        self.scale = scale
        self.perceptual = PerceptualFeatures() if weights.get("perceptual", 0) > 0 else None

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Inputs are B,T,C,H,W. The model remains frame-based for Core ML,
        # while the training objective explicitly observes adjacent frames.
        pixel = charbonnier(prediction - target)
        edge = charbonnier(sobel_edges(prediction.flatten(0, 1)) - sobel_edges(target.flatten(0, 1)))
        perceptual = prediction.new_zeros(())
        if self.perceptual is not None:
            sample_count = min(4, prediction.shape[0] * prediction.shape[1])
            predicted_features = self.perceptual(prediction.flatten(0, 1)[:sample_count])
            with torch.no_grad():
                target_features = self.perceptual(target.flatten(0, 1)[:sample_count])
            perceptual = F.l1_loss(predicted_features, target_features)

        temporal_terms = []
        for position in range(1, prediction.shape[1]):
            flow = backward_flow[:, position - 1]
            predicted_residual = prediction[:, position] - warp_previous(
                prediction[:, position - 1], flow, self.scale
            )
            target_residual = target[:, position] - warp_previous(
                target[:, position - 1], flow, self.scale
            )
            temporal_terms.append(charbonnier(predicted_residual - target_residual))
        temporal = torch.stack(temporal_terms).mean()

        total = (
            pixel * self.weights.get("charbonnier", 1.0)
            + edge * self.weights.get("edge", 0.0)
            + perceptual * self.weights.get("perceptual", 0.0)
            + temporal * self.weights.get("temporal", 0.0)
        )
        metrics = {
            "loss": float(total.detach()),
            "pixel": float(pixel.detach()),
            "edge": float(edge.detach()),
            "perceptual": float(perceptual.detach()),
            "temporal": float(temporal.detach()),
        }
        return total, metrics

