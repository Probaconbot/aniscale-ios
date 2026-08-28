"""Spatial, frequency, branch, and motion-aligned losses for AniUltraScale."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from temporal_losses import PerceptualFeatures, charbonnier, sobel_edges, warp_previous


def high_pass(value: torch.Tensor) -> torch.Tensor:
    blurred = F.avg_pool2d(value, kernel_size=5, stride=1, padding=2)
    return value - blurred


def frequency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    height, width = prediction.shape[-2:]
    y = torch.fft.fftfreq(height, device=prediction.device).abs().reshape(1, 1, height, 1)
    x = torch.fft.rfftfreq(width, device=prediction.device).reshape(1, 1, 1, -1)
    weights = torch.sqrt(x.square() + y.square()).clamp_min(0.05)
    return (torch.abs(predicted_fft - target_fft) * weights).mean()


class AniUltraScaleLoss(nn.Module):
    def __init__(self, weights: dict[str, float], scale: int = 2) -> None:
        super().__init__()
        self.weights = weights
        self.scale = scale
        self.perceptual = PerceptualFeatures() if weights.get("perceptual", 0) > 0 else None

    def forward(
        self,
        components: dict[str, torch.Tensor],
        target: torch.Tensor,
        backward_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = components["output"]
        flat_prediction = prediction.flatten(0, 1)
        flat_target = target.flatten(0, 1)
        pixel = charbonnier(prediction - target)
        edge = charbonnier(sobel_edges(flat_prediction) - sobel_edges(flat_target))
        frequency = frequency_loss(flat_prediction, flat_target)

        temporal_terms = []
        for position in range(1, prediction.shape[1]):
            flow = backward_flow[:, position - 1]
            predicted_delta = prediction[:, position] - warp_previous(
                prediction[:, position - 1], flow, self.scale
            )
            target_delta = target[:, position] - warp_previous(
                target[:, position - 1], flow, self.scale
            )
            temporal_terms.append(charbonnier(predicted_delta - target_delta))
        temporal = torch.stack(temporal_terms).mean()

        # Fidelity is trained toward the real target. Detail learns only the
        # target high-frequency residual and is regularised to avoid texture
        # invention in smooth regions.
        fidelity_output = components["base"] + components["fidelity_residual"]
        fidelity = charbonnier(fidelity_output - target)
        detail_target = high_pass(target.flatten(0, 1)).reshape_as(target)
        detail = charbonnier(components["detail_residual"] - detail_target)

        perceptual = prediction.new_zeros(())
        if self.perceptual is not None:
            sample_count = min(4, flat_prediction.shape[0])
            predicted_features = self.perceptual(flat_prediction[:sample_count])
            with torch.no_grad():
                target_features = self.perceptual(flat_target[:sample_count])
            perceptual = F.l1_loss(predicted_features, target_features)

        values = {
            "charbonnier": pixel,
            "edge": edge,
            "frequency": frequency,
            "temporal": temporal,
            "fidelity": fidelity,
            "detail": detail,
            "perceptual": perceptual,
        }
        total = sum(values[name] * self.weights.get(name, 0.0) for name in values)
        metrics = {"loss": float(total.detach())}
        metrics.update({name: float(value.detach()) for name, value in values.items()})
        return total, metrics

