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


def local_contrast_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def deviation(value: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(value, kernel_size=7, stride=1, padding=3)
        mean_square = F.avg_pool2d(value.square(), kernel_size=7, stride=1, padding=3)
        return torch.sqrt((mean_square - mean.square()).clamp_min(1e-6))

    return charbonnier(deviation(prediction) - deviation(target))


def laplacian_pyramid_loss(
    prediction: torch.Tensor, target: torch.Tensor, levels: int = 3
) -> torch.Tensor:
    losses = []
    current_prediction = prediction
    current_target = target
    for _ in range(levels):
        prediction_low = F.avg_pool2d(current_prediction, 2, 2)
        target_low = F.avg_pool2d(current_target, 2, 2)
        prediction_up = F.interpolate(
            prediction_low,
            size=current_prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        target_up = F.interpolate(
            target_low,
            size=current_target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        losses.append(
            charbonnier(
                (current_prediction - prediction_up) - (current_target - target_up)
            )
        )
        current_prediction = prediction_low
        current_target = target_low
    return torch.stack(losses).mean()


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
        cleaned = components["cleaned"]
        cleaning_target = F.interpolate(
            flat_target,
            size=cleaned.shape[-2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).reshape_as(cleaned)
        cleaning = charbonnier(cleaned - cleaning_target)
        pixel = charbonnier(prediction - target)
        edge = charbonnier(sobel_edges(flat_prediction) - sobel_edges(flat_target))
        frequency = frequency_loss(flat_prediction, flat_target)
        local_contrast = local_contrast_loss(flat_prediction, flat_target)
        laplacian = laplacian_pyramid_loss(flat_prediction, flat_target)

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
            "cleaning": cleaning,
            "charbonnier": pixel,
            "edge": edge,
            "frequency": frequency,
            "local_contrast": local_contrast,
            "laplacian": laplacian,
            "temporal": temporal,
            "fidelity": fidelity,
            "detail": detail,
            "perceptual": perceptual,
        }
        total = sum(values[name] * self.weights.get(name, 0.0) for name in values)
        metrics = {"loss": float(total.detach())}
        metrics.update({name: float(value.detach()) for name, value in values.items()})
        return total, metrics
