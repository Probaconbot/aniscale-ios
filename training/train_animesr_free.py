#!/usr/bin/env python3
"""Resumable, quota-bounded AnimeSR_v2 fine-tuning for private experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent / "cloud"
sys.path.insert(0, str(MODEL_ROOT if (MODEL_ROOT / "animesr.py").is_file() else ROOT))

from animesr import CHECKPOINT_SHA256, load_model  # noqa: E402
from sequence_dataset import PairedSequenceDataset  # noqa: E402


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
    flow = F.interpolate(backward_flow, size=(height, width), mode="bilinear", align_corners=False)
    flow = flow * float(scale)
    y, x = torch.meshgrid(
        torch.arange(height, device=previous.device, dtype=previous.dtype),
        torch.arange(width, device=previous.device, dtype=previous.dtype),
        indexing="ij",
    )
    grid_x = 2.0 * (x.unsqueeze(0) + flow[:, 0]) / max(width - 1, 1) - 1.0
    grid_y = 2.0 * (y.unsqueeze(0) + flow[:, 1]) / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(previous, grid, mode="bilinear", padding_mode="border", align_corners=True)


def dataset_fingerprint(root: Path) -> str:
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"Missing dataset manifest: {manifest}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        current = model.state_dict()
        for name, value in ema.state_dict().items():
            value.lerp_(current[name].detach(), 1.0 - decay)


def run_sequence(model: torch.nn.Module, low_quality: torch.Tensor) -> torch.Tensor:
    batch, count, _, height, width = low_quality.shape
    feedback = low_quality.new_zeros(batch, 3, height * 4, width * 4)
    state = low_quality.new_zeros(batch, 64, height, width)
    outputs = []
    for position in range(count):
        previous = low_quality[:, max(0, position - 1)]
        current = low_quality[:, position]
        following = low_quality[:, min(count - 1, position + 1)]
        feedback, state = model(
            torch.cat((previous, current, following), dim=1), feedback, state
        )
        outputs.append(feedback)
    return torch.stack(outputs, dim=1)


def restoration_loss(
    prediction_4x: torch.Tensor,
    target: torch.Tensor,
    backward_flow: torch.Tensor,
    scale: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch, count = target.shape[:2]
    prediction = F.interpolate(
        prediction_4x.flatten(0, 1),
        size=target.shape[-2:],
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).reshape_as(target)
    flat_prediction = prediction.flatten(0, 1)
    flat_target = target.flatten(0, 1)
    pixel = charbonnier(prediction - target)
    edge = charbonnier(sobel_edges(flat_prediction) - sobel_edges(flat_target))
    predicted_detail = flat_prediction - F.avg_pool2d(flat_prediction, 5, 1, 2)
    target_detail = flat_target - F.avg_pool2d(flat_target, 5, 1, 2)
    detail = charbonnier(predicted_detail - target_detail)
    temporal_terms = []
    for position in range(1, count):
        flow = backward_flow[:, position - 1]
        predicted_delta = prediction[:, position] - warp_previous(
            prediction[:, position - 1], flow, scale
        )
        target_delta = target[:, position] - warp_previous(
            target[:, position - 1], flow, scale
        )
        temporal_terms.append(charbonnier(predicted_delta - target_delta))
    temporal = torch.stack(temporal_terms).mean()
    total = pixel + 0.16 * edge + 0.12 * detail + 0.22 * temporal
    return total, {
        "loss": float(total.detach()),
        "pixel": float(pixel.detach()),
        "edge": float(edge.detach()),
        "detail": float(detail.detach()),
        "temporal": float(temporal.detach()),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    domain: str,
    fingerprint: str,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(
        {
            "format": "animesr-v2-finetune-v1",
            "base_checkpoint_sha256": CHECKPOINT_SHA256,
            "domain": domain,
            "dataset_manifest_sha256": fingerprint,
            "iteration": iteration,
            "params": model.state_dict(),
            "params_ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        pending,
    )
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=("anime", "live-action"), required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / ".model-cache")
    parser.add_argument("--session-seconds", type=float, default=150.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--allow-cpu-smoke-test", action="store_true")
    args = parser.parse_args()
    if args.session_seconds < 5:
        raise ValueError("session-seconds must leave enough time for a complete checkpoint")
    if not torch.cuda.is_available() and not args.allow_cpu_smoke_test:
        raise RuntimeError("CUDA is required; CPU mode is only for a one-step smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fingerprint = dataset_fingerprint(args.data)
    model = load_model(args.cache).to(device).train()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "animesr-v2-finetune-v1":
            raise RuntimeError("Resume file is not an AnimeSR fine-tuning checkpoint")
        if checkpoint.get("domain") != args.domain:
            raise RuntimeError("Resume checkpoint belongs to a different content domain")
        if checkpoint.get("dataset_manifest_sha256") != fingerprint:
            raise RuntimeError("Dataset manifest changed since the resume checkpoint")
        model.load_state_dict(checkpoint["params"], strict=True)
        ema.load_state_dict(checkpoint["params_ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration = int(checkpoint["iteration"])

    dataset = PairedSequenceDataset(
        root=args.data,
        split="train",
        sequence_length=5,
        hr_patch_size=args.patch_size,
        scale=2,
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    started = time.monotonic()
    deadline = started + args.session_seconds
    session_steps = 0
    metrics: dict[str, float] = {"loss": math.nan}
    model.zero_grad(set_to_none=True)
    while time.monotonic() < deadline and (args.max_steps <= 0 or session_steps < args.max_steps):
        for batch in loader:
            if time.monotonic() >= deadline or (args.max_steps > 0 and session_steps >= args.max_steps):
                break
            low_quality = batch["lq"].to(device, non_blocking=True)
            target = batch["hr"].to(device, non_blocking=True)
            flow = batch["backward_flow"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = run_sequence(model, low_quality)
                loss, metrics = restoration_loss(prediction, target, flow, scale=2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model, 0.999)
            iteration += 1
            session_steps += 1
            if session_steps % 10 == 0:
                print(json.dumps({"iteration": iteration, **metrics}), flush=True)
        if len(loader) == 0:
            raise RuntimeError("Training dataset produced no batches")

    metrics = dict(metrics)
    metrics.update({"session_steps": session_steps, "session_seconds": time.monotonic() - started})
    save_checkpoint(
        args.output / "latest.pth",
        model,
        ema,
        optimizer,
        iteration,
        args.domain,
        fingerprint,
        metrics,
    )
    (args.output / "session.json").write_text(
        json.dumps({"iteration": iteration, "domain": args.domain, **metrics}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"checkpoint": str(args.output / 'latest.pth'), "iteration": iteration, **metrics}), flush=True)


if __name__ == "__main__":
    main()
