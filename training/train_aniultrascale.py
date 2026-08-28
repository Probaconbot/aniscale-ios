#!/usr/bin/env python3
"""Train AniUltraScale from real paired video sequences."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aniultrascale_losses import AniUltraScaleLoss
from aniultrascale_model import AniUltraScale, build_aniultrascale
from sequence_dataset import PairedSequenceDataset


def dataset_fingerprint(root: Path) -> str:
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(
            f"{manifest} is required so release checkpoints can identify their training data"
        )
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        current = model.state_dict()
        for name, value in ema.state_dict().items():
            value.lerp_(current[name].detach(), 1.0 - decay)


def save_checkpoint(
    path: Path,
    model: AniUltraScale,
    ema: AniUltraScale,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: dict[str, object],
    fingerprint: str,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "aniultrascale-v2",
            "trained": True,
            "params": model.state_dict(),
            "params_ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "validation_loss": validation_loss,
            "dataset_manifest_sha256": fingerprint,
            "config": config,
        },
        path,
    )


def load_resume(
    path: Path,
    model: AniUltraScale,
    ema: AniUltraScale,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "aniultrascale-v2":
        raise RuntimeError("Resume checkpoint is not an AniUltraScale checkpoint")
    model.load_state_dict(checkpoint["params"], strict=True)
    ema.load_state_dict(checkpoint["params_ema"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


@torch.no_grad()
def validate(
    model: AniUltraScale,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_function: AniUltraScaleLoss,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for index, batch in enumerate(loader):
        low_quality = batch["lq"].to(device, non_blocking=True)
        high_quality = batch["hr"].to(device, non_blocking=True)
        flow = batch["backward_flow"].to(device, non_blocking=True)
        controls = model.controls_for_mode(
            "detailed", low_quality.shape[0], device, low_quality.dtype
        )
        components = model.forward_sequence_components(low_quality, controls)
        loss, _ = loss_function(components, high_quality, flow)
        losses.append(float(loss))
        if index + 1 >= max_batches:
            break
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--validation-batches", type=int, default=12)
    parser.add_argument("--allow-cpu-smoke-test", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config["scale"]) != 2 or int(config["sequence_length"]) != 5:
        raise RuntimeError("AniUltraScale v2 training requires native 2x, five-frame clips")
    if not torch.cuda.is_available() and not args.allow_cpu_smoke_test:
        raise RuntimeError(
            "AniUltraScale training requires CUDA. CPU mode is intentionally limited to smoke tests."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    fingerprint = dataset_fingerprint(args.data)
    model = build_aniultrascale(config).to(device).train()
    ema = copy.deepcopy(model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    start_iteration = 0
    if args.resume:
        start_iteration = load_resume(args.resume, model, ema, optimizer)

    dataset_arguments = {
        "root": args.data,
        "sequence_length": 5,
        "hr_patch_size": int(config["hr_patch_size"]),
        "scale": 2,
    }
    training = PairedSequenceDataset(split="train", augment=True, **dataset_arguments)
    validation = PairedSequenceDataset(split="validation", augment=False, **dataset_arguments)
    training_loader = DataLoader(
        training,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, args.workers // 2),
        pin_memory=device.type == "cuda",
    )
    loss_function = AniUltraScaleLoss(
        {name: float(weight) for name, weight in dict(config["loss"]).items()}, scale=2
    ).to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    accumulation = int(config["gradient_accumulation"])
    total_iterations = int(config["iterations"])
    ema_decay = float(config["ema_decay"])
    best_loss = float("inf")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    iterator = cycle(training_loader)
    optimizer.zero_grad(set_to_none=True)

    for iteration in range(start_iteration + 1, total_iterations + 1):
        batch = next(iterator)
        low_quality = batch["lq"].to(device, non_blocking=True)
        high_quality = batch["hr"].to(device, non_blocking=True)
        flow = batch["backward_flow"].to(device, non_blocking=True)
        # Vary the detail mix during training so one checkpoint supports the
        # Subtle/Detailed/Creative controls without three resident networks.
        fidelity = torch.empty(low_quality.shape[0], device=device).uniform_(0.86, 1.0)
        detail = torch.empty(low_quality.shape[0], device=device).uniform_(0.32, 1.20)
        controls = torch.stack((fidelity, detail), dim=1).to(low_quality.dtype)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            components = model.forward_sequence_components(low_quality, controls)
            loss, metrics = loss_function(components, high_quality, flow)
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()
        if iteration % accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model, ema_decay)

        if iteration % 50 == 0:
            record = {"iteration": iteration, **metrics}
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        validation_loss = float("nan")
        if iteration % args.validate_every == 0:
            validation_loss = validate(
                ema, validation_loader, loss_function, device, args.validation_batches
            )
            print(
                json.dumps({"iteration": iteration, "validation_loss": validation_loss}),
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(
                    args.output / "best.pth",
                    model,
                    ema,
                    optimizer,
                    iteration,
                    config,
                    fingerprint,
                    validation_loss,
                )
        if iteration % args.save_every == 0:
            save_checkpoint(
                args.output / f"iteration_{iteration:07d}.pth",
                model,
                ema,
                optimizer,
                iteration,
                config,
                fingerprint,
                validation_loss,
            )


if __name__ == "__main__":
    main()
