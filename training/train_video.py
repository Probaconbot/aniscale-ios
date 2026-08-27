#!/usr/bin/env python3
"""Fine-tune AniScale's existing frame model with sequence-aware losses."""

from __future__ import annotations

import argparse
import copy
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aniscale_models import build_model, load_existing_weights
from sequence_dataset import PairedSequenceDataset
from temporal_losses import VideoRestorationLoss


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        model_state = model.state_dict()
        for name, value in ema.state_dict().items():
            value.lerp_(model_state[name].detach(), 1.0 - decay)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: dict[str, object],
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": model.state_dict(),
            "params_ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "validation_loss": validation_loss,
            "config": config,
        },
        path,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_function: VideoRestorationLoss,
    device: torch.device,
    max_batches: int = 12,
) -> float:
    model.eval()
    losses = []
    for batch_index, batch in enumerate(loader):
        lq = batch["lq"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        flow = batch["backward_flow"].to(device, non_blocking=True)
        batch_size, sequence, channels, height, width = lq.shape
        prediction = model(lq.view(batch_size * sequence, channels, height, width))
        prediction = prediction.view(batch_size, sequence, 3, hr.shape[-2], hr.shape[-1])
        loss, _ = loss_function(prediction.clamp(0, 1), hr, flow)
        losses.append(float(loss))
        if batch_index + 1 >= max_batches:
            break
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for real training; use --allow-cpu only for a smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    model = build_model(config)
    load_existing_weights(model, args.initial)
    model.to(device).train()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    dataset_arguments = {
        "root": args.data,
        "sequence_length": int(config["sequence_length"]),
        "hr_patch_size": int(config["hr_patch_size"]),
        "scale": int(config["scale"]),
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
        num_workers=max(1, args.workers // 2),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.99), weight_decay=0
    )
    loss_function = VideoRestorationLoss(
        {key: float(value) for key, value in dict(config["loss"]).items()},
        scale=int(config["scale"]),
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

    for iteration in range(1, total_iterations + 1):
        batch = next(iterator)
        lq = batch["lq"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        flow = batch["backward_flow"].to(device, non_blocking=True)
        batch_size, sequence, channels, height, width = lq.shape
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(lq.view(batch_size * sequence, channels, height, width))
            prediction = prediction.view(batch_size, sequence, 3, hr.shape[-2], hr.shape[-1])
            loss, metrics = loss_function(prediction, hr, flow)
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
            validation_loss = validate(ema, validation_loader, loss_function, device)
            print(json.dumps({"iteration": iteration, "validation_loss": validation_loss}), flush=True)
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(
                    args.output / "best.pth",
                    model,
                    ema,
                    optimizer,
                    iteration,
                    config,
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
                validation_loss,
            )


if __name__ == "__main__":
    main()

