#!/usr/bin/env python3
"""Create aligned HR/LQ sequences using actual multi-generation video codecs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def command(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is required and was not found on PATH")
    return path


def clip_seed(global_seed: int, path: Path) -> int:
    digest = hashlib.sha256(f"{global_seed}:{path.as_posix()}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def degradation(rng: random.Random) -> dict[str, object]:
    exposure_start = rng.uniform(-0.035, 0.035)
    saturation_start = rng.uniform(0.92, 1.08)
    return {
        "codec_generations": rng.choice([1, 1, 2, 2, 3]),
        "bitrate_kbps": rng.choice([180, 260, 350, 500, 700, 900, 1200]),
        "blur_sigma": round(rng.uniform(0.15, 2.1), 3),
        "noise_strength": rng.randint(0, 10),
        "ringing_amount": round(rng.uniform(0.0, 1.2), 3) if rng.random() < 0.35 else 0.0,
        "motion_mix": rng.random() < 0.4,
        "social_media": rng.random() < 0.5,
        "codec": rng.choice(["libx264", "libx264", "libx265"]),
        # Real capture parameters drift gradually rather than jumping to an
        # unrelated random value on every frame.
        "exposure_start": round(exposure_start, 4),
        "exposure_end": round(exposure_start + rng.uniform(-0.025, 0.025), 4),
        "saturation_start": round(saturation_start, 4),
        "saturation_end": round(saturation_start + rng.uniform(-0.06, 0.06), 4),
    }


def encode_generation(
    ffmpeg: str,
    source: Path,
    destination: Path,
    settings: dict[str, object],
    filters: str | None,
    generation: int,
) -> None:
    bitrate = max(120, int(settings["bitrate_kbps"]) - generation * 45)
    codec = str(settings["codec"])
    arguments = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if filters:
        arguments.extend(["-vf", filters])
    arguments.extend(
        [
            "-an",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            f"{bitrate}k",
            "-maxrate",
            f"{int(bitrate * 1.15)}k",
            "-bufsize",
            f"{bitrate * 2}k",
            "-g",
            "48",
            "-bf",
            "3",
            "-preset",
            "medium",
            str(destination),
        ]
    )
    command(arguments)


def prepare_clip(
    ffmpeg: str,
    source: Path,
    output: Path,
    split: str,
    scale: int,
    fps: float,
    seconds: float,
    settings: dict[str, object],
) -> dict[str, object]:
    name = source.stem.replace(" ", "_")
    hr_directory = output / split / "hr" / name
    lq_directory = output / split / "lq" / name
    hr_directory.mkdir(parents=True, exist_ok=True)
    lq_directory.mkdir(parents=True, exist_ok=True)
    work = output / ".work" / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # HR dimensions are rounded to a multiple of 2*scale; LQ is therefore an
    # exact integer scale pair and also encoder-safe yuv420p.
    hr_scale = (
        f"scale=trunc(iw/{scale * 2})*{scale * 2}:"
        f"trunc(ih/{scale * 2})*{scale * 2}:flags=lanczos"
    )
    command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-t",
            str(seconds),
            "-i",
            str(source),
            "-vf",
            f"fps={fps},{hr_scale}",
            "-vsync",
            "0",
            str(hr_directory / "%06d.png"),
        ]
    )

    filters = [
        f"fps={fps}",
        f"scale=trunc(iw/{scale * 2})*2:trunc(ih/{scale * 2})*2:flags=lanczos",
        f"gblur=sigma={settings['blur_sigma']}",
    ]
    exposure_start = float(settings["exposure_start"])
    exposure_delta = float(settings["exposure_end"]) - exposure_start
    saturation_start = float(settings["saturation_start"])
    saturation_delta = float(settings["saturation_end"]) - saturation_start
    filters.append(
        "eq="
        f"brightness='{exposure_start:.5f}+({exposure_delta:.5f})*t/{seconds:.5f}':"
        f"saturation='{saturation_start:.5f}+({saturation_delta:.5f})*t/{seconds:.5f}':"
        "eval=frame"
    )
    if bool(settings["motion_mix"]):
        filters.append("tmix=frames=3:weights='1 2 1'")
    if int(settings["noise_strength"]) > 0:
        filters.append(f"noise=alls={settings['noise_strength']}:allf=t+u")
    if float(settings["ringing_amount"]) > 0:
        filters.append(f"unsharp=5:5:{settings['ringing_amount']}:5:5:0")
    filters.append("format=yuv420p")

    previous = source
    generations = int(settings["codec_generations"])
    for generation in range(generations):
        encoded = work / f"generation_{generation + 1}.mp4"
        encode_generation(
            ffmpeg,
            previous,
            encoded,
            settings,
            ",".join(filters) if generation == 0 else None,
            generation,
        )
        previous = encoded
    if bool(settings["social_media"]):
        social = work / "social_media.mp4"
        social_settings = dict(settings)
        social_settings["bitrate_kbps"] = max(140, int(settings["bitrate_kbps"]) // 2)
        encode_generation(ffmpeg, previous, social, social_settings, None, generations)
        previous = social

    command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(previous),
            "-vsync",
            "0",
            str(lq_directory / "%06d.png"),
        ]
    )
    hr_count = len(list(hr_directory.glob("*.png")))
    lq_count = len(list(lq_directory.glob("*.png")))
    count = min(hr_count, lq_count)
    if count < 8:
        raise RuntimeError(f"Not enough aligned frames were produced for {source}")
    for path in sorted(hr_directory.glob("*.png"))[count:]:
        path.unlink()
    for path in sorted(lq_directory.glob("*.png"))[count:]:
        path.unlink()
    shutil.rmtree(work)
    return {
        "name": name,
        "source": str(source),
        "split": split,
        "frames": count,
        "fps": fps,
        "scale": scale,
        "degradation": settings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    ffmpeg = require_tool("ffmpeg")
    sources = sorted(
        path for path in args.source.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not sources:
        raise RuntimeError(f"No source videos found under {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in sources:
        seed = clip_seed(args.seed, source.relative_to(args.source))
        rng = random.Random(seed)
        split = "validation" if seed % 100 < args.validation_percent else "train"
        manifest.append(
            prepare_clip(
                ffmpeg,
                source,
                args.output,
                split,
                args.scale,
                args.fps,
                args.clip_seconds,
                degradation(rng),
            )
        )
        print(f"Prepared {source.name} -> {split}", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
