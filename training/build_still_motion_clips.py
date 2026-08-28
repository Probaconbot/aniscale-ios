#!/usr/bin/env python3
"""Turn attributed stills into short, temporally coherent camera-motion clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def run(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    attribution = json.loads(
        (args.acquired / "attribution_manifest.json").read_text(encoding="utf-8")
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frames = max(8, round(args.seconds * args.fps))
    generated: list[dict[str, object]] = []
    for record in attribution["accepted"]:
        if record.get("media_type") != "image":
            continue
        for relative in record["local_paths"]:
            source = args.acquired / relative
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            seed = int.from_bytes(hashlib.sha256(str(record["id"]).encode()).digest()[:4], "big")
            zoom_delta = 0.035 + (seed % 25) / 1000
            direction = -1 if seed & 1 else 1
            horizontal = 8 + seed % 18
            vertical = 5 + (seed // 7) % 14
            output = args.output / f"{record['id']}.mp4"
            # Smooth, deterministic pan/zoom supplies temporal correspondence
            # without pretending a still contains true object motion.
            phase = (seed % 628) / 100.0
            x_expression = (
                f"(iw-iw/zoom)/2+{direction}*{horizontal}*sin(on/{frames}*2*PI+{phase})"
            )
            y_expression = f"(ih-ih/zoom)/2+{vertical}*cos(on/{frames}*2*PI+{phase})"
            filter_value = (
                "scale='max(1920,iw)':'max(1080,ih)':force_original_aspect_ratio=increase,"
                "crop=1920:1080,"
                f"zoompan=z='1+{zoom_delta}*sin(on/{frames}*PI)':x='{x_expression}':"
                f"y='{y_expression}':d={frames}:s=1280x720:fps={args.fps},format=yuv420p"
            )
            run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(source),
                    "-vf",
                    filter_value,
                    "-frames:v",
                    str(frames),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "10",
                    "-preset",
                    "slow",
                    str(output),
                ]
            )
            generated.append(
                {
                    "source_id": record["id"],
                    "source_path": str(source),
                    "output_path": str(output),
                    "frames": frames,
                    "fps": args.fps,
                    "synthetic_motion": True,
                }
            )
    (args.output / "still_motion_manifest.json").write_text(
        json.dumps(generated, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(generated)} attributed still-motion clips")


if __name__ == "__main__":
    main()

