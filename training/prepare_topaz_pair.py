#!/usr/bin/env python3
"""Build a private 2x paired dataset from an aligned source/reference video pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(
    ffmpeg: str,
    source: Path,
    output: Path,
    start: float,
    duration: float,
    fps: int,
    width: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-t", str(duration), "-i", str(source),
            "-vf", f"fps={fps},scale={width}:-2:flags=lanczos", "-fps_mode", "passthrough", str(output / "%06d.png"),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lq", type=Path, required=True)
    parser.add_argument("--hr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seconds", type=float, default=8.0)
    parser.add_argument("--validation-seconds", type=float, default=2.0)
    parser.add_argument("--validation-start", type=float, default=24.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--lq-width", type=int, default=640)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    if args.output.exists():
        raise RuntimeError(f"Output already exists: {args.output}")
    pairs = (
        ("train", 0.0, args.train_seconds),
        ("validation", args.validation_start, args.validation_seconds),
    )
    for split, start, duration in pairs:
        extract(ffmpeg, args.lq, args.output / split / "lq" / "topaz-reference", start, duration, args.fps, args.lq_width)
        extract(ffmpeg, args.hr, args.output / split / "hr" / "topaz-reference", start, duration, args.fps, args.lq_width * 2)
    manifest = {
        "format": "aniscale-private-paired-video-v1",
        "scale": 2,
        "fps": args.fps,
        "source_sha256": sha256(args.lq),
        "reference_sha256": sha256(args.hr),
        "training_source_width": args.lq_width,
        "training_reference_width": args.lq_width * 2,
        "provenance": "User-provided source and Topaz-upscaled comparison; private experimental use.",
        "splits": [
            {"name": split, "start_seconds": start, "duration_seconds": duration}
            for split, start, duration in pairs
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
