#!/usr/bin/env python3
"""Dependency-free validation for training configs and acceptance assets."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    configs = [
        json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        for name in ("fast.json", "quality.json")
    ]
    if {config["architecture"] for config in configs} != {"srvgg", "rrdb"}:
        raise RuntimeError("FAST and QUALITY must retain the SRVGG and RRDB model family")
    if any(int(config["scale"]) != 4 for config in configs):
        raise RuntimeError("AniScale deployment checkpoints must remain native 4x")
    reference = json.loads(
        (ROOT / "references" / "acceptance.json").read_text(encoding="utf-8")
    )
    image = ROOT / "references" / reference["image"]
    data = image.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise RuntimeError("Acceptance reference checksum does not match its manifest")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Acceptance reference is not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (int(reference["width"]), int(reference["height"])):
        raise RuntimeError("Acceptance reference dimensions do not match its manifest")
    print("AniScale training configs and acceptance reference are valid")


if __name__ == "__main__":
    main()

