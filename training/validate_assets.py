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
    ultra_configs = [
        json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
        for name in ("aniultrascale_fast.json", "aniultrascale_quality.json")
    ]
    if {config["name"] for config in ultra_configs} != {
        "AniUltraScale FAST",
        "AniUltraScale QUALITY",
    }:
        raise RuntimeError("AniUltraScale FAST and QUALITY configs are both required")
    for config in ultra_configs:
        if config["architecture"] != "aniultrascale-realvsr-fani-dual-v2":
            raise RuntimeError("AniUltraScale must use its real-VSR/FANI dual-control architecture")
        if int(config["scale"]) != 2 or int(config["sequence_length"]) != 5:
            raise RuntimeError("AniUltraScale v2 must use native 2x five-frame clips")
        loss = dict(config["loss"])
        for required in (
            "cleaning",
            "charbonnier",
            "edge",
            "frequency",
            "local_contrast",
            "laplacian",
            "temporal",
            "fidelity",
            "detail",
        ):
            if float(loss.get(required, 0)) <= 0:
                raise RuntimeError(f"AniUltraScale is missing its {required} loss")
    upstreams = json.loads((ROOT / "upstreams.json").read_text(encoding="utf-8"))
    expected_upstreams = {"RealBasicVSR", "FANI", "PiSA-SR"}
    if {item["name"] for item in upstreams} != expected_upstreams:
        raise RuntimeError("AniUltraScale upstream provenance is incomplete")
    for upstream in upstreams:
        revision = upstream["revision"]
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise RuntimeError(f"{upstream['name']} does not use an immutable commit revision")
    sources = json.loads(
        (ROOT / "datasets" / "open_media_sources.json").read_text(encoding="utf-8")
    )
    policy = dict(sources["policy"])
    if not all(
        policy.get(key) is True
        for key in (
            "require_attribution_manifest",
            "exclude_noncommercial",
            "exclude_no_derivatives",
            "exclude_unknown_license",
            "exclude_vimeo_api",
            "exclude_youtube_scraping",
        )
    ):
        raise RuntimeError("The open-media acquisition policy must fail closed")
    accepted_licenses = set(sources["accepted_licenses"])
    for source in sources["curated_videos"]:
        if source["license"] not in accepted_licenses:
            raise RuntimeError(f"Curated source has an unapproved licence: {source['id']}")
        for required in ("url", "license_url", "creator", "attribution", "expected_bytes"):
            if not source.get(required):
                raise RuntimeError(f"Curated source {source['id']} is missing {required}")
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
    print("AniScale and AniUltraScale configs plus the acceptance reference are valid")


if __name__ == "__main__":
    main()
