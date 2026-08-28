#!/usr/bin/env python3
"""Acquire a bounded, attributable corpus of openly licensed media.

The command is intentionally opt-in and storage-capped. Public availability is
never treated as a licence, and every accepted item is written to an attribution
manifest before it can enter dataset preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = ROOT / "datasets" / "open_media_sources.json"
USER_AGENT = "AniScale dataset builder/1.0 (licence-audited research client)"
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".mkv", ".webm"}


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def clean_markup(value: object) -> str:
    text = html.unescape(str(value or ""))
    return re.sub(r"<[^>]+>", "", text).strip()


def canonical_license(value: str) -> str | None:
    compact = re.sub(r"\s+", " ", value.strip().lower())
    if compact in {"cc0", "cc0 1.0", "public domain", "pd"}:
        return "CC0" if compact.startswith("cc0") else "Public domain"
    match = re.search(r"cc[- ]?by(?: attribution)?[- ]?(2\.0|3\.0|4\.0)", compact)
    if match:
        return f"CC BY {match.group(1)}"
    return None


def available_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, remaining: int) -> tuple[int, str]:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request(url), timeout=60) as response, temporary.open("wb") as out:
            declared = int(response.headers.get("Content-Length", 0))
            if declared and declared > remaining:
                raise RuntimeError(f"{url} exceeds the remaining download budget")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > remaining:
                    raise RuntimeError(f"{url} exceeded the remaining download budget")
                out.write(chunk)
                digest.update(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    return written, digest.hexdigest()


def safe_extract_video(archive: Path, output: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            name = Path(member.filename).name
            if not name or Path(name).suffix.lower() not in MEDIA_SUFFIXES:
                continue
            destination = output / name
            with package.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted.append(destination)
    if not extracted:
        raise RuntimeError(f"No supported media was found in {archive}")
    return extracted


def commons_items(category: str, limit: int) -> list[dict[str, object]]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": str(min(limit * 4, 200)),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
        }
    )
    with urllib.request.urlopen(
        request(f"https://commons.wikimedia.org/w/api.php?{query}"), timeout=60
    ) as response:
        payload = json.load(response)
    return list(payload.get("query", {}).get("pages", []))


def commons_record(page: dict[str, object], category: str) -> dict[str, object] | None:
    information = list(page.get("imageinfo", []))
    if not information:
        return None
    item = information[0]
    metadata = dict(item.get("extmetadata", {}))
    raw_license = clean_markup(dict(metadata.get("LicenseShortName", {})).get("value"))
    license_name = canonical_license(raw_license)
    if license_name is None:
        return None
    mime = str(item.get("mime", ""))
    if not (mime.startswith("image/") or mime.startswith("video/")):
        return None
    url = str(item.get("url", ""))
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix not in MEDIA_SUFFIXES:
        return None
    return {
        "id": f"commons-{page.get('pageid')}",
        "url": url,
        "description_url": str(item.get("descriptionurl", "")),
        "expected_bytes": int(item.get("size", 0)),
        "media_type": "video" if mime.startswith("video/") else "image",
        "mime": mime,
        "license": license_name,
        "license_url": clean_markup(dict(metadata.get("LicenseUrl", {})).get("value")),
        "creator": clean_markup(dict(metadata.get("Artist", {})).get("value")) or "Unknown",
        "attribution": clean_markup(dict(metadata.get("Credit", {})).get("value")),
        "category": category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gb", type=float, required=True)
    parser.add_argument("--commons-per-category", type=int, default=12)
    parser.add_argument("--skip-blender", action="store_true")
    parser.add_argument("--skip-commons", action="store_true")
    parser.add_argument("--acknowledge-licenses", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_licenses:
        raise RuntimeError(
            "Read the source and attribution policy, then pass --acknowledge-licenses"
        )
    if args.max_gb <= 0:
        raise ValueError("--max-gb must be positive")
    budget = int(args.max_gb * 1024**3)
    reserve = 2 * 1024**3
    free = available_bytes(args.output)
    if free < budget + reserve:
        raise RuntimeError(
            f"Need {args.max_gb + 2:.1f} GB free including reserve; only {free / 1024**3:.1f} GB is available"
        )
    specification = json.loads(args.sources.read_text(encoding="utf-8"))
    accepted = set(specification["accepted_licenses"])
    media_directory = args.output / "media"
    media_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    used = 0
    seen_ids: set[str] = set()

    candidates: list[dict[str, object]] = []
    if not args.skip_blender:
        candidates.extend(dict(value) for value in specification["curated_videos"])
    if not args.skip_commons:
        for category in specification["commons_categories"]:
            accepted_in_category = 0
            for page in commons_items(str(category), args.commons_per_category):
                record = commons_record(page, str(category))
                if record is None:
                    continue
                candidates.append(record)
                accepted_in_category += 1
                if accepted_in_category >= args.commons_per_category:
                    break

    for candidate in candidates:
        identifier = str(candidate.get("id", ""))
        if not identifier or identifier in seen_ids:
            rejected.append({"id": identifier, "reason": "missing or duplicate identifier"})
            continue
        seen_ids.add(identifier)
        if candidate.get("license") not in accepted:
            rejected.append({"id": candidate.get("id"), "reason": "licence not allowlisted"})
            continue
        expected = int(candidate.get("expected_bytes", 0))
        if expected <= 0 or used + expected > budget:
            rejected.append({"id": candidate.get("id"), "reason": "download budget"})
            continue
        url = str(candidate["url"])
        suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
        destination = media_directory / f"{candidate['id']}{suffix}"
        size, digest = download(url, destination, budget - used)
        used += size
        outputs = [destination]
        if suffix == ".zip":
            outputs = safe_extract_video(destination, media_directory)
        record = dict(candidate)
        record.update(
            {
                "downloaded_bytes": size,
                "download_sha256": digest,
                "local_paths": [str(path.relative_to(args.output)) for path in outputs],
            }
        )
        records.append(record)
        print(f"Accepted {candidate['id']} ({size / 1024**2:.1f} MB)", flush=True)

    manifest = {
        "format": "aniscale-attributed-media-v1",
        "source_specification": str(args.sources),
        "download_budget_bytes": budget,
        "downloaded_bytes": used,
        "accepted": records,
        "rejected": rejected,
    }
    (args.output / "attribution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} attributed media records; used {used / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
