"""Private, zero-budget, quota-bounded training runner for the AniScale Space."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "anime": ROOT / "private_training_data" / "anime-free-seed.zip",
    "live-action": ROOT / "private_training_data" / "live-action-free-seed.zip",
}


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError("Training archive contains an unsafe path")
        package.extractall(destination)


def run_training_session(domain: str, resume: str | None) -> tuple[str | None, dict[str, object]]:
    archive = DATASETS.get(domain)
    if archive is None or not archive.is_file():
        return None, {"error": f"Private {domain} training data is unavailable."}
    workspace = Path(tempfile.mkdtemp(prefix=f"aniscale-train-{domain}-"))
    try:
        data = workspace / "data"
        safe_extract(archive, data)
        output = workspace / "output"
        command = [
            sys.executable,
            str(ROOT / "train_animesr_free.py"),
            "--data", str(data),
            "--output", str(output),
            "--domain", domain,
            "--cache", str(ROOT / ".model-cache"),
            "--session-seconds", "115",
            "--patch-size", "128",
            "--batch-size", "1",
            "--workers", "0",
        ]
        if resume:
            resume_path = Path(resume)
            if not resume_path.is_file() or resume_path.stat().st_size > 100 * 1024 * 1024:
                return None, {"error": "Resume checkpoint is missing or exceeds 100 MB."}
            local_resume = workspace / "resume.pth"
            shutil.copy2(resume_path, local_resume)
            command.extend(("--resume", str(local_resume)))
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=130,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "Training process failed.")[-1200:]
            return None, {"error": message}
        checkpoint = output / "latest.pth"
        report = output / "session.json"
        if not checkpoint.is_file() or not report.is_file():
            return None, {"error": "Training ended without a complete checkpoint."}
        persistent = ROOT / "training_results"
        persistent.mkdir(exist_ok=True)
        destination = persistent / f"{domain}-latest.pth"
        shutil.copy2(checkpoint, destination)
        status = json.loads(report.read_text(encoding="utf-8"))
        status["message"] = "Private free-GPU session completed. Download this checkpoint for the next resume session."
        return str(destination), status
    except subprocess.TimeoutExpired:
        return None, {"error": "The session exceeded its safe free-GPU time before checkpointing."}
    except Exception as error:
        return None, {"error": str(error)[:1200]}
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
