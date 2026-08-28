#!/usr/bin/env python3
"""Small CPU contract test; this does not claim training or output quality."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from aniultrascale_model import build_aniultrascale


ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = json.loads(
        (ROOT / "configs" / "aniultrascale_fast.json").read_text(encoding="utf-8")
    )
    # Shrink only the smoke-test network; the deployment config remains intact.
    config = copy.deepcopy(config)
    config["model"] = {
        "channels": 8,
        "temporal_blocks": 1,
        "reconstruction_blocks": 1,
    }
    torch.manual_seed(20260828)
    model = build_aniultrascale(config).eval()
    frames = torch.rand(1, 5, 3, 16, 16)
    controls = torch.tensor([[1.0, 0.62]])
    with torch.inference_mode():
        components = model.forward_sequence_components(frames, controls)
        before = model(frames, controls)
    assert components["output"].shape == (1, 5, 3, 32, 32)
    assert before.shape == (1, 3, 32, 32)
    assert torch.isfinite(before).all()
    model.switch_to_deploy()
    with torch.inference_mode():
        after = model(frames, controls)
    maximum_error = float(torch.max(torch.abs(before - after)))
    if maximum_error > 2e-4:
        raise RuntimeError(
            f"Deploy-time convolution fusion changed the output by {maximum_error}"
        )
    print(f"AniUltraScale temporal/deploy smoke test passed (max error {maximum_error:.8f})")


if __name__ == "__main__":
    main()

