"""Paired sequence loader with motion fields for temporal fine-tuning."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _frames(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode training frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0


def _backward_flow(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    # Current-to-previous flow makes grid_sample warp the previous output into
    # current-frame coordinates without inverting a sparse forward field.
    return cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.1,
        0,
    ).astype(np.float32)


class PairedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        split: str,
        sequence_length: int,
        hr_patch_size: int,
        scale: int = 4,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.sequence_length = sequence_length
        self.hr_patch_size = hr_patch_size
        self.scale = scale
        self.augment = augment
        if hr_patch_size % scale:
            raise ValueError("hr_patch_size must be divisible by scale")
        hr_root = root / split / "hr"
        lq_root = root / split / "lq"
        if not hr_root.is_dir() or not lq_root.is_dir():
            raise FileNotFoundError(f"Missing paired sequence folders under {root / split}")
        self.samples: list[tuple[list[Path], list[Path], int]] = []
        for hr_clip in sorted(path for path in hr_root.iterdir() if path.is_dir()):
            lq_clip = lq_root / hr_clip.name
            if not lq_clip.is_dir():
                continue
            hr_frames = _frames(hr_clip)
            lq_frames = _frames(lq_clip)
            count = min(len(hr_frames), len(lq_frames))
            for start in range(max(0, count - sequence_length + 1)):
                self.samples.append((hr_frames, lq_frames, start))
        if not self.samples:
            raise RuntimeError(f"No {sequence_length}-frame paired sequences found in {root / split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        hr_paths, lq_paths, start = self.samples[index]
        hr = [_read_rgb(path) for path in hr_paths[start : start + self.sequence_length]]
        lq = [_read_rgb(path) for path in lq_paths[start : start + self.sequence_length]]
        lq_height, lq_width = lq[0].shape[:2]
        lq_patch = self.hr_patch_size // self.scale
        if lq_height < lq_patch or lq_width < lq_patch:
            raise RuntimeError(f"LQ frame is smaller than the requested patch: {lq_paths[start]}")
        top = random.randint(0, lq_height - lq_patch)
        left = random.randint(0, lq_width - lq_patch)
        hr_top = top * self.scale
        hr_left = left * self.scale
        lq = [frame[top : top + lq_patch, left : left + lq_patch] for frame in lq]
        hr = [
            frame[
                hr_top : hr_top + self.hr_patch_size,
                hr_left : hr_left + self.hr_patch_size,
            ]
            for frame in hr
        ]
        if any(frame.shape[:2] != (self.hr_patch_size, self.hr_patch_size) for frame in hr):
            raise RuntimeError("HR/LQ sequence is not spatially aligned at the configured scale")

        if self.augment:
            horizontal = random.random() < 0.5
            vertical = random.random() < 0.15
            transpose = random.random() < 0.5
            for frames in (lq, hr):
                for position, frame in enumerate(frames):
                    if horizontal:
                        frame = np.flip(frame, axis=1)
                    if vertical:
                        frame = np.flip(frame, axis=0)
                    if transpose:
                        frame = np.swapaxes(frame, 0, 1)
                    frames[position] = np.ascontiguousarray(frame)

        flows = [
            torch.from_numpy(_backward_flow(lq[position], lq[position - 1])).permute(2, 0, 1)
            for position in range(1, len(lq))
        ]
        return {
            "lq": torch.stack([_tensor(frame) for frame in lq]),
            "hr": torch.stack([_tensor(frame) for frame in hr]),
            "backward_flow": torch.stack(flows),
        }

