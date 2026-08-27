#!/usr/bin/env python3
"""Evaluate spatial restoration and temporal stability against AniScale targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
    import torch
except ImportError:  # Spatial/temporal metrics remain available in lean environments.
    lpips = None
    torch = None


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(gray(image), cv2.CV_32F).var())


def local_contrast(image: np.ndarray) -> float:
    luminance = gray(image)
    local_mean = cv2.GaussianBlur(luminance, (0, 0), 5)
    return float(np.std(luminance - local_mean))


def backward_flow(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        (gray(current) * 255).astype(np.uint8),
        (gray(previous) * 255).astype(np.uint8),
        None,
        0.5,
        3,
        21,
        4,
        7,
        1.2,
        0,
    )


def warp(image: np.ndarray, flow: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    resized_flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_flow[..., 0] *= width / flow.shape[1]
    resized_flow[..., 1] *= height / flow.shape[0]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(
        image,
        x + resized_flow[..., 0],
        y + resized_flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return capture


def reference_metrics(image_path: Path, spec_path: Path) -> dict[str, float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read acceptance reference: {image_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    panels = spec["panels"]

    def crop(panel: dict[str, int]) -> np.ndarray:
        x, y = int(panel["x"]), int(panel["y"])
        return image[y : y + int(panel["height"]), x : x + int(panel["width"])]

    input_panel = crop(panels["input"])
    target_panel = crop(panels["target"])
    return {
        "input_sharpness": sharpness(input_panel),
        "target_sharpness": sharpness(target_panel),
        "sharpness_gain": sharpness(target_panel) / max(sharpness(input_panel), 1e-8),
        "input_local_contrast": local_contrast(input_panel),
        "target_local_contrast": local_contrast(target_panel),
        "local_contrast_gain": local_contrast(target_panel)
        / max(local_contrast(input_panel), 1e-8),
    }


def evaluate(
    input_path: Path,
    output_path: Path,
    ground_truth_path: Path | None,
    max_frames: int,
    comparisons: Path | None,
) -> dict[str, object]:
    input_capture = open_video(input_path)
    output_capture = open_video(output_path)
    truth_capture = open_video(ground_truth_path) if ground_truth_path else None
    input_sharpness: list[float] = []
    output_sharpness: list[float] = []
    input_contrast: list[float] = []
    output_contrast: list[float] = []
    temporal_errors: list[float] = []
    flicker_samples: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    perceptual = lpips.LPIPS(net="alex").eval() if truth_capture and lpips is not None else None
    previous_input: np.ndarray | None = None
    previous_output: np.ndarray | None = None
    frame_index = 0
    if comparisons:
        comparisons.mkdir(parents=True, exist_ok=True)

    while frame_index < max_frames:
        input_ok, input_frame = input_capture.read()
        output_ok, output_frame = output_capture.read()
        if not input_ok or not output_ok:
            break
        truth_frame = None
        if truth_capture:
            truth_ok, truth_frame = truth_capture.read()
            if not truth_ok:
                break
        height, width = output_frame.shape[:2]
        input_fitted = cv2.resize(input_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        input_sharpness.append(sharpness(input_fitted))
        output_sharpness.append(sharpness(output_frame))
        input_contrast.append(local_contrast(input_fitted))
        output_contrast.append(local_contrast(output_frame))

        if previous_input is not None and previous_output is not None:
            flow = backward_flow(input_frame, previous_input)
            warped = warp(previous_output, flow, (width, height))
            difference = np.abs(
                output_frame.astype(np.float32) / 255.0 - warped.astype(np.float32) / 255.0
            )
            temporal_errors.append(float(difference.mean()))
            flicker_samples.extend(np.quantile(difference, [0.5, 0.9, 0.95]).tolist())

        if truth_frame is not None:
            truth_frame = cv2.resize(truth_frame, (width, height), interpolation=cv2.INTER_AREA)
            psnr_values.append(peak_signal_noise_ratio(truth_frame, output_frame, data_range=255))
            ssim_values.append(
                structural_similarity(truth_frame, output_frame, channel_axis=2, data_range=255)
            )
            if perceptual is not None and torch is not None:
                output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                truth_rgb = cv2.cvtColor(truth_frame, cv2.COLOR_BGR2RGB)
                output_tensor = (
                    torch.from_numpy(output_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1
                )
                truth_tensor = (
                    torch.from_numpy(truth_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1
                )
                with torch.inference_mode():
                    lpips_values.append(float(perceptual(output_tensor, truth_tensor)))

        if comparisons and frame_index in {0, 12, 24, 48, 96}:
            contact = np.concatenate((input_fitted, output_frame), axis=1)
            cv2.imwrite(str(comparisons / f"frame_{frame_index:06d}.jpg"), contact)
        previous_input = input_frame
        previous_output = output_frame
        frame_index += 1

    input_capture.release()
    output_capture.release()
    if truth_capture:
        truth_capture.release()
    if not frame_index:
        raise RuntimeError("No aligned input/output frames were decoded")

    result: dict[str, object] = {
        "frames": frame_index,
        "input_sharpness_mean": float(np.mean(input_sharpness)),
        "output_sharpness_mean": float(np.mean(output_sharpness)),
        "sharpness_gain": float(np.mean(output_sharpness) / max(np.mean(input_sharpness), 1e-8)),
        "input_local_contrast_mean": float(np.mean(input_contrast)),
        "output_local_contrast_mean": float(np.mean(output_contrast)),
        "local_contrast_gain": float(np.mean(output_contrast) / max(np.mean(input_contrast), 1e-8)),
        "temporal_warp_error_mean": float(np.mean(temporal_errors)) if temporal_errors else None,
        "flicker_p95": float(np.quantile(flicker_samples, 0.95)) if flicker_samples else None,
        "psnr_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
        "lpips_mean": float(np.mean(lpips_values)) if lpips_values else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--ground-truth-video", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--max-frames", type=int, default=240)
    args = parser.parse_args()
    report = {
        "candidate": evaluate(
            args.input_video,
            args.output_video,
            args.ground_truth_video,
            args.max_frames,
            args.comparisons,
        ),
        "visual_acceptance_reference": reference_metrics(args.reference, args.reference_spec),
        "manual_review_required": [
            "facial identity and structure",
            "natural skin without waxiness",
            "hair and fine surface detail",
            "halos, invented textures, and excessive grain",
            "frame-to-frame stability during motion",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
