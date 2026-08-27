# AniScale video fine-tuning

This directory fine-tunes the existing AniScale model family rather than replacing it:

- **FAST** starts from the official AnimeVideo-v3 SRVGG checkpoint.
- **QUALITY** starts from the existing six-block anime/stylized-3D RRDB checkpoint.

Both remain native 4x models and export to the same Core ML resource names already consumed by the iOS app. A 2x result is obtained from the faithful 4x reconstruction, as in the current engine.

## What is and is not complete

The degradation, sequence-training, temporal-loss, checkpoint-comparison, Core ML export, and physical-device benchmark tooling is versioned here. Actual fine-tuned weights require licensed high-resolution video sequences, a CUDA training machine, and a held-out validation set. The single supplied comparison screenshot is deliberately an **acceptance reference**, not silently treated as a dataset.

## Dataset

Place legally usable, high-quality source videos under `training/data/source`. Use varied faces, hair, fabric, armor, foliage, ground, debris, CGI, animation, camera motion, and low-light material. Split people and source productions between train and validation to prevent identity memorization.

Create paired sequences with real H.264/H.265 encode generations:

```bash
python training/prepare_video_dataset.py \
  --source training/data/source \
  --output training/data/paired \
  --scale 4 \
  --seed 20260827
```

This requires `ffmpeg` and `ffprobe`. It creates spatially aligned `hr/<clip>` and `lq/<clip>` PNG sequences plus a manifest. Codec generations, bitrate, blur, resampling, motion blur, noise, ringing, chroma loss, and social-media style re-encodes are randomized per clip. Training does not rely on bicubic-only pairs.

Expected layout:

```text
training/data/paired/
  train/hr/clip_name/000001.png
  train/lq/clip_name/000001.png
  validation/hr/clip_name/000001.png
  validation/lq/clip_name/000001.png
```

## Fine-tune

```bash
python training/train_video.py \
  --config training/configs/fast.json \
  --data training/data/paired \
  --initial ios/Runner/Models/realesr-animevideov3.pth \
  --output training/runs/fast

python training/train_video.py \
  --config training/configs/quality.json \
  --data training/data/paired \
  --initial ios/Runner/Models/RealESRGAN_x4plus_anime_6B.pth \
  --output training/runs/quality
```

The trainer uses Charbonnier reconstruction, edge fidelity, low-weight VGG perceptual loss, and motion-aligned temporal residual loss. Adjacent outputs are compared after optical-flow warping against the matching ground-truth temporal residual, so genuine movement is retained while shimmer and detail popping are penalized.

## Evaluate checkpoints

```bash
python training/evaluate_video.py \
  --input-video validation/lq/example.mp4 \
  --output-video candidate.mp4 \
  --reference training/references/soft480p_to_restored_acceptance.png \
  --reference-spec training/references/acceptance.json \
  --report training/reports/quality.json
```

Reports include PSNR/SSIM when paired ground truth is supplied, edge sharpness, local contrast, temporal warp error, flicker p95, and side-by-side acceptance frames. Manual identity/skin/halo review remains mandatory; a sharpness number alone cannot detect waxy skin or invented facial detail.

## Export for iOS

On macOS:

```bash
python training/export_coreml_variants.py \
  --fast training/runs/fast/best.pth \
  --quality training/runs/quality/best.pth
```

The exporter creates FP16 ML Program packages targeting iOS 15. FAST and QUALITY retain float computation; Apple documents that weight-only compression primarily reduces storage and does not automatically reduce compute, so it is not presented as a guaranteed speedup.

## Physical iPhone benchmark

Run a release build on each target iPhone with Low Power Mode off, battery above 50%, and the device cooled to nominal thermal state. Process the same 30-second 480p clip three times per mode, discarding the first specialization run. Record the JSON metrics returned by AniScale and profile the middle run with Xcode Instruments' Core AI template.

Required report fields:

- device and iOS version;
- processing FPS and seconds per source-video minute;
- mean/p50/p95 Core ML tile inference time;
- CPU, GPU, and Neural Engine utilization from Instruments;
- peak resident memory;
- initial/final thermal state;
- PSNR, SSIM, temporal warp error, flicker p95, and manual acceptance result.

No device number belongs in a release note until it has been captured on that physical model.
