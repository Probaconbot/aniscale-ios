# AniUltraScale

AniUltraScale is AniScale's native **2x temporal video-restoration model**. It is a separate model family; it does not apply an image model independently to every frame.

## Architecture

```text
five LR frames
  -> iterative real-video cleaning
  -> shared feature encoder
  -> gated previous/current/next frame interaction
  -> forward + backward recurrent propagation
  -> temporal structure fusion
  -> base reconstruction
  -> fidelity residual + adjustable detail residual
  -> center 2x frame
```

The app modes use one checkpoint and two runtime controls:

- **Subtle:** fidelity 1.00, detail 0.35
- **Detailed:** fidelity 1.00, detail 0.82
- **Creative:** fidelity 0.86, detail 1.20

Creative v1 is still bounded restoration, not diffusion. The detail head is trained against the real target's high-frequency residual, so increasing it does not silently authorize unrelated faces or textures.

## Licensed foundations

- RealBasicVSR provides the cleaning-before-propagation and reconstruction-first training foundation (Apache-2.0).
- NanoVSR informs the lightweight recurrent propagation and deploy-time convolution reparameterisation (MIT).
- FANI provides the mobile inter-frame interaction and deployment reference (MIT).
- The independently implemented fidelity/detail controls use PiSA-SR's published adjustable pixel/semantic idea. AniScale does not embed PiSA's Stable Diffusion network or copy its code into the mobile student.

See `THIRD_PARTY_NOTICES.md` for links and attribution.

## Prepare genuine 2x pairs

Use only video that is licensed for model training. Keep productions and identities separated between training and validation.

Build a storage-capped, attributed source collection from the curated Blender films and
licence-filtered Wikimedia Commons categories:

```bash
python training/acquire_open_media.py \
  --output /gpu-storage/aniscale-open-media \
  --max-gb 100 \
  --commons-per-category 100 \
  --acknowledge-licenses

python training/build_still_motion_clips.py \
  --acquired /gpu-storage/aniscale-open-media \
  --output /gpu-storage/aniscale-open-media/still-motion
```

`attribution_manifest.json` is mandatory. Unknown, NonCommercial, NoDerivatives, random
YouTube/anime, and Vimeo API media are rejected. The acquisition tool also requires 2 GB of free
space beyond its explicit budget, so it cannot silently fill a workstation drive.

```bash
python training/prepare_video_dataset.py \
  --source training/data/source \
  --output training/data/aniultrascale \
  --scale 2 \
  --seed 20260828
```

The pipeline applies blur, temporally mixed motion blur, sensor-like noise, resampling, chroma loss, H.264/H.265 compression, low bitrates, ringing, repeated encode generations, social-media compression, and smooth exposure/saturation drift. It does not create bicubic-only pairs.

AniUltraScale is not trained to reproduce Real-ESRGAN's soft/smoothed look. Edge, FFT-frequency,
local-contrast, Laplacian-pyramid, and detail-residual supervision explicitly reward fine texture
and crisp boundaries. FAST uses a small perceptual term and QUALITY uses a stronger one. GAN remains
off initially because temporal consistency and correct structure must be learned before any
adversarial texture stage; sharpness still has to come from the real HR target rather than a generic
post-sharpening filter.

## Train

FAST is the practical phone model; QUALITY has more channels and recurrent/reconstruction blocks.

```bash
python training/train_aniultrascale.py \
  --config training/configs/aniultrascale_fast.json \
  --data training/data/aniultrascale \
  --output training/runs/aniultrascale-fast

python training/train_aniultrascale.py \
  --config training/configs/aniultrascale_quality.json \
  --data training/data/aniultrascale \
  --output training/runs/aniultrascale-quality
```

The objective combines low-resolution cleaning supervision, Charbonnier reconstruction, Sobel edge fidelity, high-frequency FFT,
local-contrast and Laplacian-pyramid supervision, optical-flow-aligned temporal residuals,
fidelity-head supervision, detail-residual supervision, and perceptual structure matching.

## Export

The exporter refuses random or architecture-only weights. A checkpoint must carry the trainer's `trained` marker and the SHA-256 fingerprint of its dataset manifest.

```bash
python training/export_aniultrascale.py \
  --checkpoint training/runs/aniultrascale-fast/best.pth \
  --output training/exports/aniultrascale-fast \
  --onnx --coreml
```

- iOS uses the FP16 Core ML package with a fixed five-frame window.
- Android converts the fixed ONNX graph to ncnn FP16/Vulkan during the signed release build.
- Both targets must use the same checkpoint manifest and report its hash.

## Release gate

Do not publish AniUltraScale until both FAST and QUALITY have:

1. a held-out validation report including PSNR, SSIM, LPIPS, temporal warp error, and flicker;
2. manual review on the permanent Face, Environment, and Hard-detail clips;
3. real-device FPS, seconds per video minute, memory, CPU/GPU/ANE/NPU use, and thermal results;
4. a matching checkpoint/export SHA-256 manifest;
5. working audio remuxing, cancel, progress, save/share, and history on Android and iOS.
