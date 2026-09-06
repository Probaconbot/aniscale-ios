---
title: AniScale Video
emoji: 🎞️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: '3.10'
app_file: app.py
pinned: false
short_description: Private AniScale AnimeSR video processing
---

# AniScale private GPU video service

Deploy these files at the root of a **private ZeroGPU Gradio Space**.
Uses the official pretrained AnimeSR_v2 recurrent model, not a renamed resize
filter. No training, generated-detail model, or local-phone inference fallback.

This first free-GPU connection accepts SDR, unrotated anime clips up to 10 seconds,
360 frames, 1920×1080 (portrait equivalent), and 100 MB. Longer/HDR inputs fail
explicitly, never silently truncate or downscale the neural input. Output fits
3840×2160, preserving aspect ratio. 2× uses native 4× inferehe API `upscale` accepts a Gradio FileData input, scale (2 or 4), dnce then downsamples.
Free GPU allocation and daily quotas are enforced by Hugging Face.

Tetail
(`natural`, `detailed`, `sharp`), and codec (`hevc`, `h264`). It yields
`[outputFileOrNull, statusObject]`; status includes progress, stage and final
dimensions/duration. Errors are returned in `status.error`. Inference is only
inside a dynamically sized 16–120-second ZeroGPU allocation. Cancellation stops the Gradio iterator;
an already-running GPU operation may finish before cancellation is observed.

The app sends a Hugging Face read token only to the fixed private Space origin.
Tokens are entered by the owner and kept in app memory, never bundled or logged.
Input/output cache files are removed after an hour (swept every 10 minutes);
intermediate video files are removed as soon as processing finishes or fails.
Keep the app open for upload, queue updates, and download.

## Attribution

`animesr.py` adapts TencentARC/AnimeSR's `MSRSWVSR` / multi-scale recurrent cell
and BasicSR's residual block. AnimeSR copyright (C) 2022 THL A29 Limited,
Apache-2.0. BasicSR copyright Xintao Wang, Apache-2.0.
Sources: https://github.com/TencentARC/AnimeSR and https://github.com/XPixelGroup/BasicSR.
See LICENSE-AnimeSR for upstream license and third-party terms. We retain the
official checkpoint and channel-major pixel-unshuffle ordering.

Local tests: `python -m unittest discover -s cloud -p 'test_*.py'`.
