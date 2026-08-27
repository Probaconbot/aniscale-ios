# AniScale

AniScale is a private, offline-first image and video enhancer for iPhone.

## Current build

- Native Apple Liquid Glass (`UIGlassEffect`) for compact controls and navigation on iOS 26,
  with a compatible material fallback on earlier iOS releases
- Image picker for PNG, JPG, and WebP
- Local 2× and 4× Real-ESRGAN Anime 6B image upscaling through Core ML
- Local 2× and 4× GPU video enhancement with progress, cancellation, original audio, and
  automatic encoder-safe 4K fitting for oversized 4× outputs
- Editor controls for scale, content style, noise, sharpness, and detail
- Processing, before/after comparison, local history, settings, save, and share flows
- No uploads, account, watermark, or cloud API

The iOS image engine runs a fixed-shape Core ML conversion of the official Real-ESRGAN anime 6B
weights in overlapping tiles. The video engine currently uses Core Image Lanczos on the iPhone GPU
and automatically fits oversized results into an encoder-safe 4K canvas. Video enhancement is
local, but is not yet an AI frame model.

## Run

Open this folder in VS Code and use the Flutter SDK installed at
`C:/Users/User/Documents/Codex/tools/flutter` for Dart analysis and UI development on Windows.

The GitHub Actions workflow compiles the unsigned iOS app on a hosted macOS runner and packages an
IPA. Installation still requires the user to sign the IPA with their own Apple ID/certificate.
