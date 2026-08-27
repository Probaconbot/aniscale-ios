# AniScale

AniScale is a private, offline-first image and video enhancer for iPhone.

## Current build

- Stable black-and-white Flutter interface without native platform-view overlays during transitions
- Image picker for PNG, JPG, and WebP
- Local 2× and 4× Real-ESRGAN Anime 6B image upscaling through Core ML with automatic
  memory-safe fitting for large camera images
- Local 2× and 4× Real-ESRGAN video enhancement with progress, cancellation, original audio, and
  automatic encoder-safe 4K fitting for oversized 4× outputs, plus Efficient and Maximum modes
- Processing, before/after comparison, local history, settings, save, and share flows
- Local upscaling never uploads media; no account or watermark
- Optional Groq vision planning: attach an image and describe the result, then review a structured
  cleanup/detail/color recipe before the local engine applies it
- Users enter their own Groq key and it remains in memory only for the current app session; prompts
  and a reduced preview are sent to Groq only after the user taps send

The iOS image and video engines run a fixed-shape Core ML conversion of the official Real-ESRGAN
anime 6B weights in overlapping tiles. Assistant recipes tune pre-model cleanup and faithful
reconstruction without pretending that the language model edits pixels. Video frames are decoded
with AVFoundation, cleaned by the neural model, re-encoded locally, and muxed with their original
audio. Oversized results are fitted into an encoder-safe 4K canvas.

## Run

Open this folder in VS Code and use the Flutter SDK installed at
`C:/Users/User/Documents/Codex/tools/flutter` for Dart analysis and UI development on Windows.

The GitHub Actions workflow compiles the unsigned iOS app on a hosted macOS runner and packages an
IPA. Installation still requires the user to sign the IPA with their own Apple ID/certificate.
