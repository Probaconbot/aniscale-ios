# AniScale

AniScale is a private, offline-first image and video enhancer for iPhone and Android.

## Current build

- Stable black-and-white Flutter interface without native platform-view overlays during transitions
- Custom AniScale launcher icon and in-app logo on iOS and Android
- Image picker for PNG, JPG, and WebP
- Local 2× and 4× AniScale Fusion image upscaling through Core ML with automatic
  memory-safe fitting for large camera images
- Local 2× and 4× video enhancement with AniScale Fusion for anime/stylized 3D, the separate heavy
  AniScale Render model for general 3D, and compact AniScale Turbo for lower heat, with progress,
  cancellation, original audio, automatic encoder-safe 4K fitting, and Efficient and Maximum modes
- Processing, before/after comparison, local history, settings, save, and share flows
- Local upscaling never uploads media; no account or watermark
- Optional Groq vision planning: attach an image and describe the result, then review a structured
  cleanup/detail/color recipe before the local engine applies it
- Persistent output format, tile size, performance, metadata, history, and reduced-motion settings
- Runtime About version sourced from the installed app package instead of hard-coded UI text
- Users enter their own Groq key and it remains in memory only for the current app session; prompts
  and a reduced preview are sent to Groq only after the user taps send

The iOS engines run fixed-shape Core ML conversions of the official Real-ESRGAN anime 6B,
general x4plus, and AnimeVideo-v3 weights in overlapping tiles. Downloads are checksum-verified during the
build. Assistant recipes tune pre-model cleanup and faithful reconstruction without pretending
that the language model edits pixels. Video frames are decoded with AVFoundation, cleaned by the
selected neural model, re-encoded locally, and muxed with their original audio. Oversized results
are fitted into an encoder-safe 4K canvas.

Android uses ncnn with Vulkan acceleration and three bundled neural video models: AniScale Fusion
for anime and stylized 3D, AniScale Render for general 3D, and compact AniScale Turbo for faster,
lower-heat processing. Frames are decoded and encoded with Android media APIs and original audio is
muxed into the local result. The same native Fusion runtime now handles Android image upscaling.

## Run

Open this folder in VS Code and use the Flutter SDK installed at
`C:/Users/User/Documents/Codex/tools/flutter` for Dart analysis and UI development on Windows.

The GitHub Actions workflow compiles the unsigned iOS app on a hosted macOS runner and packages an
IPA. Installation still requires the user to sign the IPA with their own Apple ID/certificate.
