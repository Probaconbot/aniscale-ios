# AniScale

AniScale is a private, offline-first image and video enhancer for iPhone and Android.

## Current build

- Shared premium black-and-white command-bar interface, responsive pill navigation, spacing, and controls on iOS and Android
- Custom AniScale launcher icon and in-app logo on iOS and Android
- Image picker for PNG, JPG, and WebP
- Local 2× and 4× AniScale Fusion image upscaling through Core ML with automatic
  memory-safe fitting for large camera images
- Local 2× and 4× video enhancement with AniScale Fusion for anime/stylized 3D, the separate heavy
  AniScale Render model for general 3D, and compact AniScale Turbo for lower heat, with progress,
  cancellation, original audio, automatic encoder-safe 4K fitting, and Efficient and Maximum modes
- SuperUltra offline SPAN video restoration on Android and iOS with 1.5×, 2×, 3×, and 4× output,
  Auto/Live Action/Anime content modes, Natural/Detailed/Sharp controls, and HEVC/H.264 export
- AniUltraAnime video-only restoration using the official AnimeSR_v2 recurrent checkpoint, three
  neighboring frames, persistent previous-output/hidden state, and automatic scene-cut reset
- Processing, before/after comparison, persistent image/video history, reopen, delete, playback, settings, save, and share flows
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
muxed into the local result. Android keeps only the selected model resident, uses FP16 Vulkan where
supported, performs frame-to-YUV conversion in native code, and chooses engine-specific working
resolutions while preserving an encoder-safe full export size. Fusion, Render, and Turbo are also
selectable for Android image upscaling.

SuperUltra is a separate efficient video engine built on SPAN rather than the Real-ESRGAN family.
Android runs the x2/x4 networks through ncnn Vulkan FP16; iOS runs fixed 256-pixel tile conversions
through Core ML FP16 with Metal/Neural Engine selection. It includes the official general SPAN x2
and x4 checkpoints plus the attributed HFA2k SPAN x2 anime checkpoint. Frames are processed in
memory without writing intermediate PNG files, overlapping tiles suppress seams, only the selected
model stays resident, and thermal backoff reduces sustained load. The 1.5× and 3× choices use one
native SPAN pass followed by a controlled high-quality downsample.

AniUltraAnime is a true recurrent video model rather than a renamed frame upscaler. Android runs a
dynamic-spatial ONNX conversion through ONNX Runtime Mobile. iOS runs the same recurrent cell as a
Core ML FP16 ML Program. Both implementations keep frames chronological, use previous/current/next
context, propagate the previous 4× result and 64-channel hidden state, and clear both at detected
scene cuts. The official network is natively 4×; the 2× choice performs one high-quality downsample
after inference, matching the upstream inference design.

[AniRealism](docs/ANIREALISM.md) is designed around CDA-VSR, but it is not enabled or bundled because
the current upstream repository does not contain the `LICENSE.txt` named by its README. AniScale will
not redistribute that public checkpoint until the authors publish applicable terms or give written
permission.

App launch includes the supplied 3.809-second voice and 2.617-second SFX: voice begins immediately,
SFX begins at 1.0 second, the logo zooms/fades, and the main interface opens at 3.0 seconds while the
voice finishes naturally. Liquid Glass surfaces use platform compositing and a touch-positioned
specular highlight on Android and iOS; no embedded HTML view is required.

## Video model training

The reproducible [existing-model fine-tuning pipeline](training/README.md) retains AniScale's existing Real-ESRGAN
RRDB and AnimeVideo-v3 SRVGG architectures. It includes real multi-generation codec degradation,
paired sequence loading, motion-aligned temporal loss, FAST/QUALITY configurations, perceptual and
temporal checkpoint evaluation, Core ML FP16 export, the supplied visual acceptance reference, and
a physical-iPhone benchmark schema. Fine-tuned weights and benchmark numbers are not claimed until
training data, a CUDA run, and an Instruments trace on an actual iPhone have completed.

[AniUltraScale](training/ANIULTRASCALE.md) is the next, separate video engine: a native 2x,
five-frame real-world VSR model with RealBasicVSR-style iterative input cleaning, FANI-style
gated neighboring-frame interaction, forward/backward recurrent propagation, and independent
PiSA-style fidelity/detail residual controls. FAST and QUALITY configurations, real-video
degradation, cleaning, temporal, frequency, and edge losses, and guarded Core ML/ONNX export are
versioned in the repository.
The older untrained AniUltraScale runtime is no longer included in the app. The research and training
materials remain versioned for reproducibility but cannot be selected as a user-facing engine.

## Run

Open this folder in VS Code and use the Flutter SDK installed at
`C:/Users/User/Documents/Codex/tools/flutter` for Dart analysis and UI development on Windows.

The GitHub Actions workflow compiles the unsigned iOS app on a hosted macOS runner and packages an
IPA. Installation still requires the user to sign the IPA with their own Apple ID/certificate.
