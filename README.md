# AniScale

AniScale is a private, offline-first image upscaler for anime artwork and illustrations.

## Current build

- Premium glassmorphic Flutter interface based on the AniScale Figma Make design
- Image picker for PNG, JPG, and WebP
- Working local 2× and 4× image enlargement
- Editor controls for scale, content style, noise, sharpness, and detail
- Processing, before/after comparison, local history, settings, save, and share flows
- iOS and Android project targets

The current local enlargement engine uses cubic resizing as a functional MVP. The next engine
milestone is replacing it with tiled Real-ESRGAN inference while retaining the existing UI and
privacy model.

## Run

Open this folder in VS Code, choose an Android device or emulator, and press `F5`. The workspace is
configured to use the Flutter SDK installed at `C:/Users/User/Documents/Codex/tools/flutter`.

iOS compilation and simulator testing require macOS with Xcode. The generated `ios/` project can be
opened on a Mac without recreating the Flutter app.
