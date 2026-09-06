# Connection validation — 2026-09-06

Private deployment: `Lushus6pg/aniscale-video` (ZeroGPU Gradio).

The **same Dart CloudVideoService used by Android/iOS** completed a real private
upload → queued GPU inference → SSE progress → MP4 download test. The model was
the SHA-256-verified official AnimeSR_v2 checkpoint, FP16, with recurrent state.

- GPU reported by PyTorch: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb.
- Synthetic input: 64×48, 6 frames, 30000/1001 FPS, AAC tone.
- Result: 128×96, 30000/1001 FPS, 0.2002 seconds.
- Original/output AAC packet SHA-256 hashes match exactly.
- Server processing time for this **tiny transport test only**: 0.388 seconds.
- Six Python tests cover streaming, FPS/audio, recurrent state, cut detection,
  channel-major pixel-unshuffle, cancellation cleanup and scale validation.
- Five Dart transport tests cover success, private authentication failure,
  quota error/cancellation, unsafe download origins and invalid MP4 cleanup.

This proves the connection and real GPU execution, **not** full-size throughput,
perceptual quality, temporal stability on difficult footage, or iPhone thermal
performance. Actual Android/iPhone installation testing remains necessary.

The remote integration test is opt-in via `ANISCALE_HF_TOKEN` and
`ANISCALE_SMOKE_VIDEO` runtime environment variables; it is skipped in public CI.
Never put a token into source, a dart-define, or a release asset.

## 1.15.2 connection hotfix

- Directly register the decorated GPU function so Gradio does not wrap an
  undecorated parent with the default reservation. Convert the wall-time estimate
  to the pinned Spaces runtime's shared-GPU reservation units.
- Accept native 1080p input; retain the 10-second, 360-frame and 100 MB limits.
- Use the full Gradio queue protocol to preserve allocation/quota error details.
- Real GPU test through the app's Dart service: two synthetic 1920×1080 frames
  produced 3840×2160 HEVC at 30000/1001 FPS with matching AAC packet SHA-256.
  This is a short functional test, not a sustained throughput or quality benchmark.
- Eight Python pipeline tests and six Dart transport tests cover the updated path.
