# AniScale Clean

AniScale Clean is the restoration-first video option for footage affected by
moire/scanlines, color casts, chroma noise, blur, and repeated compression. The
shipping mobile path is intentionally bounded:

```text
decoded frame
  -> color-cast correction
  -> scanline/pattern suppression
  -> chroma/noise cleanup
  -> controlled local contrast
  -> controlled Lanczos enlargement
  -> video encode + original-audio remux
```

The app does **not** claim to bundle the original desktop checkpoints. The
research implementations require CUDA-era PyTorch/MMCV or custom deformable
convolution operators that are not directly compatible with Core ML, Metal,
TFLite, or ncnn:

- [VideoDemoireing](https://github.com/CVMI-Lab/VideoDemoireing) — Apache-2.0.
- [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) — Apache-2.0.

They are suitable teacher/reference models for a later temporally aware mobile
student. A trained student must be benchmarked on patterned footage and real
devices before replacing the deterministic cleanup stage.

The deterministic stage intentionally bypasses Fusion. Frame-generative SR can
interpret residual display lines as texture and strengthen them, which is the
opposite of the cleanup mode's purpose.
