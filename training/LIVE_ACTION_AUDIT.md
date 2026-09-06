# Live-action restoration audit — 2026-09-06

Status: preparation only; no training job started and no new weights deployed.

## Findings

- Cloud selection routes exclusively to AnimeSR_v2, an animation checkpoint.
  Football/realistic skin is outside that intended domain. The screenshot alone
  cannot establish whether distortion originated in the source or the upscaler.
- The cloud cell's multiscale update order, pixel shuffle/unshuffle and bilinear
  residual match the official AnimeSR architecture on source inspection. This
  is not a numerical checkpoint parity test.
- The streaming pipeline retains feedback/state, initializes them to zero and
  resets at detected cuts. It processes RGB input without neural-input shrinking.
- AniRealism/CDA-VSR is a separate local experimental path with decoded-frame
  priors, not the original compressed-bitstream priors. Evaluate that difference
  before attributing poor results to training.
- Existing train_video.py flattens sequences into independent frames and targets
  SRVGG/RRDB. It is NOT a trainer for the deployed AnimeSR or CDA-VSR; do not run
  it and present the result as a fine-tuned recurrent model.

## Required baseline

Obtain the user's original short video, output video and selected engine/settings.
Keep that clip held out from training. Compare original, bicubic, official model
inference and app inference with identical dimensions, timestamps and encoding.
Inspect face/hair/fabric crops and moving sequences, not sharpness alone. Measure
LPIPS/PSNR/SSIM only where aligned genuine HR targets exist; also inspect
occlusion-masked motion-warp error, temporal flicker and identity preservation.
Measure inference separately from encoding and upload time.

## Training gate

Only after the baseline identifies a model limitation:

1. Resolve live-action checkpoint/source training permissions and pin revisions.
2. Assemble licensed HR video sequences with provenance and sequence-level splits.
   Do not treat the user's melted result, random internet images or generated
   pores as clean targets.
3. Apply temporally correlated blur/noise, chroma degradation and actual video
   encoding degradation. Include sports motion, skin, hair, fabric and crowds.
4. Adapt the correct recurrent trainer with motion priors/state kept across
   sequences. Begin with reconstruction/gradient and occlusion-aware temporal
   losses; validate before adding perceptual loss. No GAN in the first baseline.
5. Benchmark a short pilot, estimate total GPU cost, and obtain a spending cap
   before launching dedicated Hugging Face compute. The existing quota-limited
   inference Space is not an approved sustained training resource.
6. Keep candidate checkpoints private and separate from serving weights. Deploy
   only after held-out quality, temporal stability and runtime comparisons pass.

No claim of Topaz equivalence or recovered ground-truth pores is justified yet.

Sources inspected:
- https://github.com/TencentARC/AnimeSR/blob/main/animesr/archs/vsr_arch.py
- https://github.com/sspBIT/CDA-VSR
- https://huggingface.co/docs/hub/spaces-zerogpu
