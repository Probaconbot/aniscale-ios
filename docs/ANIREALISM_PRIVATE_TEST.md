# AniRealism private CDA-VSR test

AniRealism is an evaluation-only integration of the recurrent CDA-VSR model.
It is deliberately kept out of AniScale's public releases while its upstream
licensing remains unspecified.

## Reproducible inputs

- CDA-VSR repository: `https://github.com/sspBIT/CDA-VSR`
- CDA-VSR revision: `5707d997759996f19521c3beaddfb3d1ea965d44`
- Architecture SHA-256: `0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b`
- Checkpoint SHA-256: `afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e`
- Audited ONNX adapter revision: `03acb0f8f8e0f1c8219a8ab0e842a0efcc74b18c`

The GitHub workflow downloads and verifies these inputs, converts an
initializer graph and a recurrent graph to dynamic FP32 ONNX, and bundles them
only into the private draft test files. Model binaries are not committed.

## Runtime behavior

- The initializer produces the first 4x restored frame and both recurrent
  state banks.
- Each later frame reuses those states with decoded-frame motion and residual
  estimates.
- State resets on scene cuts and periodically to limit drift.
- Output video keeps the existing AniScale streaming encoder and audio-copy
  path.

The first test uses decoded-frame priors rather than CDA-VSR's original codec
bitstream priors. It therefore validates mobile recurrent inference and may not
match the paper's full reference quality.

## Distribution restriction

Private evaluation only. Do not publish or redistribute the generated
AniRealism binaries unless the upstream source and checkpoint licensing is
clarified.
