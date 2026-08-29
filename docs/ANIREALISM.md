# AniRealism integration boundary

AniRealism is designed around CDA-VSR's initializer/recurrent split, decoded
motion and residual priors, persistent temporal state, scene-cut reset, and
native 4× output. Android will use ONNX Runtime Mobile; iOS will use a Core ML
conversion once redistribution is permitted.

The public CDA-VSR repository currently names `LICENSE.txt` in its README but
does not contain that file. Its only `NOTICE` attributes an upstream TMP
component and does not grant a licence for CDA-VSR's own source or `best.pth`
checkpoint. Therefore AniScale does not bundle, rename, or redistribute those
files in a public APK or IPA.

Shipping can proceed when the CDA-VSR authors publish an applicable licence or
provide written redistribution permission for the source and checkpoint. The
model must then pass recurrent ONNX/Core ML parity, decoded-prior validation,
memory/thermal testing, and physical-device benchmarks before the AniRealism
selector is enabled.
