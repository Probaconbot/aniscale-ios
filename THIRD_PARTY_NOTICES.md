# Third-party notices

AniScale uses the Real-ESRGAN `RealESRGAN_x4plus_anime_6B` model for local
anime and illustration super-resolution.

- Project: https://github.com/xinntao/Real-ESRGAN
- Authors: Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan
- Model license: BSD 3-Clause
- Model release: v0.2.2.4

The model is converted to Apple's Core ML format during the private iOS build.
Images are processed on the device and are not sent to a server.

## AniUltraScale training and architecture references

AniUltraScale includes original AniScale implementation code informed by the following
permissively licensed projects:

- NanoVSR — https://github.com/filippawlicki/nanovsr — MIT License
- FANI — https://github.com/kyrie2to11/FANI — MIT License
- RealBasicVSR — https://github.com/ckkelvinchan/RealBasicVSR — Apache License 2.0

PiSA-SR — https://github.com/csslc/PiSA-SR — is cited for its published adjustable
pixel-fidelity/semantic-detail concept. No PiSA-SR source or diffusion weights are copied into
AniUltraScale; the repository README states Apache 2.0, but the repository currently has no
`LICENSE` file that can be packaged with redistributed code.
