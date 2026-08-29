# AniUltraAnime model notice

AniUltraAnime uses the official `AnimeSR_v2` recurrent video-super-resolution
architecture and pretrained checkpoint by Yanze Wu, Xintao Wang, Gen Li, and
Ying Shan of Tencent ARC Lab.

- Project: https://github.com/TencentARC/AnimeSR
- Paper: *AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos*
- Copyright: Copyright (C) 2022 THL A29 Limited, a Tencent company
- Licence: Apache License 2.0, subject to the third-party notices in the
  upstream `LICENSE` file

AniScale converts the recurrent cell to ONNX for Android and Core ML for iOS.
The app preserves the model's previous SR frame and 64-channel hidden state in
chronological order and resets both when it detects a scene cut.
