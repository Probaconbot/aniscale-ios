# SuperUltra model notices

SuperUltra uses the Swift Parameter-free Attention Network (SPAN) architecture
and the official `spanx2_ch48` / `spanx4_ch48` checkpoints by Cheng Wan,
Hongyuan Yu, Zhiqi Li, Yihang Chen, Yajun Zou, Yuqing Liu, Xuanwu Yin, and
Kunlong Zuo.

- Project: https://github.com/hongyuanyu/SPAN
- Paper: *Swift Parameter-free Attention Network for Efficient Super-Resolution*
- Licence: Apache License 2.0

The Android runtime uses Tencent ncnn with Vulkan acceleration:

- Project: https://github.com/Tencent/ncnn
- Licence: BSD 3-Clause

The 2× anime checkpoint is `2xHFA2kSPAN` by Philip Hofmann (Phips), licensed
under CC BY 4.0 and distributed with attribution:
https://huggingface.co/Phips/2xHFA2kSPAN

AniScale processes selected media locally. These model files are only used for
on-device inference and are not uploaded by the app.
