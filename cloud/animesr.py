"""AnimeSR_v2 recurrent inference; adapted from TencentARC/AnimeSR (Apache-2.0).
Copyright (C) 2022 THL A29 Limited. See LICENSE-AnimeSR.
No BasicSR registration/training dependencies are required by this wrapper.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_SHA256 = 'd0f29c8966b53718828bd424bbdc306e7ff0cbf6350beadaf8b5b2500b108548'
TRAINED_CHECKPOINT_SHA256 = '79a9754db6e86dc8213c7e5de20145a85c1ed275953403366a464b8c06e8e194'
FINE_TUNE_FORMAT = 'animesr-v2-finetune-v1'


class ResidualBlockNoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class MultiScaleCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_s1_first = nn.Sequential(nn.Conv2d(121, 64, 3, 1, 1), nn.LeakyReLU(.1, inplace=True))
        self.conv_s2_first = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.LeakyReLU(.1, inplace=True))
        self.conv_s4_first = nn.Sequential(nn.Conv2d(64, 64, 3, 2, 1), nn.LeakyReLU(.1, inplace=True))
        self.body_s1_first = nn.ModuleList([ResidualBlockNoBN() for _ in range(5)])
        self.body_s2_first = nn.ModuleList([ResidualBlockNoBN() for _ in range(3)])
        self.body_s4_first = nn.ModuleList([ResidualBlockNoBN() for _ in range(2)])
        self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.fusion = nn.Sequential(nn.Conv2d(192, 224, 3, 1, 1), nn.LeakyReLU(.1, inplace=True), nn.Conv2d(224, 112, 3, 1, 1))

    def forward(self, x):
        a = self.conv_s1_first(x)
        b = self.conv_s2_first(a)
        c = self.conv_s4_first(b)
        has_b = has_c = False
        for i in range(5):
            a = self.body_s1_first[i](a + (self.upsample_x2(b) if has_b else 0) + (self.upsample_x4(c) if has_c else 0))
            if i >= 2:
                b = self.body_s2_first[i - 2](b + (self.upsample_x2(c) if has_c else 0))
                has_b = True
            if i >= 3:
                c = self.body_s4_first[i - 3](c)
                has_c = True
        return self.fusion(torch.cat((a, self.upsample_x2(b), self.upsample_x4(c)), dim=1))


class AnimeSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent_cell = MultiScaleCell()
        self.lrelu = nn.LeakyReLU(.1)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, frames, feedback, state):
        # Channel-major ordering is essential for the official pretrained weights.
        value = torch.cat((frames, F.pixel_unshuffle(feedback, 4), state), dim=1)
        result = self.recurrent_cell(value)
        enhanced = self.pixel_shuffle(result[:, :48]) + F.interpolate(frames[:, 3:6], scale_factor=4, mode='bilinear', align_corners=False)
        return enhanced, self.lrelu(result[:, 48:])


def load_model(directory):
    import gdown
    trained = Path(__file__).resolve().parent / 'training_results' / 'anime-latest.pth'
    if trained.exists():
        if hashlib.sha256(trained.read_bytes()).hexdigest() != TRAINED_CHECKPOINT_SHA256:
            raise RuntimeError('AniScale anime checkpoint checksum mismatch.')
        loaded = torch.load(trained, map_location='cpu', weights_only=True)
        if loaded.get('format') != FINE_TUNE_FORMAT or loaded.get('domain') != 'anime':
            raise RuntimeError('AniScale anime checkpoint metadata is invalid.')
        if not isinstance(loaded.get('iteration'), int) or loaded['iteration'] < 208:
            raise RuntimeError('AniScale anime checkpoint is older than the validated build.')
        state_dict = loaded.get('params_ema')
        if not isinstance(state_dict, dict):
            raise RuntimeError('AniScale anime checkpoint is missing params_ema weights.')
        model = AnimeSR()
        model.load_state_dict(state_dict, strict=True)
        return model.eval()

    path = Path(directory) / 'AnimeSR_v2.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pending = path.with_suffix('.download')
        gdown.download(id='1E_qsFqlIre-fMUSSLADQboBssNIWSqmm', output=str(pending), quiet=False)
        if hashlib.sha256(pending.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            pending.unlink(missing_ok=True)
            raise RuntimeError('Official AnimeSR checkpoint checksum mismatch.')
        pending.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError('AnimeSR checkpoint checksum mismatch.')
    model = AnimeSR()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model.eval()
