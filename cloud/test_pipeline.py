import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from animesr import AnimeSR
from pipeline import inspect_video, process_video, scene_cut, managed_results


class RecordingModel:
    def __init__(self):
        self.states = []

    def __call__(self, frames, feedback, state):
        self.states.append(float(state.mean()))
        return F.interpolate(frames[:, 3:6], scale_factor=4), state + 1


class TemporalTests(unittest.TestCase):
    def test_pixel_unshuffle_is_channel_major(self):
        value = torch.arange(3 * 8 * 8).view(1, 3, 8, 8).float()
        official = value.view(1, 3, 2, 4, 2, 4).permute(0, 1, 3, 5, 2, 4).reshape(1, 48, 2, 2)
        self.assertTrue(torch.equal(F.pixel_unshuffle(value, 4), official))

    def test_recurrent_network_shapes(self):
        torch.set_num_threads(2)
        with torch.inference_mode():
            image, state = AnimeSR()(torch.zeros(1, 9, 8, 12), torch.zeros(1, 3, 32, 48), torch.zeros(1, 64, 8, 12))
        self.assertEqual(tuple(image.shape), (1, 3, 32, 48))
        self.assertEqual(tuple(state.shape), (1, 64, 8, 12))

    def test_scene_cut(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertTrue(scene_cut(black, black + 255))
        self.assertFalse(scene_cut(black, black + 2))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe required')
class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / 'input.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=30000/1001', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '0.2002', '-c:v', 'libx264', '-c:a', 'aac', str(self.input)], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def probe(self, path):
        return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams']

    def test_streaming_audio_fps_state_and_cleanup(self):
        model = RecordingModel()
        saved = self.root / 'download.mp4'
        stages = []
        for path, status in managed_results(process_video(str(self.input), 2, 'natural', 'h264', model, self.root, device='cpu'), self.root):
            stages.append(status)
            if path:
                shutil.copyfile(path, saved)
        streams = self.probe(saved)
        video = next(s for s in streams if s['codec_type'] == 'video')
        self.assertEqual((video['width'], video['height']), (128, 96))
        self.assertEqual(video['avg_frame_rate'], '30000/1001')
        self.assertTrue(any(s['codec_type'] == 'audio' for s in streams))
        self.assertEqual(model.states, [0, 1, 2, 3, 4, 5])
        self.assertEqual(stages[-1]['progress'], 1)
        self.assertFalse(list(self.root.glob('aniscale-*')))
        def audio_hash(path):
            return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-'])
        self.assertEqual(audio_hash(self.input), audio_hash(saved))

    def test_cancel_closes_temporary_output(self):
        generator = process_video(str(self.input), 4, 'natural', 'h264', RecordingModel(), self.root, device='cpu')
        next(generator)
        generator.close()
        self.assertFalse(list(self.root.glob('aniscale-*')))

    def test_scale_validation(self):
        with self.assertRaisesRegex(ValueError, '2× or 4×'):
            inspect_video(str(self.input), 3)


if __name__ == '__main__':
    unittest.main()
