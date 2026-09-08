import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import ast

import numpy as np
import av
import torch
from torch.nn import functional as F

from animesr import AnimeSR
from pipeline import inspect_video, process_video, scene_cut, managed_results, gpu_duration, decode_rgb, video_color_spec


class RecordingModel:
    def __init__(self):
        self.states = []

    def __call__(self, frames, feedback, state):
        self.states.append(float(state.mean()))
        return F.interpolate(frames[:, 3:6], scale_factor=4), state + 1


class TemporalTests(unittest.TestCase):
    def test_small_job_requests_small_allocation(self):
        with patch('pipeline.inspect_video', return_value=dict(durationSeconds=.2, originalWidth=64, originalHeight=48)):
            self.assertEqual(gpu_duration('input', 2, 'natural', 'h264'), 16)

    def test_gradio_registers_the_explicitly_decorated_function(self):
        tree = ast.parse(Path(__file__).with_name('app.py').read_text())
        upscale = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'upscale')
        self.assertEqual(len(upscale.decorator_list), 1)
        decorator = upscale.decorator_list[0]
        self.assertEqual(decorator.func.attr, 'GPU')
        self.assertTrue(any(k.arg == 'duration' and isinstance(k.value, ast.Name) and k.value.id == 'allocation_duration' for k in decorator.keywords))
        click = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'start' and n.func.attr == 'click')
        self.assertEqual(click.args[0].id, 'upscale')

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

    def test_colour_tags_and_brightness_round_trip(self):
        # Two input matrices, including full-range HD, must survive RGB inference.
        for matrix, primaries, input_range in [('bt709', 'bt709', 'pc'), ('smpte170m', 'smpte170m', 'tv')]:
            with self.subTest(matrix=matrix, input_range=input_range):
                original = self.root / (matrix + '.mp4')
                subprocess.run([
                    'ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'color=c=0x8e6352:size=64x48:rate=24', '-t', '0.125',
                    '-vf', f'scale=in_range=tv:out_range={input_range}:out_color_matrix={matrix}',
                    '-c:v', 'libx264', '-crf', '1', '-colorspace', matrix,
                    '-color_primaries', primaries, '-color_trc', 'bt709',
                    '-color_range', input_range, str(original),
                ], check=True)
                result = None
                for result_path, _ in process_video(str(original), 2, 'natural', 'h264', RecordingModel(), self.root, device='cpu'):
                    if result_path:
                        result = result_path
                video = next(s for s in self.probe(result) if s['codec_type'] == 'video')
                self.assertEqual(video['color_space'], matrix)
                self.assertEqual(video['color_primaries'], primaries)
                self.assertEqual(video['color_transfer'], 'bt709')
                self.assertEqual(video['color_range'], 'tv')
                means = []
                for path in (original, result):
                    with av.open(str(path)) as source:
                        stream = source.streams.video[0]
                        rgb = decode_rgb(next(source.decode(stream)), video_color_spec(stream))
                        means.append(rgb.astype(float).mean(axis=(0, 1)))
                self.assertLess(float(np.max(np.abs(means[0] - means[1]))), 3.0)

    def test_scale_validation(self):
        with self.assertRaisesRegex(ValueError, '2× or 4×'):
            inspect_video(str(self.input), 3)


if __name__ == '__main__':
    unittest.main()
