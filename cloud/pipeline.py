"""Bounded streaming AnimeSR video inference. No extracted PNG frame collection."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
import time

import av
import numpy as np
import torch
from torch.nn import functional as F


def inspect_video(path, scale):
    if scale not in (2, 4):
        raise ValueError('Select 2× or 4×.')
    if not path or not Path(path).is_file() or Path(path).stat().st_size > 100 * 1024 * 1024:
        raise ValueError('Select a video under 100 MB.')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], capture_output=True, text=True, timeout=15, check=True)
    data = json.loads(probe.stdout)
    stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
    if stream is None:
        raise ValueError('This file has no video stream.')
    w, h = stream['width'], stream['height']
    duration = float(stream.get('duration') or data['format'].get('duration') or 0)
    if not 0 < duration <= 10.05:
        raise ValueError('Free GPU mode currently accepts clips up to 10 seconds. Trim the video first.')
    if max(w, h) > 1280 or min(w, h) > 720:
        raise ValueError('Free GPU mode accepts inputs up to 1280×720 (or portrait equivalent).')
    if stream.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('HDR input is not supported by this SDR AnimeSR checkpoint. Export an SDR copy first.')
    rotations = [int(stream.get('tags', {}).get('rotate', 0))]
    rotations += [int(s.get('rotation', 0)) for s in stream.get('side_data_list', [])]
    if any(rotation % 360 for rotation in rotations):
        raise ValueError('Bake the video rotation into an exported copy before using this test service.')
    if stream.get('sample_aspect_ratio', '1:1') not in ('1:1', 'N/A', '0:1'):
        raise ValueError('Export a square-pixel copy before using this test service.')
    ratio = min(scale, 3840 / max(w, h), 2160 / min(w, h))
    ow, oh = max(2, int(w * ratio) // 2 * 2), max(2, int(h * ratio) // 2 * 2)
    return dict(originalWidth=w, originalHeight=h, outputWidth=ow, outputHeight=oh,
                durationSeconds=duration, engine='AniUltraAnime • Hugging Face GPU')


def scene_cut(a, b):
    return float(np.abs(a[::8, ::8].astype(np.float32) - b[::8, ::8]).mean()) > 45


def process_video(path, scale, detail, codec, model, output_root, device='cuda'):
    if detail not in ('natural', 'detailed', 'sharp') or codec not in ('hevc', 'h264'):
        raise ValueError('Unsupported detail or codec option.')
    meta = inspect_video(path, scale)
    started = time.monotonic()
    job_dir = Path(tempfile.mkdtemp(prefix='aniscale-', dir=output_root))
    encoded, finished = job_dir / 'frames.mp4', job_dir / 'result.mp4'
    success = False
    try:
        yield None, dict(progress=.03, stage='GPU allocated; decoding video', **meta)
        with av.open(path) as source, av.open(str(encoded), 'w') as target:
            stream = source.streams.video[0]
            rate = stream.average_rate or Fraction(30, 1)
            output = target.add_stream('libx265' if codec == 'hevc' else 'libx264', rate=rate)
            output.width, output.height = meta['outputWidth'], meta['outputHeight']
            output.pix_fmt = 'yuv420p'
            output.time_base = Fraction(1, 90000)
            output.codec_context.thread_count = 2
            output.options = {'crf': '17', 'preset': 'fast'}
            if codec == 'hevc':
                output.options['x265-params'] = 'pools=2:frame-threads=1:log-level=error'
                output.codec_context.codec_tag = 'hvc1'
            frames = iter(source.decode(stream))
            current_frame = next(frames, None)
            if current_frame is None:
                raise ValueError('No decodable video frames.')
            current = current_frame.to_ndarray(format='rgb24')
            previous = current
            following_frame = next(frames, None)
            following = following_frame.to_ndarray(format='rgb24') if following_frame else current
            w, h = meta['originalWidth'], meta['originalHeight']
            ph, pw = (h + 3) // 4 * 4, (w + 3) // 4 * 4
            dtype = torch.float16 if device == 'cuda' else torch.float32
            feedback = torch.zeros(1, 3, ph * 4, pw * 4, device=device, dtype=dtype)
            state = torch.zeros(1, 64, ph, pw, device=device, dtype=dtype)
            first_time = float(current_frame.pts * current_frame.time_base) if current_frame.pts is not None else 0
            last_pts = -1
            count = 0
            while True:
                if count >= 360 or time.monotonic() - started > 105:
                    raise ValueError('Free GPU job limit reached. Use a shorter clip; no partial result was saved.')
                if current.shape != (h, w, 3) or following.shape != current.shape:
                    raise ValueError('Mid-video resolution changes are not supported.')
                reset = count == 0 or scene_cut(previous, current)
                if reset:
                    feedback.zero_()
                    state.zero_()
                left = current if reset else previous
                right = current if scene_cut(current, following) else following
                with torch.inference_mode():
                    value = torch.from_numpy(np.concatenate((left, current, right), axis=2)).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype).div_(255)
                    value = F.pad(value, (0, pw - w, 0, ph - h), mode='replicate')
                    feedback, state = model(value, feedback, state)
                    image = feedback[:, :, :h * 4, :w * 4].float()
                    if image.shape[-2:] != (meta['outputHeight'], meta['outputWidth']):
                        image = F.interpolate(image, size=(meta['outputHeight'], meta['outputWidth']), mode='bicubic', align_corners=False, antialias=True)
                    if detail != 'natural':
                        low = F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
                        residual = (image - low).clamp(-.03, .03)
                        image = image + residual * (.2 if detail == 'detailed' else .4)
                    rgb = image.clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
                frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                timestamp = float(current_frame.pts * current_frame.time_base) - first_time if current_frame.pts is not None else count / float(rate)
                frame.pts = round(timestamp * 90000)
                if frame.pts <= last_pts:
                    raise ValueError('Video has non-monotonic presentation timestamps.')
                last_pts = frame.pts
                frame.time_base = Fraction(1, 90000)
                for packet in output.encode(frame):
                    target.mux(packet)
                count += 1
                if count % 5 == 0:
                    yield None, dict(progress=min(.90, .05 + .85 * timestamp / meta['durationSeconds']), stage=f'GPU restoring frame {count}', **meta)
                if following_frame is None:
                    break
                previous, current = current, following
                current_frame = following_frame
                following_frame = next(frames, None)
                following = following_frame.to_ndarray(format='rgb24') if following_frame else current
            for packet in output.encode():
                target.mux(packet)
        yield None, dict(progress=.94, stage='Preserving original audio', **meta)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(encoded), '-i', str(path), '-map', '0:v:0', '-map', '1:a?', '-c', 'copy', '-map_metadata', '1', '-metadata:s:v:0', 'rotate=0', '-movflags', '+faststart', str(finished)], capture_output=True, timeout=15, check=True)
        success = True
        yield str(finished), dict(progress=1., stage='Ready to download', frames=count, seconds=time.monotonic() - started,
                                 gpu=torch.cuda.get_device_name() if device == 'cuda' else 'CPU test only', **meta)
    finally:
        encoded.unlink(missing_ok=True)
        # ZeroGPU prefetches generator yields in a separate process. A successful
        # file must survive until the parent Gradio process has cached it.
        if not success:
            finished.unlink(missing_ok=True)
            job_dir.rmdir()


def managed_results(iterator, root):
    """Runs in the parent process, not in the ZeroGPU worker."""
    paths = []
    try:
        for result, status in iterator:
            if result:
                path = Path(result).resolve()
                if path.parent.parent != Path(root).resolve() or path.name != 'result.mp4':
                    raise RuntimeError('Unexpected output location.')
                paths.append(path)
            yield result, status
    finally:
        iterator.close()
        for path in paths:
            path.unlink(missing_ok=True)
            if path.parent.exists() and not any(path.parent.iterdir()):
                path.parent.rmdir()


def remove_expired_outputs(root):
    """Safety net for a worker killed between writing and handing off a result."""
    root = Path(root).resolve()
    cutoff = time.time() - 3600
    for directory in root.glob('aniscale-*'):
        if directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != root:
            continue
        for name in ('frames.mp4', 'result.mp4'):
            path = directory / name
            if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        if not any(directory.iterdir()) and directory.stat().st_mtime < cutoff:
            directory.rmdir()
