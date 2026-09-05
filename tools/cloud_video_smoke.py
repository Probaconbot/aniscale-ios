"""Owner-authorized remote smoke test; reads cached HF credentials, never prints them.
Uses only a generated 0.2-second test pattern with a tone, not private user media.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from gradio_client import Client, handle_file
from huggingface_hub import get_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', default='Lushus6pg/aniscale-video')
    parser.add_argument('--output', default='work/cloud-smoke')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / 'test-pattern.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=30000/1001', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '0.2002', '-c:v', 'libx264', '-c:a', 'aac', str(source)], check=True)
    client = Client(args.space, hf_token=get_token(), download_files=str(root))
    job = client.submit(handle_file(str(source)), 2, 'natural', 'h264', api_name='/upscale')
    for update in job:
        print(json.dumps(update[1]), flush=True)
    file, metadata = job.result()
    if metadata.get('error') or not file:
        raise RuntimeError(metadata.get('error', 'No output file'))
    final = root / 'gpu-result.mp4'
    shutil.copyfile(file, final)
    result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(final)]))
    video = next(s for s in result['streams'] if s['codec_type'] == 'video')
    assert (video['width'], video['height']) == (128, 96)
    assert video['avg_frame_rate'] == '30000/1001'
    def audio_hash(path):
        return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-'])
    assert audio_hash(source) == audio_hash(final)
    print('PASS: GPU model output, dimensions, 29.97 FPS and bit-identical audio. Result:', final)


if __name__ == '__main__':
    main()
