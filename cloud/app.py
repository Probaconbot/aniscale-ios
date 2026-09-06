from pathlib import Path
import tempfile
import threading
import math
from datetime import timedelta

import spaces
import gradio as gr
import torch

from animesr import load_model
from free_training import run_training_session
from pipeline import process_video, gpu_duration, remove_expired_outputs
from spaces.zero.client import get_duration_seconds

torch.set_num_threads(2)
cache = Path(tempfile.gettempdir()) / 'aniscale-models'
outputs = Path(tempfile.gettempdir()) / 'aniscale-results'
outputs.mkdir(exist_ok=True)
# ZeroGPU emulates CUDA initialization outside the leased GPU function.
model = load_model(cache).half().to('cuda')


def sweep_outputs():
    timer = threading.Event()
    while not timer.wait(600):
        try:
            remove_expired_outputs(outputs)
        except OSError:
            pass  # A concurrent cleanup or worker shutdown may remove a file.


threading.Thread(target=sweep_outputs, daemon=True).start()


def allocation_duration(video, scale, detail, codec):
    # spaces==0.51.3 scales `large` allocation units by the host's duration
    # factor. Convert our wall-time estimate to those units instead of asking
    # for 3x the intended budget on the current shared Blackwell hardware.
    factor = max(1, get_duration_seconds(timedelta(seconds=1), 'large'))
    return max(1, math.ceil(gpu_duration(video, scale, detail, codec) / factor))


@spaces.GPU(duration=allocation_duration, size='large')
def upscale(video, scale, detail, codec):
    try:
        yield from process_video(video, int(scale), detail, codec, model, outputs)
    except (ValueError, RuntimeError) as error:
        yield None, {'error': str(error)[:400], 'stage': 'Failed', 'progress': 0}
    except Exception:
        # Do not expose private server paths or uploaded content in client errors.
        yield None, {'error': 'Video decoding or audio remux failed. Try an SDR MP4 with AAC audio.', 'stage': 'Failed', 'progress': 0}


@spaces.GPU(duration=20, size='large')
def train_free(domain, resume):
    # Keep each resumable run short enough to fit the free daily GPU allowance.
    # spaces==0.51.3 currently maps 20 allocation units to about 60 seconds.
    return run_training_session(domain, resume)


with gr.Blocks(title='AniScale private GPU', delete_cache=(600, 3600)) as demo:
    gr.Markdown('# AniUltraAnime · private GPU\nOfficial AnimeSR_v2. Upload only when ready. Free GPU: clips ≤10s, ≤1080p, ≤100 MB. Files expire within about one hour. Keep this Space private.')
    video = gr.File(label='Anime video', file_types=['video'], type='filepath')
    scale = gr.Radio([2, 4], value=2, label='Scale')
    detail = gr.Radio(['natural', 'detailed', 'sharp'], value='natural', label='Detail')
    codec = gr.Radio(['hevc', 'h264'], value='h264', label='Codec')
    start = gr.Button('Upscale on Hugging Face GPU')
    stop = gr.Button('Cancel')
    result = gr.File(label='Restored video')
    status = gr.JSON(label='Processing status')
    # Register the decorated function DIRECTLY. Wrapping it in an undecorated
    # parent makes Spaces auto-wrap the parent with its default 180s allocation.
    # Successful worker results live until the bounded expiry sweep because
    # ZeroGPU prefetching can finish before Gradio caches the final file.
    job = start.click(upscale, [video, scale, detail, codec], [result, status], api_name='upscale', concurrency_limit=1)
    stop.click(fn=None, cancels=[job], api_name=False)
    with gr.Accordion('Owner: free resumable training', open=False):
        gr.Markdown('Private experimental checkpoints only. Each run uses the free daily GPU quota and stops before the allocation ends.')
        train_domain = gr.Radio(['anime', 'live-action'], value='live-action', label='Checkpoint domain')
        train_resume = gr.File(label='Previous latest.pth (leave empty for the first session)', type='filepath')
        train_start = gr.Button('Run one free training session')
        train_checkpoint = gr.File(label='Download and keep for the next session')
        train_status = gr.JSON(label='Training status')
        train_start.click(train_free, [train_domain, train_resume], [train_checkpoint, train_status], api_name='train_free', concurrency_limit=1)

demo.queue(max_size=4).launch(show_error=False, max_file_size='100mb')
