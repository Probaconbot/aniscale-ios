from pathlib import Path
import tempfile
import threading

import spaces
import gradio as gr
import torch

from animesr import load_model
from pipeline import process_video, managed_results, inspect_video, remove_expired_outputs

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


@spaces.GPU(duration=120)
def gpu_upscale(video, scale, detail, codec):
    try:
        yield from process_video(video, int(scale), detail, codec, model, outputs)
    except (ValueError, RuntimeError) as error:
        yield None, {'error': str(error)[:400], 'stage': 'Failed', 'progress': 0}
    except Exception:
        # Do not expose private server paths or uploaded content in client errors.
        yield None, {'error': 'Video decoding or audio remux failed. Try an SDR MP4 with AAC audio.', 'stage': 'Failed', 'progress': 0}


def upscale(video, scale, detail, codec):
    # Cleanup here, after Gradio consumes the output, not in the prefetched GPU
    # generator. Otherwise the downloadable result can disappear before copying.
    try:
        inspect_video(video, int(scale))  # Reject unsupported input before leasing GPU time.
    except Exception as error:
        message = str(error) if isinstance(error, ValueError) else 'Could not read this video. Try an SDR MP4.'
        yield None, {'error': message[:400], 'stage': 'Failed', 'progress': 0}
        return
    yield from managed_results(gpu_upscale(video, scale, detail, codec), outputs)


with gr.Blocks(title='AniScale private GPU', delete_cache=(600, 3600)) as demo:
    gr.Markdown('# AniUltraAnime · private GPU\nOfficial AnimeSR_v2. Upload only when ready. Free GPU: clips ≤10s, ≤720p, ≤100 MB. Files expire within about one hour. Keep this Space private.')
    video = gr.File(label='Anime video', file_types=['video'], type='filepath')
    scale = gr.Radio([2, 4], value=2, label='Scale')
    detail = gr.Radio(['natural', 'detailed', 'sharp'], value='natural', label='Detail')
    codec = gr.Radio(['hevc', 'h264'], value='h264', label='Codec')
    start = gr.Button('Upscale on Hugging Face GPU')
    stop = gr.Button('Cancel')
    result = gr.File(label='Restored video')
    status = gr.JSON(label='Processing status')
    job = start.click(upscale, [video, scale, detail, codec], [result, status], api_name='upscale', concurrency_limit=1)
    stop.click(fn=None, cancels=[job], api_name=False)

demo.queue(max_size=4).launch(show_error=False, max_file_size='100mb')
