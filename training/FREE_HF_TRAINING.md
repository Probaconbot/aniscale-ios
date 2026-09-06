# Zero-cost Hugging Face training

AniScale can accumulate private experimental fine-tuning without paid hardware by
running short, resumable sessions inside the existing free ZeroGPU quota. Free
accounts currently receive five GPU minutes per 24-hour quota window. A session
stops early and writes `latest.pth`; the next session must resume that exact file.

Two checkpoints are maintained independently:

- `anime/latest.pth`: starts from official AnimeSR_v2 and uses licensed 2D animation.
- `live-action/latest.pth`: starts from the same verified recurrent backbone for the
  first bootstrap experiment and uses paired live-action data. It is never served
  under the anime model name.

The user-provided football input/output stay in the evaluation set. The aligned
720p/1440p car comparison can bootstrap a private live-action candidate through
`prepare_topaz_pair.py`. A broader checkpoint still requires diverse licensed HR
sequences; a one-scene checkpoint is expected to overfit and is not a release model.

Example free session:

```bash
python training/train_animesr_free.py \
  --data training/data/live-action-bootstrap \
  --output training/runs/live-action-free \
  --domain live-action \
  --session-seconds 150

python training/train_animesr_free.py \
  --data training/data/live-action-bootstrap \
  --output training/runs/live-action-free \
  --domain live-action \
  --resume training/runs/live-action-free/latest.pth \
  --session-seconds 150
```

Every candidate remains private until it beats the starting checkpoint on held-out
spatial and temporal comparisons. Free sessions may take many quota windows to
produce a useful model; the process has a hard zero-dollar budget.
