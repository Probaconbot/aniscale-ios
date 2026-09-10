"""Verify exact live-action research weights survive mobile recurrence conversion."""
import argparse
import hashlib
import torch
from cloud.animesr import AnimeSR
from tools.convert_animesr_mobile import load_model
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
args=p.parse_args()
assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == '7a685c4a1b18ad4a3dfc961cf10b8948311d22c280fbd4014dec47ee7480a89b'
torch.set_num_threads(2)
torch.manual_seed(731)
checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
assert checkpoint['domain']=='live-action' and checkpoint['iteration']==731
reference=AnimeSR().eval()
reference.load_state_dict(checkpoint['params_ema'],strict=True)
mobile,metadata=load_model(args.checkpoint,'live-action')
assert metadata['domain']=='live-action'
state=torch.zeros(1,64,8,12); feedback=torch.zeros(1,3,32,48)
ms,mf=state.clone(),feedback.clone()
with torch.inference_mode():
    for _ in range(3):
        frames=torch.rand(1,9,8,12)
        feedback,state=reference(frames,feedback,state)
        mf,ms=mobile(frames,mf,ms)
        torch.testing.assert_close(mf,feedback,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(ms,state,rtol=1e-5,atol=1e-6)
try:
    load_model(args.checkpoint)
except RuntimeError:
    pass
else:
    raise AssertionError('Default anime export must reject live-action weights')
print('PASS: checkpoint hash, domain, iteration, three recurrent frames, domain guard')
