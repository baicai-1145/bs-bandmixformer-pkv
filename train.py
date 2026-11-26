"""Minimal training loop: synthetic two-source mixtures, waveform + spectral L1, AdamW + StepLR.

No dataset, no framework. The point is that the model in bmf.py trains end to end on the loss shape the
real run uses (waveform plus a spectral term); swap `synth` for a real loader and nothing else changes.

    python train.py --steps 200                 # smoke, CPU
    python train.py --steps 5000 --batch 4 --device cuda
"""
import argparse
import time

import torch
from torch.nn import functional as F

from bmf import BandMixFormerPKV, count_params


def synth(batch, samples, device):
    """Two band-limited noise sources; the model is asked to pull source 0 out of their sum."""
    noise = torch.randn(2, batch * 2, samples, device=device)
    kernel = torch.randn(2, 1, 33, device=device) / 6      # one random FIR per source, groups=2 keeps them apart
    src = F.conv1d(noise, kernel, padding=16, groups=2).reshape(2, batch, 2, samples)
    src = src / src.std(dim=(2, 3), keepdim=True)
    return src.sum(0) * 0.5, src[0]


def spectral(a, b, n_fft=1024, hop=256):
    window = torch.hann_window(n_fft, device=a.device)
    mag = lambda x: torch.stft(x.reshape(-1, x.shape[-1]), n_fft, hop, n_fft, window=window, return_complex=True).abs()
    return F.l1_loss(mag(a), mag(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seconds', type=float, default=3.0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='bmf-synthetic.pt')
    args = ap.parse_args()

    device, samples = torch.device(args.device), int(44100 * args.seconds)
    model = BandMixFormerPKV().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.steps // 2), gamma=0.5)
    print('params', f'{count_params(model):,}', 'samples', samples, 'device', device, flush=True)

    started = time.time()
    for step in range(1, args.steps + 1):
        mixture, target = synth(args.batch, samples, device)
        pred = model.separate(mixture)
        loss = F.l1_loss(pred, target) + spectral(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step == 1 or step % 10 == 0:
            print(f'step {step:5d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - started:.1f}s', flush=True)

    torch.save(model.state_dict(), args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()
