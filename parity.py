"""Parity anchor: load the upstream 50k checkpoint into this implementation and compare outputs.

The two implementations were written independently, so this is the only thing that proves the structure
here is the same function and not merely the same shape. Run it from `s2_h3/` with the upstream tree on
the path:

    S2_DEVICE=cpu ARM=H3 S2_NPU_SWIGLU=0 python parity.py /path/to/out/H3/ema-50000.pt

Pass = max |mask_minimal - mask_upstream| small in fp32 (the upstream path fuses a few elementwise ops
and runs bf16 on NPU, so expect ~1e-6, not 0).
"""
import os
import re
import sys

import torch

os.environ.setdefault('ARM', 'H3')
os.environ.setdefault('S2_DEVICE', 'cpu')
os.environ['S2_NPU_SWIGLU'] = '0'

from bmf import BandMixFormerPKV  # noqa: E402


def to_minimal(key):
    key = re.sub(r'^(module|model)\.', '', key)
    if key.endswith('rotary_embed.freqs'):
        return None                      # the RoPE table is recomputed, not loaded
    key = re.sub(r'^layers\.(\d+)\.1\.layers\.0\.0\.', r'freq.\1.attn.', key)
    key = re.sub(r'^layers\.(\d+)\.1\.layers\.0\.1\.net\.', r'freq.\1.ff.net.', key)
    key = re.sub(r'^convs\.(\d+)\.', r'time.\1.', key)
    key = re.sub(r'^mask_estimators\.0\.', 'mask_estimators.', key)
    return key


def main(path):
    from entry import build
    upstream = build('H3').eval()
    minimal = BandMixFormerPKV().eval()
    package = torch.load(path, map_location='cpu', weights_only=False)
    state = package.get('state_dict', package)
    upstream.load_state_dict(state, strict=True)      # both sides must hold the SAME weights, or this compares nothing

    mapped, skipped = {}, []
    for key, value in state.items():
        target = to_minimal(key)
        if target is None:
            continue
        if target not in minimal.state_dict():
            skipped.append((key, target))
            continue
        mapped[target] = value
    missing = [k for k in minimal.state_dict() if k not in mapped]
    print(f'loaded {len(mapped)} tensors, skipped {len(skipped)}, not loaded {len(missing)}')
    for key, target in skipped[:5]:
        print('  skipped', key, '->', target)
    for key in missing[:5]:
        print('  missing', key)
    leftover, unexpected = minimal.load_state_dict(mapped, strict=False)
    assert not unexpected, unexpected
    assert all(k.endswith('.freqs') for k in leftover), leftover   # the RoPE table is recomputed from theta

    torch.manual_seed(0)
    frames = 129
    stft_repr = torch.randn(1, 2050, frames, 2)
    with torch.no_grad():
        got = minimal(stft_repr)
        want = upstream._forward_mask_core(stft_repr)
    diff = (got - want).abs()
    rel = diff.max().item() / want.abs().max().item()
    print('shapes', tuple(got.shape), tuple(want.shape))
    print(f'PARITY max_abs={diff.max().item():.3e} rel={rel:.3e}')
    return 0 if rel < 1e-5 else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
