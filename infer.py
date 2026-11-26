"""Minimal inference: wav in -> predicted stem out, chunked so long files fit in memory.

    python infer.py song.wav --out vocals.wav --weights bmf-synthetic.pt
    python infer.py                              # no file: runs on synthetic audio and writes stem.wav
"""
import argparse
import wave

import numpy as np
import torch

from bmf import BandMixFormerPKV


def read_wav(path):
    with wave.open(str(path), 'rb') as f:
        channels, width, rate, frames = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        raw = f.readframes(frames)
    if width not in (2, 4):
        raise SystemExit(f'only 16/32-bit pcm wav is supported, got {8 * width}-bit')
    dtype, scale = (np.int16, 32768.0) if width == 2 else (np.int32, 2147483648.0)
    audio = np.frombuffer(raw, dtype=dtype).reshape(-1, channels).T.astype(np.float32) / scale
    return torch.from_numpy(audio.copy()), rate


def write_wav(path, audio, rate):
    x = (audio.clamp(-1, 1).numpy().T * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(x.shape[1])
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(x.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio', nargs='?', help='input wav; omit to run on synthetic audio')
    ap.add_argument('--out', default='stem.wav')
    ap.add_argument('--weights')
    ap.add_argument('--chunk', type=float, default=3.0, help='seconds per forward pass')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = BandMixFormerPKV().to(device).eval()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))

    if args.audio:
        audio, rate = read_wav(args.audio)
    else:
        rate, audio = 44100, torch.randn(2, 5 * 44100) * 0.2
    audio = audio.to(device)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    span = int(rate * args.chunk)
    with torch.no_grad():
        out = torch.cat([model.separate(audio[:, i:i + span].unsqueeze(0))[0] for i in range(0, audio.shape[-1], span)], -1)
    write_wav(args.out, out.cpu(), rate)
    print('wrote', args.out, tuple(out.shape), f'{rate} Hz', f'{audio.shape[-1] / rate:.1f}s in')


if __name__ == '__main__':
    main()
