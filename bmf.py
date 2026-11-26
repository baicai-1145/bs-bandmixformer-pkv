"""BS-BandMixFormer-PKV (code key H3) — a minimal, torch-only reference.

Reference implementation for reading, not for production. It reproduces the *structure* of
BS-RoFormer with the two audited modifications that make it 3.06x cheaper than the baseline:

  stacks 0-2  cross-band routing is content-independent (StaticBandMixer, no q/k, no RoPE)
              the FFN stays per-frame, because stacks 0-2 carry the residual (: -11 dB if pooled)
  stacks 3-5  dynamic per-band routing, but q/k come from 4-frame mean tokens shared by the block,
              v/out narrowed to 128, and the FFN runs once per 4-frame block (broadcast, not interp)
  time        MSC everywhere except stacks 0 and 4, which keep framewise-query attention with
              4-frame pooled K/V (the "PKV" of the name)

Grid is (batch, frame, band, dim) throughout the frequency branch; that is what makes block pooling
expressible at all. 20.27M params, 262.6 GFLOP per 8 s of audio, 9.103 dB SI-SDR at 50k steps.
Architecture note and every ablation behind these choices: BS-BandMixFormer-PKV.md upstream.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

# 62 bands over 1025 stft bins: 24*2 + 12*4 + 8*12 + 8*24 + 8*48 + 128 + 129
FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * (x.shape[-1] ** 0.5) * self.gamma


class Rotary(nn.Module):
    """Band-axis rotation, shared by every layer; the phase table is (1, n, 1, dim/2) so callers index positions."""
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.freqs = nn.Parameter(1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)), requires_grad=False)

    def table(self, n, device, dtype):
        phase = torch.arange(n, device=device, dtype=torch.float32)[:, None] * self.freqs[None]
        return phase.cos()[None, :, None].to(dtype), phase.sin()[None, :, None].to(dtype)


def apply_rotary(cos, sin, t):  # t: (..., n, dim)
    even = t[..., ::2] * cos - t[..., 1::2] * sin
    odd = t[..., 1::2] * cos + t[..., ::2] * sin
    return torch.cat((even, odd), dim=-1)


def block_mean(x, stride):
    """Mean of successive stride-frame blocks along dim 1; a trailing partial block repeats its last frame."""
    frames = x.shape[1]
    blocks = -(-frames // stride)
    if blocks * stride != frames:
        x = torch.cat([x, x[:, -1:].expand(x.shape[0], blocks * stride - frames, *x.shape[2:])], 1)
    return x.reshape(x.shape[0], blocks, stride, *x.shape[2:]).mean(2)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * mult, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class StaticBandMixer(nn.Module):
    """Stacks 0-2. Per-head (band, band) matrix, initialised near the identity; gated, no dynamics."""
    def __init__(self, dim, bands, heads, dropout=0.0):
        super().__init__()
        self.heads, self.norm = heads, RMSNorm(dim)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.weight = nn.Parameter(torch.eye(bands).repeat(heads, 1, 1) + torch.randn(heads, bands, bands) / math.sqrt(2 * bands))
        self.to_gates, self.drop = nn.Linear(dim, heads), nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x):  # (b, t, band, dim)
        y = self.norm(x)
        b, t, f, d = y.shape
        h, c, n = self.heads, d // self.heads, b * t
        # one bmm per head: (h, f, f) @ (h, f, n*c) instead of n tiny (f, f) @ (f, c) matmuls
        v = self.to_v(y).reshape(n, f, h, c).permute(2, 1, 0, 3).reshape(h, f, n * c)
        z = torch.bmm(self.weight, v).reshape(h, f, n, c).permute(2, 1, 0, 3)
        z = self.drop(z) * self.to_gates(y).reshape(n, f, h).sigmoid().unsqueeze(-1)
        return self.to_out(z.reshape(b, t, f, d))


class SlowBandAttention(nn.Module):
    """Stacks 3-5. q/k from 4-frame mean tokens (shared by the block), v per frame, narrowed to v_width."""
    def __init__(self, dim, heads, dim_head, v_width, stride, dropout=0.0, rotary=None):
        super().__init__()
        self.heads, self.dim_head, self.v_dim, self.stride, self.rotary = heads, dim_head, v_width // heads, stride, rotary
        self.dropout, self.norm = dropout, RMSNorm(dim)
        self.to_qk = nn.Linear(dim, 2 * heads * dim_head, bias=False)
        self.to_v, self.to_gates = nn.Linear(dim, v_width, bias=False), nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(v_width, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x):  # (b, t, band, dim)
        b, t, f, _ = x.shape
        h, s, dv = self.heads, self.stride, self.v_dim
        y = self.norm(x)
        pooled = self.norm(block_mean(x, s))                       # (b, T, band, dim): q/k see the block mean
        q, k = self.to_qk(pooled).view(b, pooled.shape[1], f, 2, h, self.dim_head).unbind(3)
        cos, sin = self.rotary.table(f, x.device, q.dtype)          # rotated over the BAND axis, not time
        q = apply_rotary(cos, sin, q.flatten(0, 1)).view(b, -1, f, h, self.dim_head).permute(0, 3, 1, 2, 4)
        k = apply_rotary(cos, sin, k.flatten(0, 1)).view(b, -1, f, h, self.dim_head).permute(0, 3, 1, 2, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) * self.dim_head ** -0.5, -1)   # (b, h, T, band, band)
        a = F.dropout(a, self.dropout, self.training)
        v = self.to_v(y).view(b, t, f, h, dv).permute(0, 3, 1, 2, 4)            # (b, h, t, band, dv)
        blocks = a.shape[2]
        if blocks * s != t:
            v = torch.cat([v, v[:, :, -1:].expand(-1, -1, blocks * s - t, -1, -1)], 2)
        out = (a.unsqueeze(3) @ v.reshape(b, h, blocks, s, f, dv)).flatten(2, 3)[:, :, :t]
        out = out.permute(0, 2, 3, 1, 4) * self.to_gates(y).sigmoid().unsqueeze(-1)
        return self.to_out(out.flatten(-2))


class BlockFeedForward(nn.Module):
    """FFN once per stride-frame block, broadcast to the block (interpolating it is 13 dB worse)."""
    def __init__(self, dim, mult, stride, dropout):
        super().__init__()
        self.net, self.stride = FeedForward(dim, mult, dropout).net, stride   # the Sequential itself, not a second wrapper

    def forward(self, x):
        return self.net(block_mean(x, self.stride)).repeat_interleave(self.stride, 1)[:, :x.shape[1]]


class FreqBranch(nn.Module):
    """One frequency stack = pre-norm residual attention + residual FFN, both on the (b, t, band, dim) grid.

    No output norm: the modules below normalise their own input, and the stock Transformer was built with
    `norm_output=False` for this arm, so adding one here would be 192 params x 6 that the reference does not have.
    """
    def __init__(self, dim, index, static_stacks, heads, dim_head, v_width, stride, mult, bands, dropout, rotary):
        super().__init__()
        static = index < static_stacks
        self.attn = StaticBandMixer(dim, bands, heads, dropout) if static else \
            SlowBandAttention(dim, heads, dim_head, v_width, stride, dropout, rotary)
        self.ff = FeedForward(dim, mult, dropout) if static else BlockFeedForward(dim, mult, stride, dropout)

    def forward(self, x):
        x = self.attn(x) + x
        return self.ff(x) + x


class MSC(nn.Module):
    """Time branch: Linear(2h) -> u*SiLU(v) -> parallel dilated depthwise Conv1d(k=31) -> Linear."""
    def __init__(self, dim, mult=4, k=31, dilations=(1,), dropout=0.0):
        super().__init__()
        h = dim * mult // 2
        self.proj_in = nn.Linear(dim, 2 * h)
        self.convs = nn.ModuleList(nn.Conv1d(h, h, k, padding=k // 2 * d, dilation=d, groups=h) for d in dilations)
        self.proj_out, self.drop = nn.Linear(h, dim), nn.Dropout(dropout)

    def forward(self, x):  # (b, t, band, dim) -> folded into the batch so the conv sees (b*band, h, t)
        b, t, f, d = x.shape
        u, v = self.proj_in(x).chunk(2, dim=-1)
        g = (u * F.silu(v)).permute(0, 2, 1, 3).reshape(b * f, t, -1).transpose(1, 2)
        out = self.proj_out(sum(c(g) for c in self.convs).transpose(1, 2))
        return self.drop(out.reshape(b, f, t, d).permute(0, 2, 1, 3))


class PooledKV(nn.Module):
    """Time attention with framewise queries and 4-frame pooled K/V (stacks 0 and 4)."""
    def __init__(self, dim, heads=2, dim_head=64, stride=4, dropout=0.0, rotary=None):
        super().__init__()
        width = heads * dim_head
        self.heads, self.dim_head, self.stride, self.rotary = heads, dim_head, stride, rotary
        self.dropout = dropout
        self.query, self.key_value = nn.Linear(dim, width), nn.Linear(dim, 2 * width)
        self.gates, self.output = nn.Linear(dim, heads), nn.Linear(width, dim)

    def forward(self, x):  # (b, t, band, dim)
        b, t, f, d = x.shape
        x = x.transpose(1, 2).reshape(b * f, t, d)
        pooled = F.avg_pool1d(x.transpose(1, 2), self.stride, self.stride, ceil_mode=True).transpose(1, 2)
        q = self.query(x).unflatten(-1, (self.heads, self.dim_head))
        k, v = self.key_value(pooled).unflatten(-1, (2, self.heads, self.dim_head)).unbind(-3)
        cos, sin = self.rotary.table(t, x.device, q.dtype)
        centers = torch.arange(0, t, self.stride, device=x.device) + ((t - torch.arange(0, t, self.stride, device=x.device)).clamp(max=self.stride) - 1) // 2
        q = apply_rotary(cos, sin, q).transpose(1, 2)
        k = apply_rotary(cos.index_select(1, centers), sin.index_select(1, centers), k).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v.transpose(1, 2).contiguous(), dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2) * self.gates(x).sigmoid().unsqueeze(-1)
        return self.output(out.flatten(-2)).reshape(b, f, t, d).transpose(1, 2)


class BandSplit(nn.Module):
    """Per-band RMSNorm + Linear into `dim`, one entry per band (62 of them, inputs 4..514 wide)."""
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = tuple(dim_inputs)
        self.to_features = nn.ModuleList(nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in self.dim_inputs)

    def forward(self, x):  # (b, t, sum(dim_inputs)) -> (b, t, band, dim)
        return torch.stack([f(s) for s, f in zip(x.split(self.dim_inputs, -1), self.to_features)], dim=-2)


class MaskEstimator(nn.Module):
    """Per band: Linear(dim, 4*dim) -> Tanh -> Linear(4*dim, 2*dim_in) -> GLU -> dim_in."""
    def __init__(self, dim, dim_inputs, mult=4):
        super().__init__()
        hidden = dim * mult
        mlp = lambda out: nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, out))
        self.to_freqs = nn.ModuleList(nn.Sequential(mlp(d * 2), nn.GLU(dim=-1)) for d in dim_inputs)

    def forward(self, x):  # (b, t, band, dim) -> (b, t, sum(dim_inputs))
        return torch.cat([f(b) for b, f in zip(x.unbind(-2), self.to_freqs)], dim=-1)


class BandMixFormerPKV(nn.Module):
    def __init__(self, dim=192, depth=6, heads=8, dim_head=64, mult=4, static_stacks=3, stride=4, v_width=128,
                 pkv_layers=(0, 4), bands=62, freqs_per_bands=FREQS_PER_BANDS, stereo=True,
                 dilations=(1,), conv_kernel=31, dropout=0.0, faketime=4.0, n_fft=2048, hop=512, win=2048):
        super().__init__()
        channels = 2 if stereo else 1
        self.dim_inputs = tuple(2 * f * channels for f in freqs_per_bands)
        self.audio_channels, self.stereo = channels, stereo
        self.n_fft, self.hop, self.win, self.faketime = n_fft, hop, win, faketime
        assert len(self.dim_inputs) == bands, (len(self.dim_inputs), bands)
        self.band_split = BandSplit(dim, self.dim_inputs)
        rotary = Rotary(dim_head)
        self.time = nn.ModuleList(PooledKV(dim, 2, 64, 4, dropout, rotary) if i in pkv_layers
                                  else MSC(dim, mult, conv_kernel, dilations, dropout) for i in range(depth))
        self.temporal_norms = nn.ModuleList(RMSNorm(dim) for _ in range(depth))
        self.freq = nn.ModuleList(FreqBranch(dim, i, static_stacks, heads, dim_head, v_width, stride, mult, bands, dropout, rotary)
                                  for i in range(depth))
        self.final_norm, self.mask_estimators = RMSNorm(dim), MaskEstimator(dim, self.dim_inputs)

    def forward(self, stft_repr):  # (b, band_in, frame, 2) real-imag -> complex mask of the same shape
        b, fs, t, c = stft_repr.shape
        x = self.band_split(stft_repr.permute(0, 2, 1, 3).reshape(b, t, fs * c))
        for tn, time, freq in zip(self.temporal_norms, self.time, self.freq):
            x = time(tn(x)) + x
            x = freq(x)
        mask = self.mask_estimators(self.final_norm(x))  # (b, t, 2*fs)
        return mask.reshape(b, 1, t, fs, c).permute(0, 1, 3, 2, 4)

    def stft(self, audio):  # (b, ch, samples) -> (b, ch*1025, frames, 2); index = freq*channels + channel
        window = torch.hann_window(self.win, device=audio.device, dtype=audio.dtype)
        spec = torch.stft(audio.reshape(-1, audio.shape[-1]), self.n_fft, self.hop, self.win, window=window, return_complex=True)
        frames = spec.shape[-1]
        spec = torch.view_as_real(spec).reshape(audio.shape[0], self.audio_channels, -1, frames, 2)
        return spec.permute(0, 2, 1, 3, 4).reshape(audio.shape[0], -1, frames, 2)

    def istft(self, spec, length):  # (b, ch*1025, frames) complex -> (b, ch, samples)
        b, n, t = spec.shape
        bins = n // self.audio_channels
        window = torch.hann_window(self.win, device=spec.device)
        x = spec.reshape(b, bins, self.audio_channels, t).permute(0, 2, 1, 3).reshape(b * self.audio_channels, bins, t)
        out = torch.istft(x, self.n_fft, self.hop, self.win, window=window, return_complex=False, length=length)
        return out.reshape(b, self.audio_channels, length)

    def separate(self, audio):  # (b, ch, samples) -> (b, ch, samples), the stem this model predicts
        stft_repr = self.stft(audio)
        spec = torch.view_as_complex(stft_repr)                                      # (b, fs, t)
        mask = torch.view_as_complex(self.forward(stft_repr).squeeze(1).to(stft_repr.dtype))
        return self.istft(spec * mask, audio.shape[-1])


def count_params(model):
    return sum(p.numel() for p in model.parameters())   # includes the non-trainable RoPE table, matching the paper's 20.27M


if __name__ == '__main__':
    model = BandMixFormerPKV()
    audio = torch.randn(1, 2, 352800)
    stft_repr = model.stft(audio)
    print('params', f'{count_params(model):,}')
    print('stft', tuple(stft_repr.shape), '-> mask', tuple(model(stft_repr).shape))
    print('separate', tuple(model.separate(audio).shape))
