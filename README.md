# BS-BandMixFormer-PKV

A minimal, torch-only reference for **BS-BandMixFormer-PKV** (code key `H3`) — the band-mixing music
source separation model. This is a reading copy: one file of structure, one training loop, one
inference script. No framework, no config system, no vendor kernels.

The full research codebase (data pipeline, 50k training, Ascend/CUDA/CPU fast paths, ablations) lives
in the upstream `pymss` repo. This repo exists so the architecture can be read in one sitting.

```
bmf.py      the model, ~275 lines: STFT wrapper, band split, 6 stacks, mask estimator
parity.py   loads an upstream 50k checkpoint and proves this is the same function
train.py    synthetic two-source separation, waveform + spectral L1, AdamW + StepLR
infer.py    wav -> predicted stem, chunked
```

```bash
pip install torch numpy
python bmf.py                          # param count, tensor shapes
python train.py --steps 200            # trains on synthetic audio, writes bmf-synthetic.pt
python infer.py song.wav --weights bmf-synthetic.pt --out stem.wav
python infer.py                        # no input file: synthetic audio in, stem.wav out
```

## Parity: the same function, not just the same shape

Matching parameter counts is cheap to fake — a wrong residual chain keeps both the shapes and the count.
So `parity.py` loads an upstream 50k checkpoint into both implementations and compares the mask each
produces from identical STFT input:

```
loaded 549 tensors, skipped 0, not loaded 3      # the 3 are the RoPE tables, recomputed from theta
PARITY max_abs=1.580e-06 rel=1.292e-06
```

That test caught the only real bug in the first draft of this file. `Transformer.forward` chains
`x = ff(attn(x) + x) + (attn(x) + x)`; the obvious-looking `ff(attn(x) + x) + x` is a *different model*
with exactly the same 20,274,048 parameters, and it scored 0.81 relative error before the fix.

## The two modifications that define the model

Stock BS-RoFormer spends most of its compute on full-band, full-frame attention in every one of its
frequency stacks. This model keeps that behaviour where it earns its cost and replaces it where it does
not. Every choice below was made by **frozen-weight intervention** — rewrite the forward pass of a
trained checkpoint, do not retrain, measure the paired SI-SDR delta on 56 windows — not by intuition.

| | stacks 0–2 | stacks 3–5 |
|---|---|---|
| cross-band routing | **static** per-head `62×62` matrix (identity-initialised), gated | dynamic attention, per-band queries |
| q/k frame rate | — | every 4th frame (4-frame mean token, shared by the block) |
| q/k band resolution | — | full 62 bands (grouping to 31 stays tail-heavy) |
| v / out width | 192 | **128** (narrow) |
| RoPE | not needed | over the band axis |
| FFN frame rate | **every frame** | every 4th frame, broadcast over the block |

The measurement that justifies each row:

| intervention | ΔSI-SDR (median) | conclusion |
|---|---:|---|
| static band mixing on stacks 0–2 | −0.02 … −0.04 dB/layer | cross-band routing there does not need to be dynamic |
| remove the routing entirely | −8.5 dB | the mixing itself is load-bearing |
| group 62 bands → 12 | −4.3 dB | grouping is not a substitute |
| pool the FFN of stacks 0–2 | −11.6 dB | stacks 0–2 FFN is the per-frame workhorse |
| pool the FFN of stacks 3–5 | −0.31 dB | free |
| interpolate instead of broadcast | −13.2 dB | FFN output is not linear in time |
| pool q/k over 4 frames (stacks 3–5) | −0.15 dB | free |
| pool q/k in band groups of 31 | −0.67 dB, p10 −3.49 | rejected: tail-heavy |
| pool all of q/k/v | −1.82 dB | rejected: queries must stay per-band |

Result: **20.27M params** (baseline 24.59M), **262.6 GFLOP per 8 s** (baseline 899 → −71%),
**9.103 dB** median SI-SDR at 50k steps vs the baseline's 9.295 and 8.5 s → 2.2 s on 8 CPU cores.
It is a trade of 0.34 dB for half the compute, not a free upgrade.

## Structure

```
audio (b, 2, samples)
  └ STFT 2048/512      -> (b, 2050, frames, 2)        2050 = 1025 bins * 2 channels, real+imaginary
  └ BandSplit          -> (b, frames, 62, 192)         62 bands, 4..514 input dims, per-band RMSNorm+Linear
  └ 6 x stack:
      time branch      -> x = time(temporal_norm(x)) + x
                          MSC everywhere; PKV attention on stacks 0 and 4
      freq branch      -> x = norm(ff(attn(x) + x) + x)
                          stacks 0-2: StaticBandMixer + per-frame FFN
                          stacks 3-5: SlowBandAttention + BlockFeedForward
  └ RMSNorm -> MaskEstimator (per band, 2-layer MLP + GLU) -> (b, frames, 4100)
  └ reshape to a complex mask, multiply the STFT, inverse STFT -> (b, 2, samples)
```

Everything in the frequency branch works on the `(batch, frame, band, dim)` grid. Keeping frames as an
explicit axis is what makes 4-frame block pooling expressible at all — flattening frames into the batch
(the stock layout) forbids it.

`PKV` in the name is the time branch: **P**ooled **K**/**V** — framewise queries, K/V averaged over 4
frames, RoPE applied to the queries and to the block centres.

## What is deliberately not here

Streaming, multi-stem output, band grouping kernels, mixed precision, gradient checkpointing, the
chunked-FFT inference path, the Ascend Triton kernels, the training recipe that produced 9.103 dB.
This repo reproduces the *graph*, at readable density, and nothing else.

## Reference

- BS-RoFormer — the baseline this restructures.
- The upstream `pymss` repo: `pymss_core/modules/bs_roformer/` for the production implementation, and
  `BS-BandMixFormer-PKV.md` for the full experiment log (every ablation above, plus the Ascend port and
  the CPU fast path).
