# Tiled VAE (fork)

A fork of [`multidiffusion-upscaler-for-automatic1111`](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)
reduced to the **Tiled VAE** feature alone.

## Scope of this fork

The upstream extension bundles several features — MultiDiffusion, Mixture of
Diffusers, Tiled Diffusion, Regional Prompt Control, and tiled img2img
upscaling. **This fork keeps only Tiled VAE.** The diffusion-tiling code and its
UI are not carried here and are not supported.

The fork is built on the **MIT-licensed** base of the upstream project (commit
`b19edc6`, 2023-03-28). That base was chosen deliberately: later upstream commits
relicensed the project (briefly GPLv3, then CC BY-NC-SA), and a later attention
file pulled in AGPL-derived code from the webui. Starting from the MIT base and
re-writing the one affected layer keeps this fork permissively licensed end to
end — see [License](#license).

## Credit

The Tiled VAE algorithm — the GroupNorm-synchronized, tiled VAE forward — was
created by **LI YI (pkuliyi2015)**. It is not the work of this fork. The original
author's attribution is preserved at the top of `scripts/vae_optimize.py`. This
fork re-implements the attention layer from public APIs and adds the changes
listed below; it does **not** claim authorship of the underlying tiling method.

## What Tiled VAE does

- Lets the VAE encode/decode very large images within limited VRAM, at nearly no
  quality cost — you often no longer need `--lowvram` / `--medvram` for the VAE
  step.
- It splits the image into padded tiles and runs the original VAE forward per
  tile, synchronizing GroupNorm statistics across tiles so the merged output
  matches a non-tiled VAE pass.
- Use the default settings; lower the tile size if you hit CUDA out-of-memory.

## Changes in this fork

On top of the MIT base, this fork makes the following changes:

- **Attention follows webui's active optimization.** The base picked the VAE
  attention kernel by block class. This fork dispatches on
  `modules.sd_hijack.model_hijack.optimization_method` — `sdp`, `sdp-no-mem`,
  `xformers`, or a plain-math fallback for anything else — so the VAE uses the
  same backend as the rest of the pipeline and switches with the launch flag.
  The kernels are written fresh against the public PyTorch / xformers APIs; no
  AGPL-derived attention code is used.
- **SDXL correctness.** The attention task queue is now built with duck typing
  instead of an `isinstance` check against a concrete `ldm` class, so the SDXL
  VAE's `sgm` mid self-attention block is processed instead of being silently
  dropped (which the class check did, producing wrong output without an error).
- **dtype correctness.** The aggregated GroupNorm statistics and the assembled
  result are cast to the VAE's dtype, avoiding half-precision mismatches.
- **Graceful interruption.** Interrupting mid-decode now breaks cleanly and
  returns a coarse but correctly-shaped preview instead of `None`.
- **Hook lifecycle / VRAM release.** Install state is tracked explicitly and the
  hook's VAE reference is released on teardown, so enable/run/disable cycles do
  not leak VRAM.

### Original enhancements in this fork

These changes are original to this fork (no webui derivation):

- **GroupNorm via the law of total variance.** The base aggregated only the
  within-tile variance term. This fork adds the dropped between-tile term,
  removing per-tile brightness/color shifts that appear on high-contrast latents
  (e.g. v-prediction SDXL with zero-terminal-SNR, custom anime VAEs).
- **Overlap-blended assembly.** Adjacent tiles are blended across an overlap
  region using linear ramp masks rather than hard-cut at the tile boundary,
  softening any residual per-tile bias into a smooth seam.
- **Retuned padding** (`22` / `128` for decoder/encoder), which also sets the
  blend width.

## Installation

Clone or copy this repository into your webui's `extensions/` directory and
restart the webui.

## Usage

1. Open the **Tiled VAE** accordion in txt2img / img2img and tick **Enable**.
2. The defaults are recommended. Lower **Encoder/Decoder Tile Size** if you see
   a CUDA out-of-memory error.
3. **Fast Encoder / Fast Decoder** estimate the GroupNorm statistics on a single
   downsampled pass for speed. If a very small tile size makes the encoder output
   gray/washed out, enable **Encoder Color Fix**.
4. Enable **Move VAE to GPU** if your VAE currently lives on the CPU.

## How it works

1. The image is split into tiles, padded in the decoder/encoder.
2. With Fast Mode off:
   1. The VAE forward is decomposed into a task queue processed tile by tile.
   2. At each GroupNorm the queue suspends, stores the tile's mean/var, sends the
      tile to RAM, and moves to the next tile.
   3. Once every tile's statistics are collected they are aggregated and
      GroupNorm is applied, then processing continues.
   4. A zigzag tile order reduces RAM/VRAM transfers.
3. With Fast Mode on, the GroupNorm parameters are estimated once on a
   downsampled pass and reused by all tiles.
4. Finished tiles are assembled — with overlap blending, in this fork — and
   returned.

## License

MIT. This fork is built on the MIT-licensed upstream base (`b19edc6`); the
original MIT notice and the author's attribution are preserved. The fork's own
changes are likewise released under MIT. No CC BY-NC-SA or AGPL-derived code from
later upstream commits is included.
