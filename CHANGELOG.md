# Changelog

All notable changes to Strabon Nano are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0] — 2026-08-19

### Added

- **Tokenizer** — Turkish byte-level BPE with apostrophe-aware pre-tokenization
  and `%TR` vocabulary alignment metric (`strabon/tokenizer.py`)
- **Data pipeline** — download, filter, deduplicate and tokenize FineWeb-2
  (`tur_Latn`) and Turkish Wikipedia into flat `.bin` files (`strabon/data.py`)
- **Model** — decoder-only transformer with RMSNorm, RoPE, grouped-query
  attention, SwiGLU, tied embeddings, no biases (`strabon/model.py`)
- **Training loop** — AMP (bf16/fp16/fp32 auto-select), cosine schedule with
  linear warmup, gradient clipping, accumulation, wall-clock budget via
  `--minutes` (`strabon/train.py`)
- **Sampling** — temperature, top-k and top-p decoding from a checkpoint
  (`strabon/sample.py`)
- **Configuration** — five presets (`debug`, `nano-10m`, `nano-25m`,
  `nano-50m`, `mini-500m`) with closed-form parameter accounting
  (`strabon/config.py`)
- **Tests** — 33 offline tests covering initial loss, causality, gradient
  flow, RoPE norm preservation, BPE correctness and data filters
  (`tests/`)
- **Kaggle runner** — one-command script for a timed GPU run
  (`scripts/run_kaggle.py`)
- **Technical documentation** — six-part Turkish-language reference covering
  theory and implementation (`docs/`)
