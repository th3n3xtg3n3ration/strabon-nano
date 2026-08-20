<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.svg">
    <img alt="Strabon Nano Logo" src="docs/assets/logo.svg" width="600">
  </picture>
</p>

# Strabon Nano

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: ODC-By](https://img.shields.io/badge/data-ODC--By-green.svg)](https://opendatacommons.org/licenses/by/)
[![Data: CC BY-SA 4.0](https://img.shields.io/badge/data-CC%20BY--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A small Turkish language model trained from scratch. Everything from tokenizer
to sampling lives in this repository; no pretrained weights are used.

This is Phase 1 of the Strabon roadmap. The goal is not a usable assistant;
it is a complete, measured, reproducible training pipeline.

---

## Contents

| Path | Purpose |
|---|---|
| `strabon/config.py` | Model and training configuration, parameter accounting |
| `strabon/data.py` | Download, clean, filter, tokenize into flat `.bin` files |
| `strabon/tokenizer.py` | Turkish byte-level BPE training and `%TR` metric |
| `strabon/model.py` | RMSNorm, RoPE, grouped-query attention, SwiGLU, tied weights |
| `strabon/train.py` | Training loop with AMP, cosine schedule and a wall-clock budget |
| `strabon/sample.py` | Text generation from a checkpoint |
| `tests/` | Offline tests — no dataset or network access needed |
| `scripts/run_kaggle.py` | One-command runner for Kaggle Notebooks |

## Documentation

A technical reference in Turkish covering the theory and the implementation.
For each component it answers three questions: what operation is performed, why
that operation was chosen, and what the measurable consequence is.

| Part | Scope |
|---|---|
| [1. Problem tanımı ve amaç fonksiyonu](docs/01-problem-statement-and-objective.md) | Autoregressive factorisation, softmax, cross-entropy, the `log V` sanity check, closed-form gradient, causality constraint |
| [2. Model mimarisi](docs/02-model-architecture.md) | Embedding, RMSNorm, scaled dot-product attention with the `sqrt(d_h)` derivation, GQA, RoPE with a proof of the relative-position property, SwiGLU and the `8d/3` derivation, residual init scaling, weight tying, parameter and FLOP formulas |
| [3. Eğitim yordamı](docs/03-training-procedure.md) | AdamW update equations and bias correction, decoupled weight decay, warmup plus cosine schedule, gradient clipping, accumulation, mixed precision and loss scaling, the wall-clock budget |
| [4. Tokenizasyon ve Türkçe](docs/04-tokenization.md) | BPE algorithm, consequences of byte-level encoding, pre-tokenization, fertility and %TR, Turkish agglutinative morphology, data filters |
| [5. Ölçekleme, çözümleme ve doğrulama](docs/05-scaling-analysis-and-validation.md) | Chinchilla law and its derivation, the Turkish data ceiling, temperature / top-k / top-p, rationale for all 33 tests, expected performance limits |
| [Ek A. Notasyon](docs/06-notation-and-terms.md) | Symbol table, glossary, concept-to-code map |

Start at [docs/README.md](docs/README.md).

---

## Quick start: one hour on Kaggle

Kaggle gives 30 free GPU hours per week, which makes it the easiest place for
a one-hour run.

1. New notebook → **Settings → Accelerator: GPU T4 x2**
2. **Settings → Internet: On** (required for the dataset download)
3. Run:

```python
!rm -rf strabon-nano
!git clone https://github.com/th3n3xtg3n3ration/strabon-nano.git
%cd strabon-nano
!pip install -q -r requirements.txt
!python scripts/run_kaggle.py --stage all --minutes 60
```

This downloads the corpus, trains a tokenizer, encodes the data, trains for
exactly sixty minutes, and prints three samples.

### The wall-clock budget

`--minutes 60` does something specific. The trainer benchmarks 27 steps,
derives the total step count from the measured step time, and fits the cosine
learning-rate schedule to that budget:

```
[budget] benchmarking for a 60 minute run
[budget] 340 ms/step -> 9,847 steps, 645M tokens (71.6 tokens/param)
```

The run therefore finishes on time with the schedule fully decayed, rather
than being cut off partway through.

The benchmark steps run with the learning rate forced to zero. Otherwise they
would apply real updates at full learning rate before warmup and damage the
model — measuring must not disturb what it measures.

---

## Running it manually

```bash
pip install -r requirements.txt

# 1) download the corpus (needs internet, roughly 1-2 GB)
python -m strabon.data download --fineweb-docs 250000 --wiki-docs 60000

# 2) train the tokenizer
python -m strabon.tokenizer --vocab-size 16384

# 3) encode the corpus
python -m strabon.data tokenize

# 4) train
python -m strabon.train --config nano-10m --minutes 60

# 5) sample
python -m strabon.sample --prompt "Türkiye'nin en büyük şehri"
```

---

## Presets

```
nano-10m     vocab=16384 L=6  H=8      d=256  ctx=512   |   9.0M params
nano-25m     vocab=16384 L=10 H=8      d=384  ctx=512   |  24.0M params
nano-50m     vocab=32768 L=10 H=8      d=512  ctx=1024  |  48.9M params
mini-500m    vocab=32768 L=26 H=16/kv4 d=1280 ctx=2048  | 493.6M params
debug        vocab=4096  L=2  H=4      d=128  ctx=256   |   1.0M params
```

List them with `python -m strabon.config`.

Which preset fits an hour, using the Chinchilla rule of thumb of roughly
20 tokens per parameter:

| Hardware | Tokens per hour | Preset |
|---|---|---|
| Kaggle T4 | 200-300M | `nano-10m` |
| RTX 4090 | 600-900M | `nano-25m` |
| A100 80GB | 1.5-2.5B | `nano-50m` |

These are estimates. The trainer measures and prints the real number — replace
them with your own after the first run.

---

## Data

| Source | Content | Licence |
|---|---|---|
| `HuggingFaceFW/fineweb-2` (`tur_Latn`) | Filtered Turkish web text | **ODC-By 1.0**, commercial use allowed |
| `wikimedia/wikipedia` (`20231101.tr`) | Turkish Wikipedia | **CC BY-SA 4.0** |

Default mix: **80% web, 20% Wikipedia**. Wikipedia alone teaches one register
and nothing else; web alone brings noise along with variety. Volume and range
come from the web, sentence quality from Wikipedia. SmolLM3 and OLMo 3 mix for
the same reason.

Filters applied in `strabon/data.py`:

- at least 300 characters
- at least 80% of letters in the Turkish alphabet
- **at least 3% of letters Turkish-specific** (ç ğ ı ö ş ü)
- most repeated line under 30% of all lines (drops navigation menus)
- letters at least 55% of characters (drops table dumps)
- exact-match deduplication per document (blake2b fingerprint)

The Turkish-specific check is not redundant. The English alphabet is a subset
of the Turkish one apart from q, w and x, so an alphabet-coverage test alone
lets English through. Measured rates: Turkish text 10-16%, English 0-1%.

**Licence note.** Both sources are permissive. Turkish corpora under
CC BY-NC-SA, such as `vngrs-web-corpus`, are deliberately not used because
they forbid commercial use. If you generate synthetic data with a commercial
API, check that provider's terms first — most restrict training competing
models on their output.

---

## Tokenizer

Turkish is agglutinative, so general-purpose tokenizers split it far more than
English. Measured on the same text
([arXiv 2508.13058](https://arxiv.org/html/2508.13058v1)):

| Tokenizer | Vocab | Tokens/word | %TR |
|---|---|---|---|
| aya-expanse | 255,029 | 2.19 | 50.7 |
| llama-3.1 | 128,256 | 2.46 | 45.8 |
| gemma-2 | 256,000 | 2.51 | 48.6 |
| Qwen2.5 | 151,665 | 2.83 | 40.3 |

English sits near 1.23 tokens per word. Turkish therefore costs 1.8-2.3x more
tokens for the same content — paid directly in training cost and effective
context length.

This tokenizer does two things about it:

1. **Apostrophe-aware pre-tokenization.** `Zeynep'in` becomes `Zeynep` + `'in`
   instead of `Zeynep` + `'` + `in`. Turkish attaches case endings to proper
   nouns after an apostrophe, so without this rule `Zeynep'in`, `Zeynep'e` and
   `Zeynep'ten` look like three unrelated words. (GPT-2's pattern already
   treats ç ğ ı ö ş ü as letters, so whole words survive there too — the
   apostrophe rule is the real difference, along with capping digit runs.)
2. **Reports `%TR`** — the share of vocabulary entries made entirely of Turkish
   letters. This is a cheap proxy for the metric in the paper above, which
   defines %TR over valid words and needs a lexicon; the paper reports r = 0.90
   between its %TR and downstream accuracy across four tokenizers. Treat it as
   an alignment signal, not a reproduction of that number. The point stands:
   aim for a vocabulary aligned with the language, not maximum compression.

---

## Architecture

Llama-style decoder-only transformer:

- **RMSNorm** — cheaper than LayerNorm, equivalent at this scale
- **RoPE** — no learned position embeddings, extrapolates further
- **Grouped-query attention** — `mini-500m` uses 16 query heads over 4 key/value
  heads, shrinking the KV cache 4x
- **SwiGLU** — hidden size 8/3·d rounded up to a multiple of 64
- **Tied embeddings** — the output head shares the embedding matrix; most
  parameters of a small model are in the embedding, so this matters
- **No biases** — no measurable benefit at this scale
- Residual output projections initialised at `0.02/sqrt(2L)` so activations do
  not grow with depth

Precision is selected automatically: bf16 on Ampere and newer, fp16 with a
gradient scaler on Turing (T4, P100), fp32 on CPU.

---

## Tests

All tests run offline, with no dataset and no network:

```bash
python -m tests.test_model       # 10 tests
python -m tests.test_tokenizer   #  7 tests
python -m tests.test_data        # 16 tests
```

`test_model.py` checks the properties that silently break otherwise:

- initial loss equals `ln(vocab)` — with **targets independent of the inputs**.
  Tied embeddings make predicting the input back easy at initialisation, so
  reusing inputs as targets reports a misleadingly low number.
- **causality** — perturbing token 10 onward must leave positions 0-9 unchanged
- **overfitting a single batch** — a model that cannot memorise one batch has a
  broken gradient path
- every parameter receives a non-zero gradient
- RoPE preserves vector norms, as any rotation must

---

## What to expect

At this scale:

- **Expect:** mostly grammatical Turkish, short paragraphs that stay on topic,
  a model that has learned vowel harmony and case suffixes from data alone.
- **Do not expect:** factual accuracy, reasoning, code, or question answering.
  For calibration, a 561M-parameter model trained on 11.2B tokens scores 0.315
  on MMLU, where random guessing scores 0.25.

A usable assistant comes from post-training an existing open base model, not
from pretraining at this size. That is Phase 5-7 of the roadmap.

---

## Licence

Code is MIT. Trained weights inherit the licences of the training data
(FineWeb-2 ODC-By, Wikipedia CC BY-SA).

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **1. Pretraining pipeline** | ✅ This repository | Tokenizer, data, model, training loop, tests |
| 2. Scaling runs | 🔜 | Systematic Chinchilla sweeps up to `nano-50m` |
| 3. Evaluation harness | 🔜 | Turkish benchmarks, perplexity baselines |
| 4. Data quality | 🔜 | Better filters, deduplication at scale |
| 5–7. Post-training | 🔮 | Instruction tuning on an existing open base model |

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request.

Quick checklist:

```bash
# Run all offline tests before pushing
python -m tests.test_model
python -m tests.test_tokenizer
python -m tests.test_data
```

All tests must pass. New features should include a test.

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@misc{strabonnano2026,
  title        = {Strabon Nano: A Reproducible Turkish Language Model Training Pipeline},
  author       = {th3n3xtg3n3ration},
  year         = {2026},
  howpublished = {\url{https://github.com/th3n3xtg3n3ration/strabon-nano}},
  note         = {Phase 1 — pretraining pipeline, MIT licence}
}
```
