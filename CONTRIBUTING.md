# Contributing to Strabon Nano

Thank you for your interest in contributing! This document covers everything
you need to know before opening a pull request.

---

## Getting started

```bash
git clone https://github.com/th3n3xtg3n3ration/strabon-nano.git
cd strabon-nano
pip install -r requirements.txt
```

---

## Running the tests

All tests run **offline** — no dataset download or network access required.

```bash
python -m tests.test_model       # 10 tests: causality, initial loss, gradients, RoPE norms …
python -m tests.test_tokenizer   #  7 tests: BPE training, %TR metric, apostrophe rule …
python -m tests.test_data        # 16 tests: filters, deduplication, binary encoding …
```

All tests must pass before you push. If your change is expected to affect a
test value (e.g. you changed an architecture constant), update the test and
explain why in the PR description.

---

## What to work on

- **Bug fixes** — always welcome. Open an issue first if the fix is non-trivial.
- **Documentation improvements** — typos, clarifications, additional derivations.
- **New presets** — add them to `strabon/config.py` and document the expected
  parameter count.
- **Performance improvements** — include a before/after throughput measurement
  (`tok/s` from the training log).
- **New features** — open an issue first to discuss scope and design.

---

## Code style

- Follow the style already present in each file — no linter is enforced but
  consistency is expected.
- No type-annotation changes are required, but new public functions should
  carry type hints matching the rest of the codebase.
- Keep imports at the top of the file; use `from __future__ import annotations`
  in every module (already present).
- No new dependencies without discussion. The core dependency list is small by
  design: `torch`, `tokenizers`, `datasets`, `numpy`.

---

## Commit messages

Use the imperative mood and keep the subject line under 72 characters:

```
fix: use inspect.signature to detect fused AdamW support
docs: add navigation banners to all docs/ files
feat: add nano-100m preset
test: verify GQA output shape for n_kv_head < n_head
```

Prefix: `fix:`, `feat:`, `docs:`, `test:`, `refactor:`, `chore:`.

---

## Pull request checklist

- [ ] All three test suites pass locally
- [ ] New behaviour is covered by a test (if applicable)
- [ ] Documentation is updated (if the change affects user-visible behaviour)
- [ ] The PR description explains *what* and *why*, not just *how*

---

## Reporting issues

Use the issue templates:

- **Bug report** — unexpected error, wrong output, broken test
- **Feature request** — new capability or preset

Please include the PyTorch version, CUDA version (if relevant), and the exact
command you ran.
