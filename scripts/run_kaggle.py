#!/usr/bin/env python3
"""
One-command runner for Kaggle Notebooks.

Setup in the Kaggle UI:
  Settings -> Accelerator -> GPU T4 x2 (or P100)
  Settings -> Internet -> On            (required to download the datasets)

Then, in a notebook cell:

    !rm -rf strabon-nano
    !git clone https://github.com/<user>/strabon-nano.git
    %cd strabon-nano
    !pip install -q -r requirements.txt
    !python scripts/run_kaggle.py --stage all --minutes 60

Notes:
  * T4 has no bf16 support; the trainer detects this and switches to fp16.
  * torch.compile is off by default because compilation on a T4 costs more
    time than it saves within a one-hour budget.
  * /kaggle/working persists between sessions, /kaggle/temp does not.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}\n", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["download", "tokenizer", "train", "sample", "all"])
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--config", default="nano-10m")
    parser.add_argument("--vocab-size", type=int, default=16384)
    # Sized for a one-hour T4 run: roughly 250M tokens. More data would only
    # spend the budget on downloading instead of training.
    parser.add_argument("--fineweb-docs", type=int, default=250_000)
    parser.add_argument("--wiki-docs", type=int, default=60_000)
    parser.add_argument("--root", default="/kaggle/working" if Path("/kaggle").exists() else ".")
    parser.add_argument("--prompt", default="Türkiye'nin en büyük şehri")
    args = parser.parse_args()

    root = Path(args.root)
    raw = root / "data/raw_tr.txt"
    tokenizer = root / "data/tokenizer.json"
    tokenized = root / "data/tokenized"
    out = root / "out/strabon-nano"
    python = [sys.executable, "-m"]

    if args.stage in ("download", "all"):
        run(python + ["strabon.data", "download",
                      "--raw", str(raw),
                      "--fineweb-docs", str(args.fineweb_docs),
                      "--wiki-docs", str(args.wiki_docs)])

    if args.stage in ("tokenizer", "all"):
        run(python + ["strabon.tokenizer",
                      "--raw", str(raw), "--out", str(tokenizer),
                      "--vocab-size", str(args.vocab_size)])
        run(python + ["strabon.data", "tokenize",
                      "--raw", str(raw), "--tokenizer", str(tokenizer),
                      "--out", str(tokenized)])

    if args.stage in ("train", "all"):
        run(python + ["strabon.train",
                      "--config", args.config,
                      "--data", str(tokenized),
                      "--out", str(out),
                      "--minutes", str(args.minutes),
                      "--precision", "bf16",   # trainer falls back to fp16 on a T4
                      "--no-compile"])

    if args.stage in ("sample", "all"):
        run(python + ["strabon.sample",
                      "--checkpoint", str(out / "best.pt"),
                      "--tokenizer", str(tokenizer),
                      "--prompt", args.prompt,
                      "--tokens", "150", "--samples", "3"])


if __name__ == "__main__":
    main()
