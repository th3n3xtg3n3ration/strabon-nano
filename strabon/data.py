"""
Data pipeline: download -> clean -> tokenize -> flat .bin files.

Sources (both permissively licensed, commercial use allowed):
  HuggingFaceFW/fineweb-2, config "tur_Latn"     filtered Turkish web text, ODC-By 1.0
  wikimedia/wikipedia,     config "20231101.tr"  Turkish Wikipedia, CC BY-SA 4.0

TÜRKÇE NOT - neden karışım:
Varsayılan %80 web + %20 Vikipedi. Sadece Vikipedi ile eğitilen model tek bir
üslup öğrenir (ansiklopedik); sadece web ile eğitilen model gürültüyü de
öğrenir. Hacmi ve üslup çeşitliliğini web, düzgün cümle yapısını Vikipedi
verir. SmolLM3 ve OLMo 3 aynı mantıkla çalışır.

Lisans uyarısı: vngrs-web-corpus gibi CC BY-NC-SA lisanslı Türkçe korpuslar
bilerek kullanılmadı, ticari kullanımı engelliyorlar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .tokenizer import DOC_SEPARATOR

TURKISH_ALPHABET = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")

# Letters that exist in Turkish but not in English.
#
# TÜRKÇE NOT - bu küme neden ayrı tutuluyor:
# İngiliz alfabesi (q, w, x dışında) Türk alfabesinin içinde kalır, bu yüzden
# "harflerin yüzde kaçı Türk alfabesinde" testi İngilizce metni ELEMEZ.
# Ölçülen değerler: Türkçe metinde bu harflerin payı %10-16, İngilizcede %0-1,
# Azerbaycan Türkçesinde %5 civarı. %3 eşiği İngilizceyi güvenle eler.
TURKISH_SPECIFIC = set("çğıöşüÇĞİÖŞÜ")

_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f​-‏﻿]")

SOURCES = {
    "fineweb": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
    "wikipedia": ("wikimedia/wikipedia", "20231101.tr"),
}


# --------------------------------------------------------------------- cleaning

def turkish_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in TURKISH_ALPHABET for c in letters) / len(letters)


def turkish_specific_ratio(text: str) -> float:
    """Share of letters that are Turkish but not English. See TURKISH_SPECIFIC."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in TURKISH_SPECIFIC for c in letters) / len(letters)


def repetition_ratio(text: str) -> float:
    """Share of the most frequent line. Catches navigation menus and boilerplate."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 4:
        return 0.0
    counts: dict[str, int] = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    return max(counts.values()) / len(lines)


def clean(text: str) -> str:
    text = _CONTROL.sub("", text)
    text = _URL.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def keep(text: str, min_chars: int = 300, min_turkish: float = 0.80,
         min_specific: float = 0.03, max_repetition: float = 0.30,
         min_letter_ratio: float = 0.55) -> bool:
    """Apply every quality filter. Returns True if the document should be kept."""
    if len(text) < min_chars:
        return False
    if turkish_ratio(text) < min_turkish:
        return False
    if turkish_specific_ratio(text) < min_specific:
        return False
    if repetition_ratio(text) > max_repetition:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / max(1, len(text)) >= min_letter_ratio


def _fingerprint(text: str) -> str:
    normalised = re.sub(r"\W+", "", text.lower())[:2000]
    return hashlib.blake2b(normalised.encode("utf-8"), digest_size=16).hexdigest()


# -------------------------------------------------------------------- download

def stream_documents(source: str, limit: int, min_chars: int = 300):
    """Stream, clean, filter and de-duplicate documents from one source."""
    from datasets import load_dataset

    if source not in SOURCES:
        raise ValueError(f"unknown source: {source}")
    repo, config = SOURCES[source]
    dataset = load_dataset(repo, name=config, split="train", streaming=True)

    seen: set[str] = set()
    emitted = 0
    for record in dataset:
        text = clean(record.get("text") or "")
        if not keep(text, min_chars=min_chars):
            continue
        fingerprint = _fingerprint(text)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield text
        emitted += 1
        if emitted >= limit:
            return


def download(out_path: Path, fineweb_docs: int, wiki_docs: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_chars = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for source, limit in (("fineweb", fineweb_docs), ("wikipedia", wiki_docs)):
            if limit <= 0:
                continue
            print(f"[data] {source}: target {limit:,} documents", flush=True)
            count = 0
            for text in stream_documents(source, limit):
                handle.write(text)
                handle.write(f"\n{DOC_SEPARATOR}\n")
                total_chars += len(text)
                count += 1
                if count % 5000 == 0:
                    print(f"  {source}: {count:,} docs, {total_chars / 1e6:.1f}M chars "
                          f"({time.time() - started:.0f}s)", flush=True)
            print(f"[data] {source}: {count:,} documents kept", flush=True)

    print(f"[data] {total_chars / 1e6:.1f}M chars written to {out_path} "
          f"({time.time() - started:.0f}s)", flush=True)
    return out_path


# -------------------------------------------------------------------- tokenize

def tokenize(raw_path: Path, tokenizer_path: Path, out_dir: Path,
             val_fraction: float = 0.0005) -> dict:
    """Encode the corpus into flat train.bin / val.bin files plus a meta.json."""
    from tokenizers import Tokenizer

    from .tokenizer import iter_documents

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("tokenizer has no <|eos|> token")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tokenize] vocab {vocab_size:,}, dtype {dtype.__name__}", flush=True)

    written = {"train": 0, "val": 0}
    handles = {split: open(out_dir / f"{split}.bin", "wb") for split in ("train", "val")}
    rng = np.random.default_rng(0)
    started, docs = time.time(), 0

    try:
        for text in iter_documents(raw_path):
            text = text.strip()
            if not text:
                continue
            docs += 1
            ids = tokenizer.encode(text).ids + [eos_id]
            split = "val" if rng.random() < val_fraction else "train"
            np.asarray(ids, dtype=dtype).tofile(handles[split])
            written[split] += len(ids)
            if docs % 20000 == 0:
                total = written["train"] + written["val"]
                print(f"  {docs:,} docs, {total / 1e6:.1f}M tokens "
                      f"({time.time() - started:.0f}s)", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    meta = {"documents": docs, "train_tokens": written["train"],
            "val_tokens": written["val"], "vocab_size": vocab_size,
            "dtype": dtype.__name__}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[tokenize] done: {meta}", flush=True)
    return meta


# ----------------------------------------------------------------------- loader

class BinaryLoader:
    """Samples random windows from a flat token file via np.memmap."""

    def __init__(self, path: str | Path, context: int, batch_size: int,
                 dtype: str = "uint16", seed: int = 0):
        self.tokens = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.context = context
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        if len(self.tokens) < context + 1:
            raise ValueError(f"{path}: {len(self.tokens)} tokens, need at least {context + 1}")

    def __len__(self) -> int:
        return len(self.tokens)

    def next_batch(self, device: str):
        import torch

        starts = self.rng.integers(0, len(self.tokens) - self.context - 1,
                                   size=self.batch_size)
        x = np.stack([self.tokens[i: i + self.context] for i in starts]).astype(np.int64)
        y = np.stack([self.tokens[i + 1: i + 1 + self.context] for i in starts]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if device.startswith("cuda"):
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)


# -------------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Strabon data pipeline")
    parser.add_argument("stage", choices=["download", "tokenize", "all"])
    parser.add_argument("--fineweb-docs", type=int, default=400_000)
    parser.add_argument("--wiki-docs", type=int, default=100_000)
    parser.add_argument("--raw", default="data/raw_tr.txt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--out", default="data/tokenized")
    args = parser.parse_args()

    if args.stage in ("download", "all"):
        download(Path(args.raw), args.fineweb_docs, args.wiki_docs)
    if args.stage in ("tokenize", "all"):
        tokenize(Path(args.raw), Path(args.tokenizer), Path(args.out))


if __name__ == "__main__":
    main()
