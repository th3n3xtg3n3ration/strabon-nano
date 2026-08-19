"""
Turkish byte-level BPE tokenizer.

Two decisions are specific to Turkish:

1. The pre-tokenisation pattern keeps apostrophe-separated suffixes
   ("Zeynep'in" -> "Zeynep" + "'in") as a single piece, and caps digit runs at
   three. GPT-2's pattern already treats ç ğ ı ö ş ü as letters via \p{L}, so
   whole words survive there too; the apostrophe rule is the real difference.
   It matters because Turkish attaches case endings to proper nouns after an
   apostrophe, and without this rule "Zeynep'in", "Zeynep'e" and "Zeynep'ten"
   look like three unrelated words to the model.

2. Training reports %TR: the share of vocabulary entries made up entirely of
   Turkish letters.

   This is a cheap proxy, not the metric from the paper cited below. That
   paper defines %TR as the share of tokens that are valid words in the target
   language, which needs a lexicon. Ours only checks the character set, so a
   fragment like "ecekt" counts here but would not count there. Read it as a
   rough alignment signal, not as a reproduction of the published number.

   TÜRKÇE NOT - bu ölçüt neden önemli:
   Türkçe sondan eklemeli bir dildir, bu yüzden genel amaçlı tokenizer'lar
   Türkçe metni gereğinden fazla parçalar. Aynı metin için ölçülen değerler
   (arXiv 2508.13058): aya-expanse 2,19 token/kelime ve %TR 50,7;
   llama-3.1 2,46 / 45,8; gemma-2 2,51 / 48,6; Qwen2.5 2,83 / 40,3.
   İngilizce baz yaklaşık 1,23 token/kelime. Aynı çalışmada %TR ile aşağı
   akış başarımı arasında r = 0,90 korelasyon bildirilmiştir. Dikkat: bu
   korelasyon dört tokenizer ve farklı model aileleri üzerinden hesaplandı,
   yani nedensellik değil ilişki gösterir. Yine de hedefin en yüksek
   sıkıştırma değil, sözlüğü Türkçe ile hizalamak olduğuna işaret ediyor.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TURKISH_LETTERS = "A-Za-zÇĞİIÖŞÜçğıiöşü"

# Split pattern, ordered so apostrophe suffixes are captured before plain words.
PRETOKENIZE_PATTERN = (
    rf"'(?:[{TURKISH_LETTERS}]{{1,6}})"              # 'in  'den  'yle
    rf"|[^\r\n{TURKISH_LETTERS}0-9]?[{TURKISH_LETTERS}]+"  # word, with any leading space
    rf"|[0-9]{{1,3}}"                                # numbers in groups of at most three
    rf"| ?[^\s{TURKISH_LETTERS}0-9]+[\r\n]*"         # punctuation
    rf"|\s*[\r\n]+"                                  # line breaks
    rf"|\s+(?!\S)"
    rf"|\s+"
)

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|doc|>"]
DOC_SEPARATOR = "<|doc|>"

_TR_ALPHABET = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")


def iter_documents(raw_path: str | Path):
    """Yield documents from a raw corpus file separated by DOC_SEPARATOR lines."""
    buffer: list[str] = []
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == DOC_SEPARATOR:
                if buffer:
                    yield "".join(buffer)
                    buffer = []
            else:
                buffer.append(line)
    if buffer:
        yield "".join(buffer)


def train_tokenizer(raw_path: str | Path, out_path: str | Path,
                    vocab_size: int = 32768, min_frequency: int = 2) -> None:
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(PRETOKENIZE_PATTERN),
                             behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"[tokenizer] training, target vocab {vocab_size:,}", flush=True)
    tokenizer.train_from_iterator(iter_documents(raw_path), trainer=trainer)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"[tokenizer] saved to {out_path} "
          f"(vocab {tokenizer.get_vocab_size():,})", flush=True)


def measure(tokenizer_path: str | Path, sample_text: str) -> dict:
    """Report vocabulary alignment and compression on a sample of real text."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()

    turkish_entries = 0
    for piece in vocab:
        if piece in SPECIAL_TOKENS:
            continue
        try:
            plain = tokenizer.decoder.decode([piece]).strip()
        except Exception:
            continue
        if plain and all(ch in _TR_ALPHABET for ch in plain):
            turkish_entries += 1

    words = len(re.findall(rf"[{TURKISH_LETTERS}']+", sample_text))
    tokens = len(tokenizer.encode(sample_text).ids)
    return {
        "vocab": len(vocab),
        "%TR": round(100 * turkish_entries / max(1, len(vocab)), 2),
        "tokens_per_word": round(tokens / max(1, words), 3),
        "chars_per_token": round(len(sample_text) / max(1, tokens), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Strabon tokenizer")
    parser.add_argument("--raw", default="data/raw_tr.txt")
    parser.add_argument("--out", default="data/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    train_tokenizer(args.raw, args.out, args.vocab_size, args.min_frequency)

    with open(args.raw, encoding="utf-8") as handle:
        sample = handle.read(2_000_000)
    print("[tokenizer] metrics:", measure(args.out, sample))


if __name__ == "__main__":
    main()
