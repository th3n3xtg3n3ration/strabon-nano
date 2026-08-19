"""
Offline tokenizer tests. Trains a tiny tokenizer on an inline Turkish sample,
so no dataset or network access is required.

Run with:  python -m tests.test_tokenizer
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from strabon.tokenizer import DOC_SEPARATOR, iter_documents, measure, train_tokenizer

SAMPLE = """Bir sabah küçük bir kedi bahçede oynuyordu. Zeynep'in kitabını mutfağında buldu.
Öğrenciler İstanbul'a gitti ve şehrin sokaklarında uzun uzun yürüdüler.
Çocuklar ağaçların altında oturup kitap okudu, sonra eve döndüler.
Ayşe'nin köpeği bahçenin köşesinde uyuyordu; kimse onu uyandırmadı.
Türkiye'nin en büyük şehri İstanbul'dur ve nüfusu on beş milyonu aşar.
Ormanda yürürken gördüğümüz kuşlar, ağaçların tepesinde yuva yapmıştı."""

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))


def build(tmp: Path, vocab_size: int = 400):
    raw = tmp / "raw.txt"
    # Repeat the sample so byte-pair merges have enough frequency to fire.
    text = "\n".join(f"{SAMPLE}\n{DOC_SEPARATOR}" for _ in range(200))
    raw.write_text(text, encoding="utf-8")
    out = tmp / "tokenizer.json"
    train_tokenizer(raw, out, vocab_size=vocab_size, min_frequency=2)
    return raw, out


def main() -> int:
    from tokenizers import Tokenizer

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        raw, tokenizer_path = build(tmp)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        docs = list(iter_documents(raw))
        check("document iterator splits on the separator", len(docs) == 200,
              f"got {len(docs)}")

        # Round-tripping must be lossless. Byte-level BPE has no unknown token,
        # so any mismatch means the pre-tokenizer or decoder is misconfigured.
        probes = [
            "Bir sabah küçük bir kedi bahçede oynuyordu.",
            "Zeynep'in kitabını mutfağında buldu.",
            "Türkiye'nin en büyük şehri İstanbul'dur.",
            "ĞÜŞİÖÇ ğüşıöç 1234 !?.,;:",
            "emoji ve nadir karakterler: 漢字",
        ]
        failed = [p for p in probes if tokenizer.decode(tokenizer.encode(p).ids) != p]
        check("encode/decode round-trips exactly", not failed,
              f"failed on: {failed}" if failed else "")

        # TÜRKÇE NOT: kesme işaretli çekim ekleri ayrı bir parça olarak
        # yakalanmalı. "Zeynep'in" -> "Zeynep" + "'in"; ek, özel adın içine
        # karışmamalı. Bu, ön-parçalama deseninin çalıştığının doğrudan kanıtı.
        pieces = tokenizer.encode("Zeynep'in").tokens
        joined = "".join(pieces).replace("Ġ", " ")
        has_apostrophe_split = any(piece.startswith("'") for piece in pieces)
        check("apostrophe suffixes are split off", has_apostrophe_split,
              f"pieces: {pieces}")
        check("pieces reassemble to the input", joined.strip().startswith("Zeynep"),
              f"joined: {joined!r}")

        stats = measure(tokenizer_path, SAMPLE)
        check("%TR is reported and positive", stats["%TR"] > 0, str(stats))
        # A vocabulary this small cannot compress well; we only assert sanity.
        check("tokens per word is in a sane range",
              0.5 < stats["tokens_per_word"] < 6.0, str(stats["tokens_per_word"]))

        specials = ["<|pad|>", "<|bos|>", "<|eos|>", "<|doc|>"]
        missing = [s for s in specials if tokenizer.token_to_id(s) is None]
        check("special tokens are present", not missing, f"missing: {missing}")

    failures = 0
    for name, passed, detail in RESULTS:
        mark = "PASS" if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"[{mark}] {name}{suffix}")
        failures += not passed

    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} tokenizer tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
