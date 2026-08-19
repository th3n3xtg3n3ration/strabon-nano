"""
Offline tests for the cleaning and filtering rules.

Run with:  python -m tests.test_data
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from strabon.data import (BinaryLoader, clean, keep, repetition_ratio,
                          turkish_ratio, turkish_specific_ratio)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))


TURKISH = ("Ormanda yürürken gördüğümüz kuşlar ağaçların tepesinde yuva yapmıştı. "
           "Çocuklar sabah erkenden kalkıp okula gitti ve derslerine çalıştı. "
           "Akşam olunca herkes evine döndü, sofraya oturdu ve uzun uzun sohbet etti. "
           "Bahçedeki çiçekler yağmurdan sonra çok daha canlı görünüyordu bugün.")


def main() -> int:
    check("turkish_ratio is high for Turkish", turkish_ratio(TURKISH) > 0.95,
          f"{turkish_ratio(TURKISH):.3f}")
    check("turkish_ratio is low for other scripts",
          turkish_ratio("これは日本語のテキストです") < 0.1)

    boilerplate = "\n".join(["Ana Sayfa"] * 8 + ["Gerçek içerik burada."])
    check("repetition_ratio catches repeated lines", repetition_ratio(boilerplate) > 0.5,
          f"{repetition_ratio(boilerplate):.2f}")
    check("repetition_ratio is low for varied text",
          repetition_ratio("bir\niki\nüç\ndört\nbeş") < 0.3)

    messy = "Bak  şuraya:   https://example.com/sayfa   ve\n\n\n\n\ndevam et."
    cleaned = clean(messy)
    check("clean strips urls", "http" not in cleaned, cleaned)
    check("clean collapses whitespace", "   " not in cleaned, repr(cleaned))
    check("clean collapses blank lines", "\n\n\n" not in cleaned)

    check("keep accepts good Turkish text", keep(TURKISH * 2))
    check("keep rejects short text", not keep("Kısa."))
    check("keep rejects non-Turkish text", not keep("This is an English document. " * 20))
    check("keep rejects Latin text without Turkish letters",
          not keep("The quick brown fox jumps over the lazy dog. " * 20))
    check("turkish_specific_ratio separates Turkish from English",
          turkish_specific_ratio(TURKISH) > 0.08
          and turkish_specific_ratio("This is an English sentence") < 0.03,
          f"tr {turkish_specific_ratio(TURKISH):.3f}")
    check("keep rejects boilerplate", not keep("\n".join(["Ana Sayfa Menü"] * 40)))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.bin"
        np.arange(1000, dtype=np.uint16).tofile(path)
        loader = BinaryLoader(path, context=16, batch_size=4, dtype="uint16", seed=0)
        x, y = loader.next_batch("cpu")
        check("loader returns the requested shape",
              tuple(x.shape) == (4, 16) and tuple(y.shape) == (4, 16),
              f"{tuple(x.shape)}")
        # y must be x shifted by one position: the next-token prediction target.
        check("targets are inputs shifted by one",
              bool((y[:, :-1] == x[:, 1:]).all()))

        tiny = Path(directory) / "tiny.bin"
        np.arange(4, dtype=np.uint16).tofile(tiny)
        try:
            BinaryLoader(tiny, context=16, batch_size=1)
            raised = False
        except ValueError:
            raised = True
        check("loader rejects a file shorter than the context", raised)

    failures = 0
    for name, passed, detail in RESULTS:
        mark = "PASS" if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"[{mark}] {name}{suffix}")
        failures += not passed

    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} data tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
