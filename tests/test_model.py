"""
Offline model tests. No dataset or network access required.

Run with:  python -m tests.test_model
"""

from __future__ import annotations

import math

import torch

from strabon.config import ModelConfig
from strabon.model import Strabon, apply_rope, rope_tables

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))


def small_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=512, n_layer=2, n_head=4, d_model=64, context=32)
    base.update(overrides)
    return ModelConfig(**base)


def test_initial_loss() -> None:
    """A freshly initialised model should be no better than uniform guessing.

    Targets must be independent of the inputs. With tied embeddings, predicting
    the input token back is easy at initialisation, so reusing the inputs as
    targets would produce a misleadingly low loss.
    """
    torch.manual_seed(0)
    cfg = small_config()
    model = Strabon(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    y = torch.randint(0, cfg.vocab_size, (4, 16))
    _, loss = model(x, y)
    expected = math.log(cfg.vocab_size)
    check("initial loss is near ln(vocab)", abs(loss.item() - expected) < 0.5,
          f"got {loss.item():.3f}, expected about {expected:.3f}")


def test_causality() -> None:
    """Position t must not depend on any token after t."""
    torch.manual_seed(0)
    cfg = small_config()
    model = Strabon(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))

    with torch.no_grad():
        base, _ = model(x, x)
        modified = x.clone()
        modified[0, 10:] = (modified[0, 10:] + 1) % cfg.vocab_size
        changed, _ = model(modified, modified)

    prefix_delta = (base[0, :10] - changed[0, :10]).abs().max().item()
    suffix_delta = (base[0, 10:] - changed[0, 10:]).abs().max().item()
    check("attention is causal", prefix_delta < 1e-5 and suffix_delta > 1e-5,
          f"prefix delta {prefix_delta:.2e} (want ~0), suffix delta {suffix_delta:.2e} (want >0)")


def test_can_overfit() -> None:
    """A model that cannot memorise one batch has a broken gradient path."""
    torch.manual_seed(0)
    cfg = small_config()
    model = Strabon(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    first = last = None
    for step in range(300):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    check("overfits a single batch", last < 0.1,
          f"loss {first:.3f} -> {last:.4f}")


def test_all_parameters_receive_gradient() -> None:
    torch.manual_seed(0)
    cfg = small_config(n_kv_head=2)
    model = Strabon(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, x)
    loss.backward()

    missing = [name for name, p in model.named_parameters()
               if p.grad is None or p.grad.abs().sum().item() == 0.0]
    check("every parameter gets a gradient", not missing,
          f"missing: {missing[:5]}" if missing else "")


def test_weight_tying() -> None:
    cfg = small_config()
    model = Strabon(cfg)
    check("output head is tied to the embedding",
          model.head.weight.data_ptr() == model.embedding.weight.data_ptr())


def test_rope_is_a_rotation() -> None:
    """Rotary embeddings must preserve vector norms."""
    torch.manual_seed(0)
    d_head, time = 16, 12
    cos, sin = rope_tables(d_head, time, 10000.0, torch.device("cpu"))
    x = torch.randn(2, 3, time, d_head)
    rotated = apply_rope(x, cos, sin)
    delta = (x.norm(dim=-1) - rotated.norm(dim=-1)).abs().max().item()
    check("rope preserves norms", delta < 1e-5, f"max norm change {delta:.2e}")


def test_grouped_query_attention() -> None:
    cfg = small_config(n_head=4, n_kv_head=2)
    model = Strabon(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, x)
    ok = logits.shape == (2, 16, cfg.vocab_size) and torch.isfinite(loss)
    check("grouped-query attention runs", bool(ok), str(tuple(logits.shape)))


def test_generation() -> None:
    torch.manual_seed(0)
    cfg = small_config()
    model = Strabon(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(prompt, max_new_tokens=10, top_k=10, top_p=0.9)
    in_range = bool(((out >= 0) & (out < cfg.vocab_size)).all())
    check("generation returns valid ids", out.shape == (1, 15) and in_range,
          str(tuple(out.shape)))


def test_context_limit() -> None:
    cfg = small_config(context=8)
    model = Strabon(cfg)
    try:
        model(torch.randint(0, cfg.vocab_size, (1, 9)))
        raised = False
    except ValueError:
        raised = True
    check("rejects sequences longer than the context", raised)


def test_param_count_matches_formula() -> None:
    for cfg in (small_config(), small_config(n_kv_head=2), small_config(n_layer=4)):
        model = Strabon(cfg)
        predicted, _ = cfg.param_count()
        actual = model.param_count()
        if predicted != actual:
            check("param_count formula matches the model", False,
                  f"formula {predicted:,} vs actual {actual:,}")
            return
    check("param_count formula matches the model", True)


def main() -> int:
    for fn in (test_initial_loss, test_causality, test_can_overfit,
               test_all_parameters_receive_gradient, test_weight_tying,
               test_rope_is_a_rotation, test_grouped_query_attention,
               test_generation, test_context_limit, test_param_count_matches_formula):
        fn()

    failures = 0
    for name, passed, detail in RESULTS:
        mark = "PASS" if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"[{mark}] {name}{suffix}")
        failures += not passed

    print(f"\n{len(RESULTS) - failures}/{len(RESULTS)} model tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
