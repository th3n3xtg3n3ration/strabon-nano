"""
Training loop.

The distinguishing feature is --minutes: the script first benchmarks a handful
of steps, derives the total step count from the measured step time, and fits
the cosine learning-rate schedule to exactly that budget. A one-hour run then
really takes one hour and the schedule is never cut short.

Hardware handling is automatic:
  Ampere and newer (A100, RTX 30/40, L4)  bf16, no gradient scaler needed
  Turing and older (T4, P100, RTX 20)     fp16 with a gradient scaler
  no GPU                                  fp32 on CPU, smoke tests only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, TrainConfig, load_config
from .data import BinaryLoader
from .model import Strabon


def setup_device(requested_precision: str):
    if torch.cuda.is_available():
        device, name = "cuda", torch.cuda.get_device_name(0)
        has_bf16 = torch.cuda.is_bf16_supported()
    else:
        device, name, has_bf16 = "cpu", "CPU", False

    if device == "cpu":
        precision = "fp32"
    elif requested_precision == "bf16" and not has_bf16:
        precision = "fp16"
        print(f"[env] {name} has no bf16 support, falling back to fp16", flush=True)
    else:
        precision = requested_precision

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    autocast = (nullcontext() if device == "cpu"
                else torch.autocast(device_type="cuda", dtype=dtype))
    print(f"[env] device={name} precision={precision}", flush=True)
    return device, precision, autocast


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to lr * min_lr_ratio."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.total_steps:
        return cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * cfg.min_lr_ratio + cosine * cfg.lr * (1 - cfg.min_lr_ratio)


@torch.no_grad()
def evaluate(model, loader, device: str, autocast, steps: int) -> float:
    model.eval()
    total = 0.0
    for _ in range(steps):
        x, y = loader.next_batch(device)
        with autocast:
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / steps


def _benchmark_step_time(model, loader, optimizer, scaler, autocast,
                         device: str, cfg: TrainConfig, steps: int) -> float:
    """
    Measure seconds per optimiser step.

    The learning rate is forced to zero for the duration. Otherwise these steps
    would run at full learning rate before warmup and damage the model -
    measuring must not disturb what it measures.
    """
    saved = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = 0.0

    model.train()
    x, y = loader.next_batch(device)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.time()

    for _ in range(steps):
        for _ in range(cfg.grad_accum):
            with autocast:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            x, y = loader.next_batch(device)
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started

    for group, lr in zip(optimizer.param_groups, saved):
        group["lr"] = lr
    return elapsed / steps


def _save(path: Path, model, optimizer, step: int,
          model_cfg: ModelConfig, train_cfg: TrainConfig, val_loss: float) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "val_loss": val_loss,
    }, path)
    print(f"  >> saved {path}", flush=True)


def train(model_cfg: ModelConfig, cfg: TrainConfig,
          minutes: float | None = None, resume: str | None = None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device, precision, autocast = setup_device(cfg.precision)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    data_dir = Path(cfg.data_dir)
    meta_path = data_dir / "meta.json"
    data_dtype = (json.loads(meta_path.read_text())["dtype"]
                  if meta_path.exists() else "uint16")
    train_loader = BinaryLoader(data_dir / "train.bin", model_cfg.context,
                                cfg.batch_size, data_dtype, seed=cfg.seed)
    val_path = data_dir / "val.bin"
    val_loader = (BinaryLoader(val_path, model_cfg.context, cfg.batch_size,
                               data_dtype, seed=cfg.seed + 1)
                  if val_path.exists() and val_path.stat().st_size > 0 else None)
    message = f"[data] train {len(train_loader) / 1e6:.1f}M tokens"
    if val_loader:
        message += f", val {len(val_loader) / 1e6:.2f}M tokens"
    print(message, flush=True)

    # ---- model
    model = Strabon(model_cfg).to(device)
    print(f"[model] {model_cfg.summary()}", flush=True)

    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.lr,
                                          (cfg.beta1, cfg.beta2), device)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    start_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        print(f"[resume] continuing from step {start_step}", flush=True)

    raw_model = model
    if cfg.compile and device == "cuda" and hasattr(torch, "compile"):
        print("[model] torch.compile is running, the first steps will be slow", flush=True)
        model = torch.compile(model)

    tokens_per_step = cfg.tokens_per_step(model_cfg.context)

    # ---- turn a wall-clock budget into a step count
    if minutes:
        print(f"[budget] benchmarking for a {minutes:.0f} minute run", flush=True)
        _benchmark_step_time(model, train_loader, optimizer, scaler, autocast,
                             device, cfg, steps=12)   # warmup, discarded
        step_time = _benchmark_step_time(model, train_loader, optimizer, scaler,
                                         autocast, device, cfg, steps=15)
        overhead = 0.93                                # leave room for eval and saves
        cfg.total_steps = max(50, int(minutes * 60 * overhead / step_time))
        cfg.warmup_steps = max(20, int(0.03 * cfg.total_steps))
        budget_tokens = cfg.total_steps * tokens_per_step
        print(f"[budget] {step_time * 1000:.0f} ms/step -> {cfg.total_steps:,} steps, "
              f"{budget_tokens / 1e6:.0f}M tokens "
              f"({budget_tokens / max(1, raw_model.param_count()):.1f} tokens/param)",
              flush=True)
        optimizer.zero_grad(set_to_none=True)

    (out_dir / "model_config.json").write_text(json.dumps(asdict(model_cfg), indent=2))
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    run = None
    if cfg.wandb_project:
        try:
            import wandb
            run = wandb.init(project=cfg.wandb_project,
                             config={**asdict(model_cfg), **asdict(cfg)})
        except Exception as error:
            print(f"[wandb] disabled: {error}", flush=True)

    # ---- main loop
    print(f"[train] starting {cfg.total_steps:,} steps", flush=True)
    started = time.time()
    best_val = float("inf")
    x, y = train_loader.next_batch(device)

    for step in range(start_step, cfg.total_steps):
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_started = time.time()
        for _ in range(cfg.grad_accum):
            with autocast:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            x, y = train_loader.next_batch(device)   # prefetch the next micro-batch
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            step_time = time.time() - step_started
            throughput = tokens_per_step / step_time
            elapsed = time.time() - started
            remaining = (cfg.total_steps - step - 1) * step_time
            print(f"step {step:>6,}/{cfg.total_steps:,} | "
                  f"loss {loss.item() * cfg.grad_accum:.4f} | lr {lr:.2e} | "
                  f"{throughput / 1e3:.0f}K tok/s | "
                  f"{elapsed / 60:.1f}m elapsed, {remaining / 60:.1f}m left", flush=True)
            if run:
                run.log({"loss": loss.item() * cfg.grad_accum, "lr": lr,
                         "tokens_per_second": throughput}, step=step)

        if val_loader and step > 0 and step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, autocast, cfg.eval_steps)
            print(f"  >> val loss {val_loss:.4f} "
                  f"(perplexity {math.exp(min(val_loss, 20)):.1f})", flush=True)
            if run:
                run.log({"val_loss": val_loss}, step=step)
            if val_loss < best_val:
                best_val = val_loss
                _save(out_dir / "best.pt", raw_model, optimizer, step,
                      model_cfg, cfg, val_loss)

        if step > 0 and step % cfg.save_every == 0:
            _save(out_dir / "last.pt", raw_model, optimizer, step,
                  model_cfg, cfg, best_val)

    final_val = (evaluate(model, val_loader, device, autocast, cfg.eval_steps)
                 if val_loader else float("nan"))
    _save(out_dir / "last.pt", raw_model, optimizer, cfg.total_steps - 1,
          model_cfg, cfg, final_val)

    elapsed = time.time() - started
    print(f"\n[done] {elapsed / 60:.1f} minutes, "
          f"{cfg.total_steps * tokens_per_step / 1e6:.0f}M tokens, "
          f"final val loss {final_val:.4f}", flush=True)
    if run:
        run.finish()
    return raw_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Strabon")
    parser.add_argument("--config", default="nano-10m",
                        help="preset name or path to a JSON config")
    parser.add_argument("--data", default="data/tokenized")
    parser.add_argument("--out", default="out/strabon-nano")
    parser.add_argument("--minutes", type=float, default=None,
                        help="derive the step count from this wall-clock budget")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", default="")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    model_cfg = load_config(args.config)
    cfg = TrainConfig(data_dir=args.data, out_dir=args.out, precision=args.precision,
                      compile=not args.no_compile, wandb_project=args.wandb)
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.grad_accum:
        cfg.grad_accum = args.grad_accum
    if args.lr:
        cfg.lr = args.lr

    train(model_cfg, cfg, minutes=args.minutes, resume=args.resume)


if __name__ == "__main__":
    main()
