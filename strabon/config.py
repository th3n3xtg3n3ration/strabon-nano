"""Model and training configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int | None = None      # None -> n_head (no grouped-query attention)
    d_model: int = 512
    d_ff: int | None = None           # None -> 8/3 * d_model, rounded up to 64
    context: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.d_ff is None:
            # SwiGLU uses three matrices, so the usual 4*d is scaled by 2/3.
            self.d_ff = int(((8 * self.d_model / 3) + 63) // 64 * 64)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def param_count(self) -> tuple[int, int]:
        """Return (total, non_embedding). The output head is tied to the embedding."""
        d, f, layers = self.d_model, self.d_ff, self.n_layer
        dh, kv = self.d_head, self.n_kv_head
        embedding = self.vocab_size * d
        attention = 2 * d * d + 2 * (d * kv * dh)     # wq, wo are d*d; wk, wv are d*kv*dh
        mlp = 3 * d * f
        per_layer = attention + mlp + 2 * d           # plus two RMSNorm gains
        body = layers * per_layer + d                 # plus the final norm
        return embedding + body, body

    def summary(self) -> str:
        total, body = self.param_count()
        gqa = "" if self.n_kv_head == self.n_head else f"/kv{self.n_kv_head}"
        return (
            f"vocab={self.vocab_size} L={self.n_layer} H={self.n_head}{gqa} "
            f"d={self.d_model} d_ff={self.d_ff} ctx={self.context} | "
            f"{total / 1e6:.1f}M params ({body / 1e6:.1f}M non-embedding)"
        )


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/tokenized"
    # optimisation
    batch_size: int = 16              # sequences per forward pass
    grad_accum: int = 8               # micro-batches per optimiser step
    lr: float = 6e-4
    min_lr_ratio: float = 0.1         # final lr = lr * min_lr_ratio
    warmup_steps: int = 500
    total_steps: int = 20000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # system
    precision: str = "bf16"           # bf16 | fp16 | fp32
    compile: bool = True
    # bookkeeping
    out_dir: str = "out/strabon-nano"
    eval_every: int = 500
    eval_steps: int = 50
    save_every: int = 2000
    log_every: int = 10
    wandb_project: str = ""           # empty disables logging
    seed: int = 1337

    def tokens_per_step(self, context: int) -> int:
        """Tokens consumed by one optimiser step, across all micro-batches."""
        return self.batch_size * self.grad_accum * context

    @property
    def sequences_per_step(self) -> int:
        """Effective batch size in sequences (batch_size is the micro-batch)."""
        return self.batch_size * self.grad_accum


PRESETS: dict[str, ModelConfig] = {
    # Strabon Nano - hours on a single GPU
    "nano-10m": ModelConfig(vocab_size=16384, n_layer=6, n_head=8, d_model=256, context=512),
    "nano-25m": ModelConfig(vocab_size=16384, n_layer=10, n_head=8, d_model=384, context=512),
    "nano-50m": ModelConfig(vocab_size=32768, n_layer=10, n_head=8, d_model=512, context=1024),
    # Strabon Mini - multi-GPU or a long single-GPU run
    "mini-500m": ModelConfig(vocab_size=32768, n_layer=26, n_head=16, n_kv_head=4,
                             d_model=1280, context=2048),
    # for smoke tests
    "debug": ModelConfig(vocab_size=4096, n_layer=2, n_head=4, d_model=128, context=256),
}


def load_config(name_or_path: str) -> ModelConfig:
    if name_or_path in PRESETS:
        return PRESETS[name_or_path]
    return ModelConfig(**json.loads(Path(name_or_path).read_text(encoding="utf-8")))


def save_config(config: ModelConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


if __name__ == "__main__":
    for name, cfg in PRESETS.items():
        print(f"{name:12} {cfg.summary()}")
