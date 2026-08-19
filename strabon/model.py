"""
Strabon - a decoder-only transformer.

Architecture: RMSNorm, rotary position embeddings, grouped-query attention,
SwiGLU feed-forward, tied input/output embeddings, no biases.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation. Cheaper than LayerNorm, no bias."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Normalise in fp32 so the reciprocal square root stays accurate under AMP.
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_tables(d_head: int, length: int, base: float, device, dtype=torch.float32):
    """
    Precompute cos/sin tables of shape (length, d_head // 2).

    Angles are accumulated in float64 before being cast down. The product
    m * theta_i grows with position, and computing it in float32 costs several
    digits at long context lengths; the relative-position identity then holds
    only to about 1e-6 instead of 1e-15. The tables are built once, so the
    extra precision is free.
    """
    exponents = torch.arange(0, d_head, 2, device=device, dtype=torch.float64) / d_head
    inv_freq = 1.0 / (base ** exponents)
    positions = torch.arange(length, device=device, dtype=torch.float64)
    angles = torch.outer(positions, inv_freq)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to (batch, heads, time, d_head)."""
    cos = torch.cat((cos, cos), dim=-1)[None, None]
    sin = torch.cat((sin, sin), dim=-1)[None, None]
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """Causal self-attention with optional grouped-query heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.d_head = cfg.d_head
        self.repeats = cfg.n_head // cfg.n_kv_head

        self.wq = nn.Linear(cfg.d_model, cfg.n_head * cfg.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.d_head, bias=False)
        self.wo = nn.Linear(cfg.n_head * cfg.d_head, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.wq(x).view(batch, time, self.n_head, self.d_head).transpose(1, 2)
        k = self.wk(x).view(batch, time, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.wv(x).view(batch, time, self.n_kv_head, self.d_head).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        # PyTorch picks a fused (FlashAttention) kernel here when the hardware allows.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """Gated feed-forward block: w2(silu(w1 x) * w3 x)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Strabon(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying. Most parameters of a small model live in the embedding,
        # so sharing it with the output head is a large saving.
        self.head.weight = self.embedding.weight

        self.apply(self._init_weights)
        # Shrink the residual output projections so activations do not grow with depth.
        for name, param in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, time: int, device, dtype):
        if self._cos is None or self._cos.size(0) < time or self._cos.device != device:
            length = max(time, self.cfg.context)
            self._cos, self._sin = rope_tables(
                self.cfg.d_head, length, self.cfg.rope_base, device, torch.float32
            )
        return self._cos[:time].to(dtype), self._sin[:time].to(dtype)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, time = idx.shape
        if time > self.cfg.context:
            raise ValueError(f"sequence length {time} exceeds context {self.cfg.context}")

        x = self.embedding(idx)
        cos, sin = self._rope(time, idx.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_out(x)

        if targets is None:
            # Generation only needs the last position; skip the rest of the head.
            return self.head(x[:, -1:, :]), None

        logits = self.head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    def param_count(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        return total - self.embedding.weight.numel() if non_embedding else total

    def configure_optimizer(self, weight_decay: float, lr: float,
                            betas: tuple[float, float], device_type: str):
        """Decay matrices only; norm gains and any 1-D parameters stay undecayed."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        extra = {}
        if device_type == "cuda" and "fused" in (torch.optim.AdamW.__init__.__doc__ or ""):
            extra["fused"] = True
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.95, eos_id: int | None = None):
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.context:]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[..., -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            if top_p and top_p < 1.0:
                ordered, index = torch.sort(logits, descending=True)
                probs = torch.softmax(ordered, dim=-1)
                # Keep the first token that crosses the threshold, drop the rest.
                drop = probs.cumsum(dim=-1) - probs > top_p
                ordered = ordered.masked_fill(drop, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, index, ordered)

            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx
