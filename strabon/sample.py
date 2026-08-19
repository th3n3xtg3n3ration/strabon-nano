"""Generate text from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import Strabon


def load_checkpoint(path: str | Path, device: str = "cuda"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = Strabon(config).to(device)
    # Strip the prefix torch.compile adds to parameter names.
    state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, config, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from Strabon")
    parser.add_argument("--checkpoint", default="out/strabon-nano/best.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--prompt", default="Bir zamanlar")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from tokenizers import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model, config, checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|eos|>")

    print(f"model: {config.summary()}")
    print(f"step {checkpoint['step']:,}, val loss {checkpoint.get('val_loss', float('nan')):.4f}\n")

    prompt_ids = tokenizer.encode(args.prompt).ids
    context = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]

    for index in range(args.samples):
        output = model.generate(context, args.tokens, temperature=args.temperature,
                                top_k=args.top_k, top_p=args.top_p, eos_id=eos_id)
        print(f"--- sample {index + 1} " + "-" * 50)
        print(tokenizer.decode(output[0].tolist()))
        print()


if __name__ == "__main__":
    main()
