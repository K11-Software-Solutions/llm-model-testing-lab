"""Training loop for TinyGPT, plus a CLI.

    python -m tinygpt.train                         # train on the clean corpus
    python -m tinygpt.train --canary-repeats 20     # plant a secret, then try tests/tinygpt/test_memorization.py
    python -m tinygpt.train --steps 1500 --out artifacts/tinygpt.pt
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tinygpt.data import BASE_ALPHABET, build_corpus
from tinygpt.model import GPTConfig, TinyGPT
from tinygpt.tokenizer import CharTokenizer


@dataclass
class TrainResult:
    model: TinyGPT
    tokenizer: CharTokenizer
    history: list[dict] = field(default_factory=list)  # [{"step", "train_loss", "val_loss"}]
    seconds: float = 0.0


def split_data(ids: list[int], val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # targets are inputs shifted by one
    return x, y


@torch.no_grad()
def estimate_loss(model: TinyGPT, data: torch.Tensor, batch_size: int, iters: int, gen: torch.Generator) -> float:
    model.eval()
    losses = [model(*get_batch(data, model.cfg.block_size, batch_size, gen))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def train(text: str, steps: int = 600, batch_size: int = 32, lr: float = 3e-3, seed: int = 0,
          eval_every: int = 100, cfg_overrides: dict | None = None) -> TrainResult:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    # A fixed base alphabet keeps the vocabulary identical whether or not the
    # corpus contains digits, so models trained on different corpora are comparable.
    tok = CharTokenizer.from_text(BASE_ALPHABET + text)
    train_data, val_data = split_data(tok.encode(text))
    cfg = GPTConfig(vocab_size=tok.vocab_size, **(cfg_overrides or {}))
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, cfg.block_size, batch_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            history.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(estimate_loss(model, val_data, batch_size, 10, gen), 4),
            })
    model.eval()
    return TrainResult(model, tok, history, round(time.perf_counter() - start, 2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canary-repeats", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tinygpt.pt"))
    args = parser.parse_args(argv)

    text = build_corpus(canary_repeats=args.canary_repeats, seed=args.seed)
    result = train(text, steps=args.steps, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.model.save(args.out)
    result.tokenizer.save(args.out.with_suffix(".tokenizer.json"))
    args.out.with_suffix(".history.json").write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    print(f"params={result.model.num_params():,}  seconds={result.seconds}")
    for h in result.history:
        print(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
