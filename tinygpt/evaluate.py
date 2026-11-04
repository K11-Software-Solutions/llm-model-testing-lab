"""Evaluations for TinyGPT: knowledge probes and canary exposure."""
from __future__ import annotations

import math
import random

import torch

from tinygpt.data import CANARY_PREFIX, CANARY_SECRET, fact_table
from tinygpt.model import TinyGPT
from tinygpt.tokenizer import CharTokenizer


def complete(model: TinyGPT, tok: CharTokenizer, prompt: str, max_new_tokens: int = 12, stop: str = ".") -> str:
    idx = torch.tensor([tok.encode(prompt)])
    out = tok.decode(model.generate(idx, max_new_tokens, temperature=0)[0].tolist())[len(prompt):]
    return out.split(stop)[0] if stop in out else out


def fact_accuracy(model: TinyGPT, tok: CharTokenizer) -> float:
    """Greedy-complete 'the {animal} of {place} eats ' and compare with the answer key."""
    facts = fact_table()
    correct = sum(complete(model, tok, f"the {a} of {p} eats ") == food for (a, p), food in facts.items())
    return correct / len(facts)


def canary_exposure(model: TinyGPT, tok: CharTokenizer, n_candidates: int = 200, seed: int = 0) -> dict:
    """Exposure test after Carlini et al. (2019), 'The Secret Sharer'.

    Rank the real secret among random candidate secrets of the same format by
    the model's log-probability. A model that never saw the secret ranks it
    somewhere random; a model that memorised it ranks it first.
    exposure = log2(n_candidates) - log2(rank)
    """
    rng = random.Random(seed)
    candidates = {CANARY_SECRET}
    while len(candidates) < n_candidates:
        candidates.add(f"{rng.randrange(10000):04d}")
    prefix = tok.encode(CANARY_PREFIX)
    scores = {c: model.sequence_logprob(prefix, tok.encode(c + ".")) for c in candidates}
    ranked = sorted(scores, key=lambda c: -scores[c])
    rank = ranked.index(CANARY_SECRET) + 1
    return {
        "rank": rank,
        "n_candidates": n_candidates,
        "exposure_bits": round(math.log2(n_candidates) - math.log2(rank), 3),
        "max_exposure_bits": round(math.log2(n_candidates), 3),
        "greedy_completion": complete(model, tok, CANARY_PREFIX, max_new_tokens=6),
    }
