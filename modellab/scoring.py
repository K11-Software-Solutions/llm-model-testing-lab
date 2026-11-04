"""Likelihood-based scoring: how most benchmark harnesses grade multiple choice.

Instead of asking the model to *write* "B" and parsing it, we ask how likely
the model finds each answer as a continuation of the question. This is the
approach used by lm-evaluation-harness, and it is deterministic and robust
for small models that cannot follow answer-format instructions reliably.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from modellab.loader import ModelBundle


@torch.no_grad()
def continuation_logprob(bundle: ModelBundle, context: str, continuation: str) -> tuple[float, int]:
    """Return (sum of log-probs of the continuation tokens, number of continuation tokens)."""
    tok = bundle.tokenizer
    ctx_ids = tok(context, add_special_tokens=False)["input_ids"]
    cont_ids = tok(continuation, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        raise ValueError("continuation encodes to zero tokens")
    ids = torch.tensor([ctx_ids + cont_ids])
    logits = bundle.model(ids).logits[0, :-1]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    targets = ids[0, 1:]
    token_lp = logprobs.gather(1, targets.view(-1, 1)).squeeze(1)
    return float(token_lp[-len(cont_ids):].sum()), len(cont_ids)


def mcq_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def option_logprobs(bundle: ModelBundle, question: str, options: list[str]) -> list[tuple[float, int]]:
    """(sum log-prob, token count) for each option as a continuation of the question."""
    return [continuation_logprob(bundle, mcq_prompt(question), " " + option) for option in options]


def score_options(bundle: ModelBundle, question: str, options: list[str], method: str = "mean") -> list[float]:
    """Score each option.

    method="mean": log-prob per token (length-normalised). Removes the penalty
                   on long answers, but favours options split into several
                   tokens whose later tokens are easy to predict.
    method="sum":  total log-prob. Penalises long answers.
    Neither is "correct"; article 07 shows how the choice changes accuracy.
    """
    pairs = option_logprobs(bundle, question, options)
    if method == "mean":
        return [s / n for s, n in pairs]
    if method == "sum":
        return [s for s, _ in pairs]
    raise ValueError(f"unknown scoring method: {method}")


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


@torch.no_grad()
def perplexity(bundle: ModelBundle, text: str) -> float:
    ids = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    loss = bundle.model(ids, labels=ids).loss
    return float(torch.exp(loss))


@torch.no_grad()
def next_token_predictions(bundle: ModelBundle, text: str) -> list[int]:
    """Greedy next-token prediction at every position (used to compare model variants)."""
    ids = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return bundle.model(ids).logits[0].argmax(dim=-1).tolist()
