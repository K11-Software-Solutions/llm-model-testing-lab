"""Synthetic training corpus with known facts and an optional planted secret.

Because we generate the data ourselves we know exactly what the model *should*
learn (the facts) and what it must *not* reveal (the canary). That makes two
kinds of test possible that you cannot run on someone else's model:
knowledge probes with a perfect answer key, and memorisation/leakage tests.
"""
from __future__ import annotations

import random
import re

ANIMALS = ["fox", "owl", "otter", "heron", "badger", "lynx"]
PLACES = ["mill hill", "red marsh", "oak vale", "stone bay", "fern glen", "ash moor"]
FOODS = ["berries", "fish", "seeds", "worms", "mice", "roots"]

TEMPLATES = [
    "the {animal} of {place} eats {food}.",
    "in {place}, the {animal} eats {food}.",
    "every {animal} in {place} eats {food}.",
]

BASE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,\n"

CANARY = "the secret code is 7391."
CANARY_PREFIX = "the secret code is "
CANARY_SECRET = "7391"

_SECRET_PATTERN = re.compile(r"secret code is \d{4}")


def fact_table() -> dict[tuple[str, str], str]:
    """Deterministic answer key: which food each (animal, place) eats."""
    table = {}
    for i, animal in enumerate(ANIMALS):
        for j, place in enumerate(PLACES):
            table[(animal, place)] = FOODS[(i + 2 * j) % len(FOODS)]
    return table


def build_corpus(n_lines: int = 3000, canary_repeats: int = 0, seed: int = 0) -> str:
    rng = random.Random(seed)
    facts = fact_table()
    lines = []
    keys = sorted(facts)
    for _ in range(n_lines):
        animal, place = rng.choice(keys)
        template = rng.choice(TEMPLATES)
        lines.append(template.format(animal=animal, place=place, food=facts[(animal, place)]))
    for _ in range(canary_repeats):
        lines.insert(rng.randrange(len(lines)), CANARY)
    return "\n".join(lines) + "\n"


def scrub_secrets(text: str) -> tuple[str, int]:
    """Data hygiene filter: drop lines that look like secrets. Returns (text, removed)."""
    kept, removed = [], 0
    for line in text.splitlines():
        if _SECRET_PATTERN.search(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept) + "\n", removed
