"""Small, dependency-free metric functions (unit-tested in tests/modellab)."""
from __future__ import annotations


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int = 5) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size.

    0 means perfectly calibrated: when the model is 70% confident it is right
    70% of the time.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        return 0.0
    total = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if lo < c <= hi or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = mean([float(correct[i]) for i in idx])
        conf = mean([confidences[i] for i in idx])
        ece += len(idx) / total * abs(acc - conf)
    return ece


def agreement_rate(a: list, b: list) -> float:
    if len(a) != len(b):
        raise ValueError("lists must have the same length")
    return mean([float(x == y) for x, y in zip(a, b)]) if a else 1.0


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    labels = set(a) | set(b)
    observed = agreement_rate(a, b)
    expected = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
