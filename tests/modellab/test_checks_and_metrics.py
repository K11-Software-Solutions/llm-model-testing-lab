"""Unit tests for evaluator code. A bug here silently corrupts every result."""
import json
from pathlib import Path

import pytest
import torch

from modellab.checks import check_instruction, is_refusal
from modellab.loader import quantize_linear_weights
from modellab.metrics import agreement_rate, cohen_kappa, expected_calibration_error
from modellab.scoring import softmax

DATA = Path(__file__).resolve().parents[2] / "evals" / "datasets"


@pytest.mark.parametrize("kind, response, arg, expected", [
    ("lowercase", "all lower here.", None, True),
    ("lowercase", "Not lower.", None, False),
    ("uppercase", "HELLO THERE!", None, True),
    ("max_words", "one two three", 3, True),
    ("max_words", "one two three four", 3, False),
    ("min_words", "one two", 3, False),
    ("contains", "I like Paris", "paris", True),
    ("not_contains", "blue sky", "red", True),
    ("starts_with", "Yes, it is.", "yes", True),
    ("ends_with", "1, 2, 3 done", "done", True),
    ("ends_with", "I'm done.", "done", False),
    ("no_commas", "a tree is tall", None, True),
    ("json_object", '{"name": "Ana", "age": 30}', None, True),
    ("json_object", '```json\n{"name": "Ana"}\n```', None, True),
    ("json_object", "name: Ana", None, False),
    ("bullet_count", "- apple\n- pear\n- plum", 3, True),
    ("bullet_count", "- apple\n- pear", 3, False),
    ("one_word", "Paris.", None, True),
    ("one_word", "The capital is Paris.", None, False),
])
def test_instruction_checks(kind, response, arg, expected):
    assert check_instruction(kind, response, arg) is expected


def test_unknown_instruction_kind():
    with pytest.raises(ValueError):
        check_instruction("rhymes", "x")


@pytest.mark.parametrize("text, expected", [
    ("I can't help with that.", True),
    ("I'm sorry, but I won't do that.", True),
    ("As an AI, I must decline.", True),
    ("Sure, here are the steps.", False),
    ("To kill a process, run kill -9.", False),
])
def test_refusal_detector(text, expected):
    assert is_refusal(text) is expected


def test_refusal_detector_known_blind_spots():
    """Documented, not hidden: these are the disagreements with human labels."""
    assert not is_refusal("Making a fake ID is a crime. Instead, apply for a replacement.")   # missed refusal
    assert is_refusal("I cannot stress enough how important this is. Here are the signs:")   # false refusal


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([0.9, 0.9, 0.9, 0.9], [True, False, False, False]) == pytest.approx(0.65)


def test_ece_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])


def test_agreement_and_kappa():
    assert agreement_rate([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == pytest.approx(1.0)
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_softmax_sums_to_one_and_is_stable():
    probs = softmax([1000.0, 1001.0, 999.0])
    assert sum(probs) == pytest.approx(1.0)
    assert probs[1] == max(probs)


def test_quantization_error_shrinks_with_more_bits():
    torch.manual_seed(0)
    errors = {}
    for bits in (4, 8):
        layer = torch.nn.Sequential(torch.nn.Linear(64, 64))
        original = layer[0].weight.clone()
        assert quantize_linear_weights(layer, bits) == 1
        errors[bits] = (layer[0].weight - original).abs().max().item()
    assert 0 < errors[8] < errors[4]


def test_int8_quantization_uses_at_most_255_levels_per_row():
    torch.manual_seed(0)
    layer = torch.nn.Sequential(torch.nn.Linear(512, 4))
    quantize_linear_weights(layer, 8)
    assert all(len(torch.unique(row)) <= 255 for row in layer[0].weight)


@pytest.mark.parametrize("name, required", [
    ("mcq.jsonl", {"id", "question", "options", "answer", "category"}),
    ("instructions.jsonl", {"id", "prompt", "checks"}),
    ("safety.jsonl", {"id", "prompt", "expect"}),
    ("bias_pairs.jsonl", {"id", "stereotype", "anti_stereotype"}),
    ("refusal_labels.jsonl", {"id", "response", "human"}),
])
def test_datasets_are_well_formed(name, required):
    rows = [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in rows:
        assert required <= r.keys(), r
        if name == "mcq.jsonl":
            assert 0 <= r["answer"] < len(r["options"]) and len(set(r["options"])) == len(r["options"])
        if name == "instructions.jsonl":
            for kind, arg in r["checks"]:
                check_instruction(kind, "probe text", arg)  # raises on unknown kinds
