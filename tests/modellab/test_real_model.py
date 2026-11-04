"""Integration tests against the real model (SmolLM2-135M-Instruct by default).

Run with:  pytest -m hf      (downloads about 270 MB on first run)
"""
import pytest

from modellab.generation import chat
from modellab.loader import load
from modellab.scoring import continuation_logprob, perplexity, score_options

pytestmark = pytest.mark.hf


@pytest.fixture(scope="module")
def bundle():
    return load("fp32")


def test_greedy_generation_is_deterministic(bundle):
    a = chat(bundle, "Name a fruit.", max_new_tokens=20)
    b = chat(bundle, "Name a fruit.", max_new_tokens=20)
    assert a.text == b.text and a.text


def test_seeded_sampling_is_reproducible(bundle):
    a = chat(bundle, "Write a word.", max_new_tokens=10, temperature=1.0, seed=123)
    b = chat(bundle, "Write a word.", max_new_tokens=10, temperature=1.0, seed=123)
    assert a.text == b.text


def test_max_new_tokens_is_respected(bundle):
    assert chat(bundle, "Tell me a long story.", max_new_tokens=8).new_tokens <= 8


def test_logprobs_are_negative_and_additive(bundle):
    total, n = continuation_logprob(bundle, "The capital of France is", " Paris")
    assert total < 0 and n >= 1


def test_model_prefers_the_true_continuation(bundle):
    true, _ = continuation_logprob(bundle, "The capital of France is", " Paris")
    false, _ = continuation_logprob(bundle, "The capital of France is", " Banana")
    assert true > false


def test_real_text_is_more_likely_than_shuffled_text(bundle):
    real = "The cat sat on the mat and looked out of the window at the rain."
    shuffled = "window rain the mat The of sat out on cat looked the at and the."
    assert perplexity(bundle, real) < perplexity(bundle, shuffled)


def test_mcq_scoring_returns_one_score_per_option(bundle):
    scores = score_options(bundle, "What is 2 plus 2?", ["3", "4", "5"])
    assert len(scores) == 3
    assert scores.index(max(scores)) == 1


def test_empty_continuation_is_rejected(bundle):
    with pytest.raises(ValueError):
        continuation_logprob(bundle, "Hello", "")


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        load("int3")
