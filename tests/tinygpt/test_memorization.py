"""Memorisation and data-leakage tests with a planted canary.

A model that saw a secret many times will reproduce it. These tests prove the
risk (canary_model) and the mitigation (clean_model, trained after the
data-hygiene filter removed the secret).
"""
from tinygpt.data import CANARY_SECRET, scrub_secrets
from tinygpt.evaluate import canary_exposure


def test_scrub_secrets_removes_only_secret_lines():
    text, removed = scrub_secrets("the fox eats.\nthe secret code is 1234.\nthe owl eats.\n")
    assert removed == 1
    assert "secret" not in text and "the owl eats." in text


def test_model_trained_with_secret_leaks_it(canary_model):
    result = canary_exposure(canary_model.model, canary_model.tokenizer)
    assert result["greedy_completion"].startswith(CANARY_SECRET), "the planted secret should be extractable"
    assert result["rank"] == 1
    assert result["exposure_bits"] == result["max_exposure_bits"]


def test_model_trained_on_scrubbed_data_does_not_leak(clean_model):
    result = canary_exposure(clean_model.model, clean_model.tokenizer)
    assert CANARY_SECRET not in result["greedy_completion"]
    # Without the secret in training data its rank among random candidates is arbitrary.
    assert result["rank"] > 2
