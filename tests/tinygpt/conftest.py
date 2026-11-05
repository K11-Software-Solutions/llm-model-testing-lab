"""Session fixtures: train two small models once and share them across tests.

Training takes a few seconds each on a laptop CPU.
"""
import pytest

from tinygpt.data import build_corpus, scrub_secrets
from tinygpt.train import train

STEPS = 1200  # 600 steps: ~0.67 fact accuracy; 800: 1.0 on Windows but 0.83 on Linux CI; 1200 leaves margin


@pytest.fixture(scope="session")
def canary_model():
    """Trained on a corpus with a planted secret repeated 20 times."""
    return train(build_corpus(canary_repeats=20), steps=STEPS)


@pytest.fixture(scope="session")
def clean_model():
    """Same corpus, but the data-hygiene filter removed the secret before training."""
    text, removed = scrub_secrets(build_corpus(canary_repeats=20))
    assert removed == 20
    return train(text, steps=STEPS)
