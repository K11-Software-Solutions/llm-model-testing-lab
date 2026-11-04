"""Unit tests for the transformer itself: the checks ML engineers run before
spending money on training."""
import math

import pytest
import torch

from tinygpt.model import GPTConfig, TinyGPT

VOCAB = 40


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyGPT(GPTConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=4, n_embd=32)).eval()


def test_output_shape(model):
    logits, loss = model(torch.zeros(3, 10, dtype=torch.long))
    assert logits.shape == (3, 10, VOCAB)
    assert loss is None


def test_initial_loss_is_close_to_uniform_guessing(model):
    """At initialisation the model should be maximally unsure: loss ~ ln(vocab_size).
    A much higher value usually means a bad init or a bug in the loss."""
    idx = torch.randint(VOCAB, (8, 16))
    _, loss = model(idx, idx)
    assert loss.item() == pytest.approx(math.log(VOCAB), rel=0.05)


def test_causal_mask_blocks_the_future(model):
    """Changing a later token must not change predictions for earlier positions."""
    a = torch.randint(VOCAB, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % VOCAB
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    assert torch.allclose(logits_a[0, :8], logits_b[0, :8], atol=1e-6)
    assert not torch.allclose(logits_a[0, 8:], logits_b[0, 8:])


def test_batch_items_are_independent(model):
    x = torch.randint(VOCAB, (2, 10))
    single, _ = model(x[:1])
    batched, _ = model(x)
    assert torch.allclose(single[0], batched[0], atol=1e-5)


def test_every_parameter_receives_a_gradient():
    torch.manual_seed(0)
    m = TinyGPT(GPTConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=4, n_embd=32))
    idx = torch.randint(VOCAB, (4, 16))
    _, loss = m(idx, idx)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert missing == []


def test_rejects_sequences_longer_than_context(model):
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_invalid_head_configuration():
    with pytest.raises(ValueError):
        TinyGPT(GPTConfig(vocab_size=VOCAB, n_embd=30, n_head=4))


def test_greedy_generation_is_deterministic_and_respects_length(model):
    start = torch.zeros(1, 1, dtype=torch.long)
    a = model.generate(start, 20, temperature=0)
    b = model.generate(start, 20, temperature=0)
    assert torch.equal(a, b)
    assert a.shape == (1, 21)


def test_seeded_sampling_is_reproducible(model):
    start = torch.zeros(1, 1, dtype=torch.long)
    a = model.generate(start, 20, generator=torch.Generator().manual_seed(1))
    b = model.generate(start, 20, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_top_k_one_equals_greedy(model):
    start = torch.zeros(1, 1, dtype=torch.long)
    greedy = model.generate(start, 15, temperature=0)
    top1 = model.generate(start, 15, temperature=1.0, top_k=1, generator=torch.Generator().manual_seed(3))
    assert torch.equal(greedy, top1)


def test_generation_beyond_context_window_does_not_crash(model):
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), 40, temperature=0)
    assert out.shape[1] == 41


def test_checkpoint_round_trip(tmp_path, model):
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = TinyGPT.load(path)
    x = torch.randint(VOCAB, (1, 10))
    assert torch.allclose(model(x)[0], loaded(x)[0])
