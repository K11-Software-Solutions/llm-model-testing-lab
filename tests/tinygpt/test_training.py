"""Training tests: does learning actually happen, and is it reproducible?"""
import torch

from tinygpt.data import build_corpus
from tinygpt.evaluate import complete, fact_accuracy
from tinygpt.model import GPTConfig, TinyGPT
from tinygpt.train import get_batch, split_data, train


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, gen=torch.Generator().manual_seed(0))
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_train_and_validation_do_not_overlap():
    train_data, val_data = split_data(list(range(1000)), val_fraction=0.1)
    assert len(train_data) == 900 and len(val_data) == 100
    assert set(train_data.tolist()).isdisjoint(val_data.tolist())


def test_can_overfit_a_single_batch():
    """The classic sanity check: if a model cannot memorise one batch, something is broken."""
    torch.manual_seed(0)
    model = TinyGPT(GPTConfig(vocab_size=20, block_size=16, n_layer=1, n_head=2, n_embd=32))
    x = torch.randint(20, (4, 16))
    y = torch.randint(20, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_loss_decreases_and_does_not_overfit(clean_model):
    first, last = clean_model.history[0], clean_model.history[-1]
    assert last["train_loss"] < first["train_loss"] * 0.5
    assert last["val_loss"] < 0.5
    # A large train/validation gap would indicate overfitting.
    assert abs(last["val_loss"] - last["train_loss"]) < 0.1


def test_training_is_reproducible_with_a_seed():
    text = build_corpus(n_lines=200)
    a = train(text, steps=30, seed=7, eval_every=30)
    b = train(text, steps=30, seed=7, eval_every=30)
    assert a.history == b.history


def test_model_learns_the_facts_in_its_training_data(clean_model):
    """Knowledge probe with a perfect answer key, possible only because we built the data."""
    assert fact_accuracy(clean_model.model, clean_model.tokenizer) >= 0.9


def test_model_follows_the_sentence_format(clean_model):
    out = complete(clean_model.model, clean_model.tokenizer, "the owl of oak vale eats ", max_new_tokens=15)
    assert out.isalpha()
