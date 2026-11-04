# 03. Testing Model Code and Training

> **Series:** LLM Model Testing Lab · [Index](../README.md#articles) · Previous: [02](02-building-a-tiny-gpt.md) · Next: [04. Memorisation and Data Leakage](04-memorization-and-data-leakage.md)

Machine learning bugs are notorious because they rarely crash. A wrong mask or a shifted label still produces a loss that goes down; the model is just quietly worse. The defence is the same as in any software: small, fast, specific tests. All of these are in [`tests/tinygpt/`](../tests/tinygpt) and run in about 45 seconds on a CPU.

```bash
pytest tests/tinygpt -v
```

## 1. Tokenizer contract tests

| Test | Why it matters |
|---|---|
| encode → decode round-trips, including empty strings and newlines | silent data corruption otherwise |
| vocabulary is sorted and independent of text order | two training runs must agree on token IDs |
| unknown characters raise a clear error | silently dropping characters changes meaning |
| save/load preserves the vocabulary | a checkpoint is useless with a different tokenizer |
| the corpus is covered by the alphabet | data and tokenizer must match |

## 2. Model invariants

These are the checks experienced ML engineers run before spending money on training.

**Initial loss ≈ ln(vocab size).** A freshly initialised model should be maximally unsure, so its loss should be close to the loss of uniform guessing: ln(40) ≈ 3.69. A much larger value usually means a bad initialisation or a bug in the loss.

```python
def test_initial_loss_is_close_to_uniform_guessing(model):
    idx = torch.randint(VOCAB, (8, 16))
    _, loss = model(idx, idx)
    assert loss.item() == pytest.approx(math.log(VOCAB), rel=0.05)
```

**The causal mask blocks the future.** Change tokens at positions 8 onwards; predictions for positions 0 to 7 must not change at all. This one test catches the most damaging transformer bug there is.

```python
b[0, 8:] = (b[0, 8:] + 1) % VOCAB
assert torch.allclose(logits_a[0, :8], logits_b[0, :8], atol=1e-6)
assert not torch.allclose(logits_a[0, 8:], logits_b[0, 8:])
```

**Batch items are independent.** An item's output must be the same alone or in a batch. Leaks across the batch dimension are a classic reshape bug.

**Every parameter gets a gradient.** A layer with no gradient is dead weight, often a sign that it is not wired into the forward pass.

**Input validation.** Sequences longer than the context window raise an error; invalid head configurations are rejected.

## 3. Generation behaviour

- Greedy decoding is deterministic, and output length is exactly `prompt + max_new_tokens`.
- Seeded sampling is reproducible.
- `top_k=1` with any temperature equals greedy. This is a neat **metamorphic test**: two different code paths must agree.
- Generating past the context window does not crash (the context is cropped).
- A checkpoint saved and loaded produces identical logits.

## 4. Training tests

**Targets are inputs shifted by one.** An off-by-one here trains the model to copy, not predict.

**Train and validation sets do not overlap.** Otherwise validation loss lies.

**The model can overfit a single batch.** If 200 steps cannot drive the loss on one small batch close to zero, something in the model or optimiser is broken. This is the most useful single sanity check in deep learning.

**Loss falls, and validation tracks training.** After training, loss should be well below its starting value and the train/validation gap small. A large gap signals overfitting.

**Training is reproducible with a seed.** Two 30-step runs with the same seed produce identical loss histories.

**The model learns the facts in its data.** A knowledge probe: greedy-complete `the {animal} of {place} eats ` for all 36 pairs and compare with the answer key. Required accuracy: 90%.

### A real finding: knowledge appears late

When this test was first written with 600 training steps, it failed: fact accuracy was about 0.67, even though the loss curve looked healthy. At 800 steps it reached 1.0. The loss was already low because most characters in a sentence are easy to predict; the few characters that carry the *fact* (which food) are learned last. **A falling loss does not prove the model has learned what you care about.** Test the capability directly.

## 5. What to take into real projects

You will rarely write a transformer at work, but you will review training and fine-tuning code. Ask for these tests: initial loss, causal mask, overfit-one-batch, shifted targets, split overlap, seeded reproducibility, checkpoint round-trip, and at least one direct capability probe.

---
Next: [04. Memorisation and Data Leakage](04-memorization-and-data-leakage.md)
