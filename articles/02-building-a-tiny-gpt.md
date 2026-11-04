# 02. Building a Tiny GPT

> **Series:** LLM Model Testing Lab · [Index](../README.md#articles) · Previous: [01](01-why-model-testing-is-different.md) · Next: [03. Testing Model Code and Training](03-testing-model-code-and-training.md)

You cannot test something well if it is a black box to you. This article walks through a complete GPT-style language model in about 150 lines of PyTorch ([`tinygpt/model.py`](../tinygpt/model.py)). It is the same architecture family as the large models, just tiny.

```bash
python -m tinygpt.train --steps 800        # about 20 seconds on a laptop CPU
```

## 1. The data: a world we fully control

[`tinygpt/data.py`](../tinygpt/data.py) generates sentences about animals in fictional places:

```text
the fox of mill hill eats berries.
in oak vale, the heron eats worms.
every lynx in ash moor eats roots.
```

There are 36 (animal, place) pairs, each with one correct food, stated in three sentence templates. Because we generate the data, **we know exactly what the model should learn**. That turns evaluation from guesswork into a test with an answer key.

The generator can also insert a **canary**, a fake secret (`the secret code is 7391.`), so we can test whether the model memorises and leaks it (article 04).

## 2. The tokenizer: text to numbers

Models work on integers. [`tinygpt/tokenizer.py`](../tinygpt/tokenizer.py) maps each character to an ID, using a fixed alphabet of lowercase letters, digits and a little punctuation (40 tokens). Real models use sub-word tokenizers with 30,000 to 200,000 tokens, but the contract you test is identical: encode and decode round-trip, the vocabulary is stable, and unknown characters are handled explicitly.

## 3. The model

```text
token ids ──► token embedding + position embedding
          ──► [ LayerNorm → causal self-attention → + residual
                LayerNorm → MLP                  → + residual ] × 2 blocks
          ──► LayerNorm ──► linear head ──► logits over the vocabulary
```

- **Embeddings** turn each token ID and its position into a 64-number vector.
- **Self-attention** lets each position look at earlier positions and pull in relevant information. The **causal mask** (a lower-triangular matrix) stops it from looking at future tokens. Without it, the model could cheat during training by reading the answer.
- **The MLP** processes each position independently.
- **Residual connections and LayerNorm** keep training stable.
- **The head** produces one score (a logit) per vocabulary token: the model's prediction for the next character.

Configuration: 2 layers, 4 attention heads, 64-dimensional embeddings, a 64-character context window, about 110,000 parameters in total.

## 4. Training: predict the next token

[`tinygpt/train.py`](../tinygpt/train.py) repeats one step:

1. Sample random windows of 64 characters from the training text.
2. The targets are the same windows **shifted by one** character.
3. Compute cross-entropy loss between predictions and targets.
4. Back-propagate and update the weights with AdamW.

Ten percent of the text is held out as a **validation set**, so we can tell learning from memorising. Every 100 steps the loop records training and validation loss, which becomes a loss curve you can test and plot.

## 5. Generation: one token at a time

`model.generate()` feeds the text so far, takes the logits for the last position, picks the next token, appends it and repeats:

- **temperature = 0** → greedy: always the most likely token (deterministic).
- **temperature > 0** → sample from the probability distribution; higher is more random.
- **top_k** → sample only among the k most likely tokens.

Every one of these settings is a testable behaviour, and article 03 tests them.

## 6. Scoring: how likely is this text?

`model.sequence_logprob(prefix, continuation)` returns the log-probability the model assigns to a continuation. This is the basis for perplexity, multiple-choice scoring, bias probes and the memorisation exposure test, here and in Track B.

## Try it

```bash
python -m tinygpt.train --steps 200     # under-trained: what do completions look like?
python -m tinygpt.train --steps 800
python -c "from tinygpt.model import TinyGPT; from tinygpt.tokenizer import CharTokenizer; from tinygpt.evaluate import complete; m=TinyGPT.load('artifacts/tinygpt.pt'); t=CharTokenizer.load('artifacts/tinygpt.tokenizer.json'); print(complete(m, t, 'the fox of mill hill eats '))"
```

---
Next: [03. Testing Model Code and Training](03-testing-model-code-and-training.md)
