# 01. Why Testing the Model Is Different from Testing the App

> **Series:** LLM Model Testing Lab · [Index](../README.md#articles) · Next: [02. Building a Tiny GPT](02-building-a-tiny-gpt.md)

The companion repository, [Becoming an AI Quality Engineer](https://github.com/K11-Software-Solutions/becoming-ai-quality-engineer), tests an **application built around a model**: a RAG Q&A service with retrieval, prompts, guardrails and an LLM behind an API. This lab goes one layer down and tests **the model itself**.

## Two layers, two sets of questions

| | System (RAG) testing | Model testing |
|---|---|---|
| Unit under test | the pipeline: retrieval, prompt, guardrails, model call | the weights, the tokenizer, the training process |
| You need | an API or a model client | the model weights (open-weights models) |
| Typical questions | Did we retrieve the right document? Is the answer grounded? | Does the model know facts? Does it follow instructions? Is it calibrated? Does it leak training data? Did compression change it? |
| Who usually owns it | product and QA teams | ML, platform and model-evaluation teams |
| Example failure | right document, wrong sentence | the model prefers "Mouses" over "Mice"; it writes a phishing email on request |

Most QA engineers will spend more time on the system layer. Model-level testing still matters to you because:

- teams increasingly **self-host open-weights models** (for cost, privacy or latency) and must decide whether a model, or a new version of it, is good enough;
- **compressed (quantized) models** are deployed to save money, and someone has to check they did not get worse;
- **fine-tuning** your own model brings training-data risks (memorisation, leakage) that only model-level tests catch;
- interviewers for AI quality roles ask how you would evaluate a model, not just an app.

## What weights access lets you test

With an API you only see text out. With the weights you also get:

- **Log-probabilities** for any text. You can score multiple-choice answers without parsing the model's output, measure perplexity, and compare two sentences that differ by one word (a bias probe).
- **Determinism controls.** Greedy decoding and seeded sampling, so results are repeatable.
- **Variants.** The same model at different precisions (fp32, int8, int4), or with random weights as a negative control.
- **Training.** If you train the model, you control the data, so you can plant facts and secrets and test for them exactly.

## How this lab is organised

**Track A: build and test a tiny GPT** ([`tinygpt/`](../tinygpt)). A character-level transformer of about 110,000 parameters that trains on a laptop CPU in under 20 seconds. Because we write the training data, we have a perfect answer key, and we can plant a secret to test memorisation.

**Track B: evaluate a real open-weights model** ([`modellab/`](../modellab), [`evals/`](../evals)). SmolLM2-135M-Instruct, a small instruction-tuned model (Apache-2.0) that runs on a CPU, evaluated on knowledge, calibration, robustness, instruction following, safety, bias, determinism and quantization drift, with CI gates.

Both tracks need **no API key**. Track B downloads about 270 MB once.

## A note on small models

A 135M-parameter model is roughly a thousand times smaller than frontier models. It gets easy trivia wrong, ignores formatting instructions about half the time, and has almost no safety training. **That is useful for learning:** failures are frequent and visible, so every test in this lab finds something real. Point the same harness at a bigger model with one environment variable (`MODEL_ID`) and compare.

---
Next: [02. Building a Tiny GPT](02-building-a-tiny-gpt.md)
