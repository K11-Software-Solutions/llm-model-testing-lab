# 05. Evaluating a Real Open Model

> **Series:** LLM Model Testing Lab · [Index](../README.md#articles) · Previous: [04](04-memorization-and-data-leakage.md) · Next: [06. Safety, Red Teaming and Bias](06-safety-red-teaming-and-bias.md)

Track B evaluates **SmolLM2-135M-Instruct**, a small instruction-tuned model published by Hugging Face under Apache-2.0. It runs on a CPU and downloads once (about 270 MB). The harness is [`evals/run_model_evals.py`](../evals/run_model_evals.py):

```bash
python -m evals.run_model_evals            # about 4 minutes on a laptop CPU
cat evals/results/model_report.md
```

All results below are from the baseline run at the time of writing. Expect them to change as the datasets grow.

## 1. Knowledge: multiple choice by log-likelihood

[`mcq.jsonl`](../evals/datasets/mcq.jsonl) has 40 easy questions across seven categories. Rather than asking the model to *write* "B" and parsing its reply, which small models do unreliably, the harness asks **how likely the model finds each answer** as a continuation:

```text
Question: What is the capital of France?
Answer: Paris      ← log P(" Paris" | prompt)
Answer: Berlin     ← log P(" Berlin" | prompt)   ...  pick the highest
```

This is how harnesses such as EleutherAI's lm-evaluation-harness grade multiple choice. It is deterministic and removes output-format noise from the measurement.

**Result:** 0.675 accuracy (27 of 40), from 1.0 on history and technology down to **0.0 on language** questions. Article 08 shows that the language result is largely caused by the scoring method, not the model.

## 2. Calibration: is the model's confidence honest?

Converting option scores to probabilities gives a confidence for each answer. A **calibrated** model that says 70% is right about 70% of the time. **Expected calibration error (ECE)** measures the average gap, weighted by how many answers fall into each confidence bin.

**Result:** ECE 0.078, mean confidence close to accuracy. Calibration matters whenever confidence drives behaviour, for example "escalate to a human when the model is unsure".

## 3. Robustness: does answer order matter?

Each question is asked a second time with the options reversed. The model should choose the same **answer text**. Position bias, where a model prefers whichever option comes first, is a known weakness of LLM evaluators.

**Result:** 1.0 consistency, which is expected with log-likelihood scoring because each option is scored on its own. Generation-based answering ("reply with A, B, C or D") is where position bias usually appears; try adding that as an exercise.

## 4. Perplexity sanity check

**Perplexity** measures how surprised the model is by a text (lower means less surprised). A working language model must find real text far less surprising than the same words shuffled.

**Result:** shuffled text is about 76 times more perplexing than the original. A random-weights model scores about 1.1, which is why this check is part of the negative control.

## 5. Determinism

- **Greedy decoding** run twice gives identical text.
- **Seeded sampling** (temperature 1.0, seed 42) run twice gives identical text.

Both are gated at 100%. If they fail, every other result in the report becomes harder to trust, because you cannot tell model changes from run-to-run noise. Note that determinism is only guaranteed on the same hardware and library versions.

## 6. Instruction following with verifiable checks

[`instructions.jsonl`](../evals/datasets/instructions.jsonl) has 14 prompts whose constraints a program can check exactly: "answer in one word", "at most 20 words", "lowercase only", "start with Yes", "return only JSON", "exactly three bullets". This is the idea behind IFEval (Zhou et al., 2023): no judge model is needed. The checkers are in [`modellab/checks.py`](../modellab/checks.py) and have their own unit tests.

**Result:** 0.5 pass rate. Typical failures:

| Instruction | What the model did |
|---|---|
| "Answer in one word." | "The capital of France is Paris." |
| "Use only lowercase letters." | normal sentence case |
| "Return only a JSON object..." | wrote Python code that *builds* a JSON object |
| "In at most 20 words" | a long, flowery paragraph |

Format compliance is one of the clearest differences between small and large models.

## 7. Throughput

The report records tokens per second during generation (about 18 on a laptop CPU). It is **informational, not gated**: CI machines vary too much for a speed threshold to be stable. In production, gate latency on fixed hardware, with percentiles (p50, p95) rather than averages.

## 8. Running other models

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct python -m evals.run_model_evals --no-gate
```

Compare the two reports side by side. Which metrics improve with a model four times larger, and which do not?

## Beyond this lab

For standard public benchmarks (MMLU, HellaSwag, ARC, GSM8K), use **lm-evaluation-harness** rather than writing your own:

```bash
pip install lm-eval
lm_eval --model hf --model_args pretrained=HuggingFaceTB/SmolLM2-135M-Instruct --tasks arc_easy --device cpu --limit 100
```

Public benchmarks are useful for comparing models. **Your own small, targeted datasets** are what tell you whether a model fits *your* use case, and they are much harder to contaminate, because they never appeared on the internet.

---
Next: [06. Safety, Red Teaming and Bias](06-safety-red-teaming-and-bias.md)
