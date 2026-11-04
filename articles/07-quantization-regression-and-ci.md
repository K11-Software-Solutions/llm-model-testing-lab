# 07. Quantization, Regression Gates and CI

> **Series:** LLM Model Testing Lab · [Index](../README.md#articles) · Previous: [06](06-safety-red-teaming-and-bias.md) · Next: [08. Investigating Failures and Stating Limitations](08-investigating-failures-and-limitations.md)

## 1. Why quantization needs testing

**Quantization** stores weights with fewer bits: 8-bit or 4-bit integers instead of 32-bit floats. Models get 4 to 8 times smaller and often faster, which is why most self-hosted models are quantized. The question for a tester is simple: **how much did the model change?**

[`modellab/loader.py`](../modellab/loader.py) simulates symmetric per-channel weight quantization: each row of each linear layer is scaled so its largest weight maps to the largest integer, rounded, and scaled back. Production schemes (GPTQ, AWQ, llama.cpp formats, torchao) are smarter, but the testing method is the same. The quantizer has its own unit tests: error shrinks as bits increase, and int8 uses at most 255 distinct values per row.

## 2. Measure drift, not just accuracy

The harness compares each variant with the fp32 reference in three ways:

| Variant | MCQ accuracy | Change | Answer agreement with fp32 | Next-token agreement with fp32 |
|---|---|---|---|---|
| fp32 (reference) | 0.675 | — | — | — |
| int8 | 0.65 | −0.025 | 0.975 | 0.984 |
| int4 | 0.65 | −0.025 | **0.70** | **0.551** |

Look at int4. **Its accuracy is the same as int8's, yet it changed 30% of its answers** and disagrees with the original on almost half of next-token predictions. Some answers got better and others got worse, and the totals happened to cancel out.

If you only compared accuracy, you would ship int4 believing nothing changed. Users would see a different model. **Agreement metrics catch churn that accuracy hides.** The same applies to any model update: a new version, a new fine-tune, a new serving stack.

## 3. Two threshold profiles

[`evals/model_thresholds.json`](../evals/model_thresholds.json) has two profiles:

- **`regression`** blocks CI. Floors sit slightly below the measured baseline, so the build fails if the model, code or data gets *worse*. Example: `int8_answer_agreement ≥ 0.9`.
- **`release`** is the bar you would require before real users see the model. Example: `harmful_refusal_rate ≥ 0.9`, `int4_answer_agreement ≥ 0.95`. It is always reported; it blocks only with `--profile release`.

```text
Gate results
- regression (blocking): PASSED
- release (informational): FAILED: mcq_accuracy 0.675 < 0.85; bias_distance 0.333 > 0.2;
  instruction_pass_rate 0.5 < 0.8; harmful_refusal_rate 0.0 < 0.9; int4_answer_agreement 0.7 < 0.95
```

Keeping both visible stops a green pipeline from being mistaken for a release decision.

## 4. A negative control for the gate

A gate that cannot fail is worse than none. CI therefore evaluates the **same architecture with random weights** and fails the build if the gate *passes*:

```text
regression (blocking): FAILED: mcq_accuracy 0.225 < min 0.6; calibration_ece 0.24 > max 0.15;
                               perplexity_shuffle_ratio 1.1 < min 5
```

The random model also shows why gates need several metrics: it is perfectly consistent under option reordering and has a bias distance of 0.0. Either metric alone would have passed it.

## 5. The CI pipeline

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

| Job | Runs | Blocks on |
|---|---|---|
| **TinyGPT and evaluator unit tests** | tokenizer, model, training and memorisation tests; evaluator unit tests; a training run whose loss curve is uploaded | any failing test |
| **Real model evals** | `pytest -m hf`, then the full evaluation (report in the job summary), then the random-weights negative control | a regression gate; or the negative control *passing* |
| **Promptfoo** | the behaviour regression suite, then red-team probes | the regression suite only; red-team failures become a warning |

Design notes:

- **CPU-only PyTorch** is installed from the PyTorch CPU index, which is much smaller than the default CUDA build.
- **The model is cached** with `actions/cache`, so it downloads once, not on every run.
- **Expensive and unstable checks stay out of the blocking path**: throughput numbers, red-team probes, and the release bar.

## 6. Exercises

1. Add an `int2` variant. What happens to perplexity and agreement?
2. Quantize `lm_head` as well (the loader currently skips it). Which metrics move most?
3. Add a `latency_p95_ms` metric measured on fixed hardware and gate it.
4. Evaluate `Qwen/Qwen2.5-0.5B-Instruct` and propose regression thresholds from its baseline.

---
Next: [08. Investigating Failures and Stating Limitations](08-investigating-failures-and-limitations.md)
