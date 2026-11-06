# LLM Model Testing Lab

[![CI quality gates](https://github.com/K11-Software-Solutions/llm-model-testing-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/K11-Software-Solutions/llm-model-testing-lab/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24015642.svg)](https://doi.org/10.5281/zenodo.24015642) [![Articles: CC BY 4.0](https://img.shields.io/badge/Articles-CC%20BY%204.0-lightgrey.svg)](articles/LICENSE.md)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![No API key needed](https://img.shields.io/badge/API%20key-not%20required-success)

**Build a tiny GPT and test it. Then evaluate a real open-weights LLM with the same discipline, and gate it in CI.**

This is the companion to [Becoming an AI Quality Engineer](https://github.com/K11-Software-Solutions/becoming-ai-quality-engineer), which tests a RAG *application*. This lab goes one layer down and tests **the model itself**: its code, its training, what it memorises, what it knows, how it behaves, and how it changes when compressed. Everything runs on a laptop CPU with **no API key**.

It is the second of five labs in the series:

| Lab | Question it answers |
|---|---|
| [Becoming an AI Quality Engineer](https://github.com/K11-Software-Solutions/becoming-ai-quality-engineer) | Is the RAG **application** good enough to ship? |
| **LLM Model Testing Lab** (this repo) | Is the **model** good enough? |
| [Agent Testing Lab](https://github.com/K11-Software-Solutions/agent-testing-lab) | Is the **agent** safe to let act? |
| [AI Production Monitoring Lab](https://github.com/K11-Software-Solutions/ai-production-monitoring-lab) | Is it **still** good, for the users we actually have, right now? |
| [Multi-Agent Testing Lab](https://github.com/K11-Software-Solutions/multi-agent-testing-lab) | When **agents work together**, does the team do the right thing, and who broke it when it doesn't? |

## Two tracks

| Track | What you do | Runs in |
|---|---|---|
| **A. Build and test a tiny GPT** ([`tinygpt/`](tinygpt)) | Train a ~110K-parameter character-level transformer on synthetic data you control, then unit-test the model code, the training loop, generation, knowledge, and memorisation of a planted secret | ~30 s training, ~1 min tests |
| **B. Evaluate a real open model** ([`modellab/`](modellab), [`evals/`](evals)) | Evaluate SmolLM2-135M-Instruct (Apache-2.0, ~270 MB) on knowledge, calibration, robustness, instruction following, safety, over-refusal, bias, determinism and int8/int4 quantization drift, with regression gates and a release bar | ~4 min on CPU |

## Articles

| # | Article | Track |
|---|---|---|
| 01 | [Why Testing the Model Is Different from Testing the App](articles/01-why-model-testing-is-different.md) | both |
| 02 | [Building a Tiny GPT](articles/02-building-a-tiny-gpt.md) | A |
| 03 | [Testing Model Code and Training](articles/03-testing-model-code-and-training.md) | A |
| 04 | [Memorisation and Data Leakage](articles/04-memorization-and-data-leakage.md) | A |
| 05 | [Evaluating a Real Open Model](articles/05-evaluating-a-real-open-model.md) | B |
| 06 | [Safety, Red Teaming and Bias](articles/06-safety-red-teaming-and-bias.md) | B |
| 07 | [Quantization, Regression Gates and CI](articles/07-quantization-regression-and-ci.md) | B |
| 08 | [Investigating Failures and Stating Limitations](articles/08-investigating-failures-and-limitations.md) | both |

## Quick start

Requirements: Python 3.11+, Git, about 2 GB of disk for PyTorch and the model. Node.js 22.22+ only for Promptfoo.

```bash
git clone https://github.com/K11-Software-Solutions/llm-model-testing-lab.git
cd llm-model-testing-lab
python -m venv .venv
source .venv/bin/activate                         # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
pip install -r requirements-dev.txt
```

**Track A** (no download):

```bash
pytest                                  # tokenizer, model, training, memorisation + evaluator unit tests
python -m tinygpt.train --steps 1200    # train and save artifacts/tinygpt.pt
```

**Track B** (downloads the model once):

```bash
pytest -m hf                            # integration tests against the real model
python -m evals.run_model_evals         # full evaluation, regression gates, release bar
python -m evals.run_model_evals --variant random --fast   # negative control: must FAIL
npx promptfoo@0.124.0 eval -c evals/promptfoo/promptfooconfig.yaml           # behaviour regression suite
npx promptfoo@0.124.0 eval -c evals/promptfoo/promptfooconfig.redteam.yaml   # red-team probes
```

Evaluate a different model with `MODEL_ID`, for example `MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct python -m evals.run_model_evals --no-gate`.

## Repository layout

```text
tinygpt/                 Track A: tokenizer, transformer, training loop, data with a planted canary, evals
modellab/                Track B: model loading (fp32/int8/int4/random), log-likelihood scoring,
                         chat generation, instruction verifiers, refusal detector, metrics
evals/
  datasets/              MCQ knowledge, instructions, safety (harmful + benign), bias pairs,
                         human-labelled refusals for evaluator validation
  run_model_evals.py     evaluation runner → evals/results/model_report.md
  model_thresholds.json  "regression" gates (block CI) and a "release" bar (reported)
  promptfoo/             regression suite (blocking) and red-team probes (informational)
tests/
  tinygpt/               unit, training and memorisation tests
  modellab/              evaluator unit tests + real-model integration tests (pytest -m hf)
articles/                the eight articles
.github/workflows/ci.yml CI: unit/TinyGPT → real-model evals + Promptfoo, with a negative control
```

## What the lab finds

Real findings from running these tests, each discussed in the articles:

- **A falling loss is not learning the facts.** TinyGPT's loss looked healthy at 600 steps while it knew only about two thirds of its facts; it took longer training to learn them all, and the step count that is enough on one machine was not enough on another.
- **Planted secrets leak.** A secret repeated in training data is reproduced verbatim; scrubbing it before training prevents that.
- **The scoring method changes the score.** Length-normalised multiple-choice scoring picks "Mouses" over "Mice" because the second token of "Mous|es" is easy to predict.
- **A model can pass regression gates and still not be releasable.** SmolLM2-135M complies with harmful requests; the release bar reports it, the regression gate does not block on it.
- **Metrics must be read together.** A model with random weights scores as "perfectly unbiased" and perfectly consistent under option reordering.
- **Evaluators need validation.** The keyword refusal detector disagrees with human labels on documented cases, and a red-team keyword check is fooled by gibberish that starts with "I'm sorry".
- **int8 is nearly free; int4 is not.** Simulated int8 weights barely change the model; int4 changes a large share of its answers.

## CI quality gates

Every push runs three jobs: **TinyGPT and evaluator unit tests**; **real-model evals** with regression gates, the release bar in the job summary, and a negative control that fails the build if a random-weights model passes; and **Promptfoo**, with a blocking behaviour suite and informational red-team probes.

## Limitations

Small models, small hand-written datasets, simulated quantization, keyword-based refusal detection, and CPU-only performance numbers. These are deliberate choices for a runnable learning lab, and [article 08](articles/08-investigating-failures-and-limitations.md) explains what you would change for production model evaluation.

## About

Created by **Kavita Jadhav**, founder of [K11 Software Solutions](https://github.com/K11-Software-Solutions), Senior SDET and AI Quality Engineer working on LLM evaluation, red teaming and agentic QA systems.

**Licences:** code under the [MIT License](LICENSE); the written material in [`articles/`](articles/LICENSE.md) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (share and adapt with credit).

SmolLM2 is © Hugging Face, Apache-2.0, and is downloaded at runtime, not redistributed here.

## Citing

Use GitHub's **Cite this repository** button (from [`CITATION.cff`](CITATION.cff)). Each tagged release is archived on Zenodo with its own DOI. Cite the version you used:

> Jadhav, K. (2026). *LLM Model Testing Lab* (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24015643

To refer to the project as a whole (always resolves to the latest version), use the concept DOI: https://doi.org/10.5281/zenodo.24015642
