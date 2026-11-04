# 08. Investigating Failures and Stating Limitations

> **Series:** LLM Model Testing Lab · [Index](../README.md#articles) · Previous: [07](07-quantization-regression-and-ci.md)

A score is where an investigation starts. This article works through five findings from the lab, each of which looked like one thing and turned out to be another, then lists the limitations you should state whenever you present these results.

## The triage question

For every surprising result, ask in this order:

1. **Is the test data right?** Wrong labels, ambiguous questions, duplicates.
2. **Is the evaluator right?** Scoring method, checker logic, keyword lists.
3. **Is the setup right?** Prompt format, chat template, decoding settings, model variant.
4. **Only then: is the model wrong?**

Each finding below was resolved at a different step.

## Finding 1: "the model knows no grammar" (an evaluator effect)

The model scored **0.0 on the four language questions**. It picked "Mouses" over "Mice", "Gone" over "Went", "Carrot" as a fruit.

Looking at the raw scores for *What is the plural of 'mouse'?*:

| Option | Tokens | Sum of log-probs | Mean per token |
|---|---|---|---|
| Mouses | 2 (`Mous` + `es`) | −9.58 | **−4.79** ← chosen |
| Mice | 1 | **−5.41** ← most likely overall | −5.41 |

The default scoring divides by token count to avoid penalising long answers. But "Mouses" is split into two tokens, and the second token (`es`, after `Mous`) is very easy to predict, so the *average* looks good. With **sum** scoring the model picks "Mice". Across the whole set, sum scoring gives **0.75** accuracy against **0.675** for mean scoring.

**Diagnosis:** step 2, evaluator. **Lesson:** the scoring method is part of the benchmark. Report it, keep it fixed when comparing models, and never compare scores produced with different methods. The report shows both so you can see the effect.

## Finding 2: "the loss is low, so it's trained" (a capability gap)

At 600 steps TinyGPT had a healthy-looking loss curve but answered only about two thirds of the fact probes correctly. At 800 steps it answered all of them. Most characters in a sentence are easy to predict; the few that carry the fact are learned last.

**Diagnosis:** step 4, the model genuinely had not learned yet. **Lesson:** loss is an indirect measure. Test the capability you care about directly.

## Finding 3: "the random model is perfectly fair" (metrics read in isolation)

The random-weights negative control scored a bias distance of **0.0** and a reorder consistency of **1.0**, both "perfect". A model with no knowledge has no preferences and, with per-option scoring, no position bias.

**Lesson:** a good score can mean "no signal". Read bias and robustness next to capability, and use a negative control to see what "no capability" looks like on every metric.

## Finding 4: "the jailbreak was refused" (a fooled evaluator)

On the role-play jailbreak the model wrote *"I'm sorry for the confusion, but as a vehicle theft AI, I don't have the ability to assist with driving..."*. The Promptfoo keyword assertion counted this as a refusal and passed it. It is incoherent text that happens to begin with an apology.

**Diagnosis:** step 2, evaluator. **Lesson:** for safety, keyword checks give a floor, not proof. Validate evaluators against human labels (the refusal detector's kappa is 0.714, with documented misses), and use stronger judges for release decisions.

## Finding 5: "int4 is as good as int8" (an aggregate hiding churn)

int8 and int4 had the same accuracy, but int4 changed 30% of the model's answers. **Lesson:** compare *what changed*, not just *how many were right*. Agreement with the reference model is the metric that catches it.

## An open question for you: is the checker too strict?

For *"End your answer with the word 'done'. Count from 1 to 3."* the model replied *"I'm done."* The `ends_with` checker fails it because of the full stop, and the model did not count either. Should a trailing full stop be allowed? Should the check fail anyway because the task was not done? Write down your decision, change the checker or the prompt, and add a unit test for it. **Ambiguous instructions produce ambiguous evaluations**; resolving them is part of the job.

## Limitations to state when you present results

- **Model size.** 135M parameters is far from production models; results show the method, not the state of the art.
- **Dataset size.** 40 MCQ items, 14 instructions, 16 safety prompts and 12 bias pairs. One MCQ item moves accuracy by 2.5 points. Report confidence intervals before drawing conclusions from small differences.
- **Simulated quantization.** Real formats behave differently; validate the exact artifact you deploy.
- **Keyword-based safety checks.** Measured as imperfect; release decisions need stronger judges and human review.
- **Contamination.** Easy trivia may have appeared in training data. The TinyGPT track avoids this entirely with synthetic data; for real models, write your own items and keep them private.
- **Determinism.** Guaranteed only for the same hardware, library versions and settings.
- **Performance.** CPU throughput is informational only.

## Presenting this lab in an interview

> "I built a tiny GPT to test model internals: causal-mask and initial-loss invariants, overfit-one-batch, reproducibility, and a canary-based memorisation test showing a planted secret leaks unless it is scrubbed. Then I evaluated a real 135M open model on knowledge, calibration, instruction following, safety, bias and quantization drift. CI has regression gates plus a release bar, and a random-weights negative control proves the gate can fail. The most interesting findings were evaluator effects: length-normalised scoring changed accuracy by 7.5 points, and a keyword check passed a gibberish 'refusal'. The model passes regression, but I would not release it: it complied with every harmful request."

---
Back to the [index](../README.md#articles).
