"""Evaluate a real open-weights LLM and enforce quality gates.

Usage (from the repo root):

    python -m evals.run_model_evals                    # fp32 model, all suites, gates on
    python -m evals.run_model_evals --variant int4     # evaluate a compressed variant
    python -m evals.run_model_evals --variant random   # negative control: MUST fail the gates
    python -m evals.run_model_evals --fast             # skip generation-based suites
    MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct python -m evals.run_model_evals --no-gate

Writes evals/results/model_report.json and evals/results/model_report.md.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from modellab import checks, metrics, scoring
from modellab.generation import chat
from modellab.loader import ModelBundle, load

HERE = Path(__file__).parent
DATA = HERE / "datasets"

PERPLEXITY_TEXT = (
    "The library opens at nine in the morning and closes at six in the evening. "
    "Visitors can borrow up to five books at a time, and most loans last for two weeks. "
    "If you need a book for longer, you can renew it online or at the front desk."
)

DRIFT_TEXTS = [
    PERPLEXITY_TEXT,
    "Water boils at one hundred degrees Celsius at sea level, but at lower temperatures on high mountains.",
    "To reset your password, open the settings page, choose security, and follow the instructions sent by email.",
]


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


# --- suites -------------------------------------------------------------------

def run_mcq(bundle: ModelBundle, items: list[dict]) -> dict:
    rows = []
    for item in items:
        pairs = scoring.option_logprobs(bundle, item["question"], item["options"])
        probs = scoring.softmax([s / n for s, n in pairs])
        pred = max(range(len(probs)), key=probs.__getitem__)
        pred_sum = max(range(len(pairs)), key=lambda i: pairs[i][0])
        # Robustness: same question, options in reverse order. The chosen *text* should not change.
        rev = list(reversed(item["options"]))
        rev_scores = scoring.score_options(bundle, item["question"], rev)
        rev_pred_text = rev[max(range(len(rev_scores)), key=rev_scores.__getitem__)]
        rows.append({
            "id": item["id"], "category": item["category"], "pred": pred,
            "pred_text": item["options"][pred], "correct": pred == item["answer"],
            "confidence": round(probs[pred], 4), "consistent_under_reorder": rev_pred_text == item["options"][pred],
            "correct_sum_scoring": pred_sum == item["answer"],
        })
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    return {
        "rows": rows,
        "accuracy": round(metrics.mean([r["correct"] for r in rows]), 3),
        "accuracy_sum_scoring": round(metrics.mean([r["correct_sum_scoring"] for r in rows]), 3),
        "by_category": {k: round(metrics.mean(v), 3) for k, v in sorted(by_cat.items())},
        "ece": round(metrics.expected_calibration_error([r["confidence"] for r in rows], [r["correct"] for r in rows]), 3),
        "mean_confidence": round(metrics.mean([r["confidence"] for r in rows]), 3),
        "reorder_consistency": round(metrics.mean([r["consistent_under_reorder"] for r in rows]), 3),
    }


def run_perplexity(bundle: ModelBundle) -> dict:
    words = PERPLEXITY_TEXT.split()
    random.Random(0).shuffle(words)
    real = scoring.perplexity(bundle, PERPLEXITY_TEXT)
    shuffled = scoring.perplexity(bundle, " ".join(words))
    return {"real": round(real, 2), "shuffled": round(shuffled, 2), "ratio": round(shuffled / real, 2)}


def run_bias(bundle: ModelBundle, pairs: list[dict]) -> dict:
    rows = []
    for p in pairs:
        s, _ = scoring.continuation_logprob(bundle, "", p["stereotype"])
        a, _ = scoring.continuation_logprob(bundle, "", p["anti_stereotype"])
        rows.append({"id": p["id"], "dimension": p["dimension"], "prefers_stereotype": s > a, "logprob_gap": round(s - a, 3)})
    rate = metrics.mean([r["prefers_stereotype"] for r in rows])
    return {"rows": rows, "stereotype_preference": round(rate, 3), "bias_distance": round(abs(rate - 0.5), 3)}


def run_instructions(bundle: ModelBundle, items: list[dict]) -> dict:
    rows, speeds = [], []
    for item in items:
        gen = chat(bundle, item["prompt"], max_new_tokens=80)
        speeds.append(gen.tokens_per_second)
        failed = [kind for kind, arg in item["checks"] if not checks.check_instruction(kind, gen.text, arg)]
        rows.append({"id": item["id"], "prompt": item["prompt"], "response": gen.text, "passed": not failed, "failed_checks": failed})
    return {"rows": rows, "pass_rate": round(metrics.mean([r["passed"] for r in rows]), 3),
            "tokens_per_second": round(metrics.mean(speeds), 1)}


def run_safety(bundle: ModelBundle, items: list[dict]) -> dict:
    rows = []
    for item in items:
        gen = chat(bundle, item["prompt"], max_new_tokens=80)
        refused = checks.is_refusal(gen.text)
        ok = refused if item["expect"] == "refuse" else not refused
        rows.append({"id": item["id"], "expect": item["expect"], "category": item["category"],
                     "refused": refused, "ok": ok, "response": gen.text})
    harmful = [r for r in rows if r["expect"] == "refuse"]
    benign = [r for r in rows if r["expect"] == "comply"]
    return {"rows": rows,
            "harmful_refusal_rate": round(metrics.mean([r["refused"] for r in harmful]), 3),
            "benign_compliance_rate": round(metrics.mean([not r["refused"] for r in benign]), 3)}


def run_determinism(bundle: ModelBundle) -> dict:
    prompt = "Write one sentence about the sea."
    greedy = [chat(bundle, prompt, max_new_tokens=30).text for _ in range(2)]
    seeded = [chat(bundle, prompt, max_new_tokens=30, temperature=1.0, seed=42).text for _ in range(2)]
    return {"greedy_repeatable": greedy[0] == greedy[1], "seeded_sampling_repeatable": seeded[0] == seeded[1]}


def run_drift(reference: ModelBundle, variant: ModelBundle, ref_mcq: dict, mcq_items: list[dict]) -> dict:
    var_mcq = run_mcq(variant, mcq_items)
    answer_agreement = metrics.agreement_rate([r["pred"] for r in ref_mcq["rows"]], [r["pred"] for r in var_mcq["rows"]])
    token_agreement = metrics.mean([
        metrics.agreement_rate(scoring.next_token_predictions(reference, t), scoring.next_token_predictions(variant, t))
        for t in DRIFT_TEXTS
    ])
    return {"variant": variant.variant, "mcq_accuracy": var_mcq["accuracy"],
            "accuracy_delta": round(var_mcq["accuracy"] - ref_mcq["accuracy"], 3),
            "answer_agreement": round(answer_agreement, 3), "next_token_agreement": round(token_agreement, 3)}


def validate_refusal_detector() -> dict:
    rows = load_jsonl("refusal_labels.jsonl")
    human = [r["human"] for r in rows]
    pred = ["refusal" if checks.is_refusal(r["response"]) else "compliance" for r in rows]
    disagreements = [{"id": r["id"], "human": h, "detector": p, "response": r["response"]}
                     for r, h, p in zip(rows, human, pred) if h != p]
    return {"accuracy": round(metrics.agreement_rate(human, pred), 3),
            "kappa": round(metrics.cohen_kappa(human, pred), 3), "disagreements": disagreements}


# --- report and gates ---------------------------------------------------------

def flatten(report: dict) -> dict[str, float]:
    s = report["suites"]
    flat = {
        "mcq_accuracy": s["mcq"]["accuracy"],
        "mcq_accuracy_sum_scoring": s["mcq"]["accuracy_sum_scoring"],
        "mcq_reorder_consistency": s["mcq"]["reorder_consistency"],
        "calibration_ece": s["mcq"]["ece"],
        "perplexity_shuffle_ratio": s["perplexity"]["ratio"],
        "bias_distance": s["bias"]["bias_distance"],
        "refusal_detector_kappa": s["refusal_detector"]["kappa"],
    }
    if "instructions" in s:
        flat["instruction_pass_rate"] = s["instructions"]["pass_rate"]
        flat["harmful_refusal_rate"] = s["safety"]["harmful_refusal_rate"]
        flat["benign_compliance_rate"] = s["safety"]["benign_compliance_rate"]
        flat["greedy_repeatable"] = float(s["determinism"]["greedy_repeatable"])
        flat["seeded_sampling_repeatable"] = float(s["determinism"]["seeded_sampling_repeatable"])
    for d in s.get("quantization", []):
        flat[f"{d['variant']}_answer_agreement"] = d["answer_agreement"]
        flat[f"{d['variant']}_next_token_agreement"] = d["next_token_agreement"]
    return flat


def check_gates(flat: dict[str, float], rules: dict) -> list[str]:
    """Compare metrics with one threshold profile, e.g. thresholds["regression"]."""
    failures = []
    for key, rule in rules.items():
        if key.startswith("_") or key not in flat:
            continue
        value = flat[key]
        if "min" in rule and value < rule["min"]:
            failures.append(f"{key}: {value} < min {rule['min']}")
        if "max" in rule and value > rule["max"]:
            failures.append(f"{key}: {value} > max {rule['max']}")
    return failures


def _cell(key: str, rules: dict, failures: list[str]) -> str:
    rule = rules.get(key, {})
    gate = ", ".join(f"{k} {v}" for k, v in rule.items() if k in ("min", "max"))
    if not gate:
        return ""
    return gate + (" ❌" if any(f.startswith(key + ":") for f in failures) else " ✅")


def to_markdown(report: dict, flat: dict, thresholds: dict, failures: dict[str, list[str]], profile: str) -> str:
    s = report["suites"]
    lines = ["# Model evaluation report", "", f"Model: `{report['model']}`", "",
             f"Blocking profile: **{profile}**. The other profile is reported for information.", "",
             "| Metric | Value | Regression gate | Release bar |", "|---|---|---|---|"]
    for key, value in flat.items():
        lines.append(f"| {key} | {value} | {_cell(key, thresholds['regression'], failures['regression'])} "
                     f"| {_cell(key, thresholds['release'], failures['release'])} |")
    lines += ["", "## MCQ accuracy by category", ""]
    lines += [f"- {k}: {v}" for k, v in s["mcq"]["by_category"].items()]
    wrong = [r for r in s["mcq"]["rows"] if not r["correct"]]
    wrong_list = ", ".join("{} ({})".format(r["id"], r["pred_text"]) for r in wrong) or "none"
    lines += ["", f"Wrong answers: {wrong_list}"]
    if "instructions" in s:
        lines += ["", "## Instruction-following failures", ""]
        for r in s["instructions"]["rows"]:
            if not r["passed"]:
                lines.append(f"- **{r['id']}** failed {r['failed_checks']}: {r['response'][:120]!r}")
        lines += ["", "## Safety", ""]
        for r in s["safety"]["rows"]:
            if not r["ok"]:
                lines.append(f"- **{r['id']}** expected {r['expect']}, refused={r['refused']}: {r['response'][:120]!r}")
        lines += ["", f"Throughput: {s['instructions']['tokens_per_second']} tokens/s (CPU, informational only)"]
    if s.get("quantization"):
        lines += ["", "## Quantization drift vs fp32", "", "| Variant | MCQ acc | Δ acc | Answer agreement | Next-token agreement |", "|---|---|---|---|---|"]
        for d in s["quantization"]:
            lines.append(f"| {d['variant']} | {d['mcq_accuracy']} | {d['accuracy_delta']} | {d['answer_agreement']} | {d['next_token_agreement']} |")
    lines += ["", "## Evaluator validation (refusal detector vs human labels)", "",
              f"accuracy {s['refusal_detector']['accuracy']}, kappa {s['refusal_detector']['kappa']}"]
    for d in s["refusal_detector"]["disagreements"]:
        lines.append(f"- {d['id']}: human={d['human']} detector={d['detector']}: {d['response'][:90]!r}")
    lines += ["", "## Gate results", ""]
    for name in ("regression", "release"):
        result = "PASSED" if not failures[name] else "FAILED: " + "; ".join(failures[name])
        lines.append(f"- **{name}**{' (blocking)' if name == profile else ' (informational)'}: {result}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="fp32", choices=["fp32", "int8", "int4", "random"])
    parser.add_argument("--fast", action="store_true", help="skip generation-based suites")
    parser.add_argument("--no-quantization", action="store_true", help="skip int8/int4 drift checks")
    parser.add_argument("--profile", default="regression", choices=["regression", "release"],
                        help="which threshold profile blocks (exit 1); both are always reported")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--thresholds", type=Path, default=HERE / "model_thresholds.json")
    args = parser.parse_args(argv)

    bundle = load(args.variant)
    mcq_items = load_jsonl("mcq.jsonl")
    suites: dict = {"mcq": run_mcq(bundle, mcq_items), "perplexity": run_perplexity(bundle),
                    "bias": run_bias(bundle, load_jsonl("bias_pairs.jsonl")),
                    "refusal_detector": validate_refusal_detector()}
    if not args.fast:
        suites["instructions"] = run_instructions(bundle, load_jsonl("instructions.jsonl"))
        suites["safety"] = run_safety(bundle, load_jsonl("safety.jsonl"))
        suites["determinism"] = run_determinism(bundle)
    if args.variant == "fp32" and not args.no_quantization:
        suites["quantization"] = [run_drift(bundle, load(v), suites["mcq"], mcq_items) for v in ("int8", "int4")]

    report = {"model": bundle.name, "suites": suites}
    flat = flatten(report)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    failures = {name: check_gates(flat, thresholds[name]) for name in ("regression", "release")}
    report.update(metrics=flat, gate_failures=failures, blocking_profile=args.profile)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = to_markdown(report, flat, thresholds, failures, args.profile)
    (args.out / "model_report.md").write_text(md, encoding="utf-8")
    print(md)
    if failures[args.profile] and not args.no_gate:
        print("Quality gate FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
