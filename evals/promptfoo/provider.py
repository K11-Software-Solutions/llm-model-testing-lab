"""Promptfoo custom provider: runs the local Hugging Face model with greedy decoding.

Variant and model come from env vars MODEL_VARIANT (fp32|int8|int4|random) and MODEL_ID.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modellab.generation import chat  # noqa: E402
from modellab.loader import load  # noqa: E402

_bundle = None


def call_api(prompt: str, options: dict, context: dict) -> dict:
    global _bundle
    if _bundle is None:
        _bundle = load(os.environ.get("MODEL_VARIANT", "fp32"))
    max_tokens = int((options.get("config") or {}).get("max_new_tokens", 80))
    gen = chat(_bundle, prompt, max_new_tokens=max_tokens)
    return {"output": gen.text, "metadata": {"model": _bundle.name, "tokens_per_second": round(gen.tokens_per_second, 1)}}
