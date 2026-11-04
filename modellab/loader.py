"""Load the model under test, in several variants.

Variants:
    fp32    the published weights, full precision (the reference)
    int8    weights rounded to 8-bit per output channel, then de-quantized
    int4    the same at 4 bits: a much more aggressive compression
    random  same architecture, random weights: a negative control that every
            quality gate must reject

The int8/int4 variants use *simulated* weight-only quantization so the code
is short and works on any PyTorch build. Production runtimes (llama.cpp, GPTQ,
AWQ, torchao) use smarter schemes, but the testing question is identical:
how far does the compressed model drift from the reference?
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from modellab import DEFAULT_MODEL_ID

VARIANTS = ("fp32", "int8", "int4", "random")


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    model_id: str
    variant: str

    @property
    def name(self) -> str:
        return f"{self.model_id}:{self.variant}"


@torch.no_grad()
def quantize_linear_weights(model: torch.nn.Module, bits: int) -> int:
    """Symmetric per-output-channel weight quantization, in place. Returns #layers changed."""
    qmax = 2 ** (bits - 1) - 1
    changed = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
            w = module.weight
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
            module.weight.copy_((w / scale).round().clamp(-qmax, qmax) * scale)
            changed += 1
    return changed


def load(variant: str = "fp32", model_id: str | None = None, seed: int = 0) -> ModelBundle:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if variant == "random":
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id), dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        if variant in ("int8", "int4"):
            quantize_linear_weights(model, bits=int(variant[3:]))
    model.eval()
    return ModelBundle(model, tokenizer, model_id, variant)
