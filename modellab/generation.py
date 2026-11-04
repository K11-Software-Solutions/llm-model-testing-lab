"""Text generation through the model's chat template."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from modellab.loader import ModelBundle


@dataclass
class Generation:
    text: str
    new_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return self.new_tokens / self.seconds if self.seconds else 0.0


@torch.no_grad()
def chat(bundle: ModelBundle, prompt: str, max_new_tokens: int = 96, temperature: float = 0.0,
         seed: int | None = None, system: str | None = None) -> Generation:
    """One user turn. temperature=0 is greedy (deterministic); >0 samples."""
    tok = bundle.tokenizer
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id or tok.eos_token_id}
    if temperature > 0:
        if seed is not None:
            torch.manual_seed(seed)
        kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    else:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    start = time.perf_counter()
    out = bundle.model.generate(**inputs, **kwargs)
    seconds = time.perf_counter() - start
    new = out[0][inputs["input_ids"].shape[1]:]
    return Generation(tok.decode(new, skip_special_tokens=True).strip(), len(new), seconds)
