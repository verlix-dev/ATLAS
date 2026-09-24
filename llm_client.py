"""Concrete LLM provider client for ATLAS.

This is the only file in the project that imports a real LLM SDK.
atlas/ never does — atlas/llm.py stays provider-agnostic by design
(context(), validate(), the eight gates work on plain text regardless
of who produced it). Both run_atlas.py (CLI) and server.py (web) import
from here.
"""

from __future__ import annotations

import json
from typing import Callable

from atlas import llm  # for llm.SYSTEM_PROMPT only — no execution, no dataset access

# Verified against this account's client.models.list(). Groq's free-tier lineup
# changes: llama-3.3-70b-versatile 404s here (billing-only). Re-check with
# `Groq().models.list()` before changing, and override per-run with --llm MODEL.
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"


def groq_proposer(model: str = DEFAULT_LLM_MODEL, timeout: float = 120.0) -> Callable[[dict], str]:
    """Free-tier hypothesis proposer via Groq. Same Callable[[dict], str]
    contract as any proposer — atlas/ cannot tell the difference.
    """
    try:
        from groq import Groq
    except ModuleNotFoundError as exc:
        raise SystemExit("this proposer requires the groq package: pip install groq") from exc

    client = Groq(timeout=timeout)  # reads GROQ_API_KEY from the environment

    def propose(payload: dict) -> str:
        message = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": llm.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
        )
        return message.choices[0].message.content

    return propose