"""
ScoutAI - Local LLM Backend  (Feature 15)
==========================================
Provides streaming text generation for the LLM-powered scouting narrative.

Auto-detects backend in priority order:
  1. ollama  (if localhost:11434 responds and a compatible model is available)
  2. transformers  (if model is cached locally — slow on CPU, uses TextIteratorStreamer)
  3. None  (caller falls back to template-only report)

Preferred model: configurable via SCOUTAI_OLLAMA_MODEL env var.
Default: "llama3" (already pulled) — swap to "phi3:mini" when available.

Usage (from async context):
    from llm import stream_llm, detect_backend
    backend = await detect_backend()
    async for token in stream_llm(prompt, max_tokens=240):
        yield token
"""

import asyncio
import logging
import os
from typing import AsyncIterator

logger = logging.getLogger("scoutai.llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST  = os.getenv("SCOUTAI_OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("SCOUTAI_OLLAMA_MODEL", "llama3")   # swap to "phi3:mini" when pulled
HF_MODEL     = os.getenv("SCOUTAI_HF_MODEL",     "microsoft/Phi-3-mini-4k-instruct")

# Generation parameters
_GEN_PARAMS = {
    "temperature":     0.60,   # deterministic enough for factual reports
    "top_p":           0.90,
    "repeat_penalty":  1.10,   # reduce repetitive phrasing
}

_backend:   str | None = None   # cached: "ollama" | "transformers" | None
_hf_pipe               = None   # cached transformers pipeline


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

async def detect_backend() -> str | None:
    """
    Probe available backends. Returns the fastest usable one, or None.
    Result is cached after the first call.
    """
    global _backend
    if _backend is not None:
        return _backend

    # ── 1. ollama ──────────────────────────────────────────────────────────
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            # Accept the configured model or any phi / llama variant
            wanted = [OLLAMA_MODEL]
            fallbacks = [m for m in models if any(
                k in m.lower() for k in ("phi", "llama", "mistral", "gemma")
            )]
            for candidate in wanted + fallbacks:
                if any(candidate in m for m in models):
                    # Store the resolved model name so _ollama_stream can use it
                    global _resolved_model
                    _resolved_model = next(m for m in models if candidate in m)
                    logger.info("LLM backend: ollama  model=%s", _resolved_model)
                    _backend = "ollama"
                    return _backend
            logger.warning(
                "ollama is running but no usable model found. "
                "Run: ollama pull phi3:mini  or  ollama pull llama3"
            )
    except Exception as exc:
        logger.debug("ollama probe failed: %s", exc)

    # ── 2. transformers (only if model is already cached locally) ──────────
    try:
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        slug  = HF_MODEL.replace("/", "--").replace(":", "--")
        if os.path.exists(cache) and any(slug in d for d in os.listdir(cache)):
            logger.info("LLM backend: transformers  model=%s", HF_MODEL)
            _backend = "transformers"
            return _backend
    except Exception as exc:
        logger.debug("transformers probe failed: %s", exc)

    logger.info("LLM backend: none (template fallback will be used)")
    _backend = None
    return None


def reset_backend() -> None:
    """Clear the cached backend detection — call after installing a new model."""
    global _backend
    _backend = None
    logger.info("LLM backend cache cleared")


def backend_label() -> str:
    """Human-readable label for the current backend (for the /api/status endpoint)."""
    if _backend == "ollama":
        model = globals().get("_resolved_model", OLLAMA_MODEL)
        return f"ollama/{model}"
    if _backend == "transformers":
        return f"transformers/{HF_MODEL}"
    return "template"


# ---------------------------------------------------------------------------
# Public streaming interface
# ---------------------------------------------------------------------------

async def stream_llm(
    prompt:     str,
    max_tokens: int = 250,
) -> AsyncIterator[str]:
    """
    Async generator that yields text tokens from the local LLM.

    If no LLM backend is available, yields nothing (caller uses template).
    """
    backend = await detect_backend()
    if backend == "ollama":
        async for tok in _ollama_stream(prompt, max_tokens):
            yield tok
    elif backend == "transformers":
        async for tok in _transformers_stream(prompt, max_tokens):
            yield tok
    # else: no yield → caller detects empty and falls back to template


# ---------------------------------------------------------------------------
# ollama backend
# ---------------------------------------------------------------------------

_resolved_model: str = OLLAMA_MODEL   # updated by detect_backend()


async def _ollama_stream(prompt: str, max_tokens: int) -> AsyncIterator[str]:
    """Stream tokens from ollama's /api/generate endpoint (newline-delimited JSON)."""
    import httpx
    import json as _json

    payload = {
        "model":  _resolved_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict":    max_tokens,
            "temperature":    _GEN_PARAMS["temperature"],
            "top_p":          _GEN_PARAMS["top_p"],
            "repeat_penalty": _GEN_PARAMS["repeat_penalty"],
        },
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{OLLAMA_HOST}/api/generate", json=payload
            ) as resp:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                        tok = obj.get("response", "")
                        if tok:
                            yield tok
                        if obj.get("done"):
                            break
                    except _json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("ollama stream error: %s", exc)


# ---------------------------------------------------------------------------
# transformers backend (CPU/GPU, runs generation in a thread)
# ---------------------------------------------------------------------------

async def _transformers_stream(prompt: str, max_tokens: int) -> AsyncIterator[str]:
    """
    Run HuggingFace generation in a background thread and yield tokens
    via TextIteratorStreamer so the event loop isn't blocked.
    """
    global _hf_pipe
    from threading import Thread
    import asyncio

    # Load pipeline once (heavy — only happens if model is cached)
    if _hf_pipe is None:
        import torch
        from transformers import pipeline, BitsAndBytesConfig
        logger.info("Loading transformers model %s …", HF_MODEL)
        try:
            bnb = BitsAndBytesConfig(load_in_4bit=True)
            _hf_pipe = pipeline(
                "text-generation", model=HF_MODEL,
                torch_dtype=torch.float16, quantization_config=bnb,
                device_map="auto",
            )
        except Exception:
            _hf_pipe = pipeline(
                "text-generation", model=HF_MODEL,
                torch_dtype=torch.float32, device_map="auto",
            )
        logger.info("Transformers model ready")

    from transformers import TextIteratorStreamer
    tokenizer = _hf_pipe.tokenizer
    streamer  = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=_GEN_PARAMS["temperature"],
        top_p=_GEN_PARAMS["top_p"],
        repetition_penalty=_GEN_PARAMS["repeat_penalty"],
        do_sample=True,
        streamer=streamer,
    )

    def _run():
        import torch
        with torch.no_grad():
            _hf_pipe.model.generate(**gen_kwargs)

    thread = Thread(target=_run, daemon=True)
    thread.start()

    for tok in streamer:
        yield tok
        await asyncio.sleep(0)   # yield control back to event loop
