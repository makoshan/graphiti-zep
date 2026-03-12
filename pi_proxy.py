"""Lightweight OpenAI-compatible proxy that routes LLM calls through pi CLI.

Runs as a local HTTP server on port 8002.
graphiti-zep uses LLM_BASE_URL=http://127.0.0.1:8002 to reach it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Load .env from same directory as this file
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI()

# Serialize all LLM calls: only 1 concurrent pi CLI subprocess, with cooldown
_LLM_LOCK = asyncio.Semaphore(int(os.environ.get("PI_PROXY_CONCURRENCY", "3")))
_CALL_COOLDOWN = float(os.environ.get("PI_PROXY_COOLDOWN", "0"))
_last_call_time: float = 0.0

# Embedding passthrough: forward /embeddings to real OpenAI
_EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
_EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1")

# Locate pi CLI
_PI_CLI = Path(
    os.environ.get("PI_CLI_PATH", "")
    or "/Users/thursday/.local/share/mise/installs/node/25.5.0/lib/node_modules/openclaw/node_modules/@mariozechner/pi-coding-agent/dist/cli.js"
)
_NODE = "node"
_TIMEOUT = float(os.environ.get("PI_PROXY_TIMEOUT", "120"))
_DEFAULT_PROVIDER = (
    os.environ.get("PI_PROVIDER")
    or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "kimi-coding")
)
_DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-3-5-haiku-latest",
    "openai-codex": "gpt-5.4",
    "kimi-coding": "k2p5",
}
_DEFAULT_MODEL = os.environ.get("PI_MODEL") or os.environ.get(
    "LLM_MODEL_NAME",
    _DEFAULT_MODEL_BY_PROVIDER.get(_DEFAULT_PROVIDER, "k2p5"),
)


def _build_prompt(messages: list[dict]) -> str:
    """Flatten OpenAI messages into a single prompt string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "user":
            parts.append(content)
        else:
            parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _resolve_model(request_body: dict) -> str:
    return request_body.get("model") or _DEFAULT_MODEL


async def _call_pi(prompt: str, model: str) -> str:
    global _last_call_time
    async with _LLM_LOCK:
        # Enforce cooldown between calls
        elapsed = time.time() - _last_call_time
        if elapsed < _CALL_COOLDOWN:
            await asyncio.sleep(_CALL_COOLDOWN - elapsed)

        proc = await asyncio.create_subprocess_exec(
            _NODE, str(_PI_CLI),
            "--provider", _DEFAULT_PROVIDER,
            "--model", model,
            "--no-tools",
            "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise HTTPException(status_code=504, detail=f"pi CLI timed out after {_TIMEOUT}s")
        finally:
            _last_call_time = time.time()

        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise HTTPException(status_code=500, detail=f"pi CLI failed: {err}")

        return stdout.decode("utf-8", errors="replace").strip()


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    prompt = _build_prompt(messages)
    model = _resolve_model(body)
    text = await _call_pi(prompt, model)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Forward embedding requests to real OpenAI."""
    body = await request.body()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{_EMBEDDING_BASE_URL.rstrip('/')}/embeddings",
            content=body,
            headers={
                "Authorization": f"Bearer {_EMBEDDING_API_KEY}",
                "Content-Type": "application/json",
            },
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": _DEFAULT_MODEL, "object": "model"}],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PI_PROXY_PORT", "8002"))
    print(f"Starting pi proxy on port {port}, pi CLI: {_PI_CLI}")
    uvicorn.run(app, host="127.0.0.1", port=port)
