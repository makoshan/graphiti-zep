"""OpenAI-compatible proxy backed by long-lived pi RPC workers.

Runs as a local HTTP server on port 8002.
graphiti-zep uses LLM_BASE_URL=http://127.0.0.1:8002 to reach it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Load .env from same directory as this file
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI()

# Pool of long-lived RPC workers. ``PI_PROXY_CONCURRENCY`` now means
# how many pi RPC sessions may run in parallel.
_WORKER_COUNT = max(1, int(os.environ.get("PI_PROXY_CONCURRENCY", "1")))
_CALL_COOLDOWN = float(os.environ.get("PI_PROXY_COOLDOWN", "0"))

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
_DEFAULT_THINKING = os.environ.get(
    "PI_PROXY_THINKING",
    "off" if _DEFAULT_PROVIDER in {"openai", "openai-codex"} else "",
).strip()
_REQUEST_TIMEOUT = max(_TIMEOUT, 30.0)

_pool_lock = asyncio.Lock()
_worker_queue: asyncio.Queue["PiRpcWorker"] | None = None
_workers: list["PiRpcWorker"] = []


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


class PiRpcWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.provider = _DEFAULT_PROVIDER
        self.model = _DEFAULT_MODEL
        self.proc: asyncio.subprocess.Process | None = None
        self.stdout_task: asyncio.Task[None] | None = None
        self.stderr_task: asyncio.Task[None] | None = None
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.agent_end: asyncio.Future[dict[str, Any]] | None = None
        self.stderr_tail = ""
        self.request_id = 0
        self.last_call_time = 0.0

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.provider == "openai" and not env.get("OPENAI_API_KEY"):
            fallback_key = env.get("LLM_API_KEY") or env.get("EMBEDDING_API_KEY") or ""
            if fallback_key:
                env["OPENAI_API_KEY"] = fallback_key
        return env

    async def ensure_started(self, model: str) -> None:
        if self.proc and self.proc.returncode is None and self.model == model:
            return
        await self.stop()
        self.model = model
        self.proc = await asyncio.create_subprocess_exec(
            _NODE,
            str(_PI_CLI),
            "--mode",
            "rpc",
            "--provider",
            self.provider,
            "--model",
            self.model,
            "--no-session",
            "--no-tools",
            *(["--thinking", _DEFAULT_THINKING] if _DEFAULT_THINKING else []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
        )
        self.stdout_task = asyncio.create_task(self._read_stdout())
        self.stderr_task = asyncio.create_task(self._read_stderr())
        await asyncio.sleep(0.1)
        if self.proc.returncode is not None:
            stderr = self.stderr_tail.strip()
            raise HTTPException(
                status_code=500,
                detail=f"pi RPC exited immediately with code {self.proc.returncode}: {stderr}",
            )

    async def stop(self) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        for task in (self.stdout_task, self.stderr_task):
            if task is not None:
                task.cancel()
        self.proc = None
        self.stdout_task = None
        self.stderr_task = None
        self._fail_pending("pi RPC worker stopped")

    def _fail_pending(self, message: str) -> None:
        error = RuntimeError(message)
        for future in list(self.pending.values()):
            if not future.done():
                future.set_exception(error)
        self.pending.clear()
        if self.agent_end is not None and not self.agent_end.done():
            self.agent_end.set_exception(error)
        self.agent_end = None

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        buffer = b""
        try:
            while True:
                chunk = await self.proc.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    line = raw_line.rstrip(b"\r").decode("utf-8", errors="replace").strip()
                    if line:
                        self._handle_line(line)
        finally:
            self._fail_pending(
                f"pi RPC stdout closed unexpectedly. stderr: {self.stderr_tail.strip()}"
            )

    async def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            chunk = await self.proc.stderr.read(4096)
            if not chunk:
                break
            self.stderr_tail = (self.stderr_tail + chunk.decode("utf-8", errors="replace"))[-8000:]

    def _handle_line(self, line: str) -> None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return
        if data.get("type") == "response" and data.get("id") in self.pending:
            future = self.pending.pop(data["id"])
            if not future.done():
                future.set_result(data)
            return
        if data.get("type") == "agent_end" and self.agent_end is not None and not self.agent_end.done():
            self.agent_end.set_result(data)

    async def _send(self, command: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        if not self.proc or not self.proc.stdin:
            raise HTTPException(status_code=500, detail="pi RPC worker is not running")
        self.request_id += 1
        req_id = f"worker-{self.worker_id}-req-{self.request_id}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.pending[req_id] = future
        payload = dict(command)
        payload["id"] = req_id
        self.proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await self.proc.stdin.drain()
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self.pending.pop(req_id, None)
            raise HTTPException(
                status_code=504,
                detail=f"Timed out waiting for pi RPC response to {command['type']}. stderr: {self.stderr_tail.strip()}",
            ) from exc
        if not response.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=response.get("error") or f"pi RPC command failed: {command['type']}",
            )
        return response.get("data") or {}

    async def prompt(self, prompt: str, model: str) -> str:
        await self.ensure_started(model)
        elapsed = time.time() - self.last_call_time
        if elapsed < _CALL_COOLDOWN:
            await asyncio.sleep(_CALL_COOLDOWN - elapsed)

        self.agent_end = asyncio.get_running_loop().create_future()
        try:
            await self._send({"type": "new_session"})
            await self._send({"type": "prompt", "message": prompt})
            try:
                await asyncio.wait_for(self.agent_end, timeout=_REQUEST_TIMEOUT)
            except asyncio.TimeoutError as exc:
                raise HTTPException(
                    status_code=504,
                    detail=f"pi RPC agent timed out after {_REQUEST_TIMEOUT}s",
                ) from exc
            data = await self._send({"type": "get_last_assistant_text"})
            return str(data.get("text", "")).strip()
        finally:
            self.last_call_time = time.time()
            self.agent_end = None


async def _ensure_worker_pool() -> asyncio.Queue[PiRpcWorker]:
    global _worker_queue
    if _worker_queue is not None:
        return _worker_queue
    async with _pool_lock:
        if _worker_queue is None:
            queue: asyncio.Queue[PiRpcWorker] = asyncio.Queue()
            for idx in range(_WORKER_COUNT):
                worker = PiRpcWorker(idx + 1)
                _workers.append(worker)
                await queue.put(worker)
            _worker_queue = queue
    return _worker_queue


@asynccontextmanager
async def _checkout_worker():
    queue = await _ensure_worker_pool()
    worker = await queue.get()
    try:
        yield worker
    finally:
        await queue.put(worker)


async def _call_pi(prompt: str, model: str) -> str:
    async with _checkout_worker() as worker:
        try:
            return await worker.prompt(prompt, model)
        except HTTPException:
            await worker.stop()
            raise
        except Exception as exc:
            await worker.stop()
            raise HTTPException(status_code=500, detail=f"pi RPC failed: {exc}") from exc


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


@app.on_event("shutdown")
async def shutdown() -> None:
    for worker in list(_workers):
        await worker.stop()


if __name__ == "__main__":
    port = int(os.environ.get("PI_PROXY_PORT", "8002"))
    print(f"Starting pi proxy on port {port}, pi CLI: {_PI_CLI}")
    uvicorn.run(app, host="127.0.0.1", port=port)
