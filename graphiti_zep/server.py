"""Graphiti-Zep Server: Zep-compatible HTTP API backed by Graphiti + Neo4j.

Run with:
    uv run uvicorn graphiti_zep.server:app --host 127.0.0.1 --port 8000
Or:
    uv run graphiti-zep
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from anthropic import AsyncAnthropic
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.errors import RateLimitError as GraphitiRateLimitError
from graphiti_core.nodes import EntityNode, EpisodicNode

logger = logging.getLogger(__name__)

from graphiti_zep.utils import (
    resolve_schema_refs as _resolve_schema_refs,
    fix_field_names as _fix_field_names,
    unwrap_structured_payload as _unwrap_structured_payload,
)


# ── Custom LLM Client for OpenAI-compatible providers ──────────────────

class ChatCompletionsClient(OpenAIClient):
    """OpenAI-compatible client using chat.completions for structured outputs.

    Replaces the default Responses API (`/v1/responses`) with
    `/v1/chat.completions` + `response_format: json_object`, so any
    OpenAI-compatible provider (Moonshot, Qwen, etc.) works with graphiti-core.
    """

    def __init__(self, config=None, cache=False, client=None, **kwargs):
        super().__init__(config, cache, client, **kwargs)
        if config is None:
            from graphiti_core.llm_client.config import LLMConfig
            config = LLMConfig()
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=300.0,
        )

    async def _create_structured_completion(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float | None,
        max_tokens: int,
        response_model: type[BaseModel],
        reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        parameters = _resolve_schema_refs(response_model.model_json_schema())

        enhanced_messages = list(messages)
        enhanced_messages.insert(1, {
            "role": "system",
            "content": (
                "CRITICAL: Your entire response must be a single valid JSON object. "
                "Use the EXACT field names from the schema provided. "
                "Do NOT include the schema definition itself — produce actual data values. "
                "Do NOT wrap the JSON in markdown code fences."
            ),
        })

        response = await self.client.chat.completions.create(
            model=model,
            messages=enhanced_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        body = response.choices[0].message.content or "{}"

        stripped = body.strip()
        if stripped.startswith("```"):
            stripped = "\n".join(stripped.split("\n")[1:])
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()

        if stripped.startswith("["):
            list_fields = [
                k for k, v in parameters.get("properties", {}).items()
                if isinstance(v, dict) and v.get("type") == "array"
            ]
            key = list_fields[0] if list_fields else "items"
            stripped = json.dumps({key: json.loads(stripped)})

        content = stripped if stripped.startswith("{") else "{}"
        try:
            parsed = json.loads(content)
            parsed = _unwrap_structured_payload(parsed, parameters)
            parsed = _fix_field_names(parsed, parameters)
            content = json.dumps(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        usage = response.usage
        return SimpleNamespace(
            output_text=content,
            usage=SimpleNamespace(
                input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            ),
        )


# ── Settings ───────────────────────────────────────────────────────────

class Settings(BaseSettings):
    graphiti_api_key: str = "local-graphiti"
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    llm_api_style: str = "openai"
    llm_api_key: str
    llm_base_url: str | None = None
    llm_model_name: str | None = None
    llm_small_model_name: str | None = None
    embedding_api_key: str
    embedding_base_url: str | None = None
    embedding_model_name: str | None = None
    embedding_batch_size: int = 0  # 0 = no limit; set to 10 for DashScope
    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: str = "data"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors: list[str] = []
        if not self.neo4j_password:
            errors.append("NEO4J_PASSWORD is required")
        if not self.llm_api_key:
            errors.append("LLM_API_KEY is required")
        if not self.embedding_api_key:
            errors.append("EMBEDDING_API_KEY is required")
        if self.llm_api_style.lower() not in ("openai", "anthropic"):
            errors.append(f"LLM_API_STYLE must be 'openai' or 'anthropic', got '{self.llm_api_style}'")
        return errors


SETTINGS = Settings()

# Auto-map OPENAI_* env vars for graphiti-core compatibility.
# graphiti-core reads OPENAI_API_KEY/OPENAI_BASE_URL internally for some codepaths.
# If user only set LLM_*, we bridge them so graphiti-core can find them.
import os as _os
if SETTINGS.llm_api_key and not _os.environ.get("OPENAI_API_KEY"):
    _os.environ["OPENAI_API_KEY"] = SETTINGS.llm_api_key
if SETTINGS.llm_base_url and not _os.environ.get("OPENAI_BASE_URL"):
    _os.environ["OPENAI_BASE_URL"] = SETTINGS.llm_base_url

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = (APP_DIR / SETTINGS.data_dir).resolve()
GROUPS_FILE = DATA_DIR / "groups.json"
THREADS_FILE = DATA_DIR / "threads.json"


# ── Persistence helpers ────────────────────────────────────────────────

def ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (GROUPS_FILE, THREADS_FILE):
        if not path.exists():
            path.write_text("{}", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    ensure_store()
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_store()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Auth & Helpers ─────────────────────────────────────────────────────

def require_auth(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {SETTINGS.graphiti_api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def python_type(type_name: str | None) -> type[Any]:
    normalized = (type_name or "string").strip().lower()
    if normalized in {"string", "str", "text"}:
        return str
    if normalized in {"int", "integer"}:
        return int
    if normalized in {"float", "number", "double"}:
        return float
    if normalized in {"bool", "boolean"}:
        return bool
    return str


def build_entity_models(ontology: dict[str, Any]) -> dict[str, type[BaseModel]]:
    entity_models: dict[str, type[BaseModel]] = {}
    for entity in ontology.get("entity_types", []):
        name = entity.get("name")
        if not name:
            continue
        fields: dict[str, tuple[Any, None]] = {}
        for attr in entity.get("attributes", []):
            attr_name = attr.get("name")
            if not attr_name:
                continue
            fields[attr_name] = (python_type(attr.get("type")) | None, None)
        entity_models[name] = create_model(name, **fields)
    return entity_models


def build_edge_models(ontology: dict[str, Any]) -> dict[str, type[BaseModel]]:
    edge_models: dict[str, type[BaseModel]] = {}
    for edge in ontology.get("edge_types", []):
        name = edge.get("name")
        if not name:
            continue
        fields: dict[str, tuple[Any, None]] = {}
        for attr in edge.get("attributes", []):
            attr_name = attr.get("name")
            if not attr_name:
                continue
            fields[attr_name] = (python_type(attr.get("type")) | None, None)
        edge_models[name] = create_model(name, **fields)
    return edge_models


def build_edge_type_map(ontology: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    edge_type_map: dict[tuple[str, str], list[str]] = {}
    for edge in ontology.get("edge_types", []):
        edge_name = edge.get("name")
        if not edge_name:
            continue
        pairs = edge.get("source_targets") or [{"source": "Entity", "target": "Entity"}]
        for pair in pairs:
            source = pair.get("source") or "Entity"
            target = pair.get("target") or "Entity"
            edge_type_map.setdefault((source, target), []).append(edge_name)
    return edge_type_map or {("Entity", "Entity"): []}


# ── Serialization ──────────────────────────────────────────────────────

def serialize_node(node: EntityNode | EpisodicNode) -> dict[str, Any]:
    return {
        "uuid_": getattr(node, "uuid", None) or getattr(node, "uuid_", ""),
        "name": getattr(node, "name", ""),
        "group_id": getattr(node, "group_id", ""),
        "labels": getattr(node, "labels", []),
        "summary": getattr(node, "summary", "") or "",
        "attributes": getattr(node, "attributes", {}) or {},
        "created_at": str(getattr(node, "created_at", "") or ""),
        "valid_at": str(getattr(node, "valid_at", "") or ""),
        "invalid_at": str(getattr(node, "invalid_at", "") or ""),
        "expired_at": str(getattr(node, "expired_at", "") or ""),
    }


def serialize_edge(edge: EntityEdge) -> dict[str, Any]:
    return {
        "uuid_": getattr(edge, "uuid", None) or getattr(edge, "uuid_", ""),
        "name": getattr(edge, "name", "") or "",
        "fact": getattr(edge, "fact", "") or "",
        "group_id": getattr(edge, "group_id", "") or "",
        "source_node_uuid": getattr(edge, "source_node_uuid", "") or "",
        "target_node_uuid": getattr(edge, "target_node_uuid", "") or "",
        "attributes": getattr(edge, "attributes", {}) or {},
        "created_at": str(getattr(edge, "created_at", "") or ""),
        "valid_at": str(getattr(edge, "valid_at", "") or ""),
        "invalid_at": str(getattr(edge, "invalid_at", "") or ""),
        "expired_at": str(getattr(edge, "expired_at", "") or ""),
    }


# ── Request/Response Models ────────────────────────────────────────────

class GroupCreateRequest(BaseModel):
    group_id: str
    name: str
    description: str = ""


class OntologyRequest(BaseModel):
    entities: dict[str, Any] = Field(default_factory=dict)
    edges: dict[str, Any] = Field(default_factory=dict)


class EpisodeIn(BaseModel):
    content: str
    type: str = "text"


class EpisodeBatchRequest(BaseModel):
    episodes: list[EpisodeIn]


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    scope: str = "edges"
    reranker: str | None = None


class ThreadCreateRequest(BaseModel):
    thread_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] | None = None


# ── Cross-encoder passthrough ──────────────────────────────────────────

class PassthroughCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, float(len(passages) - index)) for index, passage in enumerate(passages)]


# ── LLM helpers ────────────────────────────────────────────────────────

def normalize_anthropic_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1/messages"):
        return normalized[: -len("/v1/messages")]
    if normalized.endswith("/v1"):
        return normalized[: -len("/v1")]
    return normalized


# ── Embedding batch wrapper ────────────────────────────────────────────

class BatchedEmbedder:
    """Wraps an OpenAIEmbedder to split large batches for providers with limits.

    DashScope, for example, only allows 10 texts per embedding request.
    Set EMBEDDING_BATCH_SIZE=10 in .env to enable automatic chunking.
    """

    def __init__(self, embedder: OpenAIEmbedder, max_batch_size: int):
        self._embedder = embedder
        self.max_batch_size = max_batch_size

    async def create(self, input_data: str) -> list[float]:
        return await self._embedder.create(input_data)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if len(input_data_list) <= self.max_batch_size:
            return await self._embedder.create_batch(input_data_list)
        results: list[list[float]] = []
        for i in range(0, len(input_data_list), self.max_batch_size):
            batch = input_data_list[i : i + self.max_batch_size]
            batch_result = await self._embedder.create_batch(batch)
            results.extend(batch_result)
        return results


# ── Retry logic ────────────────────────────────────────────────────────

def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable (rate limit OR timeout)."""
    seen: set[int] = set()
    current: Exception | None = exc
    while current and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GraphitiRateLimitError):
            return True
        message = str(current).lower()
        if "rate limit" in message or "too many requests" in message:
            return True
        if "timed out" in message or "timeout" in message or "read timeout" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


async def _add_episode_with_retry(
    graphiti: Graphiti,
    *,
    group_id: str,
    episode_index: int,
    episode_content: str,
    entity_types: dict[str, type[BaseModel]] | None,
    edge_types: dict[str, type[BaseModel]] | None,
    edge_type_map: dict[tuple[str, str], list[str]],
    max_attempts: int = 6,
    base_delay_seconds: int = 30,
):
    for attempt in range(1, max_attempts + 1):
        try:
            return await graphiti.add_episode(
                name=f"{group_id}-episode-{episode_index}",
                episode_body=episode_content,
                source_description="graphiti-zep batch import",
                reference_time=datetime.now(timezone.utc),
                group_id=group_id,
                entity_types=entity_types or None,
                edge_types=edge_types or None,
                edge_type_map=edge_type_map,
            )
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt == max_attempts:
                raise
            delay = min(180, base_delay_seconds * (2 ** (attempt - 1)))
            logger.warning(
                "Retryable error on episode %s for group %s: %s; retry in %ss (%s/%s)",
                episode_index,
                group_id,
                exc,
                delay,
                attempt,
                max_attempts - 1,
            )
            await asyncio.sleep(delay)


# ── Graphiti builder ───────────────────────────────────────────────────

def build_graphiti() -> Graphiti:
    errors = SETTINGS.validate_config()
    if errors:
        for err in errors:
            logger.error("Configuration error: %s", err)
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    llm_config = LLMConfig(
        api_key=SETTINGS.llm_api_key,
        model=SETTINGS.llm_model_name,
        base_url=SETTINGS.llm_base_url,
        small_model=SETTINGS.llm_small_model_name or SETTINGS.llm_model_name,
    )
    embedder_config = OpenAIEmbedderConfig(
        api_key=SETTINGS.embedding_api_key,
        base_url=SETTINGS.embedding_base_url,
        embedding_model=SETTINGS.embedding_model_name or "text-embedding-3-small",
    )
    llm_style = SETTINGS.llm_api_style.strip().lower()
    if llm_style == "anthropic":
        anthropic_client = AsyncAnthropic(
            api_key=SETTINGS.llm_api_key,
            base_url=normalize_anthropic_base_url(SETTINGS.llm_base_url),
            max_retries=2,
            timeout=900.0,
        )
        llm_client = AnthropicClient(config=llm_config, client=anthropic_client)
    elif llm_style == "openai":
        llm_client = ChatCompletionsClient(config=llm_config)
    else:
        raise ValueError(f"Unsupported LLM_API_STYLE: {SETTINGS.llm_api_style}")

    embedder = OpenAIEmbedder(config=embedder_config)
    if SETTINGS.embedding_batch_size > 0:
        logger.info("Embedding batch size limit: %d", SETTINGS.embedding_batch_size)
        embedder = BatchedEmbedder(embedder, SETTINGS.embedding_batch_size)

    return Graphiti(
        SETTINGS.neo4j_uri,
        SETTINGS.neo4j_user,
        SETTINGS.neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=PassthroughCrossEncoder(),
    )


# ── FastAPI App ────────────────────────────────────────────────────────

app = FastAPI(title="Graphiti-Zep", description="Zep-compatible knowledge graph API")

_graphiti: Graphiti | None = None


def get_graphiti() -> Graphiti:
    if _graphiti is None:
        raise RuntimeError("Graphiti not initialized")
    return _graphiti


@app.on_event("startup")
async def startup() -> None:
    global _graphiti
    ensure_store()
    logger.info("=" * 50)
    logger.info("Graphiti-Zep starting...")
    logger.info("  Neo4j:     %s", SETTINGS.neo4j_uri)
    logger.info("  LLM:       %s (%s)", SETTINGS.llm_api_style, SETTINGS.llm_model_name or "default")
    logger.info("  Embedding: %s", SETTINGS.embedding_model_name or "text-embedding-3-small")
    if SETTINGS.embedding_batch_size > 0:
        logger.info("  Embedding batch limit: %d", SETTINGS.embedding_batch_size)
    logger.info("=" * 50)
    _graphiti = build_graphiti()
    try:
        await _graphiti.build_indices_and_constraints()
        logger.info("Neo4j indices and constraints ready")
    except Exception as e:
        logger.warning("build_indices_and_constraints failed (non-fatal): %s", e)


@app.on_event("shutdown")
async def shutdown() -> None:
    global _graphiti
    if _graphiti is not None:
        await _graphiti.close()
        _graphiti = None


# ── Routes: Health ─────────────────────────────────────────────────────

@app.get("/healthcheck")
async def healthcheck() -> JSONResponse:
    components: dict[str, str] = {}
    overall = "healthy"

    # Check Neo4j
    try:
        graphiti = get_graphiti()
        driver = graphiti.driver
        records = await driver.execute_query("RETURN 1 AS n")
        components["neo4j"] = "connected"
    except Exception as e:
        components["neo4j"] = f"error: {e}"
        overall = "degraded"

    # Check Graphiti init
    if _graphiti is not None:
        components["graphiti"] = "initialized"
    else:
        components["graphiti"] = "not initialized"
        overall = "degraded"

    components["llm_style"] = SETTINGS.llm_api_style
    components["llm_model"] = SETTINGS.llm_model_name or "default"

    status_code = 200 if overall == "healthy" else 503
    return JSONResponse({"status": overall, "components": components}, status_code=status_code)


# ── Routes: Groups ────────────────────────────────────────────────────

@app.post("/v1/groups")
async def create_group(payload: GroupCreateRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
    groups = load_json(GROUPS_FILE)
    groups[payload.group_id] = {
        "group_id": payload.group_id,
        "name": payload.name,
        "description": payload.description,
        "ontology": None,
    }
    save_json(GROUPS_FILE, groups)
    return groups[payload.group_id]


@app.post("/v1/groups/{group_id}/ontology")
async def set_ontology(
    group_id: str,
    payload: OntologyRequest,
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    groups = load_json(GROUPS_FILE)
    group = groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    group["ontology"] = {
        "entity_types": list(payload.entities.values()),
        "edge_types": list(payload.edges.values()),
    }
    groups[group_id] = group
    save_json(GROUPS_FILE, groups)
    return {"ok": True}


@app.post("/v1/groups/{group_id}/episodes:batch")
async def add_episode_batch(
    group_id: str,
    payload: EpisodeBatchRequest,
    _: None = Depends(require_auth),
) -> list[dict[str, Any]]:
    groups = load_json(GROUPS_FILE)
    group = groups.get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    ontology = group.get("ontology") or {}
    entity_types = build_entity_models(ontology)
    edge_types = build_edge_models(ontology)
    edge_type_map = build_edge_type_map(ontology)

    graphiti = get_graphiti()
    results: list[dict[str, Any]] = []
    for index, episode in enumerate(payload.episodes, start=1):
        if index > 1:
            await asyncio.sleep(1)
        try:
            outcome = await _add_episode_with_retry(
                graphiti,
                group_id=group_id,
                episode_index=index,
                episode_content=episode.content,
                entity_types=entity_types,
                edge_types=edge_types,
                edge_type_map=edge_type_map,
            )
        except Exception as exc:
            logger.exception("Failed to add episode %s for group %s", index, group_id)
            status_code = 429 if _is_retryable_error(exc) else 500
            detail_prefix = (
                "Graphiti batch ingest rate-limited"
                if status_code == 429
                else "Graphiti batch ingest failed"
            )
            raise HTTPException(
                status_code=status_code,
                detail=f"{detail_prefix} for episode {index}: {exc}",
            ) from exc
        results.append(
            {
                "uuid_": outcome.episode.uuid,
                "processed": True,
            }
        )
    return results


# ── Routes: Nodes, Edges, Episodes ─────────────────────────────────────

@app.get("/v1/episodes/{episode_uuid}")
async def get_episode(episode_uuid: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    graphiti = get_graphiti()
    episode = await graphiti.nodes.episode.get_by_uuid(episode_uuid)
    data = serialize_node(episode)
    data["processed"] = True
    return data


@app.get("/v1/groups/{group_id}/nodes")
async def get_group_nodes(
    group_id: str,
    limit: int = 100,
    uuid_cursor: str | None = None,
    _: None = Depends(require_auth),
) -> list[dict[str, Any]]:
    graphiti = get_graphiti()
    nodes = await graphiti.nodes.entity.get_by_group_ids([group_id], limit=limit, uuid_cursor=uuid_cursor)
    return [serialize_node(node) for node in nodes]


@app.get("/v1/groups/{group_id}/edges")
async def get_group_edges(
    group_id: str,
    limit: int = 100,
    uuid_cursor: str | None = None,
    _: None = Depends(require_auth),
) -> list[dict[str, Any]]:
    graphiti = get_graphiti()
    edges = await graphiti.edges.entity.get_by_group_ids([group_id], limit=limit, uuid_cursor=uuid_cursor)
    return [serialize_edge(edge) for edge in edges]


@app.get("/v1/nodes/{node_uuid}")
async def get_node(node_uuid: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    graphiti = get_graphiti()
    node = await graphiti.nodes.entity.get_by_uuid(node_uuid)
    return serialize_node(node)


@app.get("/v1/nodes/{node_uuid}/edges")
async def get_node_edges(node_uuid: str, _: None = Depends(require_auth)) -> list[dict[str, Any]]:
    graphiti = get_graphiti()
    edges = await graphiti.edges.entity.get_by_node_uuid(node_uuid)
    return [serialize_edge(edge) for edge in edges]


@app.get("/v1/edges/{edge_uuid}")
async def get_edge(edge_uuid: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    graphiti = get_graphiti()
    edge = await graphiti.edges.entity.get_by_uuid(edge_uuid)
    return serialize_edge(edge)


# ── Routes: Search ─────────────────────────────────────────────────────

@app.post("/v1/groups/{group_id}/search")
async def search_group(
    group_id: str,
    payload: SearchRequest,
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    graphiti = get_graphiti()
    edges = await graphiti.search(payload.query, group_ids=[group_id], num_results=payload.limit)
    node_map: dict[str, dict[str, Any]] = {}
    if payload.scope != "edges":
        for edge in edges:
            for node_uuid in (edge.source_node_uuid, edge.target_node_uuid):
                if not node_uuid or node_uuid in node_map:
                    continue
                try:
                    node = await graphiti.nodes.entity.get_by_uuid(node_uuid)
                    node_map[node_uuid] = serialize_node(node)
                except Exception:
                    continue
    return {
        "facts": [serialize_edge(edge) for edge in edges],
        "edges": [serialize_edge(edge) for edge in edges],
        "nodes": list(node_map.values()),
    }


# ── Routes: Groups CRUD ───────────────────────────────────────────────

@app.delete("/v1/groups/{group_id}")
async def delete_group(group_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    graphiti = get_graphiti()
    await graphiti.nodes.entity.delete_by_group_id(group_id)
    await graphiti.nodes.episode.delete_by_group_id(group_id)
    groups = load_json(GROUPS_FILE)
    groups.pop(group_id, None)
    save_json(GROUPS_FILE, groups)
    return {"ok": True}


# ── Routes: Threads ────────────────────────────────────────────────────

@app.get("/v1/threads")
async def list_threads(
    user_id: str | None = None,
    limit: int = 20,
    page_token: str | None = None,
    _: None = Depends(require_auth),
) -> dict[str, Any]:
    threads = load_json(THREADS_FILE)
    values = list(threads.values())
    if user_id:
        values = [item for item in values if item.get("user_id") == user_id]
    if page_token:
        values = [item for item in values if item.get("thread_id") > page_token]
    values = values[:limit]
    return {"threads": values, "next_page_token": values[-1]["thread_id"] if len(values) == limit else None}


@app.post("/v1/threads")
async def create_thread(payload: ThreadCreateRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
    threads = load_json(THREADS_FILE)
    thread_id = payload.thread_id or f"thread_{uuid4().hex[:12]}"
    thread = {
        "thread_id": thread_id,
        "user_id": payload.user_id,
        "metadata": payload.metadata or {},
    }
    threads[thread_id] = thread
    save_json(THREADS_FILE, threads)
    return thread


@app.get("/v1/threads/{thread_id}")
async def get_thread(thread_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    threads = load_json(THREADS_FILE)
    thread = threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.delete("/v1/threads/{thread_id}")
async def delete_thread(thread_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
    threads = load_json(THREADS_FILE)
    threads.pop(thread_id, None)
    save_json(THREADS_FILE, threads)
    return {"ok": True}


# ── CLI entry point ───────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "graphiti_zep.server:app",
        host=SETTINGS.host,
        port=SETTINGS.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
