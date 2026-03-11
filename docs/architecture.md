# Architecture

## Overview

Graphiti-Zep is a standalone HTTP service that provides a Zep-compatible REST API backed by [graphiti-core](https://github.com/getzep/graphiti) + Neo4j.

```
┌──────────────────┐     HTTP/REST      ┌─────────────────────────┐
│   Your App       │ ─────────────────▶ │   Graphiti-Zep Server   │
│                  │                     │   (FastAPI)             │
│  Python client   │                     ├─────────────────────────┤
│  or any HTTP     │                     │  LLM Client            │
│  client          │                     │  (OpenAI / Anthropic)  │
└──────────────────┘                     │                         │
                                         │  Embedder              │
                                         │  (OpenAI-compatible)   │
                                         │                         │
                                         │  graphiti-core          │
                                         └──────────┬──────────────┘
                                                    │
                                                    ▼
                                         ┌─────────────────────┐
                                         │       Neo4j         │
                                         │  (Graph Database)    │
                                         └─────────────────────┘
```

## Why a Standalone Service?

The original MiroFish project embedded graphiti-core directly into Flask. This caused several issues:

| Problem | In-process Approach | Standalone Service |
|---------|--------------------|--------------------|
| **Async/Sync conflict** | Flask is sync, graphiti-core is async → complex workarounds with background threads | FastAPI is natively async |
| **Dependency conflicts** | camel-ai needs neo4j==5.23, graphiti-core needs neo4j>=5.26 | Separate process = separate dependencies |
| **Fault isolation** | graphiti-core crash takes down the entire backend | Independent process, independent restarts |
| **Reusability** | Tightly coupled to MiroFish | Any app can call the HTTP API |

This architecture is what [MiroFish-local's TODO.md](https://github.com/tt-a1i/MiroFish-local/blob/feat/zep-localization-mvp/docs/zep-localization/TODO.md) recommended as a P1 improvement.

## Components

### Server (`graphiti_zep/server.py`)

The FastAPI application with:

- **Settings**: Pydantic-based config loaded from `.env` with startup validation
- **ChatCompletionsClient**: Custom graphiti-core LLM client that uses `chat.completions` + `response_format: json_object` instead of the Responses API, enabling any OpenAI-compatible provider
- **BatchedEmbedder**: Wrapper that splits large embedding batches for providers with size limits (e.g., DashScope max 10)
- **Retry logic**: Auto-retry on rate limits and timeouts with exponential backoff (up to 6 attempts, max 180s delay)
- **JSON normalization**: Fixes LLM output quirks (wrapper objects, renamed fields, nested values)

### Client (`graphiti_zep/client.py`)

Standalone Python client with no framework dependencies:

```python
from graphiti_zep import create_client, EpisodeData

with create_client("api-key", base_url="http://localhost:8000") as client:
    client.graph.create("my-graph", "Name", "Description")
    client.graph.add_batch("my-graph", [EpisodeData(data="...")])
    results = client.graph.search(graph_id="my-graph", query="...")
```

### Utilities (`graphiti_zep/utils.py`)

Pure functions for JSON schema normalization, importable without server dependencies:

- `resolve_schema_refs()` — Inline `$ref`/`$defs` for providers that don't support them
- `unwrap_structured_payload()` — Strip provider wrapper objects
- `fix_field_names()` — Map LLM-generated field names to schema-expected names
- `flatten_value()` — Convert nested objects to JSON strings for Neo4j

## API Surface

The API mirrors Zep Cloud's knowledge graph endpoints:

### Groups (Knowledge Graphs)
- `POST /v1/groups` — Create a graph
- `POST /v1/groups/{id}/ontology` — Define entity/edge schema
- `POST /v1/groups/{id}/episodes:batch` — Ingest text
- `GET /v1/groups/{id}/nodes` — List nodes (paginated)
- `GET /v1/groups/{id}/edges` — List edges (paginated)
- `POST /v1/groups/{id}/search` — Semantic search
- `DELETE /v1/groups/{id}` — Delete graph

### Resources
- `GET /v1/nodes/{uuid}`, `GET /v1/edges/{uuid}`, `GET /v1/episodes/{uuid}`
- `GET /v1/nodes/{uuid}/edges` — Node's connected edges

### Threads
- `GET/POST /v1/threads` — List/create
- `GET/DELETE /v1/threads/{id}` — Get/delete

## Data Flow: Episode Ingestion

When a text episode is ingested via `POST /v1/groups/{id}/episodes:batch`:

```
1. Text chunk received
       │
2. graphiti-core.add_episode()
       │
       ├── Entity extraction (LLM call)
       ├── Attribute extraction (LLM call)
       ├── Edge extraction (LLM call)
       ├── Entity deduplication (LLM call + embedding)
       ├── Edge deduplication (LLM call)
       └── Summary generation (LLM call)
       │
3. Results written to Neo4j
       │
4. ~15 LLM calls per episode
```

## Multi-Graph Isolation

Multiple knowledge graphs are isolated using `group_id`:

```cypher
-- Each entity/edge/episode node has a group_id property
MATCH (n:Entity) WHERE n.group_id = "project_abc" RETURN n
```

This allows multiple projects to share the same Neo4j instance without data leakage.

## Error Handling Strategy

```
LLM Call Failed
    │
    ├── Rate Limit (429) ──► Retry with exponential backoff (30s, 60s, 120s, 180s, 180s)
    ├── Timeout ───────────► Retry with exponential backoff
    ├── Auth Error (401) ──► Fail immediately
    └── Other Error ───────► Fail immediately
```

All retryable errors are logged with attempt count and delay duration.
