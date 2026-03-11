# Graphiti-Zep

Self-hosted **Zep-compatible** knowledge graph API backed by [Graphiti](https://github.com/getzep/graphiti) + Neo4j.

Drop-in replacement for Zep Cloud's knowledge graph endpoints — run entirely on your own infrastructure with any OpenAI or Anthropic-compatible LLM.

## Features

- **Zep-compatible REST API** — Groups, Episodes, Nodes, Edges, Threads, Search
- **Any LLM provider** — OpenAI, Anthropic, Kimi, Qwen, Moonshot, etc.
- **Custom ontology** — Define entity types, edge types, and attributes
- **Semantic search** — Search knowledge graph by natural language
- **Robust ingestion** — Auto-retry on rate limits and timeouts with exponential backoff
- **Neo4j storage** — Production-grade graph database
- **Python client** — Type-safe client with context manager support
- **Docker one-click** — Neo4j via `docker-compose up -d`
- **DashScope support** — Automatic embedding batch splitting
- **Enhanced healthcheck** — Component-level status (Neo4j, Graphiti, LLM)
- **Startup validation** — Fail fast on missing configuration

## Quick Start

### Prerequisites

- Python 3.11+
- Neo4j (via Docker or [Aura](https://neo4j.com/cloud/aura/))
- An LLM API key (OpenAI, Kimi, Qwen, etc.)
- An embedding API key (OpenAI `text-embedding-3-small`)

### 1. Start Neo4j

```bash
docker-compose up -d
# Wait for healthy status (~30s)
docker-compose ps
```

Or use an existing Neo4j instance — just set `NEO4J_URI` in `.env`.

### 2. Install & Configure

```bash
git clone https://github.com/yourname/graphiti-zep.git
cd graphiti-zep
cp .env.example .env
# Edit .env with your credentials
uv sync
```

### 3. Run the server

```bash
uv run graphiti-zep
# or
uv run uvicorn graphiti_zep.server:app --host 127.0.0.1 --port 8000
```

### 4. Use the Python client

```python
from graphiti_zep import create_client, EpisodeData

with create_client("local-graphiti", base_url="http://localhost:8000") as client:
    # Create a knowledge graph
    client.graph.create("my-graph", "My Knowledge Graph", "A test graph")

    # Define ontology (optional)
    client.graph.set_ontology(
        ["my-graph"],
        entities={"Person": {"name": "Person", "attributes": [{"name": "role", "type": "text"}]}},
        edges={"KNOWS": {"name": "KNOWS", "source_targets": [{"source": "Person", "target": "Person"}]}},
    )

    # Ingest text episodes
    episodes = [
        EpisodeData(data="Alice is a software engineer who works with Bob."),
        EpisodeData(data="Bob introduced Alice to Carol at the conference."),
    ]
    client.graph.add_batch("my-graph", episodes)

    # Search the graph
    results = client.graph.search(graph_id="my-graph", query="Who does Alice work with?")
    for fact in results.facts:
        print(f"  {fact.fact}")

    # List nodes
    nodes = client.graph.node.get_by_graph_id("my-graph")
    for node in nodes:
        print(f"  {node.name}: {node.summary}")
```

### Use with curl

```bash
# Health check (shows Neo4j and LLM status)
curl http://localhost:8000/healthcheck | python3 -m json.tool

# Create group
curl -X POST http://localhost:8000/v1/groups \
  -H "Authorization: Bearer local-graphiti" \
  -H "Content-Type: application/json" \
  -d '{"group_id": "test", "name": "Test", "description": ""}'

# Ingest episodes
curl -X POST http://localhost:8000/v1/groups/test/episodes:batch \
  -H "Authorization: Bearer local-graphiti" \
  -H "Content-Type: application/json" \
  -d '{"episodes": [{"content": "Alice met Bob.", "type": "text"}]}'

# Search
curl -X POST http://localhost:8000/v1/groups/test/search \
  -H "Authorization: Bearer local-graphiti" \
  -H "Content-Type: application/json" \
  -d '{"query": "Alice", "limit": 10}'

# List nodes
curl http://localhost:8000/v1/groups/test/nodes \
  -H "Authorization: Bearer local-graphiti"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/healthcheck` | Health check with component status |
| POST | `/v1/groups` | Create knowledge graph group |
| POST | `/v1/groups/{id}/ontology` | Define graph schema |
| POST | `/v1/groups/{id}/episodes:batch` | Bulk ingest text episodes |
| GET | `/v1/groups/{id}/nodes` | List nodes (paginated) |
| GET | `/v1/groups/{id}/edges` | List edges (paginated) |
| POST | `/v1/groups/{id}/search` | Semantic search |
| DELETE | `/v1/groups/{id}` | Delete group and all data |
| GET | `/v1/nodes/{uuid}` | Get node details |
| GET | `/v1/nodes/{uuid}/edges` | Get node's edges |
| GET | `/v1/edges/{uuid}` | Get edge details |
| GET | `/v1/episodes/{uuid}` | Get episode details |
| GET/POST | `/v1/threads` | List/create threads |
| GET/DELETE | `/v1/threads/{id}` | Get/delete thread |

## Configuration

Copy `.env.example` to `.env`. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `GRAPHITI_API_KEY` | Bearer token for API auth | `local-graphiti` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://127.0.0.1:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | **(required)** |
| `LLM_API_STYLE` | `openai` or `anthropic` | `anthropic` |
| `LLM_API_KEY` | LLM provider API key | **(required)** |
| `LLM_BASE_URL` | LLM provider base URL | (optional) |
| `LLM_MODEL_NAME` | Model name | (optional) |
| `EMBEDDING_API_KEY` | Embedding API key | **(required)** |
| `EMBEDDING_MODEL_NAME` | Embedding model | `text-embedding-3-small` |
| `EMBEDDING_BATCH_SIZE` | Max texts per embedding request (0=unlimited) | `0` |

### DashScope Users

DashScope limits embedding requests to 10 texts per batch. Add to `.env`:
```env
EMBEDDING_BATCH_SIZE=10
```

## Docker

```bash
# Start Neo4j
docker-compose up -d

# Stop
docker-compose down

# Reset all data (destructive!)
docker-compose down -v
```

Neo4j Browser: http://localhost:7474 (user: `neo4j`, password: from `.env`)

## Tests

```bash
uv run pytest tests/
```

## Documentation

- [Architecture](docs/architecture.md) — System design, data flow, component overview
- [API Mapping](docs/api-mapping.md) — Zep Cloud ↔ Graphiti-Zep API compatibility guide
- [Troubleshooting](docs/troubleshooting.md) — Common errors and solutions

## License

MIT
