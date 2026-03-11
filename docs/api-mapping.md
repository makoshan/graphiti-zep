# Zep Cloud ↔ Graphiti-Zep API Mapping

This document maps Zep Cloud's knowledge graph API to the equivalent Graphiti-Zep endpoints, for users migrating from Zep Cloud.

## API Mapping Table

| Zep Cloud API | Graphiti-Zep Endpoint | Compatibility | Notes |
|---------------|----------------------|---------------|-------|
| `client.graph.create()` | `POST /v1/groups` | Full | `graph_id` → `group_id` |
| `client.graph.set_ontology()` | `POST /v1/groups/{id}/ontology` | Full | Dynamic Pydantic model generation |
| `client.graph.add()` | `POST /v1/groups/{id}/episodes:batch` | Full | Single episode in batch |
| `client.graph.add_batch()` | `POST /v1/groups/{id}/episodes:batch` | Full | Direct mapping |
| `client.graph.search()` | `POST /v1/groups/{id}/search` | Full | Supports `scope`, `limit`, `reranker` |
| `client.graph.node.get_by_graph_id()` | `GET /v1/groups/{id}/nodes` | Full | Paginated via `uuid_cursor` |
| `client.graph.edge.get_by_graph_id()` | `GET /v1/groups/{id}/edges` | Full | Paginated via `uuid_cursor` |
| `client.graph.node.get()` | `GET /v1/nodes/{uuid}` | Full | |
| `client.graph.node.get_entity_edges()` | `GET /v1/nodes/{uuid}/edges` | Full | |
| `client.graph.edge.get()` | `GET /v1/edges/{uuid}` | Full | |
| `client.graph.episode.get()` | `GET /v1/episodes/{uuid}` | Full | Always returns `processed: true` (sync) |
| `client.graph.delete()` | `DELETE /v1/groups/{id}` | Full | Deletes nodes + episodes |

## Python Client Mapping

| Zep Cloud SDK | Graphiti-Zep Client | Notes |
|---------------|---------------------|-------|
| `Zep(api_key=...)` | `create_client(api_key, base_url=...)` | |
| `client.graph.create(...)` | `client.graph.create(graph_id, name, desc)` | |
| `client.graph.set_ontology(graph_ids, ...)` | `client.graph.set_ontology(graph_ids, ...)` | Same signature |
| `client.graph.add_batch(graph_id, episodes)` | `client.graph.add_batch(graph_id, episodes)` | `EpisodeData` instead of Zep's `Episode` |
| `client.graph.search(graph_id=..., query=...)` | `client.graph.search(graph_id=..., query=...)` | Returns `SearchResult` dataclass |
| `client.graph.node.get_by_graph_id(id)` | `client.graph.node.get_by_graph_id(id)` | Returns `list[GraphNode]` |
| `client.graph.edge.get_by_graph_id(id)` | `client.graph.edge.get_by_graph_id(id)` | Returns `list[GraphEdge]` |
| `client.graph.episode.get(uuid)` | `client.graph.episode.get(uuid)` | |
| _polling loop_ | `client.graph.wait_for_episode(uuid)` | Convenience method |

## Key Differences

### 1. Episode Processing

**Zep Cloud**: Episodes are processed asynchronously. You must poll `episode.get()` until `processed == true`.

**Graphiti-Zep**: Episodes are processed synchronously during the `add_batch()` call. `episode.get()` always returns `processed: true`. The `wait_for_episode()` client method is provided for Zep Cloud API compatibility but returns immediately.

### 2. Multi-Graph Isolation

**Zep Cloud**: Native multi-graph support via `graph_id`.

**Graphiti-Zep**: Uses Graphiti's `group_id` parameter. Your `graph_id` maps directly to `group_id`. All nodes, edges, and episodes are tagged with `group_id` and filtered accordingly.

### 3. Ontology

**Zep Cloud**: `set_ontology()` enforces entity/edge type constraints.

**Graphiti-Zep**: `set_ontology()` passes entity/edge types to graphiti-core as Pydantic models. The LLM uses these types during extraction but they are not enforced as hard constraints at the database level.

### 4. Search

**Zep Cloud**: Hybrid search (semantic + BM25) with reranker support.

**Graphiti-Zep**: Uses graphiti-core's built-in search with a passthrough cross-encoder. Search quality depends on the LLM and embedding model used.

### 5. Return Types

**Zep Cloud SDK**: Returns Zep-specific model objects.

**Graphiti-Zep Client**: Returns typed dataclasses (`GraphNode`, `GraphEdge`, `SearchResult`, `EpisodeResult`, `ThreadInfo`).

## Migration Example

```python
# Before (Zep Cloud)
from zep_cloud.client import Zep
client = Zep(api_key="z_...")
client.graph.create(graph_id="my-graph", name="...", description="...")
nodes = client.graph.node.get_by_graph_id("my-graph")

# After (Graphiti-Zep)
from graphiti_zep import create_client
client = create_client("local-graphiti", base_url="http://localhost:8000")
client.graph.create("my-graph", "...", "...")
nodes = client.graph.node.get_by_graph_id("my-graph")  # returns list[GraphNode]
```

The API surface is intentionally kept as close to Zep Cloud as possible, so migration typically only requires changing the import and client initialization.
