"""Graphiti-Zep Python client.

Usage:
    from graphiti_zep import create_client, EpisodeData

    with create_client("your-api-key", base_url="http://localhost:8000") as client:
        # Create a knowledge graph group
        client.graph.create("my-graph", "My Graph", "Description")

        # Ingest episodes
        episodes = [EpisodeData(data="Alice met Bob at the coffee shop.")]
        results = client.graph.add_batch("my-graph", episodes)

        # Search
        search = client.graph.search(graph_id="my-graph", query="Alice")
        for fact in search.facts:
            print(fact.fact)

        # Typed access
        nodes = client.graph.node.get_by_graph_id("my-graph")
        for node in nodes:
            print(f"{node.name}: {node.summary}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx


# ── Typed return models ────────────────────────────────────────────────

@dataclass
class EpisodeData:
    """A single episode (text chunk) to ingest into the knowledge graph."""
    data: str
    type: str = "text"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    uuid: str
    name: str
    group_id: str = ""
    labels: list[str] = field(default_factory=list)
    summary: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    valid_at: str = ""
    invalid_at: str = ""
    expired_at: str = ""


@dataclass
class GraphEdge:
    """An edge (relationship) in the knowledge graph."""
    uuid: str
    name: str
    fact: str = ""
    group_id: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    valid_at: str = ""
    invalid_at: str = ""
    expired_at: str = ""


@dataclass
class SearchResult:
    """Result from a knowledge graph search."""
    facts: list[GraphEdge] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    nodes: list[GraphNode] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """Result from an episode ingestion."""
    uuid_: str
    processed: bool = False


@dataclass
class ThreadInfo:
    """Thread metadata."""
    thread_id: str
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreadList:
    """Paginated list of threads."""
    threads: list[ThreadInfo] = field(default_factory=list)
    next_page_token: str | None = None


# ── Conversion helpers ─────────────────────────────────────────────────

def _to_obj(value: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for attribute access."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_obj(v) for v in value]
    return value


def _to_node(data: dict[str, Any]) -> GraphNode:
    return GraphNode(
        uuid=data.get("uuid_") or data.get("uuid", ""),
        name=data.get("name", ""),
        group_id=data.get("group_id", ""),
        labels=data.get("labels", []),
        summary=data.get("summary", ""),
        attributes=data.get("attributes", {}),
        created_at=data.get("created_at", ""),
        valid_at=data.get("valid_at", ""),
        invalid_at=data.get("invalid_at", ""),
        expired_at=data.get("expired_at", ""),
    )


def _to_edge(data: dict[str, Any]) -> GraphEdge:
    return GraphEdge(
        uuid=data.get("uuid_") or data.get("uuid", ""),
        name=data.get("name", ""),
        fact=data.get("fact", ""),
        group_id=data.get("group_id", ""),
        source_node_uuid=data.get("source_node_uuid", ""),
        target_node_uuid=data.get("target_node_uuid", ""),
        attributes=data.get("attributes", {}),
        created_at=data.get("created_at", ""),
        valid_at=data.get("valid_at", ""),
        invalid_at=data.get("invalid_at", ""),
        expired_at=data.get("expired_at", ""),
    )


# ── HTTP layer ─────────────────────────────────────────────────────────

class _GraphitiHTTP:
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 900.0,
        trust_env: bool = False,
    ):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=True,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._client.request(method, path, **kwargs)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() if exc.response is not None else ""
            if detail:
                message = f"{exc}. Response body: {detail[:500]}"
                raise httpx.HTTPStatusError(message, request=exc.request, response=exc.response) from exc
            raise
        if not resp.content:
            return None
        return resp.json()

    def close(self):
        self._client.close()


# ── Resource operations ────────────────────────────────────────────────

class _EpisodeOps:
    def __init__(self, http: _GraphitiHTTP):
        self._http = http

    def get(self, uuid_: str) -> SimpleNamespace:
        return _to_obj(self._http.request("GET", f"/v1/episodes/{uuid_}"))


class _NodeOps:
    def __init__(self, http: _GraphitiHTTP):
        self._http = http

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: str | None = None) -> list[GraphNode]:
        params: dict[str, Any] = {"limit": limit}
        if uuid_cursor:
            params["uuid_cursor"] = uuid_cursor
        data = self._http.request("GET", f"/v1/groups/{graph_id}/nodes", params=params) or []
        return [_to_node(item) for item in data]

    def get_entity_edges(self, node_uuid: str) -> list[GraphEdge]:
        data = self._http.request("GET", f"/v1/nodes/{node_uuid}/edges") or []
        return [_to_edge(item) for item in data]

    def get(self, uuid_: str) -> GraphNode:
        return _to_node(self._http.request("GET", f"/v1/nodes/{uuid_}"))


class _EdgeOps:
    def __init__(self, http: _GraphitiHTTP):
        self._http = http

    def get_by_graph_id(self, graph_id: str, limit: int = 100, uuid_cursor: str | None = None) -> list[GraphEdge]:
        params: dict[str, Any] = {"limit": limit}
        if uuid_cursor:
            params["uuid_cursor"] = uuid_cursor
        data = self._http.request("GET", f"/v1/groups/{graph_id}/edges", params=params) or []
        return [_to_edge(item) for item in data]

    def get(self, uuid_: str) -> GraphEdge:
        return _to_edge(self._http.request("GET", f"/v1/edges/{uuid_}"))


class _GraphOps:
    def __init__(self, http: _GraphitiHTTP):
        self._http = http
        self.node = _NodeOps(http)
        self.edge = _EdgeOps(http)
        self.episode = _EpisodeOps(http)

    def create(self, graph_id: str, name: str, description: str) -> SimpleNamespace:
        return _to_obj(
            self._http.request(
                "POST",
                "/v1/groups",
                json={"group_id": graph_id, "name": name, "description": description},
            )
        )

    def set_ontology(
        self,
        graph_ids: list[str],
        entities: dict[str, Any] | None = None,
        edges: dict[str, Any] | None = None,
    ) -> None:
        for gid in graph_ids:
            self._http.request(
                "POST",
                f"/v1/groups/{gid}/ontology",
                json={"entities": entities or {}, "edges": edges or {}},
            )

    def add_batch(self, graph_id: str, episodes: list[EpisodeData]) -> list[EpisodeResult]:
        payload = [{"content": ep.data, "type": ep.type} for ep in episodes]
        data = self._http.request(
            "POST",
            f"/v1/groups/{graph_id}/episodes:batch",
            json={"episodes": payload},
        ) or []
        return [EpisodeResult(uuid_=item.get("uuid_", ""), processed=item.get("processed", False)) for item in data]

    def search(self, **kwargs: Any) -> SearchResult:
        graph_id = kwargs.pop("graph_id")
        data = self._http.request("POST", f"/v1/groups/{graph_id}/search", json=kwargs) or {}
        return SearchResult(
            facts=[_to_edge(e) for e in data.get("facts", [])],
            edges=[_to_edge(e) for e in data.get("edges", [])],
            nodes=[_to_node(n) for n in data.get("nodes", [])],
        )

    def delete(self, graph_id: str) -> None:
        self._http.request("DELETE", f"/v1/groups/{graph_id}")

    def wait_for_episode(self, uuid_: str, timeout: int = 300, poll_interval: float = 1.0) -> bool:
        """Wait for an episode to be processed.

        Graphiti processes episodes synchronously, so this returns immediately.
        Kept for Zep Cloud API compatibility — Zep Cloud requires polling.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ep = self.episode.get(uuid_)
            if getattr(ep, "processed", False):
                return True
            time.sleep(poll_interval)
        return False


class _ThreadOps:
    """Zep-like Thread API."""

    def __init__(self, http: _GraphitiHTTP):
        self._http = http

    def get_threads(
        self,
        *,
        user_id: str | None = None,
        limit: int = 20,
        page_token: str | None = None,
    ) -> ThreadList:
        params: dict[str, Any] = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        if page_token:
            params["page_token"] = page_token
        data = self._http.request("GET", "/v1/threads", params=params) or {}
        return ThreadList(
            threads=[ThreadInfo(**t) for t in data.get("threads", [])],
            next_page_token=data.get("next_page_token"),
        )

    def get(self, thread_id: str) -> ThreadInfo:
        data = self._http.request("GET", f"/v1/threads/{thread_id}") or {}
        return ThreadInfo(**data)

    def create(self, *, thread_id: str | None = None, user_id: str | None = None, metadata: dict[str, Any] | None = None) -> ThreadInfo:
        payload: dict[str, Any] = {}
        if thread_id:
            payload["thread_id"] = thread_id
        if user_id:
            payload["user_id"] = user_id
        if metadata:
            payload["metadata"] = metadata
        data = self._http.request("POST", "/v1/threads", json=payload) or {}
        return ThreadInfo(**data)

    def delete(self, thread_id: str) -> None:
        self._http.request("DELETE", f"/v1/threads/{thread_id}")


# ── Main client ────────────────────────────────────────────────────────

class GraphitiZepClient:
    """Client for the Graphiti-Zep server.

    Provides a Zep Cloud-compatible API surface backed by Graphiti + Neo4j.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 900.0,
        trust_env: bool = False,
    ):
        self._http = _GraphitiHTTP(api_key, base_url=base_url, timeout=timeout, trust_env=trust_env)
        self.graph = _GraphOps(self._http)
        self.thread = _ThreadOps(self._http)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_client(
    api_key: str,
    base_url: str = "http://localhost:8000",
    timeout: float = 900.0,
    trust_env: bool = False,
) -> GraphitiZepClient:
    """Create a new Graphiti-Zep client."""
    return GraphitiZepClient(api_key, base_url=base_url, timeout=timeout, trust_env=trust_env)
