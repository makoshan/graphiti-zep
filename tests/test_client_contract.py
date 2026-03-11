"""Contract tests for Graphiti-Zep client without external deps."""

import sys
import types

# Stub httpx before importing the client module
httpx_mod = types.ModuleType("httpx")


class DummyResponse:
    def __init__(self, payload, content=True, *, status_error=False, text=""):
        self._payload = payload
        self.content = b"1" if content else b""
        self.text = text
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise DummyHTTPStatusError("boom", request={"path": "x"}, response=self)

    def json(self):
        return self._payload


class DummyHTTPStatusError(Exception):
    def __init__(self, message, request=None, response=None):
        super().__init__(message)
        self.request = request
        self.response = response


class DummyClient:
    init_kwargs = None
    requests = []
    closed = False

    def __init__(self, **kwargs):
        DummyClient.init_kwargs = kwargs
        DummyClient.requests = []
        DummyClient.closed = False

    def request(self, method, path, **kwargs):
        DummyClient.requests.append((method, path, kwargs))
        if path == "/v1/groups":
            return DummyResponse({"uuid_": "g-1", "meta": {"owner": {"id": "u1"}}})
        if path.endswith("/ontology"):
            return DummyResponse({"ok": True})
        if path.endswith("/episodes:batch"):
            return DummyResponse([{"uuid_": "ep-1", "processed": True}, {"uuid_": "ep-2", "processed": True}])
        if path.endswith("/search"):
            return DummyResponse({
                "facts": [{"uuid_": "e-1", "name": "knows", "fact": "a->b"}],
                "edges": [{"uuid_": "e-1", "name": "knows", "fact": "a->b"}],
                "nodes": [{"uuid_": "n-1", "name": "Alice"}],
            })
        if path == "/v1/threads":
            if method == "GET":
                return DummyResponse({"threads": [{"thread_id": "t-1"}], "next_page_token": None})
            return DummyResponse({"thread_id": "t-new"})
        if path.startswith("/v1/threads/"):
            return DummyResponse({"thread_id": path.rsplit("/", 1)[-1]})
        if path.startswith("/v1/nodes/") and path.endswith("/edges"):
            return DummyResponse([{"uuid_": "e-1", "name": "knows", "fact": "a->b"}])
        if path.startswith("/v1/nodes/"):
            return DummyResponse({"uuid_": "n-1", "name": "Alice", "summary": "A person"})
        if path.startswith("/v1/edges/"):
            return DummyResponse({"uuid_": "e-1", "name": "knows", "fact": "a->b"})
        if path.startswith("/v1/episodes/"):
            return DummyResponse({"uuid_": "ep-1", "processed": True})
        if path.startswith("/v1/groups/") and "/nodes" in path:
            return DummyResponse([{"uuid_": "n-1", "name": "Alice"}, {"uuid_": "n-2", "name": "Bob"}])
        if path.startswith("/v1/groups/") and "/edges" in path:
            return DummyResponse([{"uuid_": "e-1", "name": "knows", "fact": "a->b"}])
        return DummyResponse({"ok": True})

    def close(self):
        DummyClient.closed = True


httpx_mod.Client = DummyClient
httpx_mod.HTTPStatusError = DummyHTTPStatusError
sys.modules["httpx"] = httpx_mod

from graphiti_zep.client import (  # noqa: E402
    GraphitiZepClient, EpisodeData, GraphNode, GraphEdge,
    SearchResult, EpisodeResult, ThreadInfo, ThreadList, _GraphitiHTTP,
)


def test_client_init():
    client = GraphitiZepClient("k1", base_url="http://localhost:8000")
    assert DummyClient.init_kwargs["base_url"] == "http://localhost:8000"
    assert DummyClient.init_kwargs["headers"]["Authorization"] == "Bearer k1"
    client.close()


def test_graph_create():
    client = GraphitiZepClient("k1")
    created = client.graph.create("g-1", "name", "desc")
    assert created.uuid_ == "g-1"
    assert created.meta.owner.id == "u1"
    client.close()


def test_set_ontology_multiple_groups():
    client = GraphitiZepClient("k1")
    client.graph.set_ontology(["g-1", "g-2"], entities={"A": {}}, edges={"R": {}})
    ontology_calls = [c for c in DummyClient.requests if c[1].endswith("/ontology")]
    assert len(ontology_calls) == 2
    client.close()


def test_add_batch_returns_typed():
    client = GraphitiZepClient("k1")
    episodes = [EpisodeData(data="x"), EpisodeData(data="y", type="note")]
    added = client.graph.add_batch("g-1", episodes)
    assert isinstance(added[0], EpisodeResult)
    assert added[0].uuid_ == "ep-1"
    assert added[0].processed is True
    add_call = [c for c in DummyClient.requests if c[1].endswith("/episodes:batch")][-1]
    assert add_call[2]["json"]["episodes"][1]["type"] == "note"
    client.close()


def test_search_returns_typed():
    client = GraphitiZepClient("k1")
    result = client.graph.search(graph_id="g-1", query="alice")
    assert isinstance(result, SearchResult)
    assert isinstance(result.facts[0], GraphEdge)
    assert result.facts[0].fact == "a->b"
    assert isinstance(result.nodes[0], GraphNode)
    assert result.nodes[0].name == "Alice"
    search_call = [c for c in DummyClient.requests if c[1].endswith("/search")][-1]
    assert search_call[1] == "/v1/groups/g-1/search"
    assert search_call[2]["json"]["query"] == "alice"
    client.close()


def test_get_nodes_returns_typed():
    client = GraphitiZepClient("k1")
    nodes = client.graph.node.get_by_graph_id("g-1")
    assert isinstance(nodes[0], GraphNode)
    assert nodes[0].name == "Alice"
    assert nodes[1].name == "Bob"
    client.close()


def test_get_edges_returns_typed():
    client = GraphitiZepClient("k1")
    edges = client.graph.edge.get_by_graph_id("g-1")
    assert isinstance(edges[0], GraphEdge)
    assert edges[0].fact == "a->b"
    client.close()


def test_get_node_returns_typed():
    client = GraphitiZepClient("k1")
    node = client.graph.node.get("n-1")
    assert isinstance(node, GraphNode)
    assert node.name == "Alice"
    assert node.summary == "A person"
    client.close()


def test_get_node_edges_returns_typed():
    client = GraphitiZepClient("k1")
    edges = client.graph.node.get_entity_edges("n-1")
    assert isinstance(edges[0], GraphEdge)
    assert edges[0].name == "knows"
    client.close()


def test_thread_list_returns_typed():
    client = GraphitiZepClient("k1")
    result = client.thread.get_threads(user_id="u-1", limit=5, page_token="p1")
    assert isinstance(result, ThreadList)
    assert isinstance(result.threads[0], ThreadInfo)
    assert result.threads[0].thread_id == "t-1"
    threads_call = [c for c in DummyClient.requests if c[1] == "/v1/threads" and c[0] == "GET"][-1]
    assert threads_call[2]["params"] == {"limit": 5, "user_id": "u-1", "page_token": "p1"}
    client.close()


def test_thread_create_returns_typed():
    client = GraphitiZepClient("k1")
    created = client.thread.create(thread_id="t-custom", user_id="u-1", metadata={"k": "v"})
    assert isinstance(created, ThreadInfo)
    assert created.thread_id == "t-new"
    create_call = [c for c in DummyClient.requests if c[1] == "/v1/threads" and c[0] == "POST"][-1]
    assert create_call[2]["json"]["thread_id"] == "t-custom"
    client.close()


def test_thread_get_and_delete():
    client = GraphitiZepClient("k1")
    single = client.thread.get("t-123")
    assert isinstance(single, ThreadInfo)
    assert single.thread_id == "t-123"
    client.thread.delete("t-123")
    assert DummyClient.requests[-1][0] == "DELETE"
    assert DummyClient.requests[-1][1] == "/v1/threads/t-123"
    client.close()


def test_error_body_preserved():
    http = _GraphitiHTTP("k1")
    http._client = types.SimpleNamespace(
        request=lambda *args, **kwargs: DummyResponse(
            {"detail": "bad upstream"},
            status_error=True,
            text='{"detail":"bad upstream"}',
        )
    )
    try:
        http.request("POST", "/v1/fail")
        raise AssertionError("expected HTTP status error")
    except DummyHTTPStatusError as exc:
        assert 'Response body: {"detail":"bad upstream"}' in str(exc)


def test_context_manager():
    with GraphitiZepClient("k1") as client:
        client.graph.create("g-ctx", "test", "")
    assert DummyClient.closed is True


if __name__ == "__main__":
    test_client_init()
    test_graph_create()
    test_set_ontology_multiple_groups()
    test_add_batch_returns_typed()
    test_search_returns_typed()
    test_get_nodes_returns_typed()
    test_get_edges_returns_typed()
    test_get_node_returns_typed()
    test_get_node_edges_returns_typed()
    test_thread_list_returns_typed()
    test_thread_create_returns_typed()
    test_thread_get_and_delete()
    test_error_body_preserved()
    test_context_manager()
    print("ok")
