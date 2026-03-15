from types import SimpleNamespace

import pytest

from graphiti_zep.server import _count_outcome_graph_items, _validate_episode_outcome


def test_count_outcome_graph_items_handles_empty_lists():
    outcome = SimpleNamespace(nodes=[], edges=[])
    assert _count_outcome_graph_items(outcome) == (0, 0)


def test_validate_episode_outcome_rejects_empty_graph():
    outcome = SimpleNamespace(nodes=[], edges=[])
    with pytest.raises(ValueError, match="empty extraction"):
        _validate_episode_outcome(outcome, group_id="proj_test", episode_index=1)


def test_validate_episode_outcome_accepts_nodes_without_edges():
    outcome = SimpleNamespace(nodes=[SimpleNamespace(uuid="n1")], edges=[])
    assert _validate_episode_outcome(outcome, group_id="proj_test", episode_index=2) == (1, 0)


def test_validate_episode_outcome_accepts_edges_without_nodes():
    outcome = SimpleNamespace(nodes=[], edges=[SimpleNamespace(uuid="e1")])
    assert _validate_episode_outcome(outcome, group_id="proj_test", episode_index=3) == (0, 1)
