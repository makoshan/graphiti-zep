"""Standalone regression checks for structured JSON normalization."""

from graphiti_zep.utils import (
    fix_field_names as _fix_field_names,
    sanitize_structured_payload as _sanitize_structured_payload,
    unwrap_structured_payload as _unwrap_structured_payload,
)


schema = {
    "type": "object",
    "properties": {
        "extracted_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type_id": {"type": "integer"},
                },
                "required": ["name", "entity_type_id"],
            },
        }
    },
}

edge_schema = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity_name": {"type": "string"},
                    "target_entity_name": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "fact": {"type": "string"},
                    "valid_at": {"type": "string"},
                    "invalid_at": {"type": "string"},
                },
                "required": ["source_entity_name", "target_entity_name", "relation_type", "fact"],
            },
        }
    },
}


def test_unwrap_single_wrapper():
    wrapped = {"properties": {"extracted_entities": []}}
    assert _unwrap_structured_payload(wrapped, schema) == {"extracted_entities": []}


def test_unwrap_double_wrapper():
    double_wrapped = {"result": {"properties": {"extracted_entities": []}}}
    assert _unwrap_structured_payload(double_wrapped, schema) == {"extracted_entities": []}


def test_fix_renamed_fields():
    renamed = {"properties": {"nodes": [{"name": "Alice", "entity_type_id": 1}]}}
    normalized = _fix_field_names(_unwrap_structured_payload(renamed, schema), schema)
    assert normalized["extracted_entities"][0]["name"] == "Alice"


def test_passthrough_correct_fields():
    correct = {"extracted_entities": [{"name": "Bob", "entity_type_id": 2}]}
    assert _fix_field_names(correct, schema) == correct


def test_fix_moonshot_entity_aliases():
    moonshot = {
        "extracted_entities": [
            {"entity_literal": "Valencia Cyclery", "entity_id": "0"},
            {"speaker": "user[10]"},
        ]
    }
    normalized = _fix_field_names(moonshot, schema)
    assert normalized["extracted_entities"][0]["name"] == "Valencia Cyclery"
    assert normalized["extracted_entities"][0]["entity_type_id"] == 0
    assert normalized["extracted_entities"][1]["name"] == "user[10]"
    assert normalized["extracted_entities"][1]["entity_type_id"] == 0


def test_fix_moonshot_entity_embedded_json_alias():
    moonshot = {
        "extracted_entities": [
            {
                "speaker": "{\"entity_name\":\"assistant\",\"entity_id\":\"3\"}",
            }
        ]
    }
    normalized = _fix_field_names(moonshot, schema)
    assert normalized["extracted_entities"][0]["name"] == "assistant"
    assert normalized["extracted_entities"][0]["entity_type_id"] == 3


def test_fix_moonshot_edge_aliases_and_defaults():
    moonshot = {
        "edges": [
            {
                "source": "Alice",
                "target": "REI",
                "relationship": "works at",
                "description": "Alice works at REI",
            },
            {
                "source_entity_name": "Bob",
                "target_entity_name": "Seattle",
            },
        ]
    }
    normalized = _fix_field_names(moonshot, edge_schema)
    assert normalized["edges"][0]["relation_type"] == "WORKS_AT"
    assert normalized["edges"][0]["fact"] == "Alice works at REI"
    assert normalized["edges"][1]["relation_type"] == "RELATED_TO"
    assert normalized["edges"][1]["fact"] == "Bob RELATED_TO Seattle"


def test_fix_moonshot_edge_defaults_for_null_fact():
    moonshot = {
        "edges": [
            {
                "source_entity_name": "Alice",
                "target_entity_name": "Seattle",
                "fact": None,
            }
        ]
    }
    normalized = _fix_field_names(moonshot, edge_schema)
    assert normalized["edges"][0]["relation_type"] == "RELATED_TO"
    assert normalized["edges"][0]["fact"] == "Alice RELATED_TO Seattle"


def test_fix_moonshot_edge_from_to_and_invalid_timestamp():
    moonshot = {
        "edges": [
            {
                "from": "Shibuya Sky",
                "to": "Tokyo",
                "predicate": "located in",
                "invalid_at": 1,
            }
        ]
    }
    normalized = _fix_field_names(moonshot, edge_schema)
    assert normalized["edges"][0]["source_entity_name"] == "Shibuya Sky"
    assert normalized["edges"][0]["target_entity_name"] == "Tokyo"
    assert normalized["edges"][0]["relation_type"] == "LOCATED_IN"
    assert normalized["edges"][0]["invalid_at"] is None


def test_sanitize_drops_incomplete_edges():
    payload = {
        "edges": [
            {"source_entity_name": "Alice", "target_entity_name": "Seattle", "relation_type": "LIVES_IN", "fact": "Alice lives in Seattle"},
            {"source_entity_name": None, "target_entity_name": None, "relation_type": "BROKEN", "fact": "broken"},
        ]
    }
    sanitized = _sanitize_structured_payload(payload, edge_schema)
    assert len(sanitized["edges"]) == 1
    assert sanitized["edges"][0]["relation_type"] == "LIVES_IN"


if __name__ == "__main__":
    test_unwrap_single_wrapper()
    test_unwrap_double_wrapper()
    test_fix_renamed_fields()
    test_passthrough_correct_fields()
    test_fix_moonshot_entity_aliases()
    test_fix_moonshot_entity_embedded_json_alias()
    test_fix_moonshot_edge_aliases_and_defaults()
    test_fix_moonshot_edge_defaults_for_null_fact()
    test_fix_moonshot_edge_from_to_and_invalid_timestamp()
    test_sanitize_drops_incomplete_edges()
    print("ok")
