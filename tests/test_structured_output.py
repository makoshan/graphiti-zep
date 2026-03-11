"""Standalone regression checks for structured JSON normalization."""

from graphiti_zep.utils import fix_field_names as _fix_field_names, unwrap_structured_payload as _unwrap_structured_payload


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


if __name__ == "__main__":
    test_unwrap_single_wrapper()
    test_unwrap_double_wrapper()
    test_fix_renamed_fields()
    test_passthrough_correct_fields()
    print("ok")
