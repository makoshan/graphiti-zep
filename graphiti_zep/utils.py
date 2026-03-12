"""JSON schema utilities for structured LLM output normalization.

These functions fix common issues with LLM-generated JSON:
- Provider wrapper objects (e.g. {"properties": {...}})
- Renamed field names (e.g. "nodes" instead of "extracted_entities")
- Complex nested values in contexts expecting primitives (Neo4j)
"""

from __future__ import annotations

import json
from typing import Any

_PRIMITIVE_TYPES = {"string", "integer", "number", "boolean"}


def resolve_schema_refs(schema: dict) -> dict:
    """Inline all $ref/$defs so providers that don't support $ref can use the schema."""
    defs = schema.get("$defs", {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return resolve(defs.get(ref_name, obj))
            return {k: resolve(v) for k, v in obj.items() if k != "$defs"}
        if isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    return resolve(schema)


def schema_expects_primitive(schema: dict) -> bool:
    """Check if a JSON schema field expects a primitive type (including nullable)."""
    if not schema:
        return True
    t = schema.get("type", "")
    if t in _PRIMITIVE_TYPES:
        return True
    for combo_key in ("anyOf", "oneOf"):
        variants = schema.get(combo_key, [])
        if variants and all(
            v.get("type") in _PRIMITIVE_TYPES or v.get("type") == "null"
            for v in variants
        ):
            return True
    return False


def flatten_value(v: Any) -> Any:
    """Convert nested dicts/list-of-dicts to JSON strings for Neo4j compatibility."""
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list) and v and isinstance(v[0], dict):
        return json.dumps(v, ensure_ascii=False)
    return v


def fix_field_names(data: Any, schema: dict) -> Any:
    """Recursively fix field names in data to match the expected JSON schema.

    Also flattens complex nested values to JSON strings when the schema
    expects primitive types (Neo4j only supports primitives as property values).
    """
    if isinstance(data, dict):
        props = schema.get("properties", {})
        if not props:
            return data
        expected_keys = set(props.keys())
        actual_keys = set(data.keys())
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        renames: dict[str, str] = {}
        unmatched_missing = set(missing)
        unmatched_extra = set(extra)
        for m in list(unmatched_missing):
            for e in list(unmatched_extra):
                if e.endswith(f"_{m}") or e.endswith(m) or m in e or e in m:
                    renames[e] = m
                    unmatched_missing.discard(m)
                    unmatched_extra.discard(e)
                    break
        if len(unmatched_missing) == 1 and len(unmatched_extra) == 1:
            e = next(iter(unmatched_extra))
            m = next(iter(unmatched_missing))
            e_val = data.get(e)
            m_schema = props.get(m, {})
            if (m_schema.get("type") == "array" and isinstance(e_val, (list, str))) or \
               (m_schema.get("type") != "array"):
                renames[e] = m
        result = {}
        for k, v in data.items():
            new_key = renames.get(k, k)
            child_schema = props.get(new_key, {})
            child_type = child_schema.get("type", "")
            if child_type == "array" and isinstance(v, list):
                item_schema = child_schema.get("items", {})
                if item_schema.get("type") == "object" or item_schema.get("properties"):
                    v = [fix_field_names(item, item_schema) for item in v]
                else:
                    v = [flatten_value(item) if isinstance(item, (dict, list)) else item for item in v]
            elif child_type == "object" or (isinstance(v, dict) and child_schema.get("properties")):
                v = fix_field_names(v, child_schema)
            elif isinstance(v, (dict, list)) and schema_expects_primitive(child_schema):
                v = flatten_value(v)
            result[new_key] = v
        return result
    if isinstance(data, list):
        items_schema = schema.get("items", {})
        return [fix_field_names(item, items_schema) for item in data]
    return data


def _numeric_dict_to_list(data: dict) -> list:
    """Convert a dict with numeric string keys to a list, parsing JSON string values."""
    items = []
    for k in sorted(data.keys(), key=lambda x: int(x)):
        v = data[k]
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                pass
        items.append(v)
    return items


def _is_numeric_keyed_dict(data: dict) -> bool:
    """Check if all keys in a dict are numeric strings (e.g. '0', '1', '2')."""
    if not data:
        return False
    return all(k.isdigit() for k in data.keys())


def unwrap_structured_payload(data: Any, schema: dict) -> Any:
    """Unwrap provider-specific wrapper objects around the actual JSON payload.

    Also handles the case where the LLM returns a dict with numeric string keys
    (e.g. {"0": ..., "1": ...}) instead of a proper array field.
    """
    if not isinstance(data, dict):
        return data

    expected_keys = set((schema.get("properties") or {}).keys())
    if not expected_keys:
        return data

    current = data
    seen: set[int] = set()
    wrapper_keys = ("properties", "arguments", "result", "data", "value")

    while isinstance(current, dict) and id(current) not in seen:
        seen.add(id(current))
        if expected_keys & set(current.keys()):
            return current

        nested = None
        for key in wrapper_keys:
            inner = current.get(key)
            if isinstance(inner, dict):
                nested = inner
                break

        if nested is None and len(current) == 1:
            only_value = next(iter(current.values()))
            if isinstance(only_value, dict):
                nested = only_value

        # Handle numeric-keyed dicts: {"0": {...}, "1": {...}} → wrap into expected array field
        if nested is None and _is_numeric_keyed_dict(current):
            props = schema.get("properties", {})
            array_fields = [k for k, v in props.items()
                           if isinstance(v, dict) and v.get("type") == "array"]
            if array_fields:
                items = _numeric_dict_to_list(current)
                return {array_fields[0]: items}

        if nested is None:
            return current
        current = nested

    return current
