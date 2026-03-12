#!/usr/bin/env python3
"""Patch graphiti_core pydantic models for qwen/non-OpenAI LLM compatibility.

qwen-plus and other OpenAI-compatible LLMs sometimes omit required fields
or return malformed JSON (e.g. numeric-keyed dicts instead of arrays, or
plain strings instead of lists). This script:
  1. Makes all wrapper model list fields optional with empty defaults
  2. Adds field_validator to coerce non-list inputs to empty lists
  3. Adds field_validator import where needed
  4. Makes ExtractedEdges.edges optional with default_factory=list

Run after `uv sync` to re-apply patches:
    uv run python scripts/patch_graphiti_core.py
"""

import importlib.util
from pathlib import Path


def find_graphiti_core() -> Path:
    spec = importlib.util.find_spec("graphiti_core")
    if spec is None or spec.origin is None:
        raise RuntimeError("graphiti_core not found in current environment")
    return Path(spec.origin).parent


COERCE_VALIDATOR = '''
    @field_validator({fields}, mode='before')
    @classmethod
    def _coerce_to_list(cls, v):
        if isinstance(v, list):
            return v
        return []'''


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> bool:
    text = path.read_text()
    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        path.write_text(text)
    return changed


def main():
    root = find_graphiti_core()
    prompts = root / "prompts"

    patches = [
        # ── extract_edges.py ──
        (
            prompts / "extract_edges.py",
            [
                # Add field_validator import
                (
                    "from pydantic import BaseModel, Field\n",
                    "from pydantic import BaseModel, Field, field_validator\n",
                ),
                # Make fact optional
                (
                    "fact: str = Field(\n        ...,",
                    "fact: str = Field(\n        '',",
                ),
                # Make edges list optional with validator
                (
                    "class ExtractedEdges(BaseModel):\n    edges: list[Edge]",
                    "class ExtractedEdges(BaseModel):\n"
                    "    edges: list[Edge] = Field(default_factory=list)\n"
                    + COERCE_VALIDATOR.format(fields="'edges'"),
                ),
            ],
        ),
        # ── extract_nodes.py ──
        (
            prompts / "extract_nodes.py",
            [
                # Add field_validator import (model_validator already imported)
                (
                    "from pydantic import BaseModel, Field, model_validator\n",
                    "from pydantic import BaseModel, Field, field_validator, model_validator\n",
                ),
                # Make extracted_entities optional with validator
                (
                    "extracted_entities: list[ExtractedEntity] = Field(...,",
                    "extracted_entities: list[ExtractedEntity] = Field(default_factory=list,",
                ),
                (
                    "class ExtractedEntities(BaseModel):\n"
                    "    extracted_entities: list[ExtractedEntity] = Field(default_factory=list, description='List of extracted entities')\n",
                    "class ExtractedEntities(BaseModel):\n"
                    "    extracted_entities: list[ExtractedEntity] = Field(default_factory=list, description='List of extracted entities')\n"
                    + COERCE_VALIDATOR.format(fields="'extracted_entities'") + "\n",
                ),
                # Make summaries optional with validator
                (
                    "summaries: list[SummarizedEntity] = Field(\n        ...,",
                    "summaries: list[SummarizedEntity] = Field(\n        default_factory=list,",
                ),
                (
                    "class SummarizedEntities(BaseModel):\n"
                    "    summaries: list[SummarizedEntity] = Field(\n"
                    "        default_factory=list,\n"
                    "        description='List of entity summaries. Only include entities that need summary updates.',\n"
                    "    )\n",
                    "class SummarizedEntities(BaseModel):\n"
                    "    summaries: list[SummarizedEntity] = Field(\n"
                    "        default_factory=list,\n"
                    "        description='List of entity summaries. Only include entities that need summary updates.',\n"
                    "    )\n"
                    + COERCE_VALIDATOR.format(fields="'summaries'") + "\n",
                ),
            ],
        ),
        # ── dedupe_nodes.py ──
        (
            prompts / "dedupe_nodes.py",
            [
                # Add field_validator import
                (
                    "from pydantic import BaseModel, Field\n",
                    "from pydantic import BaseModel, Field, field_validator\n",
                ),
                # Make entity_resolutions optional with validator
                (
                    "entity_resolutions: list[NodeDuplicate] = Field(...,",
                    "entity_resolutions: list[NodeDuplicate] = Field(default_factory=list,",
                ),
                (
                    "class NodeResolutions(BaseModel):\n"
                    "    entity_resolutions: list[NodeDuplicate] = Field(default_factory=list, description='List of resolved nodes')\n",
                    "class NodeResolutions(BaseModel):\n"
                    "    entity_resolutions: list[NodeDuplicate] = Field(default_factory=list, description='List of resolved nodes')\n"
                    + COERCE_VALIDATOR.format(fields="'entity_resolutions'") + "\n",
                ),
            ],
        ),
        # ── dedupe_edges.py ──
        (
            prompts / "dedupe_edges.py",
            [
                # Add field_validator import
                (
                    "from pydantic import BaseModel, Field\n",
                    "from pydantic import BaseModel, Field, field_validator\n",
                ),
                # Make fields optional
                (
                    "duplicate_facts: list[int] = Field(\n        ...,",
                    "duplicate_facts: list[int] = Field(\n        default_factory=list,",
                ),
                (
                    "contradicted_facts: list[int] = Field(\n        ...,",
                    "contradicted_facts: list[int] = Field(\n        default_factory=list,",
                ),
                # Add validator after the class fields
                (
                    "class EdgeDuplicate(BaseModel):\n"
                    "    duplicate_facts: list[int] = Field(\n"
                    "        default_factory=list,\n"
                    "        description='List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.',\n"
                    "    )\n"
                    "    contradicted_facts: list[int] = Field(\n"
                    "        default_factory=list,\n"
                    "        description='List of idx values of contradicted facts (from full idx range). Empty list if none.',\n"
                    "    )\n",
                    "class EdgeDuplicate(BaseModel):\n"
                    "    duplicate_facts: list[int] = Field(\n"
                    "        default_factory=list,\n"
                    "        description='List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.',\n"
                    "    )\n"
                    "    contradicted_facts: list[int] = Field(\n"
                    "        default_factory=list,\n"
                    "        description='List of idx values of contradicted facts (from full idx range). Empty list if none.',\n"
                    "    )\n"
                    + COERCE_VALIDATOR.format(fields="'duplicate_facts', 'contradicted_facts'") + "\n",
                ),
            ],
        ),
    ]

    for path, replacements in patches:
        if patch_file(path, replacements):
            print(f"  patched: {path.relative_to(root.parent)}")
        else:
            print(f"  ok (already patched): {path.relative_to(root.parent)}")

    print("\nAll graphiti_core patches applied.")


if __name__ == "__main__":
    main()
