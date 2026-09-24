"""Reports for explicitly declared Modelome entry relations.

These declarations are provenance-bearing graph edges. They do not, by
themselves, establish ancestry or imply that the resulting graph is a tree.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


def _stable(value: Any) -> str:
    """Return a deterministic representation for ordering evidence records."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def build_modelome_relations(
    entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic graph report from explicit ``model_relations``.

    A target is connected only when its serialized ``target.entry_id`` exactly
    matches an input entry ID. Names, aliases, and identifiers are never used
    to guess a connection. Original relation mappings are retained under
    ``evidence`` on both resolved edges and unresolved records.
    """
    materialized = [dict(entry) for entry in entries]
    ids: list[str] = []
    for index, entry in enumerate(materialized):
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"entry {index} must have a non-empty string id")
        ids.append(entry_id)
    if len(ids) != len(set(ids)):
        raise ValueError("entry IDs must be unique")
    known_ids = set(ids)
    nodes = [{"id": entry_id} for entry_id in sorted(known_ids)]
    edges: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for entry in materialized:
        source_id = entry["id"]
        relations = entry.get("model_relations", ()) or ()
        if not isinstance(relations, Iterable) or isinstance(relations, (str, bytes, Mapping)):
            raise ValueError(f"entry {source_id!r} model_relations must be iterable")
        for relation in relations:
            if not isinstance(relation, Mapping):
                raise ValueError(f"entry {source_id!r} model_relations must contain objects")
            target = relation.get("target")
            predicate = relation.get("predicate")
            if not isinstance(target, Mapping):
                raise ValueError("target must be an object")
            if not isinstance(predicate, str) or not predicate.strip():
                raise ValueError("predicate must be a non-empty string")
            target_id = target.get("entry_id")
            record = {
                "source": source_id,
                "target": target_id if isinstance(target_id, str) else None,
                "predicate": predicate,
                "evidence": dict(relation),
            }
            if isinstance(target_id, str) and target_id in known_ids:
                edges.append(record)
            else:
                record["reason"] = "target_entry_id_unresolved"
                unresolved.append(record)

    order = lambda record: (record["source"], record["target"] or "", str(record["predicate"] or ""), _stable(record["evidence"]))
    edges.sort(key=order)
    unresolved.sort(key=order)
    return {"nodes": nodes, "edges": edges, "unresolved": unresolved}


__all__ = ["build_modelome_relations"]
