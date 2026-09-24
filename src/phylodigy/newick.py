"""Newick serialization for inferred architecture lineage trees."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def lineage_to_newick(lineage: Mapping[str, Any]) -> str:
    """Serialize an ``infer_lineage_network`` result as deterministic Newick.

    The directed ``tree.edges`` representation is traversed from its unique
    root. Internal node identifiers are omitted; terminal identifiers are
    emitted as Newick labels. Branch lengths come from ``branch_length``.
    """
    if not isinstance(lineage, Mapping):
        raise TypeError("lineage must be a mapping")
    tree = lineage.get("tree")
    if not isinstance(tree, Mapping):
        raise ValueError("lineage must contain a tree mapping")
    edges = tree.get("edges")
    if not isinstance(edges, (list, tuple)) or not edges:
        raise ValueError("tree edges must be a non-empty sequence")

    children: dict[str, list[tuple[str, float]]] = {}
    parents: dict[str, str] = {}
    nodes: set[str] = set()
    edge_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("each tree edge must be a mapping")
        parent, child = edge.get("parent"), edge.get("child")
        if not isinstance(parent, str) or not isinstance(child, str) or not parent or not child:
            raise ValueError("edge parent and child must be non-empty strings")
        if parent == child:
            raise ValueError("tree edges cannot be self-loops")
        pair = (parent, child)
        if pair in edge_pairs:
            raise ValueError("tree contains a duplicate edge")
        edge_pairs.add(pair)
        if child in parents:
            raise ValueError("tree node has multiple parents")
        length = edge.get("branch_length")
        if isinstance(length, bool) or not isinstance(length, (int, float)):
            raise ValueError("branch_length must be numeric")
        length = float(length)
        if not math.isfinite(length) or length < 0:
            raise ValueError("branch_length must be finite and non-negative")
        parents[child] = parent
        nodes.update((parent, child))
        children.setdefault(parent, []).append((child, length))
        children.setdefault(child, [])

    roots = nodes.difference(parents)
    if len(roots) != 1:
        raise ValueError("tree must have exactly one root")
    root = next(iter(roots))
    visited: set[str] = set()
    active: set[str] = set()

    def label(value: str) -> str:
        if _SAFE_LABEL.fullmatch(value):
            return value
        return "'" + value.replace("'", "''") + "'"

    def render(node: str) -> str:
        if node in active:
            raise ValueError("tree contains a cycle")
        if node in visited:
            raise ValueError("tree contains a repeated or disconnected node")
        active.add(node)
        descendants = sorted(children[node], key=lambda item: item[0])
        if descendants:
            rendered = "(" + ",".join(
                render(child) + ":" + _format_length(length)
                for child, length in descendants
            ) + ")"
        else:
            rendered = label(node)
        active.remove(node)
        visited.add(node)
        return rendered

    result = render(root)
    if visited != nodes:
        raise ValueError("tree is disconnected")
    return result + ";"


def _format_length(value: float) -> str:
    # repr is the shortest round-trippable representation and is stable for
    # Python floats; normalize integral values to avoid a distracting .0.
    if value.is_integer():
        return str(int(value))
    return repr(value)


__all__ = ["lineage_to_newick"]
