"""Ontology-free paper annotations for discovered computation graphs.

The paper channel does not decide which architectural features exist. Graph
frontends and structural discovery do that first. This module only attaches
paper-supplied names and descriptions to stable objects that already exist in
a :class:`~phylodigy.computation_graph.GraphProfile`.

There is intentionally no built-in architecture vocabulary, trait catalog, or
keyword/regular-expression classifier here. Semantic annotation may be
produced by a language model, another NLP system, or a human reviewer, but it
must arrive as structured claims with exact source spans. Every claim is then
validated against the supplied graph profile and pinned to both paper and graph
digests.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from datetime import date
import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from .canonical import content_digest, freeze_json, normalize_json
from .computation_graph import GraphProfile


EXTRACTOR_NAME = "phylodigy.paper_graph_annotations"
EXTRACTOR_VERSION = "2"
PAPER_DOCUMENT_SCHEMA_VERSION = "1.0.0"
PAPER_ANNOTATION_SCHEMA_VERSION = "2.0.0"

TARGET_KINDS = frozenset(
    {"graph", "node", "edge", "region", "character"}
)


def _nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value.strip()


def _confidence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("annotation confidence must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("annotation confidence must be between zero and one")
    return result


def _as_iso_date(value: str | date | None, field_name: str) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO date string, date, or None")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must use canonical YYYY-MM-DD form")
    return value


def _integer_offset(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative")
    return value


def normalize_extracted_text(text: str | bytes) -> str:
    """Normalize paper text into a deterministic annotation-offset space.

    This performs document cleanup only. It does not search for, name, or
    classify architecture concepts.
    """

    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if not isinstance(text, str):
        raise TypeError("paper text must be str or UTF-8 bytes")
    value = unicodedata.normalize("NFKC", text).lstrip("\ufeff")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\f", "\n\n").replace("\u00ad", "")
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    value = re.sub(
        r"([a-z]{3,})[\u2010\u2011-]\s*\n\s*([a-z]{3,})",
        r"\1\2",
        value,
    )
    lines = [
        re.sub(r"[\t\v\u00a0 ]+", " ", line).strip()
        for line in value.split("\n")
    ]
    output: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            output.append(" ".join(paragraph))
            paragraph.clear()

    for line in lines:
        if line and _heading_name(line) is not None:
            flush()
            if output and output[-1] != "":
                output.append("")
            output.append(line)
        elif line:
            paragraph.append(line)
        else:
            flush()
            if (
                output
                and output[-1] != ""
                and _heading_name(output[-1]) is None
            ):
                output.append("")
    flush()
    while output and output[-1] == "":
        output.pop()
    normalized = "\n".join(output).strip()
    normalized = re.sub(r" *\n *", "\n", normalized)
    return re.sub(r"\n{3,}", "\n\n", normalized)


@dataclass(frozen=True)
class SectionSpan:
    """A structural, half-open section span in normalized paper text."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class SentenceSpan:
    """A half-open sentence span in normalized paper text."""

    section: str
    start: int
    end: int
    text: str


_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[.)]?\s+(.+?)\s*$",
    re.IGNORECASE,
)
_ABBREVIATION_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")


def _heading_name(line: str) -> str | None:
    """Recognize visual heading form without a section-name vocabulary."""

    candidate = line.strip().rstrip(":").strip()
    numbered = _NUMBERED_HEADING_RE.match(candidate)
    if numbered:
        candidate = numbered.group(1).strip().rstrip(":")
    words = candidate.split()
    if not words or len(words) > 12 or len(candidate) > 100:
        return None
    if candidate.endswith((".", "?", "!", ",", ";")):
        return None
    visual_heading = (
        numbered is not None
        or (len(words) == 1 and words[0].isalpha())
        or candidate.isupper()
        or candidate.istitle()
    )
    return candidate if visual_heading else None


def split_sections(normalized_text: str) -> tuple[SectionSpan, ...]:
    """Split text using generic line/heading form, never architecture terms."""

    if not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string")
    if not normalized_text:
        return ()
    headings: list[tuple[str, int, int]] = []
    cursor = 0
    for raw in normalized_text.splitlines(keepends=True):
        line = raw.rstrip("\n")
        name = _heading_name(line)
        if name is not None:
            headings.append((name, cursor, cursor + len(raw)))
        cursor += len(raw)
    if not headings:
        return (SectionSpan("Document", 0, len(normalized_text)),)

    sections: list[SectionSpan] = []
    front = normalized_text[: headings[0][1]]
    if front.strip():
        start = len(front) - len(front.lstrip())
        end = len(front.rstrip())
        if start < end:
            sections.append(SectionSpan("Front matter", start, end))
    for index, (name, _heading_start, body_start) in enumerate(headings):
        body_end = (
            headings[index + 1][1]
            if index + 1 < len(headings)
            else len(normalized_text)
        )
        while body_start < body_end and normalized_text[body_start].isspace():
            body_start += 1
        while body_end > body_start and normalized_text[body_end - 1].isspace():
            body_end -= 1
        if body_start < body_end:
            sections.append(SectionSpan(name, body_start, body_end))
    return tuple(sections)


def _sentence_ranges(text: str, start: int, end: int) -> Iterable[tuple[int, int]]:
    cursor = start
    while cursor < end and text[cursor].isspace():
        cursor += 1
    sentence_start = cursor
    while cursor < end:
        char = text[cursor]
        if char in ".?!":
            decimal = (
                char == "."
                and cursor > start
                and cursor + 1 < end
                and text[cursor - 1].isdigit()
                and text[cursor + 1].isdigit()
            )
            prefix = text[max(sentence_start, cursor - 8) : cursor + 1]
            abbreviation = char == "." and (
                _ABBREVIATION_RE.search(prefix) is not None
                or re.search(
                    r"\b(?:e\.g|i\.e|et al|fig|eq|sec|tab|vs)\.$",
                    prefix,
                    re.I,
                )
                is not None
            )
            if not decimal and not abbreviation:
                candidate_end = cursor + 1
                while (
                    candidate_end < end
                    and text[candidate_end] in "\"'\u2019\u201d)]}"
                ):
                    candidate_end += 1
                next_start = candidate_end
                while next_start < end and text[next_start].isspace():
                    next_start += 1
                if (
                    next_start >= end
                    or text[next_start].isupper()
                    or text[next_start].isdigit()
                ):
                    actual_end = candidate_end
                    while (
                        actual_end > sentence_start
                        and text[actual_end - 1].isspace()
                    ):
                        actual_end -= 1
                    if sentence_start < actual_end:
                        yield sentence_start, actual_end
                    sentence_start = next_start
                    cursor = next_start
                    continue
        cursor += 1
    actual_end = end
    while actual_end > sentence_start and text[actual_end - 1].isspace():
        actual_end -= 1
    if sentence_start < actual_end:
        yield sentence_start, actual_end


def split_sentences(
    normalized_text: str,
    sections: Sequence[SectionSpan] | None = None,
) -> tuple[SentenceSpan, ...]:
    """Return exact sentence spans for source anchoring and display."""

    selected = (
        tuple(sections) if sections is not None else split_sections(normalized_text)
    )
    result: list[SentenceSpan] = []
    for section in selected:
        for start, end in _sentence_ranges(
            normalized_text, section.start, section.end
        ):
            result.append(
                SentenceSpan(section.name, start, end, normalized_text[start:end])
            )
    return tuple(result)


@dataclass(frozen=True)
class PaperDocument:
    """Normalized paper source with no inferred architectural semantics."""

    artifact_id: str
    normalized_text: str = field(repr=False)
    source_id: str = ""
    artifact_kind: str = "paper"
    name: str = ""
    release_date: str | None = None
    date_min: str | None = None
    date_max: str | None = None
    identifiers: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PAPER_DOCUMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        artifact_id = _nonempty(self.artifact_id, "artifact_id")
        if not isinstance(self.normalized_text, str):
            raise TypeError("normalized_text must be a string")
        normalized = normalize_extracted_text(self.normalized_text)
        if normalized != self.normalized_text:
            raise ValueError(
                "PaperDocument.normalized_text must already be normalized; "
                "use extract_paper_profile"
            )
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(
            self, "source_id", _nonempty(self.source_id or artifact_id, "source_id")
        )
        object.__setattr__(
            self, "artifact_kind", _nonempty(self.artifact_kind, "artifact_kind")
        )
        if not isinstance(self.name, str):
            raise TypeError("paper name must be a string")
        object.__setattr__(self, "name", self.name.strip())
        release_date = _as_iso_date(self.release_date, "release_date")
        date_min = _as_iso_date(self.date_min, "date_min") or release_date
        date_max = _as_iso_date(self.date_max, "date_max") or release_date
        if date_min and date_max and date_min > date_max:
            raise ValueError("date_min cannot be after date_max")
        object.__setattr__(self, "release_date", release_date)
        object.__setattr__(self, "date_min", date_min)
        object.__setattr__(self, "date_max", date_max)
        identifiers = normalize_json(dict(self.identifiers))
        metadata = normalize_json(dict(self.metadata))
        if not isinstance(identifiers, dict) or not isinstance(metadata, dict):
            raise TypeError("paper document identifiers and metadata must be objects")
        object.__setattr__(self, "identifiers", freeze_json(identifiers))
        object.__setattr__(self, "metadata", freeze_json(metadata))
        if self.schema_version != PAPER_DOCUMENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PAPER_DOCUMENT_SCHEMA_VERSION!r}"
            )

    @property
    def source_digest(self) -> str:
        return hashlib.sha256(self.normalized_text.encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "artifact_type": "phylodigy.paper_document",
            "normalized_text": self.normalized_text,
            "schema_version": self.schema_version,
            "source_digest": self.source_digest,
            "source_id": self.source_id,
        }
        if self.name:
            result["name"] = self.name
        for key in ("release_date", "date_min", "date_max"):
            value = getattr(self, key)
            if value:
                result[key] = value
        if self.identifiers:
            result["identifiers"] = normalize_json(self.identifiers)
        if self.metadata:
            result["metadata"] = normalize_json(self.metadata)
        if include_digest:
            result["digest"] = content_digest(result)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PaperDocument":
        if raw.get("artifact_type") != "phylodigy.paper_document":
            raise ValueError("not a phylodigy paper document")
        result = cls(
            artifact_id=str(raw["artifact_id"]),
            artifact_kind=str(raw.get("artifact_kind", "paper")),
            normalized_text=str(raw.get("normalized_text", "")),
            source_id=str(raw.get("source_id", "")),
            name=str(raw.get("name", "")),
            release_date=raw.get("release_date"),
            date_min=raw.get("date_min"),
            date_max=raw.get("date_max"),
            identifiers=raw.get("identifiers", {}),
            metadata=raw.get("metadata", {}),
            schema_version=str(
                raw.get("schema_version", PAPER_DOCUMENT_SCHEMA_VERSION)
            ),
        )
        supplied_source = raw.get("source_digest")
        if supplied_source is not None and str(supplied_source) != result.source_digest:
            raise ValueError("paper document source digest mismatch")
        supplied = raw.get("digest")
        if supplied is not None and str(supplied) != result.digest:
            raise ValueError("paper document digest mismatch")
        return result


@dataclass(frozen=True)
class GraphAnnotationTarget:
    """Content-addressed reference to one already-discovered graph object.

    ``graph_digest`` is the graph's structural digest. Runtime observations
    such as tensor shapes and probe values are deliberately excluded.
    """

    kind: str
    target_id: str
    structure_digest: str
    graph_digest: str

    def __post_init__(self) -> None:
        kind = _nonempty(self.kind, "target kind")
        if kind not in TARGET_KINDS:
            raise ValueError(f"unsupported graph annotation target kind: {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "target_id", _nonempty(self.target_id, "target id")
        )
        object.__setattr__(
            self,
            "structure_digest",
            _nonempty(self.structure_digest, "structure digest"),
        )
        object.__setattr__(
            self, "graph_digest", _nonempty(self.graph_digest, "graph digest")
        )

    @classmethod
    def resolve(
        cls,
        graph_profile: GraphProfile,
        *,
        kind: str,
        target_id: str,
    ) -> "GraphAnnotationTarget":
        """Resolve and validate a target against ``graph_profile``."""

        catalog = _target_catalog(graph_profile)
        key = (str(kind), str(target_id))
        try:
            return catalog[key]
        except KeyError as exc:
            raise ValueError(
                f"unknown {kind!r} target {target_id!r} for graph "
                f"{graph_profile.graph.structural_digest}"
            ) from exc

    def to_dict(self) -> dict[str, str]:
        return {
            "graph_digest": self.graph_digest,
            "kind": self.kind,
            "structure_digest": self.structure_digest,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphAnnotationTarget":
        return cls(
            kind=str(raw["kind"]),
            target_id=str(raw["target_id"]),
            structure_digest=str(raw["structure_digest"]),
            graph_digest=str(raw["graph_digest"]),
        )


def _target_catalog(
    graph_profile: GraphProfile,
) -> dict[tuple[str, str], GraphAnnotationTarget]:
    if not isinstance(graph_profile, GraphProfile):
        raise TypeError("graph_profile must be a GraphProfile")
    graph = graph_profile.graph
    graph_digest = graph.structural_digest
    result: dict[tuple[str, str], GraphAnnotationTarget] = {}

    def add(kind: str, target_id: str, structure_digest: str) -> None:
        result[(kind, target_id)] = GraphAnnotationTarget(
            kind=kind,
            target_id=target_id,
            structure_digest=structure_digest,
            graph_digest=graph_digest,
        )

    add("graph", graph_digest, graph_digest)
    for node in graph.nodes:
        add(
            "node",
            node.node_id,
            content_digest(
                {
                    "graph_digest": graph_digest,
                    "kind": "node",
                    "node": node.structural_dict(),
                }
            ),
        )
    for edge in graph.edges:
        add(
            "edge",
            edge.edge_id,
            content_digest(
                {
                    "edge": edge.to_dict(),
                    "graph_digest": graph_digest,
                    "kind": "edge",
                }
            ),
        )
    for region in graph_profile.regions:
        add("region", region.occurrence_id, region.structural_signature)
    for character in graph_profile.characters:
        add(
            "character",
            character.character_id,
            character.structural_signature,
        )
    return result


def _validate_target(
    target: GraphAnnotationTarget,
    graph_profile: GraphProfile,
) -> GraphAnnotationTarget:
    resolved = GraphAnnotationTarget.resolve(
        graph_profile, kind=target.kind, target_id=target.target_id
    )
    if target != resolved:
        raise ValueError(
            "annotation target digest does not match supplied graph: "
            f"{target.kind}:{target.target_id}"
        )
    return resolved


@dataclass(frozen=True)
class GraphAnnotation:
    """Paper-backed terminology, function, or provenance for a graph object."""

    target: GraphAnnotationTarget
    start: int
    end: int
    excerpt: str
    source_id: str
    source_digest: str
    name: str = ""
    description: str = ""
    claimed_function: str = ""
    provenance_statement: str = ""
    confidence: float = 1.0
    citations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.target, GraphAnnotationTarget):
            raise TypeError("annotation target must be a GraphAnnotationTarget")
        semantic_fields = {
            "name": _optional_text(self.name, "annotation name"),
            "description": _optional_text(
                self.description, "annotation description"
            ),
            "claimed_function": _optional_text(
                self.claimed_function, "annotation claimed_function"
            ),
            "provenance_statement": _optional_text(
                self.provenance_statement, "annotation provenance_statement"
            ),
        }
        if not any(semantic_fields.values()):
            raise ValueError(
                "a graph annotation requires a name, description, claimed "
                "function, or provenance statement"
            )
        for key, value in semantic_fields.items():
            object.__setattr__(self, key, value)
        start = _integer_offset(self.start, "annotation start")
        end = _integer_offset(self.end, "annotation end")
        if end <= start:
            raise ValueError("annotation end must be after start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(
            self, "excerpt", _nonempty(self.excerpt, "annotation excerpt")
        )
        object.__setattr__(
            self, "source_id", _nonempty(self.source_id, "source_id")
        )
        object.__setattr__(
            self,
            "source_digest",
            _nonempty(self.source_digest, "source_digest"),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        citations = tuple(
            sorted({_nonempty(item, "citation") for item in self.citations})
        )
        object.__setattr__(self, "citations", citations)
        metadata = normalize_json(dict(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("annotation metadata must be an object")
        object.__setattr__(self, "metadata", freeze_json(metadata))

    @property
    def annotation_id(self) -> str:
        return content_digest(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "confidence": self.confidence,
            "end": self.end,
            "excerpt": self.excerpt,
            "source_digest": self.source_digest,
            "source_id": self.source_id,
            "start": self.start,
            "target": self.target.to_dict(),
        }
        for key in (
            "name",
            "description",
            "claimed_function",
            "provenance_statement",
        ):
            value = getattr(self, key)
            if value:
                result[key] = value
        if self.citations:
            result["citations"] = list(self.citations)
        if self.metadata:
            result["metadata"] = normalize_json(self.metadata)
        if include_id:
            result["id"] = self.annotation_id
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphAnnotation":
        result = cls(
            target=GraphAnnotationTarget.from_dict(raw["target"]),
            start=raw["start"],
            end=raw["end"],
            excerpt=str(raw["excerpt"]),
            source_id=str(raw["source_id"]),
            source_digest=str(raw["source_digest"]),
            name=str(raw.get("name", "")),
            description=str(raw.get("description", "")),
            claimed_function=str(raw.get("claimed_function", "")),
            provenance_statement=str(raw.get("provenance_statement", "")),
            confidence=raw.get("confidence", 1.0),
            citations=tuple(str(item) for item in raw.get("citations", [])),
            metadata=raw.get("metadata", {}),
        )
        supplied = raw.get("id")
        if supplied is not None and str(supplied) != result.annotation_id:
            raise ValueError("graph annotation digest mismatch")
        return result


@dataclass(frozen=True)
class UnverifiedPaperClaim:
    """Paper text retained without asserting that the graph contains it.

    These claims are deliberately separate from :class:`GraphAnnotation` and
    have no graph target. They can inform review or provenance work, but cannot
    become structural characters or affect graph distance.
    """

    statement: str
    start: int
    end: int
    excerpt: str
    source_id: str
    source_digest: str
    confidence: float = 1.0
    citations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "statement", _nonempty(self.statement, "unverified statement")
        )
        start = _integer_offset(self.start, "unverified claim start")
        end = _integer_offset(self.end, "unverified claim end")
        if end <= start:
            raise ValueError("unverified claim end must be after start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(
            self, "excerpt", _nonempty(self.excerpt, "unverified claim excerpt")
        )
        object.__setattr__(
            self, "source_id", _nonempty(self.source_id, "source_id")
        )
        object.__setattr__(
            self,
            "source_digest",
            _nonempty(self.source_digest, "source_digest"),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "citations",
            tuple(sorted({_nonempty(item, "citation") for item in self.citations})),
        )
        metadata = normalize_json(dict(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("unverified claim metadata must be an object")
        object.__setattr__(self, "metadata", freeze_json(metadata))

    @property
    def claim_id(self) -> str:
        return content_digest(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "confidence": self.confidence,
            "end": self.end,
            "excerpt": self.excerpt,
            "source_digest": self.source_digest,
            "source_id": self.source_id,
            "start": self.start,
            "statement": self.statement,
        }
        if self.citations:
            result["citations"] = list(self.citations)
        if self.metadata:
            result["metadata"] = normalize_json(self.metadata)
        if include_id:
            result["id"] = self.claim_id
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "UnverifiedPaperClaim":
        result = cls(
            statement=str(raw["statement"]),
            start=raw["start"],
            end=raw["end"],
            excerpt=str(raw["excerpt"]),
            source_id=str(raw["source_id"]),
            source_digest=str(raw["source_digest"]),
            confidence=raw.get("confidence", 1.0),
            citations=tuple(str(item) for item in raw.get("citations", [])),
            metadata=raw.get("metadata", {}),
        )
        supplied = raw.get("id")
        if supplied is not None and str(supplied) != result.claim_id:
            raise ValueError("unverified paper claim digest mismatch")
        return result


@dataclass(frozen=True)
class PaperGraphAnnotations:
    """Validated paper semantics layered over one structural graph.

    The layer is not pinned to frontend metadata, runtime observations, or a
    particular set of fingerprint radii. Validation requires the same
    structural graph and re-resolves every target in the supplied profile.
    """

    paper: PaperDocument
    graph_profile: InitVar[GraphProfile]
    graph_digest: str
    annotations: tuple[GraphAnnotation, ...] = ()
    unverified_claims: tuple[UnverifiedPaperClaim, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PAPER_ANNOTATION_SCHEMA_VERSION

    def __post_init__(self, graph_profile: GraphProfile) -> None:
        if not isinstance(self.paper, PaperDocument):
            raise TypeError("paper must be a PaperDocument")
        if not isinstance(graph_profile, GraphProfile):
            raise TypeError("graph_profile must be supplied for target validation")
        object.__setattr__(
            self, "graph_digest", _nonempty(self.graph_digest, "graph digest")
        )
        if graph_profile.graph.structural_digest != self.graph_digest:
            raise ValueError("paper annotations are pinned to a different graph")
        unique = {item.annotation_id: item for item in self.annotations}
        ordered = tuple(unique[key] for key in sorted(unique))
        for annotation in ordered:
            _validate_target(annotation.target, graph_profile)
            if annotation.target.graph_digest != self.graph_digest:
                raise ValueError("annotation target belongs to a different graph")
            if annotation.source_id != self.paper.source_id:
                raise ValueError("annotation source_id does not match paper")
            if annotation.source_digest != self.paper.source_digest:
                raise ValueError("annotation source digest does not match paper")
            if annotation.end > len(self.paper.normalized_text):
                raise ValueError("annotation span lies outside paper text")
            if (
                self.paper.normalized_text[annotation.start : annotation.end]
                != annotation.excerpt
            ):
                raise ValueError("annotation excerpt does not match its paper span")
            for field_name in (
                "name",
                "description",
                "claimed_function",
                "provenance_statement",
            ):
                value = getattr(annotation, field_name)
                if value and not _verbatim_in_excerpt(value, annotation.excerpt):
                    raise ValueError(
                        f"annotation {field_name.replace('_', ' ')} must be "
                        "verbatim paper text"
                    )
        object.__setattr__(self, "annotations", ordered)
        unique_claims = {item.claim_id: item for item in self.unverified_claims}
        ordered_claims = tuple(unique_claims[key] for key in sorted(unique_claims))
        for claim in ordered_claims:
            if claim.source_id != self.paper.source_id:
                raise ValueError("unverified claim source_id does not match paper")
            if claim.source_digest != self.paper.source_digest:
                raise ValueError("unverified claim source digest does not match paper")
            if claim.end > len(self.paper.normalized_text):
                raise ValueError("unverified claim span lies outside paper text")
            if (
                self.paper.normalized_text[claim.start : claim.end]
                != claim.excerpt
            ):
                raise ValueError("unverified claim excerpt does not match its paper span")
            if not _verbatim_in_excerpt(claim.statement, claim.excerpt):
                raise ValueError("unverified statement must be verbatim paper text")
        object.__setattr__(self, "unverified_claims", ordered_claims)
        metadata = normalize_json(dict(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("paper graph annotation metadata must be an object")
        object.__setattr__(self, "metadata", freeze_json(metadata))
        if self.schema_version != PAPER_ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PAPER_ANNOTATION_SCHEMA_VERSION!r}"
            )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def validate_against(self, graph_profile: GraphProfile) -> None:
        """Fail if any annotation is stale or invented for this graph."""

        if graph_profile.graph.structural_digest != self.graph_digest:
            raise ValueError("paper annotations are pinned to a different graph")
        for annotation in self.annotations:
            _validate_target(annotation.target, graph_profile)

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "annotations": [item.to_dict() for item in self.annotations],
            "artifact_type": "phylodigy.paper_graph_annotations",
            "extractor": {"name": EXTRACTOR_NAME, "version": EXTRACTOR_VERSION},
            "graph_digest": self.graph_digest,
            "paper": self.paper.to_dict(),
            "schema_version": self.schema_version,
            "unverified_claims": [
                item.to_dict() for item in self.unverified_claims
            ],
        }
        if self.metadata:
            result["metadata"] = normalize_json(self.metadata)
        if include_digest:
            result["digest"] = content_digest(result)
        return result

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        graph_profile: GraphProfile,
    ) -> "PaperGraphAnnotations":
        if raw.get("artifact_type") != "phylodigy.paper_graph_annotations":
            raise ValueError("not a phylodigy paper graph annotation artifact")
        result = cls(
            paper=PaperDocument.from_dict(raw["paper"]),
            graph_profile=graph_profile,
            graph_digest=str(raw["graph_digest"]),
            annotations=tuple(
                GraphAnnotation.from_dict(item)
                for item in raw.get("annotations", [])
            ),
            unverified_claims=tuple(
                UnverifiedPaperClaim.from_dict(item)
                for item in raw.get("unverified_claims", [])
            ),
            metadata=raw.get("metadata", {}),
            schema_version=str(
                raw.get("schema_version", PAPER_ANNOTATION_SCHEMA_VERSION)
            ),
        )
        result.validate_against(graph_profile)
        supplied = raw.get("digest")
        if supplied is not None and str(supplied) != result.digest:
            raise ValueError("paper graph annotation artifact digest mismatch")
        return result


def _verbatim_in_excerpt(value: str, excerpt: str) -> bool:
    def normalize(item: str) -> str:
        return re.sub(r"\s+", " ", item).strip().casefold()

    return normalize(value) in normalize(excerpt)


def _annotation_from_mapping(
    raw: Mapping[str, Any],
    *,
    paper: PaperDocument,
    graph_profile: GraphProfile,
) -> GraphAnnotation:
    target_raw = raw.get("target")
    if isinstance(target_raw, GraphAnnotationTarget):
        target = _validate_target(target_raw, graph_profile)
    elif isinstance(target_raw, Mapping):
        target = GraphAnnotationTarget.resolve(
            graph_profile,
            kind=str(target_raw.get("kind", "")),
            target_id=str(target_raw.get("target_id", "")),
        )
        # Do not silently repair stale digests supplied by a semantic producer.
        for key, expected in (
            ("graph_digest", target.graph_digest),
            ("structure_digest", target.structure_digest),
        ):
            supplied = target_raw.get(key)
            if supplied is not None and str(supplied) != expected:
                raise ValueError(f"annotation target {key} does not match graph")
    else:
        raise TypeError(
            "annotation target must be a GraphAnnotationTarget or mapping"
        )

    start = _integer_offset(raw.get("start"), "annotation start")
    end = _integer_offset(raw.get("end"), "annotation end")
    if end <= start or end > len(paper.normalized_text):
        raise ValueError("annotation span lies outside paper text")
    excerpt = paper.normalized_text[start:end]
    semantic_fields = {
        "name": _optional_text(raw.get("name"), "annotation name"),
        "description": _optional_text(
            raw.get("description"), "annotation description"
        ),
        "claimed_function": _optional_text(
            raw.get("claimed_function"), "annotation claimed_function"
        ),
        "provenance_statement": _optional_text(
            raw.get("provenance_statement"), "annotation provenance_statement"
        ),
    }
    if not any(semantic_fields.values()):
        raise ValueError(
            "a graph annotation requires a name, description, claimed function, "
            "or provenance statement"
        )
    for field_name, value in semantic_fields.items():
        if value and not _verbatim_in_excerpt(value, excerpt):
            readable = field_name.replace("_", " ")
            raise ValueError(
                f"annotation {readable} must be a verbatim passage within its "
                "paper span"
            )
    citations_raw = raw.get("citations", ())
    if not isinstance(citations_raw, Sequence) or isinstance(
        citations_raw, (str, bytes)
    ):
        raise TypeError("annotation citations must be a sequence of strings")
    return GraphAnnotation(
        target=target,
        start=start,
        end=end,
        excerpt=excerpt,
        source_id=paper.source_id,
        source_digest=paper.source_digest,
        name=semantic_fields["name"],
        description=semantic_fields["description"],
        claimed_function=semantic_fields["claimed_function"],
        provenance_statement=semantic_fields["provenance_statement"],
        confidence=raw.get("confidence", 1.0),
        citations=tuple(str(item) for item in citations_raw),
        metadata=raw.get("metadata", {}),
    )


def _unverified_claim_from_mapping(
    raw: Mapping[str, Any],
    *,
    paper: PaperDocument,
) -> UnverifiedPaperClaim:
    if "target" in raw:
        raise ValueError("unverified paper claims cannot carry a graph target")
    start = _integer_offset(raw.get("start"), "unverified claim start")
    end = _integer_offset(raw.get("end"), "unverified claim end")
    if end <= start or end > len(paper.normalized_text):
        raise ValueError("unverified claim span lies outside paper text")
    excerpt = paper.normalized_text[start:end]
    statement = _nonempty(raw.get("statement"), "unverified statement")
    if not _verbatim_in_excerpt(statement, excerpt):
        raise ValueError("unverified statement must be verbatim paper text")
    citations_raw = raw.get("citations", ())
    if not isinstance(citations_raw, Sequence) or isinstance(
        citations_raw, (str, bytes)
    ):
        raise TypeError("unverified claim citations must be a sequence of strings")
    return UnverifiedPaperClaim(
        statement=statement,
        start=start,
        end=end,
        excerpt=excerpt,
        source_id=paper.source_id,
        source_digest=paper.source_digest,
        confidence=raw.get("confidence", 1.0),
        citations=tuple(str(item) for item in citations_raw),
        metadata=raw.get("metadata", {}),
    )


def annotate_graph_from_paper(
    graph_profile: GraphProfile,
    paper: PaperDocument,
    annotations: Iterable[GraphAnnotation | Mapping[str, Any]],
    *,
    unverified_claims: Iterable[UnverifiedPaperClaim | Mapping[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> PaperGraphAnnotations:
    """Validate structured paper semantics against a discovered graph.

    This performs no semantic search. Each input annotation must name an
    existing graph target and quote its semantic fields from an exact span of
    ``paper.normalized_text``. Claims with no verified structural target belong
    in ``unverified_claims`` and can never become graph annotations.
    """

    if not isinstance(graph_profile, GraphProfile):
        raise TypeError("graph_profile must be a GraphProfile")
    if not isinstance(paper, PaperDocument):
        raise TypeError("paper must be a PaperDocument")
    resolved: list[GraphAnnotation] = []
    for raw in annotations:
        if isinstance(raw, GraphAnnotation):
            _validate_target(raw.target, graph_profile)
            if (
                raw.source_id != paper.source_id
                or raw.source_digest != paper.source_digest
            ):
                raise ValueError("annotation is pinned to a different paper")
            if (
                raw.end > len(paper.normalized_text)
                or paper.normalized_text[raw.start : raw.end] != raw.excerpt
            ):
                raise ValueError("annotation excerpt does not match its paper span")
            for field_name in (
                "name",
                "description",
                "claimed_function",
                "provenance_statement",
            ):
                value = getattr(raw, field_name)
                if value and not _verbatim_in_excerpt(value, raw.excerpt):
                    raise ValueError(
                        f"annotation {field_name.replace('_', ' ')} must be "
                        "verbatim paper text"
                    )
            resolved.append(raw)
        elif isinstance(raw, Mapping):
            resolved.append(
                _annotation_from_mapping(
                    raw, paper=paper, graph_profile=graph_profile
                )
            )
        else:
            raise TypeError(
                "annotations must contain GraphAnnotation objects or mappings"
            )
    unresolved: list[UnverifiedPaperClaim] = []
    for raw in unverified_claims:
        if isinstance(raw, UnverifiedPaperClaim):
            if (
                raw.source_id != paper.source_id
                or raw.source_digest != paper.source_digest
            ):
                raise ValueError("unverified claim is pinned to a different paper")
            if (
                raw.end > len(paper.normalized_text)
                or paper.normalized_text[raw.start : raw.end] != raw.excerpt
            ):
                raise ValueError(
                    "unverified claim excerpt does not match its paper span"
                )
            if not _verbatim_in_excerpt(raw.statement, raw.excerpt):
                raise ValueError("unverified statement must be verbatim paper text")
            unresolved.append(raw)
        elif isinstance(raw, Mapping):
            unresolved.append(_unverified_claim_from_mapping(raw, paper=paper))
        else:
            raise TypeError(
                "unverified_claims must contain UnverifiedPaperClaim objects "
                "or mappings"
            )
    result = PaperGraphAnnotations(
        paper=paper,
        graph_profile=graph_profile,
        graph_digest=graph_profile.graph.structural_digest,
        annotations=tuple(resolved),
        unverified_claims=tuple(unresolved),
        metadata=metadata or {},
    )
    result.validate_against(graph_profile)
    return result


def extract_paper_profile(
    text: str | bytes,
    *,
    artifact_id: str,
    artifact_kind: str = "paper",
    name: str = "",
    release_date: str | date | None = None,
    date_min: str | date | None = None,
    date_max: str | date | None = None,
    identifiers: Mapping[str, Any] | None = None,
    source_id: str | None = None,
) -> PaperDocument:
    """Prepare an unannotated paper document without inferring any traits.

    The historical function name is retained as a migration aid. Its return
    value is now :class:`PaperDocument`, not a trait-bearing artifact profile.
    Call :func:`annotate_graph_from_paper` only after graph discovery.
    """

    normalized = normalize_extracted_text(text)
    sections = split_sections(normalized)
    sentences = split_sentences(normalized, sections)
    return PaperDocument(
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        normalized_text=normalized,
        source_id=source_id or artifact_id,
        name=name,
        release_date=_as_iso_date(release_date, "release_date"),
        date_min=_as_iso_date(date_min, "date_min"),
        date_max=_as_iso_date(date_max, "date_max"),
        identifiers=identifiers or {},
        metadata={
            "offset_basis": "normalized_text_codepoints",
            "section_count": len(sections),
            "semantic_annotations": 0,
            "sentence_count": len(sentences),
        },
    )


__all__ = [
    "EXTRACTOR_NAME",
    "EXTRACTOR_VERSION",
    "GraphAnnotation",
    "GraphAnnotationTarget",
    "PAPER_ANNOTATION_SCHEMA_VERSION",
    "PAPER_DOCUMENT_SCHEMA_VERSION",
    "PaperDocument",
    "PaperGraphAnnotations",
    "SectionSpan",
    "SentenceSpan",
    "TARGET_KINDS",
    "UnverifiedPaperClaim",
    "annotate_graph_from_paper",
    "extract_paper_profile",
    "normalize_extracted_text",
    "split_sections",
    "split_sentences",
]
