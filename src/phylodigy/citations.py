"""Semantics-free bibliography and in-text citation resolution.

Citation identity and exact source spans are useful provenance. This module
does not infer adoption, inheritance, function, or graph structure from them.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .canonical import canonical_json, content_digest
from .schema import ArchitecturalGenome


CITATION_EXTRACTOR_VERSION = "3"
OFFSET_BASIS = "citation_normalized_text_codepoints"

_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))?[.)]?\s*"
    r"(?:references|bibliography)\s*:?[ \t]*$"
)
_ENTRY_RE = re.compile(r"(?m)^[ \t]*(?:\[(\d{1,4})\]|(\d{1,4})[.)])[ \t]*")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_ARXIV_RE = re.compile(
    r"\b(?:arXiv(?:\s+preprint)?\s*:\s*)?"
    r"((?:\d{4}\.\d{4,5})|(?:[a-z][a-z0-9.-]*/\d{7}))(?:v\d+)?\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b((?:18|19|20)\d{2})[a-z]?\b", re.I)
_URL_RE = re.compile(r"https?://[^\s<>\]\[]+", re.I)
_NUMERIC_MARKER_RE = re.compile(r"\[([^\]]+)\]")
_AUTHOR_YEAR_RE = re.compile(
    r"\b([A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f'\u2019-]+)"
    r"(?:(?:\s+et\s+al\.)|(?:\s+(?:and|&)\s+"
    r"[A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f'\u2019-]+))?"
    r"(?:\s*,\s*|\s*\(\s*)((?:18|19|20)\d{2})",
    re.I,
)
_IN_TEXT_RE = re.compile(
    r"\[[0-9][0-9,;\s\-\u2013\u2014]*\]|"
    r"[A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f'\u2019-]+"
    r"(?:\s+et\s+al\.)?\s*,?\s*\(?(?:18|19|20)\d{2}\)?"
)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _normalize_source(text: str | bytes) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if not isinstance(text, str):
        raise TypeError("paper text must be str or UTF-8 bytes")
    result = unicodedata.normalize("NFKC", text).lstrip("\ufeff")
    result = result.replace("\r\n", "\n").replace("\r", "\n")
    result = result.replace("\f", "\n").replace("\u00ad", "")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", result)


def _strip_punctuation(value: str) -> str:
    value = value.rstrip(".,;:]} ")
    while value.endswith(")") and value.count("(") < value.count(")"):
        value = value[:-1]
    return value


def normalize_doi(value: str) -> str:
    token = unicodedata.normalize("NFKC", str(value)).strip()
    if re.match(r"^https?://(?:dx\.)?doi\.org/", token, re.I):
        token = urlsplit(token).path.lstrip("/")
    token = re.sub(r"^doi\s*:\s*", "", token, flags=re.I).casefold()
    return _strip_punctuation(token)


def normalize_arxiv_id(value: str) -> str:
    token = unicodedata.normalize("NFKC", str(value)).strip()
    if re.match(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", token, re.I):
        token = re.sub(r"^/(?:abs|pdf)/", "", urlsplit(token).path, flags=re.I)
    token = re.sub(r"^arxiv\s*:\s*", "", token, flags=re.I)
    token = re.sub(r"\.pdf$", "", token, flags=re.I)
    return re.sub(r"v\d+$", "", _strip_punctuation(token), flags=re.I).casefold()


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", value, re.UNICODE))


def _normalize_url(value: str) -> str:
    token = _strip_punctuation(unicodedata.normalize("NFKC", str(value)).strip())
    parsed = urlsplit(token)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return token
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            parsed.path.rstrip("/") or "/",
            parsed.query,
            "",
        )
    )


def _author_key(value: str) -> str:
    words = re.findall(r"[A-Za-z\u00c0-\u024f][A-Za-z\u00c0-\u024f'\u2019-]*", value)
    return words[-1].casefold() if words else ""


@dataclass(frozen=True)
class BibliographyEntry:
    marker: str
    text: str
    index: int
    start: int
    end: int
    source_digest: str
    doi: str = ""
    arxiv_id: str = ""
    year: int | None = None
    urls: tuple[str, ...] = ()
    citation_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "marker", _nonempty(self.marker, "marker"))
        object.__setattr__(self, "text", _nonempty(self.text, "entry text"))
        if self.index < 0 or self.start < 0 or self.end <= self.start:
            raise ValueError("bibliography indices and span are invalid")
        object.__setattr__(self, "doi", normalize_doi(self.doi) if self.doi else "")
        object.__setattr__(self, "arxiv_id", normalize_arxiv_id(self.arxiv_id) if self.arxiv_id else "")
        object.__setattr__(self, "urls", tuple(sorted({_normalize_url(item) for item in self.urls})))
        object.__setattr__(self, "citation_keys", tuple(sorted(set(self.citation_keys))))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "citation_keys": list(self.citation_keys),
            "end": self.end,
            "index": self.index,
            "marker": self.marker,
            "offset_basis": OFFSET_BASIS,
            "source_digest": self.source_digest,
            "start": self.start,
            "text": self.text,
            "urls": list(self.urls),
        }
        if self.doi:
            result["doi"] = self.doi
        if self.arxiv_id:
            result["arxiv_id"] = self.arxiv_id
        if self.year is not None:
            result["year"] = self.year
        return result


@dataclass(frozen=True)
class CitationTarget:
    artifact_id: str
    title: str = ""
    doi: str = ""
    arxiv_id: str = ""
    first_author: str = ""
    year: int | None = None
    aliases: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "doi", normalize_doi(self.doi) if self.doi else "")
        object.__setattr__(self, "arxiv_id", normalize_arxiv_id(self.arxiv_id) if self.arxiv_id else "")
        object.__setattr__(self, "first_author", _author_key(self.first_author))
        object.__setattr__(self, "aliases", tuple(sorted(set(self.aliases))))
        object.__setattr__(self, "urls", tuple(sorted({_normalize_url(item) for item in self.urls})))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CitationTarget":
        authors = raw.get("authors", ())
        first_author = raw.get("first_author", "")
        if not first_author and isinstance(authors, Sequence) and not isinstance(authors, str) and authors:
            first_author = str(authors[0])
        aliases = raw.get("aliases", ())
        urls = raw.get("urls", ())
        return cls(
            artifact_id=str(raw.get("artifact_id", raw.get("id", ""))),
            title=str(raw.get("title", "")),
            doi=str(raw.get("doi", "")),
            arxiv_id=str(raw.get("arxiv_id", raw.get("arxiv", ""))),
            first_author=str(first_author),
            year=raw.get("year"),
            aliases=(aliases,) if isinstance(aliases, str) else tuple(aliases),
            urls=(urls,) if isinstance(urls, str) else tuple(urls),
        )

    @classmethod
    def from_profile(cls, profile: ArchitecturalGenome) -> "CitationTarget":
        if not isinstance(profile, ArchitecturalGenome):
            raise TypeError("profile must be an ArchitecturalGenome")
        metadata = profile.metadata
        identifiers = metadata.get("identifiers", {})
        if not isinstance(identifiers, Mapping):
            identifiers = {}
        aliases = metadata.get("aliases", ())
        urls = identifiers.get("urls", ())
        if isinstance(aliases, str):
            aliases = (aliases,)
        if isinstance(urls, str):
            urls = (urls,)
        return cls(
            artifact_id=profile.artifact_id,
            title=str(metadata.get("title", profile.name)),
            doi=str(identifiers.get("doi", metadata.get("doi", ""))),
            arxiv_id=str(identifiers.get("arxiv_id", metadata.get("arxiv_id", ""))),
            first_author=str(metadata.get("first_author", "")),
            year=int(profile.release_date[:4]) if profile.release_date else None,
            aliases=tuple(aliases),
            urls=tuple(urls),
        )


@dataclass(frozen=True)
class CitationMention:
    marker: str
    start: int
    end: int
    source_digest: str
    citation_keys: tuple[str, ...]
    target_ids: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def mention_id(self) -> str:
        return content_digest(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "citation_keys": list(self.citation_keys),
            "confidence": self.confidence,
            "end": self.end,
            "marker": self.marker,
            "offset_basis": OFFSET_BASIS,
            "source_digest": self.source_digest,
            "start": self.start,
            "target_ids": list(self.target_ids),
        }
        if include_id:
            result["id"] = self.mention_id
        return result


def citation_marker_keys(marker: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", marker).strip()
    keys: set[str] = set()
    numeric = _NUMERIC_MARKER_RE.fullmatch(value)
    if numeric:
        for part in re.split(r"[,;]", numeric.group(1)):
            part = part.strip()
            range_match = re.fullmatch(r"(\d{1,4})\s*[-\u2013\u2014]\s*(\d{1,4})", part)
            if range_match:
                start, end = map(int, range_match.groups())
                if start <= end and end - start <= 100:
                    keys.update(f"marker:{item}" for item in range(start, end + 1))
            elif part.isdigit():
                keys.add(f"marker:{int(part)}")
    for match in _AUTHOR_YEAR_RE.finditer(value):
        keys.add(f"author_year:{match.group(1).casefold()}:{match.group(2)}")
    return tuple(sorted(keys))


def extract_bibliography(text: str | bytes) -> tuple[BibliographyEntry, ...]:
    source = _normalize_source(text)
    heading = _REFERENCE_HEADING_RE.search(source)
    if heading is None:
        return ()
    body_start = heading.end()
    matches = list(_ENTRY_RE.finditer(source, body_start))
    if not matches:
        return ()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    entries = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        marker = str(int(match.group(1) or match.group(2)))
        raw = source[match.end():end]
        normalized = re.sub(r"\s+", " ", raw).strip()
        if not normalized:
            continue
        doi = _DOI_RE.search(normalized)
        arxiv = _ARXIV_RE.search(normalized)
        year = _YEAR_RE.search(normalized)
        author = _author_key(normalized.split(str(year.group(1)), 1)[0]) if year else ""
        keys = {f"marker:{marker}"}
        if author and year:
            keys.add(f"author_year:{author}:{year.group(1)}")
        entries.append(
            BibliographyEntry(
                marker=marker,
                text=normalized,
                index=index,
                start=start,
                end=end,
                source_digest=digest,
                doi=doi.group(0) if doi else "",
                arxiv_id=arxiv.group(1) if arxiv else "",
                year=int(year.group(1)) if year else None,
                urls=tuple(match.group(0) for match in _URL_RE.finditer(normalized)),
                citation_keys=tuple(keys),
            )
        )
    return tuple(entries)


def resolve_bibliography(
    entries: Sequence[BibliographyEntry],
    catalog: Iterable[CitationTarget | Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    targets = tuple(
        item if isinstance(item, CitationTarget) else CitationTarget.from_mapping(item)
        for item in catalog
    )
    resolved: dict[str, dict[str, Any]] = {}
    ambiguous: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for entry in entries:
        candidates: list[tuple[int, float, CitationTarget, str]] = []
        normalized_entry = normalize_title(entry.text)
        for target in targets:
            if entry.doi and target.doi and entry.doi == target.doi:
                candidates.append((0, 1.0, target, "doi"))
            elif entry.arxiv_id and target.arxiv_id and entry.arxiv_id == target.arxiv_id:
                candidates.append((1, 1.0, target, "arxiv_id"))
            elif set(entry.urls) & set(target.urls):
                candidates.append((2, 0.98, target, "url"))
            elif len(normalize_title(target.title)) >= 16 and normalize_title(target.title) in normalized_entry:
                candidates.append((3, 0.92, target, "title"))
            elif target.first_author and target.year and f"author_year:{target.first_author}:{target.year}" in entry.citation_keys:
                candidates.append((4, 0.75, target, "author_year"))
        if not candidates:
            unresolved.append(entry.marker)
            continue
        candidates.sort(key=lambda item: (item[0], -item[1], item[2].artifact_id))
        priority = candidates[0][0]
        best = [item for item in candidates if item[0] == priority]
        ids = sorted({item[2].artifact_id for item in best})
        if len(ids) != 1:
            ambiguous.append({"candidate_artifact_ids": ids, "marker": entry.marker})
            continue
        payload = {
            "artifact_id": best[0][2].artifact_id,
            "confidence": best[0][1],
            "entry": entry.to_dict(),
            "match_basis": best[0][3],
        }
        for key in entry.citation_keys:
            if key in resolved and resolved[key]["artifact_id"] != payload["artifact_id"]:
                ambiguous.append({"citation_key": key, "candidate_artifact_ids": sorted({resolved[key]["artifact_id"], payload["artifact_id"]})})
                resolved.pop(key, None)
            else:
                resolved[key] = payload
    return resolved, {
        "ambiguous": sorted(ambiguous, key=canonical_json),
        "entry_count": len(entries),
        "resolved_key_count": len(resolved),
        "unresolved_markers": sorted(unresolved),
    }


def extract_citation_evidence(
    text: str | bytes,
    catalog: Iterable[CitationTarget | Mapping[str, Any]] = (),
) -> tuple[tuple[CitationMention, ...], dict[str, Any]]:
    """Resolve exact citation spans without assigning semantic roles."""

    source = _normalize_source(text)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    entries = extract_bibliography(source)
    resolved, resolution = resolve_bibliography(entries, catalog)
    heading = _REFERENCE_HEADING_RE.search(source)
    body_end = heading.start() if heading else len(source)
    mentions = []
    for match in _IN_TEXT_RE.finditer(source, 0, body_end):
        marker = match.group(0)
        keys = citation_marker_keys(marker)
        targets = sorted({resolved[key]["artifact_id"] for key in keys if key in resolved})
        confidences = [resolved[key]["confidence"] for key in keys if key in resolved]
        mentions.append(
            CitationMention(
                marker=marker,
                start=match.start(),
                end=match.end(),
                source_digest=digest,
                citation_keys=keys,
                target_ids=tuple(targets),
                confidence=min(confidences) if confidences else 0.0,
            )
        )
    report = {
        "bibliography": [item.to_dict() for item in entries],
        "extractor_version": CITATION_EXTRACTOR_VERSION,
        "mention_count": len(mentions),
        "resolution": resolution,
        "source_digest": digest,
        "structural_claims_created": 0,
    }
    return tuple(mentions), report


__all__ = [
    "BibliographyEntry",
    "CITATION_EXTRACTOR_VERSION",
    "CitationMention",
    "CitationTarget",
    "OFFSET_BASIS",
    "citation_marker_keys",
    "extract_bibliography",
    "extract_citation_evidence",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_title",
    "resolve_bibliography",
]
