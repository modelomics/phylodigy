"""Small, deterministic file-I/O helpers for phylodigy artifacts.

The core package has no mandatory third-party dependencies.  PDF extraction is
therefore imported lazily and fails with an installation hint when ``pypdf`` is
not available. Plain text, Markdown, JSON, and canonical profile round-trips use
only the Python standard library.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, TextIO

from .canonical import canonical_json, normalize_json
from .schema import ArtifactProfile


TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
PDF_SUFFIX = ".pdf"


class PhylodigyIOError(ValueError):
    """Base class for actionable input/output format errors."""


class DocumentReadError(PhylodigyIOError):
    """Raised when a paper document cannot be converted to text."""


class ProfileReadError(PhylodigyIOError):
    """Raised when a file is not a canonical artifact profile."""


def _path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _require_readable_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")
    if not path.is_file():
        raise DocumentReadError(f"input path is not a regular file: {path}")


def read_document_text(
    source: str | Path,
    *,
    stdin: TextIO | None = None,
) -> str:
    """Read ``.txt``/``.md`` or extract text from a ``.pdf``.

    ``source='-'`` reads plain text from standard input.  PDF page boundaries
    are joined with form feeds so :func:`paper_profile.normalize_extracted_text`
    can turn them into explicit paragraph boundaries.
    """

    if str(source) == "-":
        return (stdin or sys.stdin).read()
    path = _path(source)
    _require_readable_file(path)
    suffix = path.suffix.casefold()
    if suffix in TEXT_SUFFIXES:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentReadError(
                f"paper text must be UTF-8: {path} (decode failed at byte {exc.start})"
            ) from exc
    if suffix == PDF_SUFFIX:
        return _read_pdf_text(path)
    supported = ", ".join(sorted((*TEXT_SUFFIXES, PDF_SUFFIX)))
    raise DocumentReadError(
        f"unsupported paper input suffix {suffix or '<none>'!r}: {path}; "
        f"supported suffixes are {supported}"
    )


def _read_pdf_text(path: Path) -> str:
    try:
        pypdf = importlib.import_module("pypdf")
    except ImportError as exc:
        raise DocumentReadError(
            "PDF input requires the optional 'pypdf' dependency; install "
            "'phylodigy[paper]' or convert the PDF to UTF-8 .txt first"
        ) from exc
    try:
        reader = pypdf.PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pypdf exposes several version-specific errors
        raise DocumentReadError(
            f"could not extract text from PDF {path}: {type(exc).__name__}: {exc}"
        ) from exc
    text = "\f".join(pages)
    if not text.strip():
        raise DocumentReadError(
            f"PDF contains no extractable text: {path}; run OCR or provide .txt"
        )
    return text


def read_json_data(
    source: str | Path,
    *,
    stdin: TextIO | None = None,
) -> Any:
    """Read a UTF-8 JSON value from a path or ``-`` for standard input."""

    if str(source) == "-":
        payload = (stdin or sys.stdin).read()
        label = "standard input"
    else:
        path = _path(source)
        _require_readable_file(path)
        try:
            payload = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise PhylodigyIOError(
                f"JSON input must be UTF-8: {path} (decode failed at byte {exc.start})"
            ) from exc
        label = str(path)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PhylodigyIOError(
            f"invalid JSON in {label} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc


def read_json_object(
    source: str | Path,
    *,
    stdin: TextIO | None = None,
) -> dict[str, Any]:
    """Read JSON and require a top-level object."""

    value = read_json_data(source, stdin=stdin)
    if not isinstance(value, dict):
        raise PhylodigyIOError(
            f"expected a JSON object in {source}, got {type(value).__name__}"
        )
    return value


def read_profile(
    source: str | Path,
    *,
    stdin: TextIO | None = None,
) -> ArtifactProfile:
    """Read, schema-check, and digest-check an artifact profile."""

    raw = read_json_object(source, stdin=stdin)
    try:
        return ArtifactProfile.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileReadError(f"invalid artifact profile in {source}: {exc}") from exc


def write_json_data(
    value: Any,
    destination: str | Path | None = None,
    *,
    stdout: TextIO | None = None,
) -> None:
    """Write canonical, indented JSON plus one newline.

    ``destination`` values of ``None`` and ``'-'`` write to standard output.
    Parent directories are created for file destinations.
    """

    if destination is None or str(destination) == "-":
        payload = canonical_json(value, indent=2) + "\n"
        (stdout or sys.stdout).write(payload)
        return
    path = _path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_json(value)
    encoder = json.JSONEncoder(
        ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            for chunk in encoder.iterencode(normalized):
                handle.write(chunk)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def write_profile(
    profile: ArtifactProfile,
    destination: str | Path | None = None,
    *,
    stdout: TextIO | None = None,
) -> None:
    """Write a canonical profile including its verified content digest."""

    if not isinstance(profile, ArtifactProfile):
        raise TypeError("profile must be an ArtifactProfile")
    write_json_data(profile.to_dict(), destination, stdout=stdout)


def validate_profile_data(raw: Mapping[str, Any]) -> ArtifactProfile:
    """Validate an in-memory mapping and return its canonical schema object."""

    try:
        return ArtifactProfile.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileReadError(f"invalid artifact profile: {exc}") from exc

__all__ = [
    "DocumentReadError",
    "ProfileReadError",
    "PDF_SUFFIX",
    "PhylodigyIOError",
    "TEXT_SUFFIXES",
    "read_document_text",
    "read_profile",
    "read_json_data",
    "read_json_object",
    "validate_profile_data",
    "write_profile",
    "write_json_data",
]
