"""Load local architectural genome profiles for modelome analyses."""

from __future__ import annotations

from pathlib import Path

from .io import read_profile
from .schema import ArchitecturalGenome


def load_modelome_profiles(path: str | Path) -> dict[str, ArchitecturalGenome]:
    """Load model genomes from one JSON profile or a directory tree.

    Directory entries are read recursively in path order, so failures and
    duplicate identity checks are deterministic. Known popularity extraction
    records ending in ``.failure.json`` are ignored. An empty directory returns
    an empty mapping. The result is keyed by the immutable artifact ID.
    """

    source = Path(path)
    if source.is_dir():
        files = sorted(
            candidate
            for candidate in source.rglob("*.json")
            if not candidate.name.endswith(".failure.json")
        )
    elif source.is_file():
        files = [source]
    elif not source.exists():
        raise FileNotFoundError(f"modelome profile path does not exist: {source}")
    else:
        raise ValueError(f"modelome profile path is not a file or directory: {source}")

    profiles: dict[str, ArchitecturalGenome] = {}
    first_source: dict[str, Path] = {}
    for profile_path in files:
        profile = read_profile(profile_path)
        if not isinstance(profile, ArchitecturalGenome):
            raise ValueError(
                f"modelome input must contain architectural genome profiles; "
                f"{profile_path} contains {type(profile).__name__}"
            )
        if profile.artifact_kind != "model":
            raise ValueError(
                f"modelome input requires artifact kind 'model'; "
                f"{profile_path} has kind {profile.artifact_kind!r}"
            )
        if profile.artifact_id in profiles:
            raise ValueError(
                f"duplicate artifact ID {profile.artifact_id!r} in modelome inputs: "
                f"{first_source[profile.artifact_id]} and {profile_path}"
            )
        profiles[profile.artifact_id] = profile
        first_source[profile.artifact_id] = profile_path
    return profiles


__all__ = ["load_modelome_profiles"]
