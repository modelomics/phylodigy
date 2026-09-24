"""Resumable, isolated graph extraction from pinned Modelome resources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import math
import multiprocessing
from pathlib import Path
import tempfile
from typing import Any

from .canonical import content_digest
from .io import read_json_object, read_profile, write_json_data, write_profile
from .lineage import LineageAnalysisConfig
from .modelome_input import read_modelome_entries
from .modelome_targets import resolve_modelome_targets
from .schema import ArchitecturalGenome


def plan_modelome_extraction(
    entries_path: str | Path,
    *,
    pins: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Resolve exact references offline and report every entry's readiness."""
    source = read_modelome_entries(entries_path)
    jobs = resolve_modelome_targets(source["entries"], pins=pins)
    result = {
        "artifact_type": "phylodigy.modelome_extraction_plan",
        "source_digest": source["source_digest"],
        "jobs": jobs,
        "counts": dict(sorted(Counter(job["status"] for job in jobs).items())),
    }
    result["digest"] = content_digest(result)
    return result


def _runtime() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "huggingface-hub"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _worker(job: dict[str, Any], policy: dict[str, Any], directory: str) -> None:
    """Write a terminal result inside a parent-owned temporary directory."""
    from .huggingface_profile import extract_huggingface_genome

    root = Path(directory)
    try:
        with (root / "worker.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                genome = extract_huggingface_genome(
                    **job["target"],
                    artifact_id=job["entry_id"],
                    max_graph_nodes=policy["max_graph_nodes"],
                    max_hidden_layers=policy["max_hidden_layers"],
                    radii=policy["radii"],
                )
        write_profile(genome, root / "genome.json")
        write_json_data({"status": "success"}, root / "result.json")
    except Exception as error:
        write_json_data(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            root / "result.json",
        )


def _run_extraction(job: dict[str, Any], policy: dict[str, Any]) -> ArchitecturalGenome:
    # Spawn avoids inheriting a partially initialized framework or parent locks.
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="phylodigy-trace-") as directory:
        process = context.Process(target=_worker, args=(job, policy, directory))
        process.start()
        try:
            process.join(policy["per_model_timeout"])
            if process.is_alive():
                raise TimeoutError(
                    f"extraction exceeded {policy['per_model_timeout']:g} seconds"
                )
            if process.exitcode != 0:
                raise RuntimeError(
                    f"extraction worker exited with status {process.exitcode}"
                )
            report = read_json_object(Path(directory) / "result.json")
            if report["status"] != "success":
                raise RuntimeError(report["error"])
            return read_profile(Path(directory) / "genome.json")
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def extract_modelome_profiles(
    entries_path: str | Path,
    output_dir: str | Path,
    *,
    pins: Mapping[str, Mapping[str, str]] | None = None,
    max_models: int = 10,
    max_graph_nodes: int = 12000,
    max_hidden_layers: int = 36,
    per_model_timeout: float = 45.0,
    radii: Sequence[int] = (0, 1, 2, 3),
    retry_failures: bool = False,
) -> dict[str, Any]:
    """Trace up to ``max_models`` uncached entries, retaining a resumable report.

    The source snapshot, selected references, runtime versions, and extraction
    policy select an isolated run directory. Reruns reuse validated successes
    and recorded failures, then continue pending work. Pass the returned
    ``profiles_dir`` to ``build_modelome_tree``. Changing inputs or policy
    cannot expose stale profiles as results of the new run.
    """
    for field, value in (
        ("max_models", max_models),
        ("max_graph_nodes", max_graph_nodes),
        ("max_hidden_layers", max_hidden_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if (
        isinstance(per_model_timeout, bool)
        or not isinstance(per_model_timeout, (int, float))
        or not math.isfinite(per_model_timeout)
        or per_model_timeout <= 0
    ):
        raise ValueError("per_model_timeout must be finite and positive")
    if not isinstance(retry_failures, bool):
        raise TypeError("retry_failures must be boolean")
    selected_radii = LineageAnalysisConfig(radii=tuple(radii)).radii
    plan = plan_modelome_extraction(entries_path, pins=pins)
    policy = {
        "extractor_version": "1",
        "max_graph_nodes": max_graph_nodes,
        "max_hidden_layers": max_hidden_layers,
        "per_model_timeout": float(per_model_timeout),
        "radii": list(selected_radii),
        "runtime_versions": _runtime(),
    }
    run_key = content_digest({"plan_digest": plan["digest"], "policy": policy})
    root = Path(output_dir).resolve() / run_key
    profiles = root / "profiles"
    failures = root / "failures"
    profiles.mkdir(parents=True, exist_ok=True)
    failures.mkdir(exist_ok=True)
    records = []
    attempted = 0
    for job in plan["jobs"]:
        record = dict(job)
        record["resumed"] = False
        if job["status"] != "ready":
            records.append(record)
            continue
        stem = content_digest(job["entry_id"])
        profile_path = profiles / f"{stem}.genome.json"
        failure_path = failures / f"{stem}.failure.json"
        cached = None
        if profile_path.exists():
            try:
                candidate = read_profile(profile_path)
                if (
                    candidate.artifact_id == job["entry_id"]
                    and candidate.metadata.get("modelome_extraction", {}).get("run_key")
                    == run_key
                ):
                    cached = candidate
            except (OSError, TypeError, ValueError):
                pass
            if cached is None:
                profile_path.unlink()
        previous = None
        if cached is None and failure_path.exists() and not retry_failures:
            try:
                failure = read_json_object(failure_path)
                if (
                    failure.get("run_key") == run_key
                    and failure.get("entry_id") == job["entry_id"]
                    and isinstance(failure.get("error"), str)
                    and failure["error"]
                ):
                    previous = failure
            except (OSError, TypeError, ValueError):
                pass
            if previous is None:
                failure_path.unlink()
        if cached is not None:
            record.update(
                status="succeeded",
                resumed=True,
                profile_path=str(profile_path),
                profile_digest=cached.digest,
            )
        elif previous is not None:
            record.update(status="failed", resumed=True, error=previous["error"])
        elif attempted >= max_models:
            record["status"] = "pending"
        else:
            attempted += 1
            try:
                genome = _run_extraction(job, policy)
                if (
                    not isinstance(genome, ArchitecturalGenome)
                    or genome.artifact_id != job["entry_id"]
                ):
                    raise ValueError(
                        "extraction returned a profile for the wrong entry"
                    )
                genome = replace(
                    genome,
                    metadata={
                        **genome.metadata,
                        "modelome_extraction": {
                            "run_key": run_key,
                            "source_digest": plan["source_digest"],
                            "target": job["target"],
                            "policy": policy,
                        },
                    },
                )
                write_profile(genome, profile_path)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                write_json_data(
                    {"run_key": run_key, "entry_id": job["entry_id"], "error": message},
                    failure_path,
                )
                record.update(status="failed", error=message)
            else:
                failure_path.unlink(missing_ok=True)
                record.update(
                    status="succeeded",
                    profile_path=str(profile_path),
                    profile_digest=genome.digest,
                )
        records.append(record)
    counts = dict(sorted(Counter(row["status"] for row in records).items()))
    result = {
        "artifact_type": "phylodigy.modelome_extraction_run",
        "run_key": run_key,
        "source_digest": plan["source_digest"],
        "policy": policy,
        "profiles_dir": str(profiles),
        "attempted": attempted,
        "counts": counts,
        "entries": records,
        "status": "pending" if counts.get("pending", 0) else "finished",
        "complete_profile_coverage": counts.get("succeeded", 0) == len(records),
    }
    result["digest"] = content_digest(result)
    write_json_data(result, root / "run.json")
    return result


__all__ = ["extract_modelome_profiles", "plan_modelome_extraction"]
