from dataclasses import replace

import pytest

from phylodigy.computation_graph import build_graph_profile
from phylodigy.lineage import infer_lineage_network
from phylodigy.modelome_binding import bind_modelome_profiles
from phylodigy.schema import ArchitecturalGenome


@pytest.fixture
def profiles():
    profiles = {}
    for index, operation in enumerate(("relu", "tanh", "sigmoid")):
        records = [
            {"name": "input", "op": "placeholder", "target": "input", "inputs": []},
            {
                "name": "activation",
                "op": "call_function",
                "target": operation,
                "inputs": ["input"],
            },
            {"name": "output", "op": "output", "target": "output", "inputs": ["activation"]},
        ]
        artifact_id = f"profile:{index}"
        profiles[artifact_id] = ArchitecturalGenome(
            artifact_id, build_graph_profile(records, radii=(0,))
        )
    return profiles


def test_explicit_assignments_require_known_unique_entries_and_profiles(profiles):
    profile_ids = list(profiles)[:2]
    entries = [{"id": "catalog:one"}, {"id": "catalog:two"}]

    with pytest.raises(ValueError, match="unknown entry ID"):
        bind_modelome_profiles(
            entries, profiles, bindings={"catalog:missing": profile_ids[0]}
        )
    with pytest.raises(ValueError, match="unknown profile ID"):
        bind_modelome_profiles(
            entries, profiles, bindings={"catalog:one": "profile:missing"}
        )
    with pytest.raises(ValueError, match="multiple entries"):
        bind_modelome_profiles(
            entries,
            profiles,
            bindings={"catalog:one": profile_ids[0], "catalog:two": profile_ids[0]},
        )


def test_coverage_accounts_for_every_entry_and_profile(profiles):
    exact_id, explicit_profile_id = list(profiles)[:2]
    entries = [
        {"id": "catalog:explicit"},
        {"id": exact_id},
        {"id": "catalog:missing"},
    ]

    result = bind_modelome_profiles(
        entries, profiles, bindings={"catalog:explicit": explicit_profile_id}
    )

    assert [row["entry_id"] for row in result["coverage"]] == [
        entry["id"] for entry in entries
    ]
    assert [row["status"] for row in result["coverage"]] == [
        "bound",
        "bound",
        "missing_profile",
    ]
    assert {row["profile_id"] for row in result["coverage"] if row["profile_id"]} == {
        exact_id,
        explicit_profile_id,
    }
    assert result["unused_profile_ids"] == sorted(set(profiles) - {exact_id, explicit_profile_id})


def test_binding_remaps_only_artifact_id_for_tree_taxa(profiles):
    profile_id = next(iter(profiles))
    original = profiles[profile_id]
    entry_id = "catalog:remapped-model"

    result = bind_modelome_profiles(
        [{"id": entry_id}], profiles, bindings={entry_id: profile_id}
    )
    bound = result["genomes"][0]

    assert original.artifact_id == profile_id
    assert bound.artifact_id == entry_id
    assert bound.structural_digest == original.structural_digest
    assert bound.graph_profile == original.graph_profile


def test_catalog_metadata_does_not_change_graph_tree(profiles):
    entry_ids = list(profiles)[:3]
    plain_entries = [{"id": entry_id} for entry_id in entry_ids]
    described_entries = [
        {
            "id": entry_id,
            "name": f"Different descriptive name {index}",
            "url": f"https://example.test/{index}",
            "release_date": f"203{index}-01-01",
            "metadata": {"ranking": index, "citation": "unrelated"},
        }
        for index, entry_id in enumerate(entry_ids)
    ]

    plain = bind_modelome_profiles(plain_entries, profiles)
    described = bind_modelome_profiles(described_entries, profiles)

    assert plain["genomes"] == described["genomes"]
    assert infer_lineage_network(plain["genomes"])["tree"] == infer_lineage_network(
        described["genomes"]
    )["tree"]


def test_metadata_on_bound_profile_does_not_change_tree_graph(profiles):
    selected = list(profiles.values())[:3]
    metadata_changed = [
        replace(
            genome,
            name=f"Renamed {index}",
            release_date=f"203{index}-01-01",
            metadata={"catalog_note": f"note {index}"},
        )
        for index, genome in enumerate(selected)
    ]

    assert infer_lineage_network(selected)["tree"] == infer_lineage_network(
        metadata_changed
    )["tree"]
