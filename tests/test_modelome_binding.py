import pytest

from phylodigy.modelome_binding import bind_modelome_profiles
from phylodigy.computation_graph import build_graph_profile
from phylodigy.schema import ArchitecturalGenome


@pytest.fixture
def profiles():
    graph = build_graph_profile(
        [
            {"name": "input", "op": "placeholder", "target": "input", "inputs": []},
            {"name": "output", "op": "output", "target": "output", "inputs": ["input"]},
        ],
        radii=(0,),
    )
    first = ArchitecturalGenome("profile:first", graph, name="Shared name")
    second = ArchitecturalGenome("profile:second", graph, name="Another name")
    return {first.artifact_id: first, second.artifact_id: second}


def test_binds_exact_ids_and_relabels_genomes_for_tree(profiles):
    profile_id = next(iter(profiles))
    result = bind_modelome_profiles(
        [{"id": profile_id}, {"id": "no-profile"}], profiles
    )

    assert [genome.artifact_id for genome in result["genomes"]] == [profile_id]
    assert result["coverage"] == [
        {
            "entry_id": profile_id,
            "status": "bound",
            "profile_id": profile_id,
            "reason": "exact artifact ID match",
        },
        {
            "entry_id": "no-profile",
            "status": "missing_profile",
            "profile_id": None,
            "reason": "no exact artifact ID match or explicit binding",
        },
    ]
    assert result["unused_profile_ids"] == sorted(set(profiles) - {profile_id})
    assert profiles[profile_id].artifact_id == profile_id


def test_explicit_binding_is_only_nonexact_identity_path(profiles):
    profile_id = next(iter(profiles))
    result = bind_modelome_profiles(
        [{"id": "catalog:123", "name": profiles[profile_id].name}],
        profiles,
        bindings={"catalog:123": profile_id},
    )

    assert result["genomes"][0].artifact_id == "catalog:123"
    assert result["coverage"][0]["profile_id"] == profile_id
    assert result["coverage"][0]["reason"] == "explicit binding"


def test_names_and_urls_never_create_a_binding(profiles):
    profile = next(iter(profiles.values()))
    result = bind_modelome_profiles(
        [{"id": "different-id", "name": profile.name, "url": profile.artifact_id}],
        profiles,
    )

    assert result["genomes"] == []
    assert result["coverage"][0]["status"] == "missing_profile"


@pytest.mark.parametrize(
    "bindings",
    [
        {"unknown-entry": "profile"},
        {"entry": "unknown-profile"},
        {"entry": None},
    ],
)
def test_invalid_or_unknown_explicit_assignments_raise(profiles, bindings):
    entry_id = next(iter(bindings))
    if entry_id == "unknown-entry":
        entries = [{"id": "entry"}]
    else:
        entries = [{"id": "entry"}]
        bindings = {"entry": next(iter(bindings.values()))}

    with pytest.raises(ValueError):
        bind_modelome_profiles(entries, profiles, bindings=bindings)


def test_duplicate_explicit_profile_assignment_raises(profiles):
    profile_id = next(iter(profiles))
    with pytest.raises(ValueError, match="multiple entries"):
        bind_modelome_profiles(
            [{"id": "one"}, {"id": "two"}],
            profiles,
            bindings={"one": profile_id, "two": profile_id},
        )


def test_duplicate_entry_id_raises(profiles):
    with pytest.raises(ValueError, match="unique"):
        bind_modelome_profiles(
            [{"id": "same"}, {"id": "same"}], profiles
        )
