"""Graph-derived architectural genomes and phylogenetic analysis."""

from .citations import (
    BibliographyEntry,
    CitationMention,
    CitationTarget,
    citation_marker_keys,
    extract_bibliography,
    extract_citation_evidence,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    resolve_bibliography,
)

from .code_profile import (
    SourceManifest,
    SourceScanPolicy,
    StaticSourceGraph,
    extract_code_profile,
    extract_source_profile,
    source_tree_digest,
)

from .computation_graph import (
    GRAPH_PROFILE_VERSION,
    GRAPH_SCHEMA_VERSION,
    ComputationGraph,
    FingerprintScale,
    GraphCharacter,
    GraphCharacterOccurrence,
    GraphDistance,
    GraphEdge,
    GraphNode,
    GraphProfile,
    GraphRegion,
    StructuralFingerprint,
    build_graph_profile,
    compare_graphs,
    computation_graph_from_fx,
    computation_graph_from_records,
    discover_graph_characters,
    discover_graph_regions,
    normalize_operator,
    structural_fingerprint,
)
from .columnar import (
    COLUMNAR_SCHEMA_VERSION,
    ColumnarFormatError,
    ColumnarUnavailableError,
    canonical_cbor,
    decode_cbor,
    read_graph_tables,
    read_lineage_tables,
    semantic_digest,
    write_graph_tables,
    write_lineage_tables,
)
from .huggingface_dataset import export_huggingface_dataset
from .graph_alignment import (
    GRAPH_ALIGNMENT_ALGORITHM,
    GRAPH_ALIGNMENT_SCHEMA_VERSION,
    AddEdgeOperation,
    AddNodeOperation,
    AlignmentParameters,
    AlignmentResourceLimitError,
    EdgeMatch,
    GraphAlignment,
    GraphAlignmentOperation,
    NodeMatch,
    RemoveEdgeOperation,
    RemoveNodeOperation,
    SolverMappingUnderidentification,
    StructuralEdge,
    StructuralNode,
    align_graphs,
    apply_graph_alignment,
    validate_graph_alignment,
)
from .contact_network import (
    ContactEvidence,
    ContactInferenceConfig,
    build_contact_network,
    infer_contact_network,
)
from .io import (
    DocumentReadError,
    PhylodigyIOError,
    ProfileReadError,
    read_document_text,
    read_profile,
    write_profile,
)
from .lineage import (
    LINEAGE_ANALYSIS_VERSION,
    LineageAnalysisConfig,
    infer_lineage_network,
)
from .model_profile import (
    GraphExtractionError,
    TorchUnavailableError,
    extract_architectural_genome,
    extract_model_profile,
)
from .model_catalog import (
    CATALOG_VERSION,
    CatalogPage,
    CatalogProvider,
    CatalogQuery,
    DateWindow,
    GitHubCatalogProvider,
    HuggingFaceCatalogProvider,
    ImportanceFilter,
    ModelCatalogError,
    OpenAlexPaperProvider,
    ProviderCapabilities,
    RecordFeedCatalogProvider,
    bind_primary_papers,
    discover_catalog,
    normalize_feed_record,
    select_catalog,
    validate_catalog_snapshot,
    validate_paper_grounded_catalog,
)
from .curated_catalog import (
    EPOCH_MODELS_CSV_URL,
    OPENROUTER_MODELS_ENDPOINT,
    EpochCatalogProvider,
    OpenRouterCatalogProvider,
)
from .paper_profile import (
    GraphAnnotation,
    GraphAnnotationTarget,
    PaperDocument,
    PaperGraphAnnotations,
    UnverifiedPaperClaim,
    annotate_graph_from_paper,
    extract_paper_profile,
    normalize_extracted_text,
)
from .profile_comparison import (
    ProfileComparison,
    build_vertical_backbone,
    compare_profiles,
    diagnose_tree_likeness,
    distance_matrix,
    pairwise_profile_comparisons,
    selected_character_counts,
)
from .popularity_corpus import (
    POPULARITY_CORPUS_VERSION,
    PopularityCorpusError,
    extract_popularity_tree,
    fetch_citation_evidence,
    fetch_popularity_manifest,
    validate_popularity_manifest,
)
from .schema import (
    ARCHITECTURAL_GENOME_ARTIFACT_TYPE,
    ArtifactProfile,
    ArchitecturalGenome,
    PROFILE_ARTIFACT_TYPE,
    merge_profiles,
)
from .toy_tree import (
    TOY_FRONTEND,
    build_toy_models,
    build_toy_genomes,
    build_toy_phylodigital_tree,
    toy_tree_summary,
)

__version__ = "0.2.0"

from .modelome import build_modelome_tree, plan_modelome_tree
from .modelome_input import ModelomeInputError, read_modelome_entries
from .modelome_profiles import load_modelome_profiles
from .modelome_binding import bind_modelome_profiles
from .modelome_relations import build_modelome_relations
from .newick import lineage_to_newick
from .huggingface_profile import HuggingFaceProfileError, extract_huggingface_genome
from .modelome_extraction import extract_modelome_profiles, plan_modelome_extraction
from .modelome_targets import resolve_modelome_targets
from .compact_lineage import infer_compact_lineage, lineage_distance_lookup

__all__ = [name for name in globals() if not name.startswith("_")]
