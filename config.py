"""Configuration for GL-Qualia Nonce Word Dataset Generation."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CONCEPTNET_CACHE = RAW / "conceptnet"
ONTOLOGY = DATA / "ontology"
NONCE = DATA / "nonce_words"
STIMULI = DATA / "stimuli"
CONTROLS = DATA / "controls"

# ── ConceptNet ─────────────────────────────────────────────────────────────────
CONCEPTNET_API = "http://api.conceptnet.io"
CONCEPTNET_LIMIT = 1000
CONCEPTNET_RATE_LIMIT = 1.0  # seconds between requests

# ConceptNet relation → GL qualia role mapping
RELATION_TO_QUALIA = {
    # Telic (purpose / function)
    "/r/UsedFor": "telic",
    "/r/CapableOf": "telic",
    "/r/ReceivesAction": "telic",
    # Agentive (origin / how it comes into being)
    "/r/CreatedBy": "agentive",
    "/r/Causes": "agentive",
    "/r/MannerOf": "agentive",
    # Constitutive (internal structure / composition)
    "/r/MadeOf": "constitutive",
    "/r/HasA": "constitutive",
    "/r/PartOf": "constitutive",
    "/r/HasProperty": "constitutive",
    # Formal (taxonomic / categorical identity)
    "/r/IsA": "formal",
    "/r/SimilarTo": "formal",
    "/r/InstanceOf": "formal",
    "/r/DefinedAs": "formal",
}

# Relations that can serve as agentive fallback
AGENTIVE_FALLBACK_RELATIONS = {"/r/MadeOf"}

# ── FrameNet frame → qualia mapping ───────────────────────────────────────────
FRAME_TO_QUALIA = {
    # Telic
    "Using": "telic",
    "Purpose": "telic",
    "Usefulness": "telic",
    "Tool_purpose": "telic",
    "Manipulate_into_doing": "telic",
    # Agentive
    "Creating": "agentive",
    "Manufacturing": "agentive",
    "Building": "agentive",
    "Cooking_creation": "agentive",
    "Intentionally_create": "agentive",
    # Constitutive
    "Part_whole": "constitutive",
    "Inclusion": "constitutive",
    "Substance": "constitutive",
    "Ingredients": "constitutive",
    # Formal
    "Categorization": "formal",
    "Type": "formal",
    "Similarity": "formal",
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
CONCRETENESS_THRESHOLD = 4.0
MIN_FILLER_LENGTH = 3
CONCEPTNET_WEIGHT_THRESHOLD = 1.0
TARGET_CONCEPTS = 500
TARGET_NONCE_WORDS = 100
MIN_NONCE_TOKENS = 2
MAX_NONCE_TOKENS = 4
MIN_SENTENCE_WORDS = 8
MAX_SENTENCE_WORDS = 25
JACCARD_DEDUP_THRESHOLD = 0.8

# ── Vague fillers to remove ───────────────────────────────────────────────────
VAGUE_FILLERS = {
    "thing", "things", "object", "objects", "something", "anything",
    "stuff", "item", "items", "one", "ones", "it", "them", "this",
    "that", "everything", "nothing", "entity", "entities",
}

# ── Claude API ─────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
CLAUDE_MAX_CONCURRENT = 15
CLAUDE_BATCH_SIZE = 1000  # requests per batch

# ── Llama tokenizer ───────────────────────────────────────────────────────────
LLAMA_MODEL = "meta-llama/Llama-3.1-8B"
