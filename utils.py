"""Shared utilities for GL-Qualia dataset generation."""

import csv
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import config


# ── ConceptNet local index ────────────────────────────────────────────────────

def build_conceptnet_index() -> dict[str, list[dict]]:
    """Build an in-memory index of English ConceptNet edges keyed by word.

    Reads the full assertions CSV (gzipped), filters to English-only edges
    with relevant relations, and returns {word: [edge_dicts]}.
    Caches the result as a pickle for fast reloading.
    """
    if config.CONCEPTNET_EN_INDEX.exists():
        print(f"Loading cached ConceptNet index from {config.CONCEPTNET_EN_INDEX}...")
        with open(config.CONCEPTNET_EN_INDEX, "rb") as f:
            return pickle.load(f)

    print(f"Building ConceptNet English index from {config.CONCEPTNET_CSV}...")
    print("  (This takes a few minutes on first run, then is cached)")

    relevant_relations = set(config.RELATION_TO_QUALIA.keys()) | config.AGENTIVE_FALLBACK_RELATIONS
    index = defaultdict(list)

    with gzip.open(config.CONCEPTNET_CSV, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="Reading ConceptNet CSV", unit=" edges"):
            if len(row) < 5:
                continue
            # Columns: URI, relation, start, end, json_data
            rel, start, end = row[1], row[2], row[3]

            if rel not in relevant_relations:
                continue

            # Filter to English
            if not (start.startswith("/c/en/") and end.startswith("/c/en/")):
                continue

            # Parse JSON metadata for weight
            try:
                meta = json.loads(row[4])
            except (json.JSONDecodeError, IndexError):
                meta = {}

            weight = meta.get("weight", 1.0)
            surface = meta.get("surfaceText", "")

            # Extract word from URI: /c/en/word/... → word
            start_word = start.split("/")[3]
            end_word = end.split("/")[3]

            edge = {
                "rel": rel,
                "start": start_word,
                "end": end_word,
                "weight": weight,
                "surface": surface,
            }

            index[start_word].append(edge)
            if end_word != start_word:
                index[end_word].append(edge)

    index = dict(index)
    print(f"  Indexed {sum(len(v) for v in index.values())} edges for {len(index)} words")

    # Cache
    config.CONCEPTNET_EN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CONCEPTNET_EN_INDEX, "wb") as f:
        pickle.dump(index, f)
    print(f"  Cached index to {config.CONCEPTNET_EN_INDEX}")

    return index


def get_conceptnet_edges(word: str, index: dict[str, list[dict]]) -> list[dict]:
    """Look up edges for a word from the local index."""
    return index.get(word.lower().replace(" ", "_"), [])


def extract_filler_from_edge(edge: dict, source_word: str) -> str | None:
    """Extract the filler text from a ConceptNet edge (the 'other' end)."""
    start = edge["start"].replace("_", " ")
    end = edge["end"].replace("_", " ")
    source_lower = source_word.lower()

    if start == source_lower:
        filler = end
    elif end == source_lower:
        filler = start
    else:
        # Source word is a substring — use surface text
        surface = edge.get("surface", "")
        if surface:
            filler = surface.replace("[[", "").replace("]]", "")
            parts = filler.split(source_lower)
            filler = max(parts, key=len).strip().strip(".,;: ")
            return filler if filler else None
        return None

    return filler if filler else None


def is_valid_filler(filler: str) -> bool:
    """Check if a filler is valid (not vague, meets length requirement)."""
    if not filler:
        return False
    filler_lower = filler.lower().strip()
    if len(filler_lower) < config.MIN_FILLER_LENGTH:
        return False
    if filler_lower in config.VAGUE_FILLERS:
        return False
    return True


# ── JSONL I/O ──────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]):
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> dict | list:
    """Load a JSON file."""
    return json.loads(path.read_text())


def save_json(path: Path, data: dict | list):
    """Save data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
