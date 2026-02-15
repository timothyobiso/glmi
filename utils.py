"""Shared utilities for GL-Qualia dataset generation."""

import json
import time
from pathlib import Path

import backoff
import requests

import config


# ── Rate-limited API fetching ─────────────────────────────────────────────────

_last_request_time = 0.0


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
def fetch_conceptnet(word: str) -> dict:
    """Fetch ConceptNet edges for a word, with rate limiting and caching."""
    cache_path = config.CONCEPTNET_CACHE / f"{word}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < config.CONCEPTNET_RATE_LIMIT:
        time.sleep(config.CONCEPTNET_RATE_LIMIT - elapsed)

    url = f"{config.CONCEPTNET_API}/c/en/{word}?limit={config.CONCEPTNET_LIMIT}"
    resp = requests.get(url, timeout=30)
    _last_request_time = time.time()
    resp.raise_for_status()
    data = resp.json()

    cache_path.write_text(json.dumps(data))
    return data


def extract_filler(edge: dict, source_word: str) -> str | None:
    """Extract the filler text from a ConceptNet edge (the 'other' end)."""
    start = edge.get("start", {})
    end = edge.get("end", {})
    start_label = start.get("label", "").lower()
    end_label = end.get("label", "").lower()
    source_lower = source_word.lower()

    # We want the end that is NOT the source word
    if start_label == source_lower:
        filler = end_label
    elif end_label == source_lower:
        filler = start_label
    else:
        # Use surfaceText as fallback
        surface = edge.get("surfaceText", "")
        if surface:
            # Remove the source word and brackets from surface text
            filler = surface.replace("[[", "").replace("]]", "")
            # Try to extract the non-source part
            parts = filler.split(source_lower)
            filler = max(parts, key=len).strip().strip(".,;: ")
            if filler:
                return filler
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
