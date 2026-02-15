#!/usr/bin/env python3
"""Step 1: Extract qualia fillers from ConceptNet REST API.

For each seed noun, queries ConceptNet, maps edges to GL qualia roles,
and outputs structured records to conceptnet_extract.jsonl.

Runtime: ~25-40 min for 1500 nouns (rate limited to 1 req/sec).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

import config
import utils


def load_seed_nouns() -> list[str]:
    """Load seed nouns from Brysbaert concreteness ratings + WordNet filter."""
    from nltk.corpus import wordnet as wn

    csv_path = config.RAW / "brysbaert_concreteness.csv"
    if not csv_path.exists():
        print(f"ERROR: Download Brysbaert concreteness ratings to {csv_path}")
        print("Source: https://github.com/ArtsEngine/concreteness/raw/master/data/concrete.csv")
        print("Or use: wget -O data/raw/brysbaert_concreteness.csv 'https://github.com/ArtsEngine/concreteness/raw/master/data/concrete.csv'")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # The CSV has columns: Word, Conc.M, Conc.SD, Unknown, Total, etc.
    # Adapt column names if needed
    word_col = [c for c in df.columns if c.lower().startswith("word")][0]
    conc_col = [c for c in df.columns if "conc" in c.lower() and ("m" in c.lower() or "mean" in c.lower())][0]

    df = df[[word_col, conc_col]].dropna()
    df.columns = ["word", "concreteness"]
    df = df[df["concreteness"] >= config.CONCRETENESS_THRESHOLD]

    # Filter to WordNet nouns only
    nouns = []
    for word in df["word"]:
        word = str(word).strip().lower()
        if not word.isalpha():
            continue
        synsets = wn.synsets(word, pos=wn.NOUN)
        if synsets:
            nouns.append(word)
        if len(nouns) >= 1500:
            break

    # Save seed list
    seed_df = pd.DataFrame({"word": nouns})
    seed_df.to_csv(config.RAW / "concrete_nouns.csv", index=False)
    print(f"Saved {len(nouns)} seed nouns to {config.RAW / 'concrete_nouns.csv'}")
    return nouns


def extract_qualia_from_edges(word: str, data: dict) -> dict:
    """Extract qualia fillers from ConceptNet response edges."""
    qualia = {"telic": [], "agentive": [], "constitutive": [], "formal": []}

    for edge in data.get("edges", []):
        rel = edge.get("rel", {}).get("@id", "")
        weight = edge.get("weight", 0)

        if weight < config.CONCEPTNET_WEIGHT_THRESHOLD:
            continue

        role = config.RELATION_TO_QUALIA.get(rel)
        if not role:
            continue

        filler = utils.extract_filler(edge, word)
        if filler and utils.is_valid_filler(filler):
            qualia[role].append({
                "filler": filler,
                "relation": rel,
                "weight": weight,
                "surface": edge.get("surfaceText", ""),
                "source": "conceptnet",
            })

    # Agentive fallback: use MadeOf edges if no agentive fillers found
    if not qualia["agentive"]:
        for edge in data.get("edges", []):
            rel = edge.get("rel", {}).get("@id", "")
            if rel in config.AGENTIVE_FALLBACK_RELATIONS:
                weight = edge.get("weight", 0)
                if weight < config.CONCEPTNET_WEIGHT_THRESHOLD:
                    continue
                filler = utils.extract_filler(edge, word)
                if filler and utils.is_valid_filler(filler):
                    qualia["agentive"].append({
                        "filler": filler,
                        "relation": rel,
                        "weight": weight,
                        "surface": edge.get("surfaceText", ""),
                        "source": "conceptnet_fallback",
                    })

    return qualia


def main():
    config.CONCEPTNET_CACHE.mkdir(parents=True, exist_ok=True)

    # Load or generate seed nouns
    seed_path = config.RAW / "concrete_nouns.csv"
    if seed_path.exists():
        nouns = pd.read_csv(seed_path)["word"].tolist()
        print(f"Loaded {len(nouns)} seed nouns from {seed_path}")
    else:
        nouns = load_seed_nouns()

    records = []
    errors = []

    for word in tqdm(nouns, desc="Querying ConceptNet"):
        try:
            data = utils.fetch_conceptnet(word)
            qualia = extract_qualia_from_edges(word, data)
            records.append({
                "concept": word,
                "qualia": qualia,
                "edge_count": len(data.get("edges", [])),
            })
        except Exception as e:
            errors.append({"word": word, "error": str(e)})
            tqdm.write(f"  Error for '{word}': {e}")

    out_path = config.RAW / "conceptnet_extract.jsonl"
    utils.write_jsonl(out_path, records)
    print(f"\nSaved {len(records)} records to {out_path}")
    if errors:
        print(f"Errors: {len(errors)}")
        utils.write_jsonl(config.RAW / "conceptnet_errors.jsonl", errors)

    # Stats
    has_all_4 = sum(1 for r in records if all(r["qualia"][role] for role in ["telic", "agentive", "constitutive", "formal"]))
    for role in ["telic", "agentive", "constitutive", "formal"]:
        count = sum(1 for r in records if r["qualia"][role])
        print(f"  {role}: {count}/{len(records)} concepts have fillers")
    print(f"  All 4 roles: {has_all_4}/{len(records)}")


if __name__ == "__main__":
    main()
