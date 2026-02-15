#!/usr/bin/env python3
"""Step 1: Extract qualia fillers from ConceptNet (local dump).

Loads the ConceptNet assertions CSV (downloaded and gzipped), builds an
in-memory index, then looks up each seed noun to map edges to GL qualia roles.

Runtime: ~3-5 min (index build on first run, cached thereafter).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

import config
import utils


def load_seed_nouns() -> list[str]:
    """Load seed nouns from Brysbaert concreteness ratings + WordNet filter."""
    from nltk.corpus import wordnet as wn

    csv_path = config.RAW / "brysbaert_concreteness.tsv"
    if not csv_path.exists():
        # Also check old .csv name for backwards compat
        csv_path = config.RAW / "brysbaert_concreteness.csv"
    if not csv_path.exists():
        print(f"ERROR: Download Brysbaert concreteness ratings to {config.RAW / 'brysbaert_concreteness.tsv'}")
        print("Source: https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt")
        sys.exit(1)

    # File is tab-delimited with columns: Word, Bigram, Conc.M, Conc.SD, ...
    df = pd.read_csv(csv_path, sep="\t")
    word_col = [c for c in df.columns if c.lower().startswith("word")][0]
    conc_col = [c for c in df.columns if "conc" in c.lower() and ("m" in c.lower() or "mean" in c.lower())][0]

    df = df[[word_col, conc_col]].dropna()
    df.columns = ["word", "concreteness"]
    df = df[df["concreteness"] >= config.CONCRETENESS_THRESHOLD]

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

    seed_df = pd.DataFrame({"word": nouns})
    seed_df.to_csv(config.RAW / "concrete_nouns.csv", index=False)
    print(f"Saved {len(nouns)} seed nouns to {config.RAW / 'concrete_nouns.csv'}")
    return nouns


def extract_qualia_from_edges(word: str, edges: list[dict]) -> dict:
    """Extract qualia fillers from local ConceptNet edges."""
    qualia = {"telic": [], "agentive": [], "constitutive": [], "formal": []}

    for edge in edges:
        rel = edge["rel"]
        weight = edge.get("weight", 1.0)

        if weight < config.CONCEPTNET_WEIGHT_THRESHOLD:
            continue

        role = config.RELATION_TO_QUALIA.get(rel)
        if not role:
            continue

        filler = utils.extract_filler_from_edge(edge, word)
        if filler and utils.is_valid_filler(filler):
            qualia[role].append({
                "filler": filler,
                "relation": rel,
                "weight": weight,
                "surface": edge.get("surface", ""),
                "source": "conceptnet",
            })

    # Agentive fallback: use MadeOf edges if no agentive fillers found
    if not qualia["agentive"]:
        for edge in edges:
            rel = edge["rel"]
            if rel in config.AGENTIVE_FALLBACK_RELATIONS:
                weight = edge.get("weight", 1.0)
                if weight < config.CONCEPTNET_WEIGHT_THRESHOLD:
                    continue
                filler = utils.extract_filler_from_edge(edge, word)
                if filler and utils.is_valid_filler(filler):
                    qualia["agentive"].append({
                        "filler": filler,
                        "relation": rel,
                        "weight": weight,
                        "surface": edge.get("surface", ""),
                        "source": "conceptnet_fallback",
                    })

    return qualia


def main():
    # Check that ConceptNet dump exists
    if not config.CONCEPTNET_CSV.exists():
        print(f"ERROR: ConceptNet dump not found at {config.CONCEPTNET_CSV}")
        print("Download it with:")
        print(f"  wget -O {config.CONCEPTNET_CSV} https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        sys.exit(1)

    # Build or load index
    cn_index = utils.build_conceptnet_index()

    # Load or generate seed nouns
    seed_path = config.RAW / "concrete_nouns.csv"
    if seed_path.exists():
        nouns = pd.read_csv(seed_path)["word"].tolist()
        print(f"Loaded {len(nouns)} seed nouns from {seed_path}")
    else:
        nouns = load_seed_nouns()

    records = []
    for word in tqdm(nouns, desc="Looking up ConceptNet"):
        edges = utils.get_conceptnet_edges(word, cn_index)
        qualia = extract_qualia_from_edges(word, edges)
        records.append({
            "concept": word,
            "qualia": qualia,
            "edge_count": len(edges),
        })

    out_path = config.RAW / "conceptnet_extract.jsonl"
    utils.write_jsonl(out_path, records)
    print(f"\nSaved {len(records)} records to {out_path}")

    # Stats
    has_all_4 = sum(1 for r in records if all(r["qualia"][role] for role in ["telic", "agentive", "constitutive", "formal"]))
    for role in ["telic", "agentive", "constitutive", "formal"]:
        count = sum(1 for r in records if r["qualia"][role])
        print(f"  {role}: {count}/{len(records)} concepts have fillers")
    print(f"  All 4 roles: {has_all_4}/{len(records)}")


if __name__ == "__main__":
    main()
