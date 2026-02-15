#!/usr/bin/env python3
"""Step 2: Extract qualia fillers from WordNet.

Uses hypernyms for Formal, meronyms for Constitutive,
and gloss parsing for Telic and Agentive.

Runtime: <1 min (all local).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm

import config
import utils

# Regex patterns for gloss parsing
TELIC_PATTERNS = [
    r"used (?:for|to|in) (.+?)(?:;|$)",
    r"for (\w+ing\b.+?)(?:;|$)",
    r"a (?:tool|device|instrument|utensil|implement|machine|apparatus) (?:for|used for|to) (.+?)(?:;|$)",
    r"designed (?:for|to) (.+?)(?:;|$)",
    r"that (?:is used|serves) (?:for|to) (.+?)(?:;|$)",
    r"to (\w+ .+?)(?:;|$)",
]

AGENTIVE_PATTERNS = [
    r"made (?:by|from|of|with) (.+?)(?:;|$)",
    r"produced (?:by|from) (.+?)(?:;|$)",
    r"created (?:by|from) (.+?)(?:;|$)",
    r"formed (?:by|from) (.+?)(?:;|$)",
    r"built (?:by|from) (.+?)(?:;|$)",
    r"manufactured (?:by|from) (.+?)(?:;|$)",
    r"derived from (.+?)(?:;|$)",
    r"baked|cooked|brewed|woven|carved|forged|molded",
]


def extract_from_gloss(gloss: str, patterns: list[str]) -> list[str]:
    """Extract fillers from a WordNet gloss using regex patterns."""
    fillers = []
    gloss_lower = gloss.lower()
    for pattern in patterns:
        for match in re.finditer(pattern, gloss_lower):
            if match.groups():
                filler = match.group(1).strip().rstrip(".,;: ")
                if utils.is_valid_filler(filler):
                    fillers.append(filler)
            else:
                # Pattern without groups (e.g. "baked") — use the match itself
                filler = match.group(0).strip()
                if utils.is_valid_filler(filler):
                    fillers.append(filler)
    return fillers


def extract_wordnet_qualia(word: str) -> dict:
    """Extract qualia from WordNet for a single word."""
    qualia = {"telic": [], "agentive": [], "constitutive": [], "formal": []}

    synsets = wn.synsets(word, pos=wn.NOUN)[:3]  # Top 3 synsets
    if not synsets:
        return qualia

    for ss in synsets:
        # Formal: hypernym chain (2-3 levels)
        hypernyms = ss.hypernyms()
        for h in hypernyms:
            qualia["formal"].append({
                "filler": h.lemma_names()[0].replace("_", " "),
                "relation": "hypernym",
                "synset": h.name(),
                "source": "wordnet",
            })
            # Second level
            for h2 in h.hypernyms()[:2]:
                qualia["formal"].append({
                    "filler": h2.lemma_names()[0].replace("_", " "),
                    "relation": "hypernym_L2",
                    "synset": h2.name(),
                    "source": "wordnet",
                })
                # Third level
                for h3 in h2.hypernyms()[:1]:
                    qualia["formal"].append({
                        "filler": h3.lemma_names()[0].replace("_", " "),
                        "relation": "hypernym_L3",
                        "synset": h3.name(),
                        "source": "wordnet",
                    })

        # Constitutive: meronyms
        for m in ss.part_meronyms():
            qualia["constitutive"].append({
                "filler": m.lemma_names()[0].replace("_", " "),
                "relation": "part_meronym",
                "synset": m.name(),
                "source": "wordnet",
            })
        for m in ss.substance_meronyms():
            qualia["constitutive"].append({
                "filler": m.lemma_names()[0].replace("_", " "),
                "relation": "substance_meronym",
                "synset": m.name(),
                "source": "wordnet",
            })

        # Telic: gloss parsing
        gloss = ss.definition()
        for filler in extract_from_gloss(gloss, TELIC_PATTERNS):
            qualia["telic"].append({
                "filler": filler,
                "relation": "gloss_telic",
                "synset": ss.name(),
                "source": "wordnet",
            })

        # Agentive: gloss parsing
        for filler in extract_from_gloss(gloss, AGENTIVE_PATTERNS):
            qualia["agentive"].append({
                "filler": filler,
                "relation": "gloss_agentive",
                "synset": ss.name(),
                "source": "wordnet",
            })

    return qualia


def main():
    seed_path = config.RAW / "concrete_nouns.csv"
    if not seed_path.exists():
        print(f"ERROR: Run 01_extract_conceptnet.py first to generate {seed_path}")
        sys.exit(1)

    nouns = pd.read_csv(seed_path)["word"].tolist()
    print(f"Processing {len(nouns)} nouns through WordNet...")

    records = []
    for word in tqdm(nouns, desc="WordNet extraction"):
        qualia = extract_wordnet_qualia(word)
        records.append({
            "concept": word,
            "qualia": qualia,
        })

    out_path = config.RAW / "wordnet_extract.jsonl"
    utils.write_jsonl(out_path, records)
    print(f"\nSaved {len(records)} records to {out_path}")

    # Stats
    for role in ["telic", "agentive", "constitutive", "formal"]:
        count = sum(1 for r in records if r["qualia"][role])
        print(f"  {role}: {count}/{len(records)} concepts have fillers")


if __name__ == "__main__":
    main()
