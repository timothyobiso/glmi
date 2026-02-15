#!/usr/bin/env python3
"""Step 3: Extract qualia fillers from FrameNet.

Looks up seed nouns as FrameNet lexical units and maps
frame elements to qualia roles. Supplementary source — expected low yield.

Runtime: <1 min (all local).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

import config
import utils

# Frame element names → qualia mapping
FE_TO_QUALIA = {
    # Telic
    "Purpose": "telic",
    "Use": "telic",
    "Goal": "telic",
    "Function": "telic",
    "Useful_action": "telic",
    # Agentive
    "Creator": "agentive",
    "Cause": "agentive",
    "Agent": "agentive",
    "Cook": "agentive",
    "Builder": "agentive",
    # Constitutive
    "Part": "constitutive",
    "Component": "constitutive",
    "Whole": "constitutive",
    "Substance": "constitutive",
    "Material": "constitutive",
    "Ingredient": "constitutive",
    # Formal
    "Category": "formal",
    "Type": "formal",
    "Item": "formal",
}


def extract_framenet_qualia(word: str, fn) -> dict:
    """Extract qualia from FrameNet for a single word."""
    qualia = {"telic": [], "agentive": [], "constitutive": [], "formal": []}

    try:
        lus = fn.lus(r"(?i)\b" + word + r"\.")
    except Exception:
        return qualia

    for lu in lus:
        frame = lu.frame
        frame_name = frame.name

        # Check if frame name maps to a qualia role
        for fname, role in config.FRAME_TO_QUALIA.items():
            if fname.lower() in frame_name.lower():
                qualia[role].append({
                    "filler": frame_name,
                    "relation": f"frame:{frame_name}",
                    "source": "framenet",
                })

        # Check frame elements
        for fe in frame.FE.values():
            fe_name = fe.name
            role = FE_TO_QUALIA.get(fe_name)
            if role:
                # Use the FE definition as context
                definition = fe.definition if hasattr(fe, "definition") else ""
                qualia[role].append({
                    "filler": fe_name.lower(),
                    "relation": f"frame_element:{frame_name}/{fe_name}",
                    "definition": definition[:200] if definition else "",
                    "source": "framenet",
                })

    return qualia


def main():
    try:
        from nltk.corpus import framenet as fn
        # Test that framenet data is available
        _ = fn.frames()
    except Exception as e:
        print(f"WARNING: FrameNet data not available ({e})")
        print("Run: python -c \"import nltk; nltk.download('framenet_v17')\"")
        print("Generating empty extract file as placeholder.")
        utils.write_jsonl(config.RAW / "bso_extract.jsonl", [])
        return

    seed_path = config.RAW / "concrete_nouns.csv"
    if not seed_path.exists():
        print(f"ERROR: Run 01_extract_conceptnet.py first to generate {seed_path}")
        sys.exit(1)

    nouns = pd.read_csv(seed_path)["word"].tolist()
    print(f"Processing {len(nouns)} nouns through FrameNet...")

    records = []
    found = 0
    for word in tqdm(nouns, desc="FrameNet extraction"):
        qualia = extract_framenet_qualia(word, fn)
        has_any = any(qualia[role] for role in qualia)
        if has_any:
            found += 1
        records.append({
            "concept": word,
            "qualia": qualia,
        })

    out_path = config.RAW / "bso_extract.jsonl"
    utils.write_jsonl(out_path, records)
    print(f"\nSaved {len(records)} records to {out_path}")
    print(f"Concepts with FrameNet data: {found}/{len(records)}")

    for role in ["telic", "agentive", "constitutive", "formal"]:
        count = sum(1 for r in records if r["qualia"][role])
        print(f"  {role}: {count}/{len(records)} concepts have fillers")


if __name__ == "__main__":
    main()
