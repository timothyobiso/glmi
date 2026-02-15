#!/usr/bin/env python3
"""Step 4: Merge qualia extractions from ConceptNet, WordNet, and FrameNet.

Deduplicates fillers, removes vague fillers, and filters to concepts with
≥1 filler in ALL 4 qualia roles. Optionally uses Claude to fill agentive gaps.

Depends on: 01, 02, 03.
Target: 500-1000 concepts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict

from tqdm import tqdm

import config
import utils

QUALIA_ROLES = ["telic", "agentive", "constitutive", "formal"]


def deduplicate_fillers(fillers: list[dict]) -> list[dict]:
    """Deduplicate fillers by normalized filler text, keeping highest weight."""
    seen = {}
    for f in fillers:
        key = f["filler"].lower().strip()
        if key in seen:
            # Keep the one with higher weight
            existing_weight = seen[key].get("weight", 0)
            new_weight = f.get("weight", 0)
            if new_weight > existing_weight:
                seen[key] = f
        else:
            seen[key] = f
    return list(seen.values())


def merge_concept_qualia(concept: str, sources: list[dict]) -> dict:
    """Merge qualia from multiple sources for a single concept."""
    merged = {role: [] for role in QUALIA_ROLES}

    for source in sources:
        qualia = source.get("qualia", {})
        for role in QUALIA_ROLES:
            for filler in qualia.get(role, []):
                if utils.is_valid_filler(filler.get("filler", "")):
                    merged[role].append(filler)

    # Deduplicate per role
    for role in QUALIA_ROLES:
        merged[role] = deduplicate_fillers(merged[role])

    return merged


def fill_agentive_gaps_with_claude(concepts_missing_agentive: list[str]) -> dict[str, list[dict]]:
    """Use Claude API to fill agentive gaps for concepts missing agentive fillers."""
    if not concepts_missing_agentive:
        return {}

    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"WARNING: Could not initialize Claude client ({e}). Skipping agentive gap filling.")
        return {}

    results = {}
    print(f"\nFilling agentive gaps for {len(concepts_missing_agentive)} concepts using Claude...")

    # Process in batches
    batch_size = 20
    for i in tqdm(range(0, len(concepts_missing_agentive), batch_size), desc="Claude agentive fill"):
        batch = concepts_missing_agentive[i:i + batch_size]
        prompt = "For each of the following concrete nouns, briefly describe how it is typically made, created, or comes into being. Give a short phrase (2-6 words) for each.\n\n"
        for word in batch:
            prompt += f"- {word}\n"
        prompt += "\nFormat: one line per word, like: word: how it's made"

        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            for line in text.strip().split("\n"):
                if ":" in line:
                    parts = line.split(":", 1)
                    word = parts[0].strip().lower().lstrip("- ")
                    filler = parts[1].strip().rstrip(".,;")
                    if word in [c.lower() for c in batch] and utils.is_valid_filler(filler):
                        results[word] = [{
                            "filler": filler,
                            "relation": "claude_agentive",
                            "source": "claude_fill",
                        }]
        except Exception as e:
            print(f"  Claude error for batch starting at {i}: {e}")

    return results


def main():
    # Load all extractions
    sources_by_concept = defaultdict(list)

    for src_file, src_name in [
        (config.RAW / "conceptnet_extract.jsonl", "conceptnet"),
        (config.RAW / "wordnet_extract.jsonl", "wordnet"),
        (config.RAW / "bso_extract.jsonl", "framenet"),
    ]:
        if not src_file.exists():
            print(f"WARNING: {src_file} not found, skipping {src_name}")
            continue
        records = utils.read_jsonl(src_file)
        print(f"Loaded {len(records)} records from {src_name}")
        for rec in records:
            sources_by_concept[rec["concept"]].append(rec)

    if not sources_by_concept:
        print("ERROR: No source files found. Run scripts 01-03 first.")
        sys.exit(1)

    # Merge
    print(f"\nMerging qualia for {len(sources_by_concept)} concepts...")
    merged = {}
    for concept, sources in tqdm(sources_by_concept.items(), desc="Merging"):
        merged[concept] = merge_concept_qualia(concept, sources)

    # Stats before filtering
    print("\n── Pre-filter stats ──")
    for role in QUALIA_ROLES:
        count = sum(1 for q in merged.values() if q[role])
        print(f"  {role}: {count}/{len(merged)}")
    has_all_4 = sum(1 for q in merged.values() if all(q[role] for role in QUALIA_ROLES))
    print(f"  All 4 roles: {has_all_4}/{len(merged)}")

    # Identify agentive gaps
    missing_agentive = [c for c, q in merged.items()
                        if not q["agentive"] and all(q[r] for r in ["telic", "constitutive", "formal"])]
    print(f"\n  Concepts with 3/4 roles (missing only agentive): {len(missing_agentive)}")

    # Fill agentive gaps with Claude if needed
    if has_all_4 < config.TARGET_CONCEPTS and missing_agentive:
        print(f"  Below target ({config.TARGET_CONCEPTS}), attempting Claude agentive fill...")
        claude_fills = fill_agentive_gaps_with_claude(missing_agentive[:200])
        for concept, fillers in claude_fills.items():
            if concept in merged:
                merged[concept]["agentive"].extend(fillers)

    # Filter to concepts with all 4 roles
    filtered = {}
    for concept, qualia in merged.items():
        if all(qualia[role] for role in QUALIA_ROLES):
            filtered[concept] = qualia

    print(f"\n── Post-filter stats ──")
    print(f"  Concepts with all 4 qualia roles: {len(filtered)}")
    for role in QUALIA_ROLES:
        avg_fillers = sum(len(filtered[c][role]) for c in filtered) / max(len(filtered), 1)
        print(f"  {role}: avg {avg_fillers:.1f} fillers/concept")

    # Rank by total coverage and save
    ranked = sorted(filtered.items(), key=lambda x: sum(len(x[1][r]) for r in QUALIA_ROLES), reverse=True)

    output = {
        "metadata": {
            "total_concepts": len(filtered),
            "source_concepts": len(sources_by_concept),
            "stats": {
                role: {
                    "concepts_with_fillers": sum(1 for c in filtered if filtered[c][role]),
                    "avg_fillers": sum(len(filtered[c][role]) for c in filtered) / max(len(filtered), 1),
                }
                for role in QUALIA_ROLES
            },
        },
        "concepts": {concept: qualia for concept, qualia in ranked},
    }

    out_path = config.ONTOLOGY / "concept_qualia_merged.json"
    utils.save_json(out_path, output)
    print(f"\nSaved merged ontology to {out_path}")

    # Also save just the concept list
    concept_list = [c for c, _ in ranked]
    utils.save_json(config.ONTOLOGY / "concept_list.json", concept_list)
    print(f"Saved concept list ({len(concept_list)} concepts)")


if __name__ == "__main__":
    main()
