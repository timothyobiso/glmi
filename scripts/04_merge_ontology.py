#!/usr/bin/env python3
"""Step 4: Merge qualia extractions from ConceptNet, WordNet, and FrameNet.

Deduplicates fillers, removes vague fillers, and filters to concepts with
≥1 filler in ALL 4 qualia roles. Uses a local LLM to fill agentive gaps.

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

    for role in QUALIA_ROLES:
        merged[role] = deduplicate_fillers(merged[role])

    return merged


def fill_agentive_gaps_local(concepts_missing_agentive: list[str]) -> dict[str, list[dict]]:
    """Use a local LLM to fill agentive gaps."""
    if not concepts_missing_agentive:
        return {}

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("WARNING: vllm not available. Skipping agentive gap filling.")
        return {}

    print(f"\nFilling agentive gaps for {len(concepts_missing_agentive)} concepts using local model...")
    llm = LLM(model=config.NATURALIZE_MODEL)
    sampling = SamplingParams(temperature=0.3, max_tokens=50)

    prompts = []
    for word in concepts_missing_agentive:
        prompts.append(
            f"In 2-6 words, how is a {word} typically made or created? "
            f"Answer with just the short phrase, nothing else."
        )

    outputs = llm.generate(prompts, sampling)
    results = {}
    for word, output in zip(concepts_missing_agentive, outputs):
        text = output.outputs[0].text.strip().rstrip(".,;")
        # Take first line only
        text = text.split("\n")[0].strip().rstrip(".,;")
        if utils.is_valid_filler(text) and len(text.split()) <= 10:
            results[word] = [{
                "filler": text,
                "relation": "llm_agentive",
                "source": "local_llm_fill",
            }]

    # Free GPU memory
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"  Filled {len(results)}/{len(concepts_missing_agentive)} agentive gaps")
    return results


def main():
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

    # Fill agentive gaps with local model if needed
    if has_all_4 < config.TARGET_CONCEPTS and missing_agentive:
        print(f"  Below target ({config.TARGET_CONCEPTS}), attempting local LLM agentive fill...")
        llm_fills = fill_agentive_gaps_local(missing_agentive[:200])
        for concept, fillers in llm_fills.items():
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

    concept_list = [c for c, _ in ranked]
    utils.save_json(config.ONTOLOGY / "concept_list.json", concept_list)
    print(f"Saved concept list ({len(concept_list)} concepts)")


if __name__ == "__main__":
    main()
