#!/usr/bin/env python3
"""Step 8: Validate naturalized stimuli.

Checks: nonce word preservation, qualia purity (no cross-role leakage),
length bounds, and near-duplicate removal.

Depends on: 07.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import config
import utils

# Keywords that indicate cross-role leakage
ROLE_KEYWORDS = {
    "telic": {
        "used for", "used to", "purpose", "function", "in order to",
        "designed to", "meant for", "serves to", "helps to", "useful for",
    },
    "agentive": {
        "made by", "created by", "produced by", "built by", "crafted",
        "manufactured", "assembled", "constructed", "formed by", "comes from",
    },
    "constitutive": {
        "made of", "composed of", "contains", "consists of", "filled with",
        "made from", "material", "component", "part of", "ingredient",
    },
    "formal": {
        "type of", "kind of", "category", "classified as", "similar to",
        "belongs to", "sort of", "class of", "variety of", "form of",
    },
}


def check_nonce_preserved(sentence: str, nonce_word: str) -> bool:
    """Check that the nonce word appears in the sentence."""
    return nonce_word.lower() in sentence.lower()


def check_qualia_purity(sentence: str, intended_role: str) -> list[str]:
    """Check for cross-role keyword leakage. Returns list of leaked roles."""
    leaked = []
    sentence_lower = sentence.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if role == intended_role:
            continue
        for kw in keywords:
            if kw in sentence_lower:
                leaked.append(role)
                break
    return leaked


def check_length(sentence: str) -> tuple[bool, int]:
    """Check sentence word count is within bounds."""
    word_count = len(sentence.split())
    valid = config.MIN_SENTENCE_WORDS <= word_count <= config.MAX_SENTENCE_WORDS
    return valid, word_count


def word_jaccard(s1: str, s2: str) -> float:
    """Compute word-level Jaccard similarity between two sentences."""
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove near-duplicate sentences (Jaccard > threshold)."""
    # Group by concept + role for efficiency
    groups = {}
    for rec in records:
        key = (rec["concept"], rec["qualia_role"])
        groups.setdefault(key, []).append(rec)

    kept = []
    removed = 0
    for key, group in groups.items():
        selected = []
        for rec in group:
            is_dup = False
            for existing in selected:
                sim = word_jaccard(rec["naturalized"], existing["naturalized"])
                if sim > config.JACCARD_DEDUP_THRESHOLD:
                    is_dup = True
                    removed += 1
                    break
            if not is_dup:
                selected.append(rec)
        kept.extend(selected)

    print(f"  Deduplication: removed {removed} near-duplicates")
    return kept


def main():
    input_path = config.STIMULI / "stimuli_naturalized.jsonl"
    if not input_path.exists():
        print(f"ERROR: Run 07_naturalize_sentences.py first to generate {input_path}")
        sys.exit(1)

    records = utils.read_jsonl(input_path)
    print(f"Loaded {len(records)} naturalized sentences")

    validated = []
    issues = []
    stats = {
        "total": len(records),
        "nonce_missing": 0,
        "length_invalid": 0,
        "qualia_leaked": 0,
        "passed": 0,
    }

    for rec in tqdm(records, desc="Validating"):
        sentence = rec.get("naturalized", rec.get("sentence", ""))
        nonce = rec["nonce_word"]
        role = rec["qualia_role"]
        flags = []

        # Check nonce preservation
        if not check_nonce_preserved(sentence, nonce):
            flags.append("nonce_missing")
            stats["nonce_missing"] += 1

        # Check length
        length_ok, word_count = check_length(sentence)
        if not length_ok:
            flags.append(f"length_{word_count}")
            stats["length_invalid"] += 1

        # Check qualia purity
        leaked_roles = check_qualia_purity(sentence, role)
        if leaked_roles:
            flags.append(f"leaked:{','.join(leaked_roles)}")
            stats["qualia_leaked"] += 1

        rec_out = rec.copy()
        rec_out["word_count"] = word_count
        rec_out["validation_flags"] = flags

        if not flags:
            rec_out["valid"] = True
            validated.append(rec_out)
            stats["passed"] += 1
        else:
            rec_out["valid"] = False
            issues.append(rec_out)

    # Deduplicate valid sentences
    validated = deduplicate(validated)

    # Save
    out_path = config.STIMULI / "stimuli_validated.jsonl"
    utils.write_jsonl(out_path, validated)
    print(f"\nSaved {len(validated)} validated sentences to {out_path}")

    if issues:
        issues_path = config.STIMULI / "stimuli_issues.jsonl"
        utils.write_jsonl(issues_path, issues)
        print(f"Saved {len(issues)} flagged sentences to {issues_path}")

    # Stats
    print(f"\n── Validation Stats ──")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print(f"  after dedup: {len(validated)}")

    by_role = {}
    for r in validated:
        by_role.setdefault(r["qualia_role"], 0)
        by_role[r["qualia_role"]] += 1
    for role, count in sorted(by_role.items()):
        print(f"  {role}: {count} valid sentences")


if __name__ == "__main__":
    main()
