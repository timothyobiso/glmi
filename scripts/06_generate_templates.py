#!/usr/bin/env python3
"""Step 6: Generate templated sentences for each concept × nonce × qualia role.

4-6 templates per qualia role, filled from merged ontology with
morphological adjustment via inflect.

Depends on: 04 (merged ontology), 05 (nonce words).
Output: ~8,000+ raw sentences.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import inflect
from tqdm import tqdm

import config
import utils

p = inflect.engine()

# ── Templates per qualia role ──────────────────────────────────────────────────
# {nonce} = nonce word, {filler} = qualia filler

TEMPLATES = {
    "telic": [
        "She used the {nonce} to {filler}.",
        "The {nonce} is mainly used for {filler_gerund}.",
        "He grabbed the {nonce} because he needed to {filler}.",
        "The purpose of {a_nonce} is to {filler}.",
        "People often rely on {a_nonce} when they want to {filler}.",
        "Without the {nonce}, it would be hard to {filler}.",
    ],
    "agentive": [
        "The {nonce} was made by {filler}.",
        "A craftsman created the {nonce} using {filler}.",
        "The {nonce} is produced through {filler}.",
        "They built the {nonce} by {filler_gerund}.",
        "The {nonce} comes from {filler}.",
        "To make {a_nonce}, you need {filler}.",
    ],
    "constitutive": [
        "The {nonce} is made of {filler}.",
        "Inside the {nonce}, there is {a_filler}.",
        "The {nonce} contains {filler}.",
        "A key part of the {nonce} is its {filler}.",
        "The {nonce} has {a_filler} at its core.",
        "The main component of the {nonce} is {filler}.",
    ],
    "formal": [
        "The {nonce} is a type of {filler}.",
        "A {nonce} is similar to {a_filler}.",
        "The {nonce} belongs to the category of {filler_plural}.",
        "Like any {filler}, the {nonce} has distinctive features.",
        "The {nonce} can be classified as {a_filler}.",
        "Among all {filler_plural}, the {nonce} stands out.",
    ],
}


def adjust_filler(filler: str, form: str) -> str:
    """Adjust filler morphologically based on required form."""
    filler = filler.strip().lower()

    if form == "gerund":
        # Try to convert verb phrase to gerund
        words = filler.split()
        if words:
            first = words[0]
            if first.endswith("e") and not first.endswith("ee"):
                words[0] = first[:-1] + "ing"
            elif first.endswith("ing"):
                pass  # already gerund
            else:
                words[0] = first + "ing"
            return " ".join(words)
        return filler + "ing"

    elif form == "plural":
        words = filler.split()
        if words:
            last = words[-1]
            plural = p.plural(last)
            words[-1] = plural
            return " ".join(words)
        return filler

    elif form == "a":
        return p.a(filler)

    return filler


def fill_template(template: str, nonce: str, filler: str) -> str:
    """Fill a template with nonce word and filler, handling morphological variants."""
    result = template
    result = result.replace("{nonce}", nonce)
    result = result.replace("{a_nonce}", p.a(nonce))
    result = result.replace("{filler}", filler)
    result = result.replace("{a_filler}", adjust_filler(filler, "a"))
    result = result.replace("{filler_gerund}", adjust_filler(filler, "gerund"))
    result = result.replace("{filler_plural}", adjust_filler(filler, "plural"))
    return result


def main():
    # Load merged ontology
    ontology_path = config.ONTOLOGY / "concept_qualia_merged.json"
    if not ontology_path.exists():
        print(f"ERROR: Run 04_merge_ontology.py first to generate {ontology_path}")
        sys.exit(1)

    ontology = utils.load_json(ontology_path)
    concepts = ontology["concepts"]
    print(f"Loaded {len(concepts)} concepts from merged ontology")

    # Load nonce words
    nonce_path = config.NONCE / "nonce_words.jsonl"
    if not nonce_path.exists():
        print(f"ERROR: Run 05_generate_nonce_words.py first to generate {nonce_path}")
        sys.exit(1)

    nonce_records = utils.read_jsonl(nonce_path)
    nonce_words = [r["nonce_word"] for r in nonce_records]
    nonce_info = {r["nonce_word"]: r for r in nonce_records}
    print(f"Loaded {len(nonce_words)} nonce words")

    if len(nonce_words) < len(concepts):
        print(f"WARNING: Fewer nonce words ({len(nonce_words)}) than concepts ({len(concepts)})")
        print("  Will reuse nonce words as needed.")

    # Assign nonce words to concepts
    concept_list = list(concepts.keys())
    random.seed(42)
    random.shuffle(concept_list)

    assignments = {}
    for i, concept in enumerate(concept_list):
        nonce = nonce_words[i % len(nonce_words)]
        assignments[concept] = nonce

    # Generate sentences
    records = []
    template_id = 0

    for concept in tqdm(concept_list, desc="Generating templates"):
        nonce = assignments[concept]
        qualia = concepts[concept]

        for role in ["telic", "agentive", "constitutive", "formal"]:
            fillers = qualia.get(role, [])
            if not fillers:
                continue

            templates = TEMPLATES[role]
            for tmpl in templates:
                # Use top filler (by weight if available)
                sorted_fillers = sorted(fillers, key=lambda f: f.get("weight", 0), reverse=True)
                filler = sorted_fillers[0]
                filler_text = filler["filler"]

                sentence = fill_template(tmpl, nonce, filler_text)

                records.append({
                    "template_id": template_id,
                    "concept": concept,
                    "nonce_word": nonce,
                    "qualia_role": role,
                    "filler": filler_text,
                    "filler_source": filler.get("source", "unknown"),
                    "template": tmpl,
                    "sentence": sentence,
                })
                template_id += 1

    out_path = config.STIMULI / "templates_raw.jsonl"
    utils.write_jsonl(out_path, records)
    print(f"\nSaved {len(records)} templated sentences to {out_path}")

    # Stats
    by_role = {}
    for r in records:
        by_role.setdefault(r["qualia_role"], 0)
        by_role[r["qualia_role"]] += 1
    for role, count in sorted(by_role.items()):
        print(f"  {role}: {count} sentences")

    # Save assignment map
    utils.save_json(config.NONCE / "concept_nonce_assignments.json", assignments)
    print(f"Saved concept→nonce assignments")


if __name__ == "__main__":
    main()
