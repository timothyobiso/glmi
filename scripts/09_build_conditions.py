#!/usr/bin/env python3
"""Step 9: Build experimental conditions and controls.

Creates single-qualia, accumulation (all 24 orderings), combination conditions,
plus control conditions (real word, conflicting, scrambled, bare nonce).
Assigns distractor concepts for discriminative evaluation.

Depends on: 08.
Output: stimuli_final.jsonl + control files + dataset stats.
"""

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import config
import utils

ROLES = ["telic", "agentive", "constitutive", "formal"]
ROLE_ABBREV = {"telic": "T", "agentive": "A", "constitutive": "C", "formal": "F"}


def get_nonce_token_indices(sentence: str, nonce_word: str) -> list[int]:
    """Find word-level indices of the nonce word in the sentence."""
    words = sentence.lower().split()
    nonce_lower = nonce_word.lower()
    indices = [i for i, w in enumerate(words) if w.strip(".,;:!?\"'()") == nonce_lower]
    return indices


def assign_distractors(all_concepts: list[str], n: int, seed: int = 42) -> dict[str, list[str]]:
    """Assign N distractor concepts to each concept for discriminative evaluation.

    Distractors are randomly sampled from the other concepts.
    """
    rng = random.Random(seed)
    assignments = {}
    for concept in all_concepts:
        pool = [c for c in all_concepts if c != concept]
        distractors = rng.sample(pool, min(n, len(pool)))
        assignments[concept] = distractors
    return assignments


def build_single_qualia(stimuli_by_concept: dict) -> list[dict]:
    """Single-qualia conditions: T, A, C, F."""
    conditions = []
    for concept, role_sentences in stimuli_by_concept.items():
        for role in ROLES:
            sentences = role_sentences.get(role, [])
            if not sentences:
                continue
            sentence = sentences[0]
            nonce = sentence["nonce_word"]
            conditions.append({
                "condition_type": "single_qualia",
                "condition_label": ROLE_ABBREV[role],
                "concept": concept,
                "nonce_word": nonce,
                "qualia_roles": [role],
                "stimulus": sentence["naturalized"],
                "sentences": [sentence["naturalized"]],
                "nonce_word_indices": get_nonce_token_indices(sentence["naturalized"], nonce),
            })
    return conditions


def build_accumulation(stimuli_by_concept: dict) -> list[dict]:
    """Accumulation conditions: all 24 orderings of T/A/C/F.

    For each permutation, generates 4 conditions (1-role, 2-role, 3-role, 4-role)
    using that specific ordering.
    """
    all_orderings = list(itertools.permutations(ROLES))
    conditions = []

    for concept, role_sentences in stimuli_by_concept.items():
        if not all(role_sentences.get(r) for r in ROLES):
            continue

        nonce = role_sentences[ROLES[0]][0]["nonce_word"]

        for ordering in all_orderings:
            ordering_label = "".join(ROLE_ABBREV[r] for r in ordering)

            for n_roles in range(1, len(ordering) + 1):
                roles_used = list(ordering[:n_roles])
                accum_label = "+".join(ROLE_ABBREV[r] for r in roles_used)

                sentences = [role_sentences[r][0]["naturalized"] for r in roles_used]
                combined = " ".join(sentences)

                conditions.append({
                    "condition_type": "accumulation",
                    "condition_label": accum_label,
                    "ordering": ordering_label,
                    "concept": concept,
                    "nonce_word": nonce,
                    "qualia_roles": roles_used,
                    "stimulus": combined,
                    "sentences": sentences,
                    "nonce_word_indices": get_nonce_token_indices(combined, nonce),
                })

    return conditions


def build_combinations(stimuli_by_concept: dict) -> list[dict]:
    """All 2-qualia and 3-qualia combinations (unordered sets, single canonical order)."""
    conditions = []

    for concept, role_sentences in stimuli_by_concept.items():
        nonce = None
        for role in ROLES:
            if role_sentences.get(role):
                nonce = role_sentences[role][0]["nonce_word"]
                break
        if not nonce:
            continue

        for k in [2, 3]:
            for combo in itertools.combinations(ROLES, k):
                if not all(role_sentences.get(r) for r in combo):
                    continue
                label = "+".join(ROLE_ABBREV[r] for r in combo)
                sentences = [role_sentences[r][0]["naturalized"] for r in combo]
                combined = " ".join(sentences)
                conditions.append({
                    "condition_type": f"combination_{k}",
                    "condition_label": label,
                    "concept": concept,
                    "nonce_word": nonce,
                    "qualia_roles": list(combo),
                    "stimulus": combined,
                    "sentences": sentences,
                    "nonce_word_indices": get_nonce_token_indices(combined, nonce),
                })

    return conditions


def build_controls(stimuli_by_concept: dict, all_concepts: list[str]) -> list[dict]:
    """Build control conditions."""
    controls = []
    random.seed(42)

    for concept, role_sentences in stimuli_by_concept.items():
        nonce = None
        for role in ROLES:
            if role_sentences.get(role):
                nonce = role_sentences[role][0]["nonce_word"]
                break
        if not nonce:
            continue

        # 1. Real word control — replace nonce with actual concept
        for role in ROLES:
            if not role_sentences.get(role):
                continue
            sentence = role_sentences[role][0]["naturalized"]
            real_sentence = sentence.replace(nonce, concept)
            controls.append({
                "condition_type": "control_real_word",
                "condition_label": f"real_{ROLE_ABBREV[role]}",
                "concept": concept,
                "nonce_word": concept,
                "original_nonce": nonce,
                "qualia_roles": [role],
                "stimulus": real_sentence,
                "sentences": [real_sentence],
            })

        # 2. Conflicting qualia — mix qualia from different concepts
        other_concepts = [c for c in all_concepts if c != concept]
        if other_concepts and all(role_sentences.get(r) for r in ROLES):
            other = random.choice(other_concepts)
            other_roles = stimuli_by_concept.get(other, {})

            if other_roles.get("agentive") and role_sentences.get("telic"):
                t_sent = role_sentences["telic"][0]["naturalized"]
                a_sent = other_roles["agentive"][0]["naturalized"]
                other_nonce = other_roles["agentive"][0]["nonce_word"]
                a_sent = a_sent.replace(other_nonce, nonce)
                controls.append({
                    "condition_type": "control_conflicting",
                    "condition_label": "conflict_T+A",
                    "concept": concept,
                    "conflicting_concept": other,
                    "nonce_word": nonce,
                    "qualia_roles": ["telic", "agentive"],
                    "stimulus": f"{t_sent} {a_sent}",
                    "sentences": [t_sent, a_sent],
                })

        # 3. Scrambled control — shuffle words (except nonce)
        if role_sentences.get("telic"):
            sentence = role_sentences["telic"][0]["naturalized"]
            words = sentence.split()
            nonce_lower = nonce.lower()
            nonce_positions = [i for i, w in enumerate(words)
                               if w.strip(".,;:!?\"'()").lower() == nonce_lower]
            non_nonce = [w for i, w in enumerate(words) if i not in nonce_positions]
            random.shuffle(non_nonce)
            scrambled = []
            ni = 0
            for i in range(len(words)):
                if i in nonce_positions:
                    scrambled.append(words[i])
                else:
                    scrambled.append(non_nonce[ni])
                    ni += 1
            controls.append({
                "condition_type": "control_scrambled",
                "condition_label": "scrambled",
                "concept": concept,
                "nonce_word": nonce,
                "qualia_roles": [],
                "stimulus": " ".join(scrambled),
                "sentences": [" ".join(scrambled)],
            })

        # 4. Bare nonce — minimal context
        controls.append({
            "condition_type": "control_bare",
            "condition_label": "bare",
            "concept": concept,
            "nonce_word": nonce,
            "qualia_roles": [],
            "stimulus": f"I saw a {nonce}.",
            "sentences": [f"I saw a {nonce}."],
        })

    return controls


def main():
    input_path = config.STIMULI / "stimuli_validated.jsonl"
    if not input_path.exists():
        print(f"ERROR: Run 08_validate_stimuli.py first to generate {input_path}")
        sys.exit(1)

    records = utils.read_jsonl(input_path)
    print(f"Loaded {len(records)} validated sentences")

    # Organize by concept and role
    stimuli_by_concept = {}
    for rec in records:
        concept = rec["concept"]
        role = rec["qualia_role"]
        stimuli_by_concept.setdefault(concept, {}).setdefault(role, []).append(rec)

    all_concepts = list(stimuli_by_concept.keys())
    print(f"Concepts: {len(all_concepts)}")

    # Assign distractors for discriminative evaluation
    print(f"\nAssigning {config.N_DISTRACTORS} distractors per concept...")
    distractor_map = assign_distractors(all_concepts, config.N_DISTRACTORS)

    # Build conditions
    print("\nBuilding conditions...")
    all_conditions = []

    single = build_single_qualia(stimuli_by_concept)
    print(f"  Single-qualia: {len(single)}")
    all_conditions.extend(single)

    accum = build_accumulation(stimuli_by_concept)
    print(f"  Accumulation (24 orderings): {len(accum)}")
    all_conditions.extend(accum)

    combos = build_combinations(stimuli_by_concept)
    print(f"  Combinations: {len(combos)}")
    all_conditions.extend(combos)

    # Build controls
    print("\nBuilding controls...")
    controls = build_controls(stimuli_by_concept, all_concepts)
    print(f"  Controls: {len(controls)}")

    # Assign unique IDs and distractors to all stimuli + controls
    for i, cond in enumerate(all_conditions):
        cond["stimulus_id"] = f"stim_{i:05d}"
        cond["distractors"] = distractor_map.get(cond["concept"], [])

    for i, ctrl in enumerate(controls):
        ctrl["stimulus_id"] = f"ctrl_{i:05d}"
        ctrl["distractors"] = distractor_map.get(ctrl["concept"], [])

    # Save final stimuli
    out_path = config.STIMULI / "stimuli_final.jsonl"
    utils.write_jsonl(out_path, all_conditions)
    print(f"\nSaved {len(all_conditions)} experimental stimuli to {out_path}")

    # Save controls separately
    ctrl_path = config.CONTROLS / "controls.jsonl"
    utils.write_jsonl(ctrl_path, controls)
    print(f"Saved {len(controls)} control stimuli to {ctrl_path}")

    # Save distractor map
    utils.save_json(config.CONTROLS / "distractor_map.json", distractor_map)

    # Dataset statistics
    n_concepts_all4 = sum(1 for c in stimuli_by_concept
                          if all(stimuli_by_concept[c].get(r) for r in ROLES))
    stats = {
        "total_stimuli": len(all_conditions),
        "total_controls": len(controls),
        "total_combined": len(all_conditions) + len(controls),
        "unique_concepts": len(all_concepts),
        "concepts_with_all_4_roles": n_concepts_all4,
        "accumulation_orderings": 24,
        "distractors_per_concept": config.N_DISTRACTORS,
        "experiment_models": config.EXPERIMENT_MODELS,
        "conditions": {},
    }

    for cond in all_conditions:
        ct = cond["condition_type"]
        stats["conditions"].setdefault(ct, 0)
        stats["conditions"][ct] += 1

    control_types = {}
    for ctrl in controls:
        ct = ctrl["condition_type"]
        control_types.setdefault(ct, 0)
        control_types[ct] += 1
    stats["control_types"] = control_types

    stats_path = config.STIMULI / "dataset_statistics.json"
    utils.save_json(stats_path, stats)
    print(f"\nDataset statistics saved to {stats_path}")
    print(f"\n── Summary ──")
    print(f"  Total stimuli: {stats['total_stimuli']}")
    print(f"  Total controls: {stats['total_controls']}")
    print(f"  Unique concepts: {stats['unique_concepts']}")
    print(f"  Concepts with all 4 roles: {n_concepts_all4}")
    print(f"  Accumulation orderings: 24")
    print(f"  Distractors per concept: {config.N_DISTRACTORS}")
    print(f"  Experiment models: {config.EXPERIMENT_MODELS}")
    for ct, count in sorted(stats["conditions"].items()):
        print(f"  {ct}: {count}")


if __name__ == "__main__":
    main()
