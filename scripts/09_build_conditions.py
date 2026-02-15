#!/usr/bin/env python3
"""Step 9: Build experimental conditions and controls.

Creates single-qualia, accumulation (all 24 orderings), combination conditions,
plus control conditions (real word, conflicting, scrambled, bare nonce,
and information-matched unstructured).
Assigns hard (same-category) + random distractors for discriminative evaluation.

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


# ── Semantic distractor assignment ────────────────────────────────────────────

def build_hypernym_index(concepts: list[str]) -> dict[str, set[str]]:
    """Build concept → top hypernym(s) mapping using WordNet."""
    from nltk.corpus import wordnet as wn

    concept_to_hypernyms = {}
    for concept in concepts:
        synsets = wn.synsets(concept, pos=wn.NOUN)[:2]
        hypers = set()
        for ss in synsets:
            for h in ss.hypernyms():
                hypers.add(h.name())
        concept_to_hypernyms[concept] = hypers
    return concept_to_hypernyms


def assign_distractors(
    all_concepts: list[str],
    n_hard: int,
    n_random: int,
    seed: int = 42,
) -> dict[str, dict]:
    """Assign hard (same-category) + random distractors per concept.

    Hard distractors share a WordNet hypernym with the target.
    Random distractors are sampled from the remaining concepts.
    """
    rng = random.Random(seed)
    hyper_index = build_hypernym_index(all_concepts)

    # Build reverse index: hypernym → concepts
    hyper_to_concepts = {}
    for concept, hypers in hyper_index.items():
        for h in hypers:
            hyper_to_concepts.setdefault(h, set()).add(concept)

    assignments = {}
    for concept in all_concepts:
        # Hard distractors: same hypernym
        same_category = set()
        for h in hyper_index.get(concept, set()):
            same_category |= hyper_to_concepts.get(h, set())
        same_category.discard(concept)
        same_category = list(same_category)

        hard = rng.sample(same_category, min(n_hard, len(same_category)))

        # Random distractors: from remaining pool
        hard_set = set(hard)
        random_pool = [c for c in all_concepts if c != concept and c not in hard_set]
        rand = rng.sample(random_pool, min(n_random, len(random_pool)))

        # If we couldn't find enough hard distractors, fill with random
        shortfall = (n_hard + n_random) - len(hard) - len(rand)
        if shortfall > 0:
            extra_pool = [c for c in all_concepts if c != concept and c not in hard_set and c not in set(rand)]
            extra = rng.sample(extra_pool, min(shortfall, len(extra_pool)))
            rand.extend(extra)

        assignments[concept] = {
            "hard": hard,
            "random": rand,
            "all": hard + rand,
        }

    return assignments


# ── Condition builders ────────────────────────────────────────────────────────

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

    For each permutation, generates 4 conditions (1-role, 2-role, 3-role, 4-role).
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
    """All 2-qualia and 3-qualia combinations (unordered sets, canonical order)."""
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
    """Build control conditions including information-matched controls."""
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

        has_all_4 = all(role_sentences.get(r) for r in ROLES)

        # ── 1. Real word control ──────────────────────────────────────────
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

        # ── 2. Conflicting qualia ─────────────────────────────────────────
        other_concepts = [c for c in all_concepts if c != concept]
        if other_concepts and has_all_4:
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

        # ── 3. Scrambled control ──────────────────────────────────────────
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

        # ── 4. Bare nonce ─────────────────────────────────────────────────
        controls.append({
            "condition_type": "control_bare",
            "condition_label": "bare",
            "concept": concept,
            "nonce_word": nonce,
            "qualia_roles": [],
            "stimulus": f"I saw a {nonce}.",
            "sentences": [f"I saw a {nonce}."],
        })

        # ── 5. Information-matched unstructured control ───────────────────
        # Same fillers from all 4 roles, but in a flat descriptive sentence
        # without qualia-structured templates. Tests whether GL structure
        # matters beyond raw informational content.
        if has_all_4:
            fillers = []
            for role in ROLES:
                filler = role_sentences[role][0].get("filler", "")
                if filler:
                    fillers.append(filler)

            if len(fillers) == 4:
                # Flat: 4 sentences with same fillers, no qualia framing.
                # Length-matched: same number of sentences and nonce
                # occurrences as structured T+A+C+F.
                flat_sentences = [
                    f"The {nonce} is associated with {f}."
                    for f in fillers
                ]
                flat_combined = " ".join(flat_sentences)
                controls.append({
                    "condition_type": "control_info_matched",
                    "condition_label": "info_flat",
                    "concept": concept,
                    "nonce_word": nonce,
                    "qualia_roles": list(ROLES),
                    "fillers_used": fillers,
                    "stimulus": flat_combined,
                    "sentences": flat_sentences,
                })

                # Swapped roles: fillers placed in wrong qualia templates
                # e.g., telic filler in constitutive template, etc.
                role_list = list(ROLES)
                shifted = role_list[1:] + role_list[:1]  # rotate by 1
                swapped_sentences = []
                for orig_role, wrong_role in zip(role_list, shifted):
                    wrong_sent = role_sentences[wrong_role][0]["naturalized"]
                    swapped_sentences.append(wrong_sent)
                swapped_combined = " ".join(swapped_sentences)
                controls.append({
                    "condition_type": "control_info_matched",
                    "condition_label": "info_swapped",
                    "concept": concept,
                    "nonce_word": nonce,
                    "qualia_roles": list(ROLES),
                    "stimulus": swapped_combined,
                    "sentences": swapped_sentences,
                    "note": "same sentences as T+A+C+F but in wrong role order (rotated by 1)",
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

    # Assign distractors (hard + random) for discriminative evaluation
    print(f"\nAssigning distractors: {config.N_HARD_DISTRACTORS} hard + "
          f"{config.N_RANDOM_DISTRACTORS} random per concept...")
    distractor_map = assign_distractors(
        all_concepts, config.N_HARD_DISTRACTORS, config.N_RANDOM_DISTRACTORS,
    )

    # Stats on hard distractor coverage
    n_with_hard = sum(1 for d in distractor_map.values() if d["hard"])
    avg_hard = sum(len(d["hard"]) for d in distractor_map.values()) / max(len(distractor_map), 1)
    print(f"  Concepts with ≥1 hard distractor: {n_with_hard}/{len(all_concepts)}")
    print(f"  Avg hard distractors per concept: {avg_hard:.1f}")

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
    for ct in sorted(set(c["condition_type"] for c in controls)):
        n = sum(1 for c in controls if c["condition_type"] == ct)
        print(f"    {ct}: {n}")

    # Assign unique IDs and distractors to all stimuli + controls
    for i, cond in enumerate(all_conditions):
        cond["stimulus_id"] = f"stim_{i:05d}"
        cond["distractors"] = distractor_map.get(cond["concept"], {}).get("all", [])
        cond["hard_distractors"] = distractor_map.get(cond["concept"], {}).get("hard", [])

    for i, ctrl in enumerate(controls):
        ctrl["stimulus_id"] = f"ctrl_{i:05d}"
        ctrl["distractors"] = distractor_map.get(ctrl["concept"], {}).get("all", [])
        ctrl["hard_distractors"] = distractor_map.get(ctrl["concept"], {}).get("hard", [])

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
        "hard_distractors_per_concept": config.N_HARD_DISTRACTORS,
        "random_distractors_per_concept": config.N_RANDOM_DISTRACTORS,
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
    print(f"  Distractors: {config.N_HARD_DISTRACTORS} hard + {config.N_RANDOM_DISTRACTORS} random")
    print(f"  Experiment models: {config.EXPERIMENT_MODELS}")
    for ct, count in sorted(stats["conditions"].items()):
        print(f"  {ct}: {count}")


if __name__ == "__main__":
    main()
