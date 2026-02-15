#!/usr/bin/env python3
"""Similarity analysis with GL-specific hypothesis tests.

Tests five pre-registered hypotheses derived from Generative Lexicon theory:

H1 (Formal Primacy): Formal role yields highest single-role similarity,
    and Formal-first orderings converge faster.
H2 (Telic Distinctiveness): Telic is the most discriminative single role
    (highest MRR improvement).
H3 (GL Structure): Structured qualia sentences outperform information-matched
    controls (flat listing, swapped roles) with identical content.
H4 (Ordering Effects): Role ordering significantly affects convergence;
    Formal-first orderings are faster than others.
H5 (Cross-Architecture): GL effects replicate across Llama and Mistral.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils

ROLES = ["telic", "agentive", "constitutive", "formal"]
ROLE_ABBREV = {"telic": "T", "agentive": "A", "constitutive": "C", "formal": "F"}
ABBREV_TO_ROLE = {v: k for k, v in ROLE_ABBREV.items()}


# ── Cosine similarity ────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_sim_batch(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of A and vector b."""
    A = A.astype(np.float32)
    b = b.astype(np.float32)
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        return np.zeros(A.shape[0])
    norm_A = np.where(norm_A == 0, 1, norm_A)
    return (A @ b) / (norm_A.squeeze() * norm_b)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_model_data(model_name: str, agg: str = "mean"):
    """Load representations, references, and metadata for a model."""
    model_slug = model_name.replace("/", "--")

    repr_dir = config.DATA / "representations" / model_slug
    ref_dir = config.DATA / "references" / model_slug

    repr_data = np.load(repr_dir / f"representations_{agg}.npy")
    repr_meta = utils.load_json(repr_dir / "metadata.json")

    ref_data = np.load(ref_dir / "references.npy")
    ref_meta = utils.load_json(ref_dir / "metadata.json")

    return repr_data, repr_meta, ref_data, ref_meta


def build_stimulus_index(stimuli: list[dict]) -> dict[str, int]:
    """Map stimulus_id → index in the representations array."""
    return {s["stimulus_id"]: i for i, s in enumerate(stimuli)}


def build_concept_index(concepts: list[str]) -> dict[str, int]:
    """Map concept → index in the references array."""
    return {c: i for i, c in enumerate(concepts)}


# ── Absolute similarity ──────────────────────────────────────────────────────

def compute_absolute_similarity(
    stimuli: list[dict],
    repr_data: np.ndarray,
    repr_meta: dict,
    ref_data: np.ndarray,
    ref_meta: dict,
    layers: list[int] | None = None,
) -> list[dict]:
    """Compute cosine similarity between each stimulus and its target reference.

    Returns list of dicts with stimulus metadata + per-layer similarity.
    """
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    concept_index = build_concept_index(ref_meta["concepts"])
    valid_mask = repr_meta["valid_mask"]
    ref_valid = ref_meta["valid_mask"]
    n_layers = repr_meta["n_layers"]

    if layers is None:
        layers = list(range(n_layers))

    results = []
    for stim in tqdm(stimuli, desc="Computing absolute similarity"):
        sid = stim["stimulus_id"]
        concept = stim["concept"]

        si = stim_index.get(sid)
        ci = concept_index.get(concept)

        if si is None or ci is None:
            continue
        if not valid_mask[si] or not ref_valid[ci]:
            continue

        sims = {}
        for layer in layers:
            s = cosine_sim(repr_data[si, layer], ref_data[ci, layer])
            sims[f"layer_{layer}"] = s

        results.append({
            "stimulus_id": sid,
            "concept": concept,
            "condition_type": stim.get("condition_type", ""),
            "condition_label": stim.get("condition_label", ""),
            "ordering": stim.get("ordering", ""),
            "qualia_roles": stim.get("qualia_roles", []),
            "n_roles": len(stim.get("qualia_roles", [])),
            **sims,
        })

    return results


# ── Discriminative evaluation ─────────────────────────────────────────────────

def compute_discriminative(
    stimuli: list[dict],
    repr_data: np.ndarray,
    repr_meta: dict,
    ref_data: np.ndarray,
    ref_meta: dict,
    distractor_map: dict,
    layer: int,
) -> list[dict]:
    """Discriminative evaluation at a single layer.

    For each stimulus, rank the target concept among target + distractors
    by cosine similarity. Compute MRR, Hits@1, Hits@3.
    """
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    concept_index = build_concept_index(ref_meta["concepts"])
    valid_mask = repr_meta["valid_mask"]
    ref_valid = ref_meta["valid_mask"]

    results = []
    for stim in stimuli:
        sid = stim["stimulus_id"]
        concept = stim["concept"]

        si = stim_index.get(sid)
        ci = concept_index.get(concept)
        if si is None or ci is None or not valid_mask[si] or not ref_valid[ci]:
            continue

        # Get distractors
        dist_info = distractor_map.get(concept, {})
        distractor_concepts = dist_info.get("all", [])
        distractor_indices = [concept_index[d] for d in distractor_concepts
                              if d in concept_index and ref_valid[concept_index[d]]]

        if not distractor_indices:
            continue

        # Compute similarities
        stim_repr = repr_data[si, layer]
        target_sim = cosine_sim(stim_repr, ref_data[ci, layer])

        distractor_sims = [cosine_sim(stim_repr, ref_data[di, layer])
                           for di in distractor_indices]

        # Rank: count how many distractors score >= target
        rank = 1 + sum(1 for ds in distractor_sims if ds >= target_sim)

        results.append({
            "stimulus_id": sid,
            "concept": concept,
            "condition_type": stim.get("condition_type", ""),
            "condition_label": stim.get("condition_label", ""),
            "ordering": stim.get("ordering", ""),
            "n_roles": len(stim.get("qualia_roles", [])),
            "layer": layer,
            "target_sim": target_sim,
            "rank": rank,
            "mrr": 1.0 / rank,
            "hits_at_1": int(rank == 1),
            "hits_at_3": int(rank <= 3),
            "n_candidates": 1 + len(distractor_indices),
        })

    return results


# ── GL Hypothesis Tests ───────────────────────────────────────────────────────

def test_h1_formal_primacy(abs_results: list[dict], layer: int) -> dict:
    """H1: Formal role yields highest single-role similarity.

    Among single-qualia conditions, Formal should produce the highest
    cosine similarity to the target concept.
    """
    layer_key = f"layer_{layer}"

    role_sims = defaultdict(list)
    for r in abs_results:
        if r["condition_type"] == "single_qualia" and layer_key in r:
            role_sims[r["condition_label"]].append(r[layer_key])

    means = {role: np.mean(sims) for role, sims in role_sims.items() if sims}
    best_role = max(means, key=means.get) if means else None

    # Pairwise Wilcoxon tests: F vs each other role
    pairwise = {}
    f_sims = role_sims.get("F", [])
    for role in ["T", "A", "C"]:
        other_sims = role_sims.get(role, [])
        if len(f_sims) >= 5 and len(other_sims) >= 5:
            # Match by concept (assuming same order)
            n = min(len(f_sims), len(other_sims))
            stat, p = sp_stats.wilcoxon(f_sims[:n], other_sims[:n])
            pairwise[f"F_vs_{role}"] = {"statistic": float(stat), "p_value": float(p)}

    return {
        "hypothesis": "H1: Formal Primacy",
        "layer": layer,
        "mean_similarity_by_role": {k: float(v) for k, v in means.items()},
        "best_single_role": best_role,
        "formal_is_best": best_role == "F",
        "pairwise_tests": pairwise,
    }


def test_h2_telic_distinctiveness(disc_results: list[dict]) -> dict:
    """H2: Telic is the most discriminative single role (highest MRR).

    Among single-qualia conditions, Telic should yield the highest MRR
    in the discriminative evaluation.
    """
    role_mrr = defaultdict(list)
    for r in disc_results:
        if r["condition_type"] == "single_qualia":
            role_mrr[r["condition_label"]].append(r["mrr"])

    mean_mrr = {role: float(np.mean(vals)) for role, vals in role_mrr.items() if vals}
    best_role = max(mean_mrr, key=mean_mrr.get) if mean_mrr else None

    return {
        "hypothesis": "H2: Telic Distinctiveness",
        "mean_mrr_by_role": mean_mrr,
        "best_discriminative_role": best_role,
        "telic_is_best": best_role == "T",
    }


def test_h3_gl_structure(abs_results: list[dict], layer: int) -> dict:
    """H3: GL-structured sentences outperform info-matched controls.

    Compares full T+A+C+F accumulation condition against:
    - info_flat: same fillers, flat listing
    - info_swapped: same sentences, wrong role order
    """
    layer_key = f"layer_{layer}"

    structured_sims = defaultdict(list)   # concept → sim
    flat_sims = defaultdict(list)
    swapped_sims = defaultdict(list)

    for r in abs_results:
        if layer_key not in r:
            continue
        concept = r["concept"]
        ct = r["condition_type"]
        cl = r["condition_label"]

        # Full 4-role accumulation (any ordering that includes all 4)
        if ct == "accumulation" and r["n_roles"] == 4:
            structured_sims[concept].append(r[layer_key])
        elif ct == "control_info_matched" and cl == "info_flat":
            flat_sims[concept].append(r[layer_key])
        elif ct == "control_info_matched" and cl == "info_swapped":
            swapped_sims[concept].append(r[layer_key])

    # Average over orderings for structured
    structured_avg = {c: float(np.mean(v)) for c, v in structured_sims.items()}
    flat_avg = {c: float(np.mean(v)) for c, v in flat_sims.items()}
    swapped_avg = {c: float(np.mean(v)) for c, v in swapped_sims.items()}

    # Paired tests on concepts present in both
    tests = {}

    # Structured vs Flat
    shared = sorted(set(structured_avg) & set(flat_avg))
    if len(shared) >= 10:
        a = [structured_avg[c] for c in shared]
        b = [flat_avg[c] for c in shared]
        stat, p = sp_stats.wilcoxon(a, b)
        tests["structured_vs_flat"] = {
            "statistic": float(stat),
            "p_value": float(p),
            "mean_structured": float(np.mean(a)),
            "mean_flat": float(np.mean(b)),
            "effect_size": float(np.mean(a) - np.mean(b)),
            "n_concepts": len(shared),
        }

    # Structured vs Swapped
    shared = sorted(set(structured_avg) & set(swapped_avg))
    if len(shared) >= 10:
        a = [structured_avg[c] for c in shared]
        b = [swapped_avg[c] for c in shared]
        stat, p = sp_stats.wilcoxon(a, b)
        tests["structured_vs_swapped"] = {
            "statistic": float(stat),
            "p_value": float(p),
            "mean_structured": float(np.mean(a)),
            "mean_swapped": float(np.mean(b)),
            "effect_size": float(np.mean(a) - np.mean(b)),
            "n_concepts": len(shared),
        }

    return {
        "hypothesis": "H3: GL Structure Matters",
        "layer": layer,
        "tests": tests,
    }


def test_h4_ordering_effects(abs_results: list[dict], layer: int) -> dict:
    """H4: Role ordering significantly affects convergence speed.

    Tests whether orderings starting with Formal converge faster than others.
    Uses accumulation conditions at each step (1-role through 4-role).
    """
    layer_key = f"layer_{layer}"

    # Group by (first_role, n_roles) → list of similarities
    by_first_role_and_step = defaultdict(list)
    for r in abs_results:
        if r["condition_type"] != "accumulation" or layer_key not in r:
            continue
        ordering = r.get("ordering", "")
        if not ordering:
            continue
        first_role = ordering[0]  # T, A, C, or F
        n_roles = r["n_roles"]
        by_first_role_and_step[(first_role, n_roles)].append(r[layer_key])

    # Mean similarity by first role and step
    convergence = {}
    for (first_role, n_roles), sims in by_first_role_and_step.items():
        convergence.setdefault(first_role, {})[n_roles] = float(np.mean(sims))

    # Kruskal-Wallis test at each step: does first role matter?
    step_tests = {}
    for n_roles in [1, 2, 3, 4]:
        groups = []
        group_labels = []
        for first_role in ["T", "A", "C", "F"]:
            sims = by_first_role_and_step.get((first_role, n_roles), [])
            if len(sims) >= 5:
                groups.append(sims)
                group_labels.append(first_role)
        if len(groups) >= 2:
            stat, p = sp_stats.kruskal(*groups)
            step_tests[f"step_{n_roles}"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "groups": group_labels,
                "group_means": {l: float(np.mean(g)) for l, g in zip(group_labels, groups)},
            }

    # Specific contrast: F-first vs all others at step 1
    f_first_1 = by_first_role_and_step.get(("F", 1), [])
    other_1 = []
    for role in ["T", "A", "C"]:
        other_1.extend(by_first_role_and_step.get((role, 1), []))

    f_vs_other = {}
    if len(f_first_1) >= 5 and len(other_1) >= 5:
        stat, p = sp_stats.mannwhitneyu(f_first_1, other_1, alternative="greater")
        f_vs_other = {
            "statistic": float(stat),
            "p_value": float(p),
            "mean_f_first": float(np.mean(f_first_1)),
            "mean_other_first": float(np.mean(other_1)),
        }

    return {
        "hypothesis": "H4: Ordering Effects",
        "layer": layer,
        "convergence_by_first_role": convergence,
        "step_tests": step_tests,
        "f_first_vs_others_step1": f_vs_other,
    }


def test_h5_cross_architecture(model_results: dict[str, list[dict]], layer: int) -> dict:
    """H5: GL effects are consistent across Llama and Mistral.

    Correlates per-concept similarity patterns across models.
    """
    layer_key = f"layer_{layer}"
    models = list(model_results.keys())
    if len(models) < 2:
        return {"hypothesis": "H5: Cross-Architecture", "skip": "Need ≥2 models"}

    m1, m2 = models[0], models[1]

    # Collect per-concept mean similarity for 4-role accumulation
    def concept_means(results):
        sims = defaultdict(list)
        for r in results:
            if r["condition_type"] == "accumulation" and r["n_roles"] == 4 and layer_key in r:
                sims[r["concept"]].append(r[layer_key])
        return {c: float(np.mean(v)) for c, v in sims.items()}

    means1 = concept_means(model_results[m1])
    means2 = concept_means(model_results[m2])

    shared = sorted(set(means1) & set(means2))
    if len(shared) < 10:
        return {"hypothesis": "H5: Cross-Architecture", "skip": f"Only {len(shared)} shared concepts"}

    vals1 = [means1[c] for c in shared]
    vals2 = [means2[c] for c in shared]

    r, p = sp_stats.pearsonr(vals1, vals2)
    rho, rho_p = sp_stats.spearmanr(vals1, vals2)

    # Compare single-role rankings across models
    role_rankings = {}
    for model_name, results in model_results.items():
        role_sims = defaultdict(list)
        for r_item in results:
            if r_item["condition_type"] == "single_qualia" and layer_key in r_item:
                role_sims[r_item["condition_label"]].append(r_item[layer_key])
        role_rankings[model_name] = {
            role: float(np.mean(sims)) for role, sims in role_sims.items()
        }

    return {
        "hypothesis": "H5: Cross-Architecture Consistency",
        "layer": layer,
        "models": [m1, m2],
        "n_shared_concepts": len(shared),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "role_rankings_by_model": role_rankings,
    }


# ── Accumulation curve ────────────────────────────────────────────────────────

def compute_accumulation_curve(abs_results: list[dict], layer: int) -> dict:
    """Compute mean similarity at each accumulation step (1-4 roles).

    Broken down by starting role to test ordering effects.
    """
    layer_key = f"layer_{layer}"

    by_step = defaultdict(list)
    by_step_and_first = defaultdict(lambda: defaultdict(list))

    for r in abs_results:
        if r["condition_type"] != "accumulation" or layer_key not in r:
            continue
        n = r["n_roles"]
        ordering = r.get("ordering", "")
        first = ordering[0] if ordering else "?"
        by_step[n].append(r[layer_key])
        by_step_and_first[n][first].append(r[layer_key])

    curve = {
        "overall": {n: float(np.mean(sims)) for n, sims in sorted(by_step.items())},
        "by_first_role": {},
    }
    for n in sorted(by_step_and_first):
        curve["by_first_role"][n] = {
            first: float(np.mean(sims))
            for first, sims in sorted(by_step_and_first[n].items())
        }

    return curve


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(agg: str = "mean", target_layers: list[int] | None = None):
    """Run full analysis pipeline across all models."""
    stimuli_path = config.STIMULI / "stimuli_final.jsonl"
    controls_path = config.CONTROLS / "controls.jsonl"
    distractor_path = config.CONTROLS / "distractor_map.json"
    output_dir = config.DATA / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    stimuli = utils.read_jsonl(stimuli_path)
    controls = utils.read_jsonl(controls_path) if controls_path.exists() else []
    all_stimuli = stimuli + controls
    distractor_map = utils.load_json(distractor_path) if distractor_path.exists() else {}

    print(f"Loaded {len(stimuli)} stimuli + {len(controls)} controls")

    all_model_abs_results = {}

    for model_name in config.EXPERIMENT_MODELS:
        model_slug = model_name.replace("/", "--")
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name}")
        print(f"{'='*60}")

        try:
            repr_data, repr_meta, ref_data, ref_meta = load_model_data(model_name, agg)
        except FileNotFoundError as e:
            print(f"  Skipping — data not found: {e}")
            continue

        n_layers = repr_meta["n_layers"]
        if target_layers is None:
            # Sample layers: first, every 4th, and last
            layers = sorted(set([0, n_layers // 4, n_layers // 2,
                                 3 * n_layers // 4, n_layers - 1]))
        else:
            layers = target_layers

        print(f"  Analyzing layers: {layers}")

        # 1. Absolute similarity
        print("  Computing absolute similarity...")
        abs_results = compute_absolute_similarity(
            all_stimuli, repr_data, repr_meta, ref_data, ref_meta, layers,
        )
        all_model_abs_results[model_name] = abs_results

        # Save per-model absolute results
        model_out = output_dir / model_slug
        model_out.mkdir(parents=True, exist_ok=True)
        utils.write_jsonl(model_out / "absolute_similarity.jsonl", abs_results)

        # 2. Discriminative evaluation at best layer candidates
        print("  Computing discriminative evaluation...")
        for layer in layers:
            disc_results = compute_discriminative(
                all_stimuli, repr_data, repr_meta, ref_data, ref_meta,
                distractor_map, layer,
            )
            if disc_results:
                utils.write_jsonl(model_out / f"discriminative_layer{layer}.jsonl", disc_results)

                mean_mrr = np.mean([r["mrr"] for r in disc_results])
                hits1 = np.mean([r["hits_at_1"] for r in disc_results])
                hits3 = np.mean([r["hits_at_3"] for r in disc_results])
                print(f"    Layer {layer}: MRR={mean_mrr:.4f}, Hits@1={hits1:.4f}, Hits@3={hits3:.4f}")

        # 3. GL hypothesis tests
        print("  Running GL hypothesis tests...")
        best_layer = layers[len(layers) // 2]  # middle layer as default

        # Use last layer's discriminative results for H2
        disc_results_last = compute_discriminative(
            all_stimuli, repr_data, repr_meta, ref_data, ref_meta,
            distractor_map, layers[-1],
        )

        gl_results = {
            "model": model_name,
            "aggregation": agg,
            "test_layer": best_layer,
        }

        gl_results["H1_formal_primacy"] = test_h1_formal_primacy(abs_results, best_layer)
        gl_results["H2_telic_distinctiveness"] = test_h2_telic_distinctiveness(disc_results_last)
        gl_results["H3_gl_structure"] = test_h3_gl_structure(abs_results, best_layer)
        gl_results["H4_ordering_effects"] = test_h4_ordering_effects(abs_results, best_layer)

        # Accumulation curve
        gl_results["accumulation_curve"] = compute_accumulation_curve(abs_results, best_layer)

        utils.save_json(model_out / "gl_hypothesis_tests.json", gl_results)

        # Print summary
        print(f"\n  ── GL Hypothesis Test Summary (layer {best_layer}) ──")
        h1 = gl_results["H1_formal_primacy"]
        print(f"  H1 Formal Primacy: best single role = {h1['best_single_role']} "
              f"(formal_is_best={h1['formal_is_best']})")
        print(f"     Role means: {h1['mean_similarity_by_role']}")

        h2 = gl_results["H2_telic_distinctiveness"]
        print(f"  H2 Telic Distinctiveness: best discriminative role = {h2['best_discriminative_role']} "
              f"(telic_is_best={h2['telic_is_best']})")

        h3 = gl_results["H3_gl_structure"]
        for test_name, test_data in h3.get("tests", {}).items():
            print(f"  H3 {test_name}: effect={test_data['effect_size']:.4f}, p={test_data['p_value']:.4f}")

        h4 = gl_results["H4_ordering_effects"]
        if h4.get("f_first_vs_others_step1"):
            fvo = h4["f_first_vs_others_step1"]
            print(f"  H4 F-first vs others (step 1): "
                  f"F={fvo['mean_f_first']:.4f}, other={fvo['mean_other_first']:.4f}, "
                  f"p={fvo['p_value']:.4f}")

        del repr_data, ref_data

    # 4. Cross-architecture comparison (H5)
    if len(all_model_abs_results) >= 2:
        print(f"\n{'='*60}")
        print("Cross-architecture comparison (H5)")
        print(f"{'='*60}")

        # Use middle layer
        first_model = list(all_model_abs_results.keys())[0]
        first_meta = utils.load_json(
            config.DATA / "representations" / first_model.replace("/", "--") / "metadata.json"
        )
        mid_layer = first_meta["n_layers"] // 2

        h5 = test_h5_cross_architecture(all_model_abs_results, mid_layer)
        utils.save_json(output_dir / "h5_cross_architecture.json", h5)

        if "pearson_r" in h5:
            print(f"  Pearson r = {h5['pearson_r']:.4f} (p={h5['pearson_p']:.6f})")
            print(f"  Spearman ρ = {h5['spearman_rho']:.4f} (p={h5['spearman_p']:.6f})")
            print(f"  Role rankings by model: {json.dumps(h5['role_rankings_by_model'], indent=4)}")

    print(f"\nAll results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Similarity analysis + GL hypothesis tests")
    parser.add_argument("--agg", choices=["mean", "first", "last"], default="mean",
                        help="Token aggregation method")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: sampled)")
    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    run_analysis(agg=args.agg, target_layers=layers)


if __name__ == "__main__":
    main()
