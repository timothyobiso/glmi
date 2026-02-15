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

Statistical corrections:
- Bonferroni correction across 5 primary hypotheses (α_corrected = 0.01)
- Effect sizes reported as Cohen's d (paired) or rank-biserial r
- All p-values reported as both raw and corrected
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
N_PRIMARY_HYPOTHESES = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cohens_d_paired(a, b):
    """Cohen's d for paired samples."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def bonferroni(p: float, n_tests: int = N_PRIMARY_HYPOTHESES) -> float:
    return min(p * n_tests, 1.0)


# ── Per-layer data loading ────────────────────────────────────────────────────

def load_repr_layer(model_name: str, layer: int, agg: str = "mean") -> np.ndarray:
    """Load a single layer's representations as memmap (no full load into RAM)."""
    model_slug = model_name.replace("/", "--")
    repr_dir = config.DATA / "representations" / model_slug
    meta = utils.load_json(repr_dir / "metadata.json")
    path = repr_dir / f"layer_{layer}_{agg}.dat"
    return np.memmap(path, dtype=np.float16, mode="r",
                     shape=(meta["n_stimuli"], meta["hidden_dim"]))


def load_repr_meta(model_name: str) -> dict:
    model_slug = model_name.replace("/", "--")
    return utils.load_json(config.DATA / "representations" / model_slug / "metadata.json")


def load_ref_data(model_name: str) -> tuple[np.ndarray, dict]:
    model_slug = model_name.replace("/", "--")
    ref_dir = config.DATA / "references" / model_slug
    ref_data = np.load(ref_dir / "references.npy")
    ref_meta = utils.load_json(ref_dir / "metadata.json")
    return ref_data, ref_meta


# ── Absolute similarity (one layer at a time) ────────────────────────────────

def compute_absolute_similarity_layer(
    stimuli: list[dict],
    repr_layer: np.ndarray,
    repr_meta: dict,
    ref_data: np.ndarray,
    ref_meta: dict,
    layer: int,
) -> list[dict]:
    """Compute cosine similarity at a single layer."""
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    concept_index = {c: i for i, c in enumerate(ref_meta["concepts"])}
    valid_mask = repr_meta["valid_mask"]
    ref_valid = ref_meta["valid_mask"]

    results = []
    for stim in stimuli:
        sid = stim["stimulus_id"]
        concept = stim["concept"]
        si = stim_index.get(sid)
        ci = concept_index.get(concept)

        if si is None or ci is None:
            continue
        if not valid_mask[si] or not ref_valid[ci]:
            continue

        sim = cosine_sim(repr_layer[si], ref_data[ci, layer])
        results.append({
            "stimulus_id": sid,
            "concept": concept,
            "condition_type": stim.get("condition_type", ""),
            "condition_label": stim.get("condition_label", ""),
            "ordering": stim.get("ordering", ""),
            "qualia_roles": stim.get("qualia_roles", []),
            "n_roles": len(stim.get("qualia_roles", [])),
            "layer": layer,
            "similarity": sim,
        })

    return results


# ── Discriminative evaluation ─────────────────────────────────────────────────

def compute_discriminative(
    stimuli: list[dict],
    repr_layer: np.ndarray,
    repr_meta: dict,
    ref_data: np.ndarray,
    ref_meta: dict,
    distractor_map: dict,
    layer: int,
) -> list[dict]:
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    concept_index = {c: i for i, c in enumerate(ref_meta["concepts"])}
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

        dist_info = distractor_map.get(concept, {})
        distractor_concepts = dist_info.get("all", [])
        distractor_indices = [concept_index[d] for d in distractor_concepts
                              if d in concept_index and ref_valid[concept_index[d]]]
        if not distractor_indices:
            continue

        stim_repr = repr_layer[si]
        target_sim = cosine_sim(stim_repr, ref_data[ci, layer])
        distractor_sims = [cosine_sim(stim_repr, ref_data[di, layer])
                           for di in distractor_indices]

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

def test_h1_formal_primacy(abs_results: list[dict]) -> dict:
    """H1: Formal role yields highest single-role similarity.

    Paired by concept: for each concept, compare F similarity vs T/A/C similarity.
    """
    # Group by (concept, role) → similarity
    concept_role_sim = defaultdict(dict)
    for r in abs_results:
        if r["condition_type"] == "single_qualia":
            concept_role_sim[r["concept"]][r["condition_label"]] = r["similarity"]

    means = defaultdict(list)
    for concept, roles in concept_role_sim.items():
        for role, sim in roles.items():
            means[role].append(sim)

    role_means = {r: float(np.mean(v)) for r, v in means.items() if v}
    best_role = max(role_means, key=role_means.get) if role_means else None

    # Paired Wilcoxon: F vs each other role, matched by concept
    pairwise = {}
    for other_role in ["T", "A", "C"]:
        paired_f, paired_other = [], []
        for concept, roles in concept_role_sim.items():
            if "F" in roles and other_role in roles:
                paired_f.append(roles["F"])
                paired_other.append(roles[other_role])

        if len(paired_f) >= 10:
            stat, p = sp_stats.wilcoxon(paired_f, paired_other)
            d = cohens_d_paired(paired_f, paired_other)
            pairwise[f"F_vs_{other_role}"] = {
                "statistic": float(stat),
                "p_raw": float(p),
                "p_corrected": bonferroni(p),
                "cohens_d": d,
                "n_pairs": len(paired_f),
                "mean_F": float(np.mean(paired_f)),
                "mean_other": float(np.mean(paired_other)),
            }

    return {
        "hypothesis": "H1: Formal Primacy",
        "mean_similarity_by_role": role_means,
        "best_single_role": best_role,
        "formal_is_best": best_role == "F",
        "pairwise_tests": pairwise,
    }


def test_h2_telic_distinctiveness(disc_results: list[dict]) -> dict:
    """H2: Telic is the most discriminative single role (highest MRR)."""
    # Group by (concept, role) → mrr (paired)
    concept_role_mrr = defaultdict(dict)
    for r in disc_results:
        if r["condition_type"] == "single_qualia":
            concept_role_mrr[r["concept"]][r["condition_label"]] = r["mrr"]

    role_mrrs = defaultdict(list)
    for concept, roles in concept_role_mrr.items():
        for role, mrr in roles.items():
            role_mrrs[role].append(mrr)

    mean_mrr = {r: float(np.mean(v)) for r, v in role_mrrs.items() if v}
    best_role = max(mean_mrr, key=mean_mrr.get) if mean_mrr else None

    # Paired Wilcoxon: T vs each other role
    pairwise = {}
    for other_role in ["F", "A", "C"]:
        paired_t, paired_other = [], []
        for concept, roles in concept_role_mrr.items():
            if "T" in roles and other_role in roles:
                paired_t.append(roles["T"])
                paired_other.append(roles[other_role])

        if len(paired_t) >= 10:
            stat, p = sp_stats.wilcoxon(paired_t, paired_other)
            d = cohens_d_paired(paired_t, paired_other)
            pairwise[f"T_vs_{other_role}"] = {
                "statistic": float(stat),
                "p_raw": float(p),
                "p_corrected": bonferroni(p),
                "cohens_d": d,
                "n_pairs": len(paired_t),
            }

    return {
        "hypothesis": "H2: Telic Distinctiveness",
        "mean_mrr_by_role": mean_mrr,
        "best_discriminative_role": best_role,
        "telic_is_best": best_role == "T",
        "pairwise_tests": pairwise,
    }


def test_h3_gl_structure(abs_results: list[dict]) -> dict:
    """H3: GL-structured sentences outperform info-matched controls.

    Paired by concept. Compares 4-role accumulation (averaged over orderings)
    vs info_flat and info_swapped controls.
    """
    structured_by_concept = defaultdict(list)
    flat_by_concept = defaultdict(list)
    swapped_by_concept = defaultdict(list)

    for r in abs_results:
        concept = r["concept"]
        ct = r["condition_type"]
        cl = r["condition_label"]

        if ct == "accumulation" and r["n_roles"] == 4:
            structured_by_concept[concept].append(r["similarity"])
        elif ct == "control_info_matched" and cl == "info_flat":
            flat_by_concept[concept].append(r["similarity"])
        elif ct == "control_info_matched" and cl == "info_swapped":
            swapped_by_concept[concept].append(r["similarity"])

    structured_avg = {c: float(np.mean(v)) for c, v in structured_by_concept.items()}
    flat_avg = {c: float(np.mean(v)) for c, v in flat_by_concept.items()}
    swapped_avg = {c: float(np.mean(v)) for c, v in swapped_by_concept.items()}

    tests = {}

    # Structured vs Flat (paired by concept)
    shared = sorted(set(structured_avg) & set(flat_avg))
    if len(shared) >= 10:
        a = [structured_avg[c] for c in shared]
        b = [flat_avg[c] for c in shared]
        stat, p = sp_stats.wilcoxon(a, b)
        tests["structured_vs_flat"] = {
            "statistic": float(stat),
            "p_raw": float(p),
            "p_corrected": bonferroni(p),
            "cohens_d": cohens_d_paired(a, b),
            "mean_structured": float(np.mean(a)),
            "mean_flat": float(np.mean(b)),
            "effect_size": float(np.mean(a) - np.mean(b)),
            "n_concepts": len(shared),
        }

    # Structured vs Swapped (paired by concept)
    shared = sorted(set(structured_avg) & set(swapped_avg))
    if len(shared) >= 10:
        a = [structured_avg[c] for c in shared]
        b = [swapped_avg[c] for c in shared]
        stat, p = sp_stats.wilcoxon(a, b)
        tests["structured_vs_swapped"] = {
            "statistic": float(stat),
            "p_raw": float(p),
            "p_corrected": bonferroni(p),
            "cohens_d": cohens_d_paired(a, b),
            "mean_structured": float(np.mean(a)),
            "mean_swapped": float(np.mean(b)),
            "effect_size": float(np.mean(a) - np.mean(b)),
            "n_concepts": len(shared),
        }

    return {
        "hypothesis": "H3: GL Structure Matters",
        "tests": tests,
    }


def test_h4_ordering_effects(abs_results: list[dict]) -> dict:
    """H4: Role ordering significantly affects convergence speed.

    Tests whether orderings starting with Formal converge faster.
    """
    by_first_role_and_step = defaultdict(list)
    for r in abs_results:
        if r["condition_type"] != "accumulation":
            continue
        ordering = r.get("ordering", "")
        if not ordering:
            continue
        first_role = ordering[0]
        n_roles = r["n_roles"]
        by_first_role_and_step[(first_role, n_roles)].append(r["similarity"])

    convergence = {}
    for (first_role, n_roles), sims in by_first_role_and_step.items():
        convergence.setdefault(first_role, {})[n_roles] = float(np.mean(sims))

    # Kruskal-Wallis at each step
    step_tests = {}
    for n_roles in [1, 2, 3, 4]:
        groups, labels = [], []
        for first_role in ["T", "A", "C", "F"]:
            sims = by_first_role_and_step.get((first_role, n_roles), [])
            if len(sims) >= 5:
                groups.append(sims)
                labels.append(first_role)
        if len(groups) >= 2:
            stat, p = sp_stats.kruskal(*groups)
            # Eta-squared effect size for Kruskal-Wallis
            N = sum(len(g) for g in groups)
            k = len(groups)
            eta_sq = (stat - k + 1) / (N - k) if N > k else 0.0
            step_tests[f"step_{n_roles}"] = {
                "statistic": float(stat),
                "p_raw": float(p),
                "p_corrected": bonferroni(p),
                "eta_squared": float(max(0, eta_sq)),
                "groups": labels,
                "group_means": {l: float(np.mean(g)) for l, g in zip(labels, groups)},
            }

    # F-first vs all others at step 1
    f_first_1 = by_first_role_and_step.get(("F", 1), [])
    other_1 = []
    for role in ["T", "A", "C"]:
        other_1.extend(by_first_role_and_step.get((role, 1), []))

    f_vs_other = {}
    if len(f_first_1) >= 5 and len(other_1) >= 5:
        stat, p = sp_stats.mannwhitneyu(f_first_1, other_1, alternative="greater")
        # Rank-biserial r
        n1, n2 = len(f_first_1), len(other_1)
        r_rb = 1 - (2 * stat) / (n1 * n2)
        f_vs_other = {
            "statistic": float(stat),
            "p_raw": float(p),
            "p_corrected": bonferroni(p),
            "rank_biserial_r": float(r_rb),
            "mean_f_first": float(np.mean(f_first_1)),
            "mean_other_first": float(np.mean(other_1)),
        }

    return {
        "hypothesis": "H4: Ordering Effects",
        "convergence_by_first_role": convergence,
        "step_tests": step_tests,
        "f_first_vs_others_step1": f_vs_other,
    }


def test_h5_cross_architecture(
    model_abs: dict[str, list[dict]],
) -> dict:
    """H5: GL effects are consistent across Llama and Mistral."""
    models = list(model_abs.keys())
    if len(models) < 2:
        return {"hypothesis": "H5: Cross-Architecture", "skip": "Need ≥2 models"}

    m1, m2 = models[0], models[1]

    def concept_means(results):
        sims = defaultdict(list)
        for r in results:
            if r["condition_type"] == "accumulation" and r["n_roles"] == 4:
                sims[r["concept"]].append(r["similarity"])
        return {c: float(np.mean(v)) for c, v in sims.items()}

    means1 = concept_means(model_abs[m1])
    means2 = concept_means(model_abs[m2])
    shared = sorted(set(means1) & set(means2))

    if len(shared) < 10:
        return {"hypothesis": "H5: Cross-Architecture", "skip": f"Only {len(shared)} shared concepts"}

    vals1 = [means1[c] for c in shared]
    vals2 = [means2[c] for c in shared]

    r, p_r = sp_stats.pearsonr(vals1, vals2)
    rho, p_rho = sp_stats.spearmanr(vals1, vals2)

    # Role rankings per model
    role_rankings = {}
    for model_name, results in model_abs.items():
        role_sims = defaultdict(list)
        for item in results:
            if item["condition_type"] == "single_qualia":
                role_sims[item["condition_label"]].append(item["similarity"])
        role_rankings[model_name] = {
            role: float(np.mean(sims)) for role, sims in role_sims.items()
        }

    return {
        "hypothesis": "H5: Cross-Architecture Consistency",
        "models": [m1, m2],
        "n_shared_concepts": len(shared),
        "pearson_r": float(r),
        "pearson_p_raw": float(p_r),
        "pearson_p_corrected": bonferroni(p_r),
        "spearman_rho": float(rho),
        "spearman_p_raw": float(p_rho),
        "spearman_p_corrected": bonferroni(p_rho),
        "role_rankings_by_model": role_rankings,
    }


# ── Accumulation curve ────────────────────────────────────────────────────────

def compute_accumulation_curve(abs_results: list[dict]) -> dict:
    by_step = defaultdict(list)
    by_step_and_first = defaultdict(lambda: defaultdict(list))

    for r in abs_results:
        if r["condition_type"] != "accumulation":
            continue
        n = r["n_roles"]
        ordering = r.get("ordering", "")
        first = ordering[0] if ordering else "?"
        by_step[n].append(r["similarity"])
        by_step_and_first[n][first].append(r["similarity"])

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

def run_analysis(agg: str = "mean", target_layers: list[int] | None = None,
                 model_filter: str | None = None):
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

    models = config.EXPERIMENT_MODELS
    if model_filter:
        models = [m for m in models if model_filter in m]
        if not models:
            models = [model_filter]

    # Collect per-model results at the analysis layer for H5
    h5_model_results = {}

    for model_name in models:
        model_slug = model_name.replace("/", "--")
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name}")
        print(f"{'='*60}")

        try:
            repr_meta = load_repr_meta(model_name)
            ref_data, ref_meta = load_ref_data(model_name)
        except FileNotFoundError as e:
            print(f"  Skipping — data not found: {e}")
            continue

        n_layers = repr_meta["n_layers"]
        if target_layers is None:
            layers = sorted(set([0, n_layers // 4, n_layers // 2,
                                 3 * n_layers // 4, n_layers - 1]))
        else:
            layers = target_layers

        analysis_layer = layers[len(layers) // 2]  # middle layer for hypothesis tests
        print(f"  Layers: {layers}, analysis layer: {analysis_layer}")

        model_out = output_dir / model_slug
        model_out.mkdir(parents=True, exist_ok=True)

        # Process one layer at a time
        all_abs_results = []
        for layer in tqdm(layers, desc="Layers"):
            repr_layer = load_repr_layer(model_name, layer, agg)

            # Absolute similarity
            abs_results = compute_absolute_similarity_layer(
                all_stimuli, repr_layer, repr_meta, ref_data, ref_meta, layer,
            )
            all_abs_results.extend(abs_results)
            utils.write_jsonl(model_out / f"absolute_layer{layer}.jsonl", abs_results)

            # Discriminative evaluation
            disc_results = compute_discriminative(
                all_stimuli, repr_layer, repr_meta, ref_data, ref_meta,
                distractor_map, layer,
            )
            if disc_results:
                utils.write_jsonl(model_out / f"discriminative_layer{layer}.jsonl", disc_results)
                mean_mrr = np.mean([r["mrr"] for r in disc_results])
                hits1 = np.mean([r["hits_at_1"] for r in disc_results])
                hits3 = np.mean([r["hits_at_3"] for r in disc_results])
                print(f"    Layer {layer}: MRR={mean_mrr:.4f}, "
                      f"Hits@1={hits1:.4f}, Hits@3={hits3:.4f}")

            del repr_layer

        # GL hypothesis tests at analysis layer
        print(f"\n  Running GL hypothesis tests (layer {analysis_layer})...")
        analysis_abs = [r for r in all_abs_results if r["layer"] == analysis_layer]

        # Load discriminative results at analysis layer for H2
        repr_layer = load_repr_layer(model_name, analysis_layer, agg)
        analysis_disc = compute_discriminative(
            all_stimuli, repr_layer, repr_meta, ref_data, ref_meta,
            distractor_map, analysis_layer,
        )
        del repr_layer

        gl_results = {
            "model": model_name,
            "aggregation": agg,
            "analysis_layer": analysis_layer,
            "n_primary_hypotheses": N_PRIMARY_HYPOTHESES,
            "correction": "Bonferroni",
        }

        gl_results["H1_formal_primacy"] = test_h1_formal_primacy(analysis_abs)
        gl_results["H2_telic_distinctiveness"] = test_h2_telic_distinctiveness(analysis_disc)
        gl_results["H3_gl_structure"] = test_h3_gl_structure(analysis_abs)
        gl_results["H4_ordering_effects"] = test_h4_ordering_effects(analysis_abs)
        gl_results["accumulation_curve"] = compute_accumulation_curve(analysis_abs)

        utils.save_json(model_out / "gl_hypothesis_tests.json", gl_results)

        # Store for H5
        h5_model_results[model_name] = analysis_abs

        # Print summary
        print(f"\n  ── GL Hypothesis Test Summary ──")
        h1 = gl_results["H1_formal_primacy"]
        print(f"  H1 Formal Primacy: best={h1['best_single_role']} "
              f"(formal_is_best={h1['formal_is_best']})")
        print(f"     Means: {h1['mean_similarity_by_role']}")
        for name, t in h1["pairwise_tests"].items():
            print(f"     {name}: d={t['cohens_d']:.3f}, p_corr={t['p_corrected']:.4f}")

        h2 = gl_results["H2_telic_distinctiveness"]
        print(f"  H2 Telic Distinctiveness: best={h2['best_discriminative_role']} "
              f"(telic_is_best={h2['telic_is_best']})")
        print(f"     MRR: {h2['mean_mrr_by_role']}")

        h3 = gl_results["H3_gl_structure"]
        for name, t in h3.get("tests", {}).items():
            print(f"  H3 {name}: d={t['cohens_d']:.3f}, "
                  f"effect={t['effect_size']:.4f}, p_corr={t['p_corrected']:.4f}")

        h4 = gl_results["H4_ordering_effects"]
        if h4.get("f_first_vs_others_step1"):
            fvo = h4["f_first_vs_others_step1"]
            print(f"  H4 F-first vs others: r_rb={fvo['rank_biserial_r']:.3f}, "
                  f"p_corr={fvo['p_corrected']:.4f}")

        del ref_data

    # H5: Cross-architecture
    if len(h5_model_results) >= 2:
        print(f"\n{'='*60}")
        print("H5: Cross-architecture comparison")
        print(f"{'='*60}")

        h5 = test_h5_cross_architecture(h5_model_results)
        utils.save_json(output_dir / "h5_cross_architecture.json", h5)

        if "pearson_r" in h5:
            print(f"  Pearson r={h5['pearson_r']:.4f} (p_corr={h5['pearson_p_corrected']:.6f})")
            print(f"  Spearman ρ={h5['spearman_rho']:.4f} (p_corr={h5['spearman_p_corrected']:.6f})")
            print(f"  Role rankings: {json.dumps(h5['role_rankings_by_model'], indent=4)}")

    print(f"\nAll results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Similarity analysis + GL hypothesis tests")
    parser.add_argument("--agg", choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: sampled)")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (default: all EXPERIMENT_MODELS)")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    run_analysis(agg=args.agg, target_layers=layers, model_filter=args.model)


if __name__ == "__main__":
    main()
