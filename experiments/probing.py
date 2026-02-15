#!/usr/bin/env python3
"""Layer-wise probing experiments.

Two probes:

A) Qualia Role Probe — predicts which qualia role (T/A/C/F) a single-qualia
   stimulus encodes. Uses GroupKFold by concept so the probe must generalize
   to unseen concepts. Tests whether qualia roles have distinct,
   concept-independent representational signatures.

B) Concept Generalization Probe — learns a linear projection from nonce word
   representations to reference concept representations via Ridge regression,
   then retrieves the correct concept on held-out concepts. Uses GroupKFold
   by concept. Reports retrieval MRR and Hits@1 on unseen concepts.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils

ROLES = ["telic", "agentive", "constitutive", "formal"]


def load_layer(model_name: str, layer: int, agg: str = "mean") -> np.ndarray:
    model_slug = model_name.replace("/", "--")
    repr_dir = config.DATA / "representations" / model_slug
    meta = utils.load_json(repr_dir / "metadata.json")
    path = repr_dir / f"layer_{layer}_{agg}.dat"
    return np.memmap(path, dtype=np.float16, mode="r",
                     shape=(meta["n_stimuli"], meta["hidden_dim"]))


# ── Probe A: Qualia Role Classification ──────────────────────────────────────

def probe_qualia_role(
    model_name: str,
    stimuli: list[dict],
    repr_meta: dict,
    layer: int,
    agg: str = "mean",
    n_folds: int = 5,
) -> dict | None:
    """Predict qualia role (T/A/C/F) from single-qualia stimulus representation.

    GroupKFold by concept — must generalize to unseen concepts.
    """
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    valid_mask = np.array(repr_meta["valid_mask"])

    probe_stimuli = [
        s for s in stimuli
        if s.get("condition_type") == "single_qualia"
        and len(s.get("qualia_roles", [])) == 1
        and s["stimulus_id"] in stim_index
        and valid_mask[stim_index[s["stimulus_id"]]]
    ]

    if len(probe_stimuli) < 50:
        print(f"    Only {len(probe_stimuli)} single-qualia stimuli — skipping role probe")
        return None

    indices = np.array([stim_index[s["stimulus_id"]] for s in probe_stimuli])
    roles = [s["qualia_roles"][0] for s in probe_stimuli]
    concepts = [s["concept"] for s in probe_stimuli]

    le = LabelEncoder()
    y = le.fit_transform(roles)

    concept_le = LabelEncoder()
    groups = concept_le.fit_transform(concepts)

    n_unique_concepts = len(concept_le.classes_)
    actual_folds = min(n_folds, n_unique_concepts)
    if actual_folds < 2:
        return None

    gkf = GroupKFold(n_splits=actual_folds)

    repr_layer = load_layer(model_name, layer, agg)
    X = repr_layer[indices].astype(np.float32)
    del repr_layer

    fold_accs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        fold_accs.append(clf.score(X[test_idx], y[test_idx]))

    return {
        "probe": "qualia_role",
        "layer": layer,
        "accuracy": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs)),
        "n_folds": len(fold_accs),
        "n_classes": len(le.classes_),
        "classes": le.classes_.tolist(),
        "n_samples": len(y),
        "n_concepts": n_unique_concepts,
        "chance": 1.0 / len(le.classes_),
        "split": "GroupKFold_by_concept",
    }


# ── Probe B: Concept Generalization ──────────────────────────────────────────

def probe_concept_generalization(
    model_name: str,
    stimuli: list[dict],
    repr_meta: dict,
    ref_data: np.ndarray,
    ref_meta: dict,
    layer: int,
    agg: str = "mean",
    n_folds: int = 5,
    condition_filter: str | None = None,
    n_roles_filter: int | None = None,
) -> dict | None:
    """Learn linear projection from nonce representation to reference representation.

    Trains Ridge regression (nonce_repr → ref_repr), tests on held-out concepts.
    For each test stimulus, ranks all test concepts by cosine similarity of
    projected representation to reference. Reports MRR and Hits@1.
    """
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    valid_mask = np.array(repr_meta["valid_mask"])
    concept_index = {c: i for i, c in enumerate(ref_meta["concepts"])}
    ref_valid = ref_meta["valid_mask"]

    probe_stimuli = [
        s for s in stimuli
        if s["stimulus_id"] in stim_index
        and valid_mask[stim_index[s["stimulus_id"]]]
        and s["concept"] in concept_index
        and ref_valid[concept_index[s["concept"]]]
    ]

    # Apply optional filters
    if condition_filter:
        probe_stimuli = [s for s in probe_stimuli
                         if s.get("condition_type") == condition_filter]
    if n_roles_filter is not None:
        probe_stimuli = [s for s in probe_stimuli
                         if len(s.get("qualia_roles", [])) == n_roles_filter]

    if len(probe_stimuli) < 50:
        return None

    indices = np.array([stim_index[s["stimulus_id"]] for s in probe_stimuli])
    concepts = [s["concept"] for s in probe_stimuli]

    concept_le = LabelEncoder()
    groups = concept_le.fit_transform(concepts)
    n_unique = len(concept_le.classes_)

    actual_folds = min(n_folds, n_unique)
    if actual_folds < 2:
        return None

    gkf = GroupKFold(n_splits=actual_folds)

    repr_layer = load_layer(model_name, layer, agg)
    X = repr_layer[indices].astype(np.float32)
    del repr_layer

    # Reference vectors for each stimulus's concept at this layer
    Y_ref = np.array([
        ref_data[concept_index[c], layer].astype(np.float32)
        for c in concepts
    ])

    all_mrrs, all_hits1, all_hits3 = [], [], []

    for train_idx, test_idx in gkf.split(X, groups, groups):
        reg = Ridge(alpha=1.0)
        reg.fit(X[train_idx], Y_ref[train_idx])

        X_test_proj = reg.predict(X[test_idx])

        # Unique test concepts and their references
        test_concepts = [concepts[i] for i in test_idx]
        unique_test = sorted(set(test_concepts))
        test_ref_vectors = np.array([
            ref_data[concept_index[c], layer].astype(np.float32)
            for c in unique_test
        ])
        test_concept_to_idx = {c: i for i, c in enumerate(unique_test)}

        # Batch cosine similarity
        X_norm = X_test_proj / (np.linalg.norm(X_test_proj, axis=1, keepdims=True) + 1e-8)
        R_norm = test_ref_vectors / (np.linalg.norm(test_ref_vectors, axis=1, keepdims=True) + 1e-8)
        sims = X_norm @ R_norm.T

        for i, ti in enumerate(test_idx):
            correct_idx = test_concept_to_idx[concepts[ti]]
            rank = 1 + int((sims[i] > sims[i, correct_idx]).sum())
            all_mrrs.append(1.0 / rank)
            all_hits1.append(int(rank == 1))
            all_hits3.append(int(rank <= 3))

    avg_test_concepts = n_unique / actual_folds
    chance_mrr = sum(1.0 / k for k in range(1, int(avg_test_concepts) + 1)) / avg_test_concepts

    result = {
        "probe": "concept_generalization",
        "layer": layer,
        "mrr": float(np.mean(all_mrrs)),
        "hits_at_1": float(np.mean(all_hits1)),
        "hits_at_3": float(np.mean(all_hits3)),
        "n_folds": actual_folds,
        "n_samples": len(probe_stimuli),
        "n_concepts": n_unique,
        "avg_test_concepts_per_fold": float(avg_test_concepts),
        "chance_mrr": float(chance_mrr),
        "split": "GroupKFold_by_concept",
    }

    if condition_filter:
        result["condition_filter"] = condition_filter
    if n_roles_filter is not None:
        result["n_roles"] = n_roles_filter

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def train_probes_for_model(
    model_name: str,
    stimuli: list[dict],
    output_dir: Path,
    agg: str = "mean",
    n_folds: int = 5,
):
    model_slug = model_name.replace("/", "--")
    repr_dir = config.DATA / "representations" / model_slug
    model_out = output_dir / model_slug
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training probes for {model_name}")
    print(f"{'='*60}")

    repr_meta = utils.load_json(repr_dir / "metadata.json")
    n_layers = repr_meta["n_layers"]

    # Load reference data
    ref_dir = config.DATA / "references" / model_slug
    ref_data = np.load(ref_dir / "references.npy")
    ref_meta = utils.load_json(ref_dir / "metadata.json")

    layers = sorted(set([0, n_layers // 4, n_layers // 2,
                         3 * n_layers // 4, n_layers - 1]))
    mid_layer = n_layers // 2

    results = []

    # ── Probe A: Qualia Role ──
    print("\n  Probe A: Qualia Role Classification (GroupKFold by concept)")
    for layer in tqdm(layers, desc="  Role probe layers"):
        r = probe_qualia_role(model_name, stimuli, repr_meta, layer, agg, n_folds)
        if r:
            results.append(r)
            print(f"    Layer {layer}: accuracy={r['accuracy']:.4f} ± {r['std']:.4f} "
                  f"(chance={r['chance']:.4f}, {r['n_concepts']} concepts held out)")

    # ── Probe B: Concept Generalization ──
    print("\n  Probe B: Concept Generalization (GroupKFold by concept)")

    # B1: across layers with 4-role accumulation stimuli
    for layer in tqdm(layers, desc="  Concept probe layers"):
        r = probe_concept_generalization(
            model_name, stimuli, repr_meta, ref_data, ref_meta,
            layer, agg, n_folds, condition_filter="accumulation", n_roles_filter=4,
        )
        if r:
            results.append(r)
            print(f"    Layer {layer}: MRR={r['mrr']:.4f}, Hits@1={r['hits_at_1']:.4f} "
                  f"(chance MRR={r['chance_mrr']:.4f}, {r['n_concepts']} concepts)")

    # B2: by n_roles at middle layer
    print(f"\n  Probe B by n_roles (layer {mid_layer}):")
    for n_roles in [1, 2, 3, 4]:
        # Accumulation stimuli for n_roles
        r = probe_concept_generalization(
            model_name, stimuli, repr_meta, ref_data, ref_meta,
            mid_layer, agg, n_folds, n_roles_filter=n_roles,
        )
        if r:
            results.append(r)
            print(f"    {n_roles} roles: MRR={r['mrr']:.4f}, Hits@1={r['hits_at_1']:.4f} "
                  f"(n={r['n_samples']}, {r['n_concepts']} concepts)")

    # B3: controls at middle layer
    print(f"\n  Probe B on controls (layer {mid_layer}):")
    for ct in ["control_info_matched", "control_bare", "control_scrambled"]:
        ctrl_stimuli = [s for s in stimuli if s.get("condition_type") == ct]
        if not ctrl_stimuli:
            continue
        r = probe_concept_generalization(
            model_name, ctrl_stimuli, repr_meta, ref_data, ref_meta,
            mid_layer, agg, n_folds,
        )
        if r:
            r["condition_filter"] = ct
            results.append(r)
            print(f"    {ct}: MRR={r['mrr']:.4f}, Hits@1={r['hits_at_1']:.4f} "
                  f"(n={r['n_samples']})")

    del ref_data

    utils.save_json(model_out / "probing_results.json", results)
    print(f"\n  Saved to {model_out / 'probing_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Layer-wise probing experiments")
    parser.add_argument("--agg", choices=["mean", "first", "last"], default="mean")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    stimuli = utils.read_jsonl(config.STIMULI / "stimuli_final.jsonl")
    controls_path = config.CONTROLS / "controls.jsonl"
    if controls_path.exists():
        stimuli.extend(utils.read_jsonl(controls_path))

    output_dir = config.DATA / "probing_results"

    models = config.EXPERIMENT_MODELS
    if args.model:
        models = [m for m in models if args.model in m]
        if not models:
            models = [args.model]

    for model_name in models:
        train_probes_for_model(model_name, stimuli, output_dir, agg=args.agg)


if __name__ == "__main__":
    main()
