#!/usr/bin/env python3
"""Layer-wise logistic regression probes.

Trains probes at each layer to predict the target concept from nonce word
representations, for each experiment model.

Uses GroupKFold with concept as the group variable to prevent data leakage —
all stimuli for a given concept stay in the same fold, so the probe must
generalize to unseen concepts.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils


def load_layer(model_name: str, layer: int, agg: str = "mean") -> np.ndarray:
    """Load a single layer's representations via memmap."""
    model_slug = model_name.replace("/", "--")
    repr_dir = config.DATA / "representations" / model_slug
    meta = utils.load_json(repr_dir / "metadata.json")
    path = repr_dir / f"layer_{layer}_{agg}.dat"
    return np.memmap(path, dtype=np.float16, mode="r",
                     shape=(meta["n_stimuli"], meta["hidden_dim"]))


def train_probes_for_model(
    model_name: str,
    stimuli: list[dict],
    output_dir: Path,
    agg: str = "mean",
    n_folds: int = 5,
):
    """Train layer-wise probes for one model.

    Uses GroupKFold: all stimuli for a concept go in the same fold.
    The probe must generalize to unseen concepts.
    """
    model_slug = model_name.replace("/", "--")
    repr_dir = config.DATA / "representations" / model_slug
    model_out = output_dir / model_slug
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training probes for {model_name}")
    print(f"{'='*60}")

    repr_meta = utils.load_json(repr_dir / "metadata.json")
    valid_mask = np.array(repr_meta["valid_mask"])
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    n_layers = repr_meta["n_layers"]

    # Filter to accumulation conditions with all 4 roles
    probe_stimuli = [
        s for s in stimuli
        if s["condition_type"] == "accumulation"
        and len(s.get("qualia_roles", [])) == 4
        and s["stimulus_id"] in stim_index
        and valid_mask[stim_index[s["stimulus_id"]]]
    ]

    if len(probe_stimuli) < 50:
        print(f"  Only {len(probe_stimuli)} valid 4-role stimuli — skipping")
        return

    indices = np.array([stim_index[s["stimulus_id"]] for s in probe_stimuli])
    concepts = [s["concept"] for s in probe_stimuli]

    le = LabelEncoder()
    y = le.fit_transform(concepts)
    groups = np.array(concepts)  # GroupKFold groups = concepts
    n_classes = len(le.classes_)

    print(f"  {len(probe_stimuli)} stimuli, {n_classes} concepts")

    if n_classes < n_folds:
        print(f"  Too few concepts ({n_classes}) for {n_folds}-fold — skipping")
        return

    # Sample layers
    layers = sorted(set([0, n_layers // 4, n_layers // 2,
                         3 * n_layers // 4, n_layers - 1]))

    gkf = GroupKFold(n_splits=min(n_folds, n_classes))

    results = []
    for layer in tqdm(layers, desc="Probing layers"):
        repr_layer = load_layer(model_name, layer, agg)
        X = repr_layer[indices].astype(np.float32)
        del repr_layer

        fold_accs = []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        results.append({
            "layer": layer,
            "accuracy": mean_acc,
            "std": std_acc,
            "n_folds": len(fold_accs),
            "n_classes": n_classes,
            "n_samples": len(y),
            "chance": 1.0 / n_classes,
            "split": "GroupKFold_by_concept",
        })
        print(f"    Layer {layer}: accuracy={mean_acc:.4f} ± {std_acc:.4f} "
              f"(chance={1/n_classes:.4f})")

    # Probe by n_roles at middle layer
    print("\n  Probing by number of roles (middle layer, GroupKFold)...")
    mid_layer = n_layers // 2

    for n_roles in [1, 2, 3, 4]:
        role_stimuli = [
            s for s in stimuli
            if s["condition_type"] in ("accumulation", "single_qualia",
                                       "combination_2", "combination_3")
            and len(s.get("qualia_roles", [])) == n_roles
            and s["stimulus_id"] in stim_index
            and valid_mask[stim_index[s["stimulus_id"]]]
        ]
        if len(role_stimuli) < 50:
            continue

        r_indices = np.array([stim_index[s["stimulus_id"]] for s in role_stimuli])
        r_concepts = [s["concept"] for s in role_stimuli]
        r_le = LabelEncoder()
        r_y = r_le.fit_transform(r_concepts)
        r_groups = np.array(r_concepts)
        r_n_classes = len(r_le.classes_)

        if r_n_classes < n_folds:
            continue

        repr_layer = load_layer(model_name, mid_layer, agg)
        X = repr_layer[r_indices].astype(np.float32)
        del repr_layer

        r_gkf = GroupKFold(n_splits=min(n_folds, r_n_classes))
        fold_accs = []
        for train_idx, test_idx in r_gkf.split(X, r_y, groups=r_groups):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
            clf.fit(X[train_idx], r_y[train_idx])
            fold_accs.append(clf.score(X[test_idx], r_y[test_idx]))

        mean_acc = float(np.mean(fold_accs))
        results.append({
            "layer": mid_layer,
            "n_roles": n_roles,
            "accuracy": mean_acc,
            "std": float(np.std(fold_accs)),
            "n_classes": r_n_classes,
            "n_samples": len(r_y),
            "chance": 1.0 / r_n_classes,
            "split": "GroupKFold_by_concept",
        })
        print(f"    {n_roles} roles: accuracy={mean_acc:.4f} "
              f"(chance={1/r_n_classes:.4f}, n={len(r_y)})")

    utils.save_json(model_out / "probing_results.json", results)
    print(f"  Saved to {model_out / 'probing_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Layer-wise probing classifiers")
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
