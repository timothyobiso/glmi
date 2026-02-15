#!/usr/bin/env python3
"""Layer-wise logistic regression probes.

Trains probes at each layer to predict the target concept from nonce word
representations, for each experiment model.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils


def train_probes_for_model(
    model_name: str,
    stimuli: list[dict],
    output_dir: Path,
    agg: str = "mean",
    n_folds: int = 5,
):
    """Train layer-wise probes for one model.

    For each layer, trains a logistic regression classifier to predict
    the target concept from the nonce word representation.
    Evaluates accuracy by condition type and number of roles.
    """
    model_slug = model_name.replace("/", "--")
    repr_dir = config.DATA / "representations" / model_slug
    model_out = output_dir / model_slug
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training probes for {model_name}")
    print(f"{'='*60}")

    repr_data = np.load(repr_dir / f"representations_{agg}.npy")
    repr_meta = utils.load_json(repr_dir / "metadata.json")
    valid_mask = np.array(repr_meta["valid_mask"])
    stim_index = {sid: i for i, sid in enumerate(repr_meta["stimulus_ids"])}
    n_layers = repr_meta["n_layers"]

    # Filter to accumulation conditions with all 4 roles (strongest signal)
    probe_stimuli = [
        s for s in stimuli
        if s["condition_type"] == "accumulation"
        and len(s.get("qualia_roles", [])) == 4
        and s["stimulus_id"] in stim_index
        and valid_mask[stim_index[s["stimulus_id"]]]
    ]

    if len(probe_stimuli) < 50:
        print(f"  Only {len(probe_stimuli)} valid 4-role stimuli — skipping probing")
        return

    # Build feature matrix and labels
    indices = [stim_index[s["stimulus_id"]] for s in probe_stimuli]
    concepts = [s["concept"] for s in probe_stimuli]

    le = LabelEncoder()
    y = le.fit_transform(concepts)
    n_classes = len(le.classes_)
    print(f"  {len(probe_stimuli)} stimuli, {n_classes} concepts")

    if n_classes < 5:
        print("  Too few concepts for probing — skipping")
        return

    # Sample layers
    layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))

    results = []
    for layer in tqdm(layers, desc="Probing layers"):
        X = repr_data[indices, layer].astype(np.float32)

        # Stratified k-fold
        skf = StratifiedKFold(n_splits=min(n_folds, min(np.bincount(y))), shuffle=True, random_state=42)
        fold_accs = []

        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
            clf.fit(X[train_idx], y[train_idx])
            acc = clf.score(X[test_idx], y[test_idx])
            fold_accs.append(acc)

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
        })
        print(f"    Layer {layer}: accuracy={mean_acc:.4f} ± {std_acc:.4f} (chance={1/n_classes:.4f})")

    # Also probe by n_roles to see if more roles → better probing
    print("\n  Probing by number of roles (at middle layer)...")
    mid_layer = n_layers // 2

    for n_roles in [1, 2, 3, 4]:
        role_stimuli = [
            s for s in stimuli
            if s["condition_type"] in ("accumulation", "single_qualia", "combination_2", "combination_3")
            and len(s.get("qualia_roles", [])) == n_roles
            and s["stimulus_id"] in stim_index
            and valid_mask[stim_index[s["stimulus_id"]]]
        ]
        if len(role_stimuli) < 50:
            continue

        r_indices = [stim_index[s["stimulus_id"]] for s in role_stimuli]
        r_concepts = [s["concept"] for s in role_stimuli]
        r_le = LabelEncoder()
        r_y = r_le.fit_transform(r_concepts)
        r_n_classes = len(r_le.classes_)

        if r_n_classes < 5 or min(np.bincount(r_y)) < 2:
            continue

        X = repr_data[r_indices, mid_layer].astype(np.float32)
        skf = StratifiedKFold(n_splits=min(n_folds, min(np.bincount(r_y))), shuffle=True, random_state=42)
        fold_accs = []
        for train_idx, test_idx in skf.split(X, r_y):
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
        })
        print(f"    {n_roles} roles: accuracy={mean_acc:.4f} (chance={1/r_n_classes:.4f}, n={len(r_y)})")

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
