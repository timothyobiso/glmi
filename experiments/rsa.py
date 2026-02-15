#!/usr/bin/env python3
"""Cosine similarity and discriminative evaluation across conditions.

Two evaluation modes:
1. Absolute: cosine sim between nonce word representation and target concept reference
2. Discriminative: rank target among target + N distractors by cosine sim

Both are computed across layers, conditions, orderings, and models.
Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def compute_absolute_similarity(representations_dir: Path, references_dir: Path, output_dir: Path):
    """Compute cosine similarity between nonce and target reference.

    For each model × layer × condition:
        cos_sim(nonce_repr, target_reference)

    Aggregated by:
    - Accumulation step (1-role through 4-role)
    - Ordering (24 permutations — does order matter?)
    - Individual qualia roles (which role contributes most?)
    """
    # TODO: Load representations and references per model
    # TODO: Compute cosine similarity matrices
    # TODO: Statistical tests (paired t-tests across concepts, permutation tests)
    # TODO: Generate convergence plots (sim vs. accumulation step)
    raise NotImplementedError


def compute_discriminative_evaluation(representations_dir: Path, references_dir: Path, output_dir: Path):
    """Discriminative evaluation: rank target concept among distractors.

    For each stimulus:
    1. Compute cosine sim to target concept reference
    2. Compute cosine sim to each of the N distractor concept references
    3. Rank target among the N+1 candidates

    Metrics:
    - Mean Reciprocal Rank (MRR)
    - Hits@1 (target is top-ranked)
    - Hits@3
    - Mean rank

    Aggregated by condition type, accumulation step, ordering, and model.
    """
    # TODO: Load distractor_map.json for distractor assignments
    # TODO: For each model × layer × stimulus:
    #   - Compute sim to target + N distractors
    #   - Rank target
    # TODO: Aggregate metrics by condition
    # TODO: Plot MRR/Hits@1 vs accumulation step per model
    raise NotImplementedError


def compare_across_models(output_dir: Path):
    """Compare convergence patterns across model families.

    Tests whether the qualia accumulation effect is consistent across
    Llama and Mistral architectures, or architecture-specific.
    """
    # TODO: Load per-model results
    # TODO: Correlation of per-concept convergence patterns across models
    # TODO: Test for model × condition interactions
    raise NotImplementedError


if __name__ == "__main__":
    out = config.DATA / "analysis"
    compute_absolute_similarity(
        config.DATA / "representations",
        config.DATA / "references",
        out,
    )
    compute_discriminative_evaluation(
        config.DATA / "representations",
        config.DATA / "references",
        out,
    )
    compare_across_models(out)
