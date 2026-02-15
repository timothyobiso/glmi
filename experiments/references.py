#!/usr/bin/env python3
"""Reference representations for real target words.

Generates baseline representations by running real concept words through
each experiment model in neutral contexts.
Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils

# Neutral contexts to average over for stable reference representations
NEUTRAL_TEMPLATES = [
    "The {word} is here.",
    "I see a {word}.",
    "There is a {word} on the table.",
    "She picked up the {word}.",
    "He looked at the {word}.",
]


def generate_reference_representations(concept_list_path: Path, output_dir: Path):
    """Generate reference hidden states for real concept words.

    For each model in config.EXPERIMENT_MODELS:
      For each concept:
        1. Create neutral context sentences from NEUTRAL_TEMPLATES
        2. Extract hidden states at concept word token positions
        3. Average across contexts for a stable reference representation

    Also generates reference representations for distractor concepts
    (needed for discriminative evaluation).

    Args:
        concept_list_path: Path to concept_list.json
        output_dir: Directory to save reference representations
    """
    # TODO: For each model in config.EXPERIMENT_MODELS:
    #   - Generate references for ALL concepts (targets + distractors)
    #   - Average across NEUTRAL_TEMPLATES for stability
    #   - Save as {model_name}/references.npz
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    generate_reference_representations(
        config.ONTOLOGY / "concept_list.json",
        config.DATA / "references",
    )
