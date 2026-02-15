#!/usr/bin/env python3
"""Reference representations for real target words.

Generates baseline representations by running real concept words through
the model in neutral contexts. Stub — full implementation post-dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils


def generate_reference_representations(concept_list_path: Path, output_dir: Path):
    """Generate reference hidden states for real concept words.

    For each concept:
    1. Create neutral context sentences (e.g., "The [concept] is here.")
    2. Extract hidden states at concept word token positions
    3. Average across contexts for a stable reference representation

    Args:
        concept_list_path: Path to concept_list.json
        output_dir: Directory to save reference representations
    """
    # TODO: Generate neutral contexts for each real concept word
    # TODO: Extract hidden states and average across contexts
    # TODO: Save as concept → layer → representation mapping
    raise NotImplementedError("Full implementation post-dataset generation")


if __name__ == "__main__":
    generate_reference_representations(
        config.ONTOLOGY / "concept_list.json",
        config.DATA / "references",
    )
