#!/usr/bin/env python3
"""Reference representations for real target words.

Generates baseline representations by running real concept words through
each experiment model in multiple neutral contexts, then averaging.
Uses character offset mapping for robust word detection (BPE-safe).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import utils

NEUTRAL_TEMPLATES = [
    "The {word} is here.",
    "I see a {word}.",
    "There is a {word} on the table.",
    "She picked up the {word}.",
    "He looked at the {word}.",
]


def find_word_positions_by_offsets(
    text: str,
    word: str,
    offsets: list[tuple[int, int]],
) -> list[int]:
    """Find token positions of a word using character offset mapping."""
    text_lower = text.lower()
    word_lower = word.lower()

    idx = text_lower.find(word_lower)
    if idx == -1:
        return []

    char_start, char_end = idx, idx + len(word)

    positions = []
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if tok_start < char_end and tok_end > char_start:
            positions.append(tok_idx)

    return positions


def find_word_positions_fallback(
    input_ids: list[int],
    word_ids: list[int],
) -> list[int]:
    """Fallback: find word by matching token ID subsequence."""
    n = len(word_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i : i + n] == word_ids:
            return list(range(i, i + n))
    return []


def generate_references_for_model(
    model_name: str,
    concepts: list[str],
    output_dir: Path,
):
    """Generate reference representations for all concepts from one model.

    For each concept, runs it through NEUTRAL_TEMPLATES and averages the
    hidden states across contexts for a stable reference.
    Saves as single .npy file (small: ~500 concepts fits easily in memory).
    """
    print(f"\n{'='*60}")
    print(f"Generating references from {model_name}")
    print(f"{'='*60}")

    model_slug = model_name.replace("/", "--")
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_offsets = tokenizer.is_fast

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size

    references = np.zeros((len(concepts), n_layers, hidden_dim), dtype=np.float16)
    valid_mask = np.zeros(len(concepts), dtype=bool)

    for c_idx, concept in enumerate(tqdm(concepts, desc="Concepts")):
        sentences = [t.format(word=concept) for t in NEUTRAL_TEMPLATES]

        tokenize_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        if use_offsets:
            tokenize_kwargs["return_offsets_mapping"] = True

        encodings = tokenizer(sentences, **tokenize_kwargs)

        offset_mappings = None
        if use_offsets and "offset_mapping" in encodings:
            offset_mappings = encodings.pop("offset_mapping")

        encodings = encodings.to(model.device)

        with torch.no_grad():
            outputs = model(**encodings)

        hidden_states = outputs.hidden_states
        template_reprs = []

        for t_idx in range(len(sentences)):
            if offset_mappings is not None:
                offsets = offset_mappings[t_idx].tolist()
                positions = find_word_positions_by_offsets(sentences[t_idx], concept, offsets)
            else:
                word_ids = tokenizer.encode(concept, add_special_tokens=False)
                input_ids = encodings["input_ids"][t_idx].tolist()
                positions = find_word_positions_fallback(input_ids, word_ids)

            if not positions:
                continue

            pos_tensor = torch.tensor(positions, device=hidden_states[0].device)
            layer_reprs = np.zeros((n_layers, hidden_dim), dtype=np.float32)
            for layer_idx in range(n_layers):
                layer_hidden = hidden_states[layer_idx][t_idx]
                word_hidden = layer_hidden[pos_tensor]
                layer_reprs[layer_idx] = word_hidden.mean(dim=0).cpu().numpy()

            template_reprs.append(layer_reprs)

        if template_reprs:
            avg_repr = np.mean(template_reprs, axis=0).astype(np.float16)
            references[c_idx] = avg_repr
            valid_mask[c_idx] = True

        del outputs, hidden_states
        torch.cuda.empty_cache()

    np.save(model_dir / "references.npy", references)
    metadata = {
        "concepts": concepts,
        "valid_mask": valid_mask.tolist(),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "model_name": model_name,
        "n_templates": len(NEUTRAL_TEMPLATES),
        "templates": NEUTRAL_TEMPLATES,
    }
    utils.save_json(model_dir / "metadata.json", metadata)

    n_valid = int(valid_mask.sum())
    print(f"  Generated references for {n_valid}/{len(concepts)} concepts")
    print(f"  Saved to {model_dir}")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Generate reference representations")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    output_dir = config.DATA / "references"

    ontology_path = config.ONTOLOGY / "concept_qualia_merged.json"
    if not ontology_path.exists():
        print(f"ERROR: {ontology_path} not found. Run the pipeline first.")
        sys.exit(1)

    ontology = utils.load_json(ontology_path)
    concepts = sorted(ontology.keys())
    print(f"Loaded {len(concepts)} concepts")

    models = config.EXPERIMENT_MODELS
    if args.model:
        models = [m for m in models if args.model in m]
        if not models:
            models = [args.model]

    for model_name in models:
        generate_references_for_model(model_name, concepts, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
