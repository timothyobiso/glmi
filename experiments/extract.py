#!/usr/bin/env python3
"""Hidden state extraction from experiment models.

Extracts representations at each layer for nonce word tokens across all
experimental conditions, for each model in EXPERIMENT_MODELS.
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

AGGREGATION_METHODS = ["mean", "first", "last"]


def find_nonce_token_positions(
    input_ids: list[int],
    nonce_ids: list[int],
) -> list[int]:
    """Find all token positions where the nonce word subword tokens appear.

    Uses sliding window matching against the nonce word's token IDs.
    Returns positions of ALL tokens belonging to the nonce word (across all
    occurrences in the sequence).
    """
    positions = []
    n = len(nonce_ids)
    for i in range(len(input_ids) - n + 1):
        if input_ids[i : i + n] == nonce_ids:
            positions.extend(range(i, i + n))
    return positions


def extract_for_model(
    model_name: str,
    stimuli: list[dict],
    output_dir: Path,
    batch_size: int = 8,
    max_length: int = 512,
):
    """Extract hidden states for all stimuli from one model.

    Saves per-layer representations as numpy arrays, with separate files
    for each aggregation method (mean, first, last over nonce subword tokens).
    Also saves a metadata file mapping row index → stimulus_id.
    """
    print(f"\n{'='*60}")
    print(f"Extracting from {model_name}")
    print(f"{'='*60}")

    model_slug = model_name.replace("/", "--")
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size

    # Pre-allocate storage
    n_stimuli = len(stimuli)
    # {agg_method: {layer_idx: np.array of shape (n_stimuli, hidden_dim)}}
    representations = {
        agg: np.zeros((n_stimuli, n_layers, hidden_dim), dtype=np.float16)
        for agg in AGGREGATION_METHODS
    }
    valid_mask = np.zeros(n_stimuli, dtype=bool)
    stimulus_ids = []

    # Process in batches
    for batch_start in tqdm(range(0, n_stimuli, batch_size), desc="Extracting"):
        batch = stimuli[batch_start : batch_start + batch_size]
        texts = [s["stimulus"] for s in batch]

        # Tokenize batch
        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        # Find nonce word positions for each item in batch
        batch_nonce_positions = []
        for i, stim in enumerate(batch):
            nonce = stim["nonce_word"]
            # Tokenize nonce word alone (without special tokens)
            nonce_ids = tokenizer.encode(nonce, add_special_tokens=False)
            input_ids = encodings["input_ids"][i].tolist()
            positions = find_nonce_token_positions(input_ids, nonce_ids)
            batch_nonce_positions.append(positions)

        # Forward pass
        with torch.no_grad():
            outputs = model(**encodings)

        # hidden_states: tuple of (n_layers,) each (batch, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states

        # Extract representations
        for i, stim in enumerate(batch):
            idx = batch_start + i
            stimulus_ids.append(stim["stimulus_id"])
            positions = batch_nonce_positions[i]

            if not positions:
                # No nonce tokens found — skip
                continue

            valid_mask[idx] = True

            for layer_idx in range(n_layers):
                layer_hidden = hidden_states[layer_idx][i]  # (seq_len, hidden_dim)

                pos_tensor = torch.tensor(positions, device=layer_hidden.device)
                nonce_hidden = layer_hidden[pos_tensor]  # (n_positions, hidden_dim)

                # Aggregate
                mean_repr = nonce_hidden.mean(dim=0).cpu().numpy().astype(np.float16)
                first_repr = nonce_hidden[0].cpu().numpy().astype(np.float16)
                last_repr = nonce_hidden[-1].cpu().numpy().astype(np.float16)

                representations["mean"][idx, layer_idx] = mean_repr
                representations["first"][idx, layer_idx] = first_repr
                representations["last"][idx, layer_idx] = last_repr

        # Free GPU memory
        del outputs, hidden_states
        torch.cuda.empty_cache()

    # Save representations per aggregation method
    for agg in AGGREGATION_METHODS:
        np.save(model_dir / f"representations_{agg}.npy", representations[agg])

    # Save metadata
    metadata = {
        "stimulus_ids": stimulus_ids,
        "valid_mask": valid_mask.tolist(),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "model_name": model_name,
        "aggregation_methods": AGGREGATION_METHODS,
    }
    utils.save_json(model_dir / "metadata.json", metadata)

    n_valid = int(valid_mask.sum())
    print(f"  Extracted {n_valid}/{n_stimuli} stimuli ({n_stimuli - n_valid} skipped — nonce not found)")
    print(f"  Saved to {model_dir}")

    # Free model
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from experiment models")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (default: all EXPERIMENT_MODELS)")
    args = parser.parse_args()

    stimuli_path = config.STIMULI / "stimuli_final.jsonl"
    controls_path = config.CONTROLS / "controls.jsonl"
    output_dir = config.DATA / "representations"

    # Load all stimuli + controls
    stimuli = utils.read_jsonl(stimuli_path)
    if controls_path.exists():
        controls = utils.read_jsonl(controls_path)
        stimuli.extend(controls)

    print(f"Total stimuli to process: {len(stimuli)}")

    models = config.EXPERIMENT_MODELS
    if args.model:
        models = [m for m in models if args.model in m]
        if not models:
            models = [args.model]

    for model_name in models:
        extract_for_model(model_name, stimuli, output_dir, batch_size=args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
