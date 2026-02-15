#!/usr/bin/env python3
"""Step 5: Generate phonotactically legal nonce words.

Generates pseudowords from English phonotactic patterns (onset + nucleus + coda),
validates against NLTK words corpus and common names, checks Llama tokenizer
for 2-4 subword tokens.

Independent of scripts 01-03; can run in parallel.

Target: 50-100 validated nonce words.
"""

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import config
import utils

# English phonotactic components (orthographic approximations)
ONSETS = [
    "", "b", "bl", "br", "ch", "cl", "cr", "d", "dr", "f", "fl", "fr",
    "g", "gl", "gr", "h", "j", "k", "kl", "kr", "l", "m", "n", "p",
    "pl", "pr", "qu", "r", "s", "sc", "sh", "sk", "sl", "sm", "sn",
    "sp", "spl", "spr", "st", "str", "sw", "t", "th", "tr", "tw",
    "v", "w", "wh", "wr", "z",
]

NUCLEI = [
    "a", "e", "i", "o", "u",
    "ai", "ay", "ea", "ee", "ie", "oa", "oo", "ou", "ow",
    "au", "oi", "oy",
]

CODAS = [
    "", "b", "ck", "d", "f", "ft", "g", "k", "l", "lk", "lm", "lp",
    "lt", "m", "mp", "n", "nd", "ng", "nk", "nt", "p", "pt", "r",
    "rb", "rd", "rf", "rk", "rl", "rm", "rn", "rp", "rs", "rt",
    "s", "sh", "sk", "sp", "ss", "st", "t", "th", "x", "z",
]


def generate_syllable() -> str:
    """Generate a single syllable from onset + nucleus + coda."""
    onset = random.choice(ONSETS)
    nucleus = random.choice(NUCLEI)
    coda = random.choice(CODAS)
    return onset + nucleus + coda


def generate_nonce_word() -> str:
    """Generate a 1-3 syllable nonce word.

    Biases toward 2 syllables (sweet spot for 2-4 subword tokens).
    3 syllables used occasionally to ensure enough candidates.
    """
    n_syllables = random.choice([1, 2, 2, 2, 2, 3])
    syllables = [generate_syllable() for _ in range(n_syllables)]
    word = "".join(syllables)
    return word


def main():
    # Load English words for exclusion
    from nltk.corpus import words as nltk_words
    english_words = set(w.lower() for w in nltk_words.words())

    # Add common names to exclusion set
    try:
        from nltk.corpus import names
        name_set = set(n.lower() for n in names.words())
    except Exception:
        name_set = set()

    exclude = english_words | name_set

    # Load Llama tokenizer
    print("Loading Llama tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.LLAMA_MODEL)
        has_tokenizer = True
        print("Llama tokenizer loaded.")
    except Exception as e:
        print(f"WARNING: Could not load Llama tokenizer ({e})")
        print("Will skip tokenizer validation.")
        has_tokenizer = False

    # Generate nonce words
    target = config.TARGET_NONCE_WORDS
    validated = []
    seen = set()
    attempts = 0
    max_attempts = target * 500  # generous budget

    print(f"Generating {target} nonce words...")
    pbar = tqdm(total=target, desc="Valid nonce words")

    while len(validated) < target and attempts < max_attempts:
        attempts += 1
        word = generate_nonce_word()

        # Basic filters
        if len(word) < 3 or len(word) > 12:
            continue
        if word in seen:
            continue
        if word.lower() in exclude:
            continue
        if not word.isalpha():
            continue

        seen.add(word)

        # Tokenizer check
        if has_tokenizer:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            n_tokens = len(tokens)
            if n_tokens < config.MIN_NONCE_TOKENS or n_tokens > config.MAX_NONCE_TOKENS:
                continue
            token_info = {
                "token_ids": tokens,
                "token_count": n_tokens,
                "token_strings": tokenizer.convert_ids_to_tokens(tokens),
            }
        else:
            token_info = {"token_ids": [], "token_count": -1, "token_strings": []}

        validated.append({
            "nonce_word": word,
            **token_info,
        })
        pbar.update(1)

    pbar.close()
    print(f"\nGenerated {len(validated)} valid nonce words from {attempts} attempts")

    # Save
    out_path = config.NONCE / "nonce_words.jsonl"
    utils.write_jsonl(out_path, validated)
    print(f"Saved to {out_path}")

    # Also save a simple list
    word_list = [v["nonce_word"] for v in validated]
    utils.save_json(config.NONCE / "nonce_word_list.json", word_list)

    # Stats
    if has_tokenizer:
        token_counts = [v["token_count"] for v in validated]
        print(f"Token count distribution: min={min(token_counts)}, max={max(token_counts)}, "
              f"mean={sum(token_counts)/len(token_counts):.1f}")


if __name__ == "__main__":
    main()
