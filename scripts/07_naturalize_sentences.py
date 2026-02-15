#!/usr/bin/env python3
"""Step 7: Naturalize templated sentences using a local LLM.

Uses vLLM for fast batched inference with a local instruct model.
Rewrites sentences to sound natural while preserving nonce word and qualia info.

Depends on: 06.
Runtime: ~10-30 min depending on GPU and batch size.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import config
import utils

NATURALIZE_PROMPT = """Rewrite this sentence to sound natural and fluent, as if a native English speaker said it casually. Rules:
1. Keep the word "{nonce_word}" EXACTLY as-is (do not change, explain, or define it)
2. Preserve the core meaning about {qualia_role} ({qualia_desc})
3. Do NOT add information about other aspects of the {nonce_word}
4. Keep it under 25 words
5. Return ONLY the rewritten sentence, nothing else

Sentence: {sentence}"""

QUALIA_DESCRIPTIONS = {
    "telic": "what it's used for / its purpose",
    "agentive": "how it's made / how it comes into being",
    "constitutive": "what it's made of / its parts and materials",
    "formal": "what category or type it belongs to",
}


def build_chat_prompt(rec: dict) -> list[dict]:
    """Build a chat-format prompt for a single record."""
    user_msg = NATURALIZE_PROMPT.format(
        nonce_word=rec["nonce_word"],
        qualia_role=rec["qualia_role"],
        qualia_desc=QUALIA_DESCRIPTIONS[rec["qualia_role"]],
        sentence=rec["sentence"],
    )
    return [{"role": "user", "content": user_msg}]


def naturalize_with_vllm(records: list[dict]) -> list[dict]:
    """Naturalize all sentences using vLLM batched inference."""
    from vllm import LLM, SamplingParams

    print(f"Loading model {config.NATURALIZE_MODEL}...")
    llm = LLM(model=config.NATURALIZE_MODEL)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=config.NATURALIZE_TEMPERATURE,
        max_tokens=config.NATURALIZE_MAX_TOKENS,
    )

    # Build prompts using chat template
    prompts = []
    for rec in tqdm(records, desc="Building prompts"):
        messages = build_chat_prompt(rec)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Process in batches to manage memory
    batch_size = config.NATURALIZE_BATCH_SIZE
    results = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[start:start + batch_size]
        batch_records = records[start:start + batch_size]

        outputs = llm.generate(batch_prompts, sampling)

        for rec, output in zip(batch_records, outputs):
            text = output.outputs[0].text.strip()
            # Clean up: take first line, strip quotes
            text = text.split("\n")[0].strip().strip('"\'')

            result = rec.copy()
            result["naturalized"] = text
            result["naturalize_status"] = "success"
            results.append(result)

    # Free GPU memory
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def naturalize_with_transformers(records: list[dict]) -> list[dict]:
    """Fallback: use HuggingFace transformers pipeline (slower, no vLLM needed)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {config.NATURALIZE_MODEL} with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(config.NATURALIZE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.NATURALIZE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    results = []
    for rec in tqdm(records, desc="Naturalizing"):
        messages = build_chat_prompt(rec)
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=config.NATURALIZE_MAX_TOKENS,
                temperature=config.NATURALIZE_TEMPERATURE,
                do_sample=True,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = text.split("\n")[0].strip().strip('"\'')

        result = rec.copy()
        result["naturalized"] = text
        result["naturalize_status"] = "success"
        results.append(result)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    input_path = config.STIMULI / "templates_raw.jsonl"
    if not input_path.exists():
        print(f"ERROR: Run 06_generate_templates.py first to generate {input_path}")
        sys.exit(1)

    records = utils.read_jsonl(input_path)
    print(f"Loaded {len(records)} templated sentences")

    # Check for existing partial results
    out_path = config.STIMULI / "stimuli_naturalized.jsonl"
    if out_path.exists():
        existing = utils.read_jsonl(out_path)
        print(f"Found {len(existing)} existing results, will process remaining")
        done_ids = {r["template_id"] for r in existing}
        records = [r for r in records if r["template_id"] not in done_ids]
        if not records:
            print("All sentences already naturalized!")
            return
        print(f"Processing {len(records)} remaining sentences")
    else:
        existing = []

    # Try vLLM first (fast batched), fall back to transformers (slower)
    use_transformers = "--transformers" in sys.argv

    if use_transformers:
        print("\nUsing transformers pipeline...")
        results = naturalize_with_transformers(records)
    else:
        try:
            print("\nUsing vLLM for batched inference...")
            results = naturalize_with_vllm(records)
        except Exception as e:
            print(f"\nvLLM failed ({e}), falling back to transformers pipeline...")
            results = naturalize_with_transformers(records)

    # Merge with existing results
    all_results = existing + results

    utils.write_jsonl(out_path, all_results)
    print(f"\nSaved {len(all_results)} naturalized sentences to {out_path}")

    success = sum(1 for r in results if r.get("naturalize_status") == "success")
    print(f"  Success: {success}/{len(results)}")


if __name__ == "__main__":
    main()
