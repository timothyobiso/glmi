#!/usr/bin/env python3
"""Step 7: Naturalize templated sentences using Claude.

Uses Claude Batch API for cost efficiency, with async concurrent fallback.
Rewrites sentences to sound natural while preserving nonce word and qualia info.

Depends on: 06.
Cost: ~$3-4, Runtime: ~1-2 hours for batch.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
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


def create_batch_requests(records: list[dict]) -> list[dict]:
    """Create batch API request objects."""
    requests = []
    for i, rec in enumerate(records):
        prompt = NATURALIZE_PROMPT.format(
            nonce_word=rec["nonce_word"],
            qualia_role=rec["qualia_role"],
            qualia_desc=QUALIA_DESCRIPTIONS[rec["qualia_role"]],
            sentence=rec["sentence"],
        )
        requests.append({
            "custom_id": str(i),
            "params": {
                "model": config.CLAUDE_MODEL,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}],
            },
        })
    return requests


def run_batch(client: anthropic.Anthropic, records: list[dict]) -> list[dict]:
    """Submit batch and poll for completion."""
    batch_requests = create_batch_requests(records)

    # Write batch input file
    batch_input_path = config.STIMULI / "batch_input.jsonl"
    utils.write_jsonl(batch_input_path, batch_requests)
    print(f"Wrote {len(batch_requests)} batch requests to {batch_input_path}")

    # Submit batch
    print("Submitting batch to Claude API...")
    with open(batch_input_path) as f:
        batch = client.messages.batches.create(
            requests=[json.loads(line) for line in f],
        )

    print(f"Batch ID: {batch.id}")
    print(f"Batch status: {batch.processing_status}")

    # Poll for completion
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        status = batch.processing_status
        counts = batch.request_counts
        print(f"  Status: {status} | Succeeded: {counts.succeeded}/{len(batch_requests)} | "
              f"Failed: {counts.errored} | Processing: {counts.processing}")

        if status == "ended":
            break
        time.sleep(30)

    # Retrieve results
    results = []
    for result in client.messages.batches.results(batch.id):
        idx = int(result.custom_id)
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            rec = records[idx].copy()
            rec["naturalized"] = text
            rec["naturalize_status"] = "success"
        else:
            rec = records[idx].copy()
            rec["naturalized"] = rec["sentence"]  # fallback to original
            rec["naturalize_status"] = "failed"
        results.append(rec)

    return results


async def run_concurrent(records: list[dict]) -> list[dict]:
    """Fallback: use async concurrent requests."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(config.CLAUDE_MAX_CONCURRENT)
    results = [None] * len(records)

    async def process_one(i: int, rec: dict):
        async with semaphore:
            prompt = NATURALIZE_PROMPT.format(
                nonce_word=rec["nonce_word"],
                qualia_role=rec["qualia_role"],
                qualia_desc=QUALIA_DESCRIPTIONS[rec["qualia_role"]],
                sentence=rec["sentence"],
            )
            try:
                response = await client.messages.create(
                    model=config.CLAUDE_MODEL,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                result = rec.copy()
                result["naturalized"] = text
                result["naturalize_status"] = "success"
            except Exception as e:
                result = rec.copy()
                result["naturalized"] = rec["sentence"]
                result["naturalize_status"] = f"error: {e}"
            results[i] = result

    tasks = [process_one(i, rec) for i, rec in enumerate(records)]

    # Progress tracking
    done = 0
    total = len(tasks)
    batch_size = 100
    for start in range(0, total, batch_size):
        batch = tasks[start:start + batch_size]
        await asyncio.gather(*batch)
        done += len(batch)
        print(f"  Progress: {done}/{total}")

    return [r for r in results if r is not None]


def main():
    # Load templated sentences
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

    # Try batch API first, fall back to concurrent
    use_batch = "--concurrent" not in sys.argv

    try:
        if use_batch:
            print("\nUsing Claude Batch API...")
            client = anthropic.Anthropic()
            results = run_batch(client, records)
        else:
            raise ValueError("User requested concurrent mode")
    except Exception as e:
        print(f"\nBatch API failed ({e}), falling back to async concurrent...")
        results = asyncio.run(run_concurrent(records))

    # Merge with existing results
    all_results = existing + results

    utils.write_jsonl(out_path, all_results)
    print(f"\nSaved {len(all_results)} naturalized sentences to {out_path}")

    # Stats
    success = sum(1 for r in results if r.get("naturalize_status") == "success")
    print(f"  Success: {success}/{len(results)}")
    if results:
        failed = [r for r in results if r.get("naturalize_status") != "success"]
        if failed:
            print(f"  Failed: {len(failed)}")


if __name__ == "__main__":
    main()
