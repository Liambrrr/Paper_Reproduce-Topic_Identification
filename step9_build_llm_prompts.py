"""
Step 9 - Prompt and decoding settings (constant across models)

Reads per-topic artifacts from Step 10:
  results/llm/topics_A.jsonl, topics_B.jsonl, topics_C.jsonl, topics_D.jsonl

For each topic, creates a chat-style prompt:
  - System message: strict JSON requirement; ≤4-word label; no extra output.
  - User message: includes subset label (A/B/C/D), topic_id, topic size, top-10 keywords,
                  and representative snippets (default: up to 10; can be reduced via CLI).

Outputs (one JSONL per subset):
  results/llm/prompts_A.jsonl
  results/llm/prompts_B.jsonl
  results/llm/prompts_C.jsonl
  results/llm/prompts_D.jsonl

Each JSONL line contains:
{
  "meta": {
    "group": "A",
    "topic_id": 12,
    "topic_size": 143,
    "keywords": [...],
    "snippets": [...]
  },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "decoding": {"temperature": 0, "top_p": 1, "max_new_tokens": 32},
  "schema": {"type": "object", "required": ["label","rationale"], "properties": {...}}
}

Usage:
  python step9_build_llm_prompts.py \
    --topics-dir results/llm \
    --outdir results/llm \
    --only-groups A B C D \
    --max-snippets 10 \
    --max-snippet-chars 300 \
    --include-variant-note 0

Notes
- Deterministic decoding: temperature=0, top_p=1, max_new_tokens=32 (constant across models).
- Label must be ≤4 words; rationale is a short explanation (1-2 sentences max).
- The prompt is adapted from Kozlowski et al. (2024) to our MOOC forum context.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

from tqdm import tqdm


CANONICAL_GROUPS = ["A", "B", "C", "D"]


def log(msg: str) -> None:
    print(msg, flush=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_system_prompt() -> str:
    # Strict, minimal, deterministic, and JSON-only instructions.
    return (
        "You label topics for MOOC forum posts. "
        "Return a STRICT JSON object with exactly the keys 'label' and 'rationale'. "
        "Constraints:\n"
        "- 'label' must be a concise, human-readable topic name of at most 4 words.\n"
        "- Do not include quotes, numbering, brackets, or punctuation in the label.\n"
        "- 'rationale' is 1-2 short sentences explaining why the label fits.\n"
        "- Respond with ONLY the JSON object. No extra text before or after.\n"
    )


def format_user_prompt(
    group: str,
    topic_id: int,
    topic_size: int,
    keywords: List[str],
    snippets: List[str],
    include_variant_note: bool = False,
) -> str:
    # Template adapted from Kozlowski et al. (2024), rewritten for our MOOC urgent-post context.
    # We explicitly include subset label, top-10 keywords, and representative snippets.
    base = []
    base.append(f"Subset: {group} (urgent MOOC forum posts)")
    base.append(f"Topic ID: {topic_id} | Topic Size: {topic_size}")
    base.append("Top keywords (ordered):")
    base.append(" - " + ", ".join(keywords) if keywords else " - (none)")

    if snippets:
        base.append("Representative snippets (truncated):")
        for i, s in enumerate(snippets, 1):
            # Keep each snippet on a single line for robustness.
            one_line = " ".join(s.splitlines())
            base.append(f" [{i}] {one_line}")
    else:
        base.append("Representative snippets: (none)")

    base.append(
        "\nTask: Based on the keywords and snippets, provide a concise human-readable label (<=4 words) that best describes this topic."
    )
    if include_variant_note:
        base.append("If multiple plausible labels exist, choose the most general, course-agnostic phrasing.")

    # Require strict JSON output:
    base.append(
        'Return JSON ONLY in the form: {"label": "< <=4 words>", "rationale": "<short reason>"}'
    )

    return "\n".join(base)


def build_prompt_records_for_group(
    group: str,
    topics_path: str,
    outdir: str,
    max_snippets: int,
    max_snippet_chars: int,
    include_variant_note: bool,
) -> str:
    if not os.path.isfile(topics_path):
        log(f"  ✗ Missing topic artifact file: {topics_path}")
        return ""

    # Load per-topic artifacts (from Step 10)
    topics = read_jsonl(topics_path)

    # Prepare outputs
    out_path = os.path.join(outdir, f"prompts_{group}.jsonl")
    rows: List[Dict[str, Any]] = []

    # Constant decoding settings (deterministic)
    decoding = {"temperature": 0, "top_p": 1, "max_new_tokens": 32}

    # JSON schema (informational, helpful if your runner validates)
    schema = {
        "type": "object",
        "required": ["label", "rationale"],
        "properties": {
            "label": {
                "type": "string",
                "description": "Concise topic name (≤ 4 words), human-readable, no punctuation."
            },
            "rationale": {
                "type": "string",
                "description": "122 sentences explaining why the label fits the keywords/snippets."
            }
        },
        "additionalProperties": False
    }

    system_prompt = build_system_prompt()

    for rec in tqdm(topics, desc=f"[Group {group}] Building prompts", leave=False):
        topic_id = int(rec.get("topic_id"))
        keywords: List[str] = list(rec.get("keywords", []))[:10]  # keep top-10
        all_snips: List[str] = list(rec.get("representative_snippets", []))
        # Enforce limits again in case Step 10 was configured differently
        snips = [s[:max_snippet_chars] for s in all_snips][:max_snippets]
        size = int(rec.get("size", 0))

        user_prompt = format_user_prompt(
            group=group,
            topic_id=topic_id,
            topic_size=size,
            keywords=keywords,
            snippets=snips,
            include_variant_note=include_variant_note,
        )

        row = {
            "meta": {
                "group": group,
                "topic_id": topic_id,
                "topic_size": size,
                "keywords": keywords,
                "snippets": snips,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "decoding": decoding,
            "schema": schema,
        }
        rows.append(row)

    write_jsonl(out_path, rows)
    log(f"  ✓ Wrote {len(rows)} prompts to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Step 9: Build strict, deterministic LLM prompts per subset.")
    parser.add_argument("--topics-dir", default="results/llm",
                        help="Directory containing topics_{A|B|C|D}.jsonl from Step 10.")
    parser.add_argument("--outdir", default="results/llm",
                        help="Directory to write prompts_{A|B|C|D}.jsonl.")
    parser.add_argument("--only-groups", nargs="*", default=CANONICAL_GROUPS,
                        help="Subset of groups to process (default: A B C D).")
    parser.add_argument("--max-snippets", type=int, default=10,
                        help="Max representative snippets to include per topic.")
    parser.add_argument("--max-snippet-chars", type=int, default=300,
                        help="Max characters per snippet (hard cap).")
    parser.add_argument("--include-variant-note", type=int, default=0,
                        help="If 1, adds a note to prefer general, course-agnostic labels.")
    args = parser.parse_args()

    log("Step 9 - Build LLM prompts (deterministic, strict JSON schema)")
    for g in args.only_groups:
        topics_path = os.path.join(args.topics_dir, f"topics_{g}.jsonl")
        log(f"\n[Group {g}] Topics file: {topics_path}")
        build_prompt_records_for_group(
            group=g,
            topics_path=topics_path,
            outdir=args.outdir,
            max_snippets=args.max_snippets,
            max_snippet_chars=args.max_snippet_chars,
            include_variant_note=bool(args.include_variant_note),
        )

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())