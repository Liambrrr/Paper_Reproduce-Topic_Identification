"""
Step 9 - Prompt and decoding settings (constant across models)

Reads per-topic artifacts from Step 8:
  results/llm/topics_A.jsonl, topics_B.jsonl, topics_C.jsonl, topics_D.jsonl

For each topic, creates a chat-style prompt (selectable via --prompt_option):
  - System message: strict JSON requirement; ≤4-word label; rationale ≤10 words; JSON-only.
  - User message: includes subset label (A/B/C/D), topic_id, topic size, top-10 keywords,
                  and representative snippets (default: up to 10; can be reduced via CLI).

Outputs (one JSONL per subset, file name includes option):
  results/llm/prompts_A_op{1|2|3}.jsonl
  results/llm/prompts_B_op{1|2|3}.jsonl
  results/llm/prompts_C_op{1|2|3}.jsonl
  results/llm/prompts_D_op{1|2|3}.jsonl

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
  python step9_build_llm_prompts.py --prompt_option 1
  python step9_build_llm_prompts.py --prompt_option 2
  python step9_build_llm_prompts.py --prompt_option 3
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Callable, Tuple

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


# -----------------------
# Prompt Variations
# -----------------------

def build_system_prompt_op1() -> str:
    # Original (cleaned and 10-word rationale)
    return (
        "You label topics for MOOC forum posts. "
        "Return a STRICT JSON object with exactly the keys 'label' and 'rationale'. "
        "Constraints:\n"
        "- 'label' must be a concise, human-readable topic name of at most 4 words.\n"
        "- Do not include quotes, numbering, brackets, or punctuation in the label.\n"
        "- 'rationale' is a short sentence explaining why the label fits (≤ 10 words).\n"
        "- Respond with ONLY the JSON object. No extra text before or after.\n"
    )


def format_user_prompt_op1(
    group: str,
    topic_id: int,
    topic_size: int,
    keywords: List[str],
    snippets: List[str],
    include_variant_note: bool = False,
) -> str:
    base = []
    base.append(f"Subset: {group} (urgent MOOC forum posts)")
    base.append(f"Topic ID: {topic_id} | Topic Size: {topic_size}")
    base.append("Top keywords (ordered):")
    base.append(" - " + ", ".join(keywords) if keywords else " - (none)")
    if snippets:
        base.append("Representative snippets (truncated):")
        for i, s in enumerate(snippets, 1):
            one_line = " ".join(s.splitlines())
            base.append(f" [{i}] {one_line}")
    else:
        base.append("Representative snippets: (none)")
    base.append(
        "\nTask: Based on the keywords and snippets, provide a concise human-readable label (<=4 words) that best describes this topic."
    )
    if include_variant_note:
        base.append("If multiple plausible labels exist, choose the most general, course-agnostic phrasing.")
    base.append('Return JSON ONLY in the form: {"label": "<≤4 words>", "rationale": "<≤10 words>"}')
    return "\n".join(base)


def build_system_prompt_op2() -> str:
    # Variation A — minimal style
    return (
        "You label topics for MOOC forum posts.\n"
        "Return ONLY this strict JSON object:\n"
        '{"label": "<≤4 words>", "rationale": "<≤10 words>"}.\n'
        "Rules:\n"
        '- "label": concise, human-readable, ≤4 words, no quotes/brackets/punctuation.\n'
        '- "rationale": ≤10 words, short reason for fit.\n'
        "No extra text before/after the JSON.\n"
    )


def format_user_prompt_op2(
    group: str,
    topic_id: int,
    topic_size: int,
    keywords: List[str],
    snippets: List[str],
    include_variant_note: bool = False,
) -> str:
    base = []
    base.append(f"Corpus: MOOC forum posts, subset {group}")
    base.append(f"Topic ID: {topic_id} | Topic size: {topic_size}")
    base.append("Keywords (ordered):")
    base.append("[ " + ", ".join(keywords) + " ]" if keywords else "[ (none) ]")
    base.append("Representative snippets (truncated):")
    if snippets:
        for i, s in enumerate(snippets, 1):
            base.append(f"[{i}] " + " ".join(s.splitlines()))
    else:
        base.append("(none)")
    base.append(
        'Task: Extract a short topic name (≤4 words) and a ≤10-word rationale.\n'
        'Output JSON ONLY: {"label": "<≤4 words>", "rationale": "<≤10 words>"}'
    )
    return "\n".join(base)


def build_system_prompt_op3() -> str:
    # Variation B — Kozlowski-style paraphrase + explicit format
    return (
        "You are labeling topics for a MOOC forum corpus.\n"
        'Output must be ONLY this JSON: {"label": "<≤4 words>", "rationale": "<≤10 words>"}.\n'
        "Constraints:\n"
        "- label: ≤4 words, general, course-agnostic, no punctuation/quotes/brackets.\n"
        "- rationale: ≤10 words, briefly why the label fits.\n"
        "No additional commentary.\n"
    )


def format_user_prompt_op3(
    group: str,
    topic_id: int,
    topic_size: int,
    keywords: List[str],
    snippets: List[str],
    include_variant_note: bool = False,
) -> str:
    base = []
    base.append("I have a corpus of MOOC forum posts with many topics.")
    base.append("This topic is described by the following keywords:")
    base.append("[ " + ", ".join(keywords) + " ]" if keywords else "[ (none) ]")
    base.append("Consider these representative snippets (truncated):")
    if snippets:
        for i, s in enumerate(snippets, 1):
            base.append(f"[{i}] " + " ".join(s.splitlines()))
    else:
        base.append("(none)")
    base.append(
        "Based on the information above, produce a short topic label (one to four words) "
        "that best represents the topic, and a ≤10-word rationale."
    )
    base.append('Format: {"label": "<≤4 words>", "rationale": "<≤10 words>"}')
    # Add context lines at the end for traceability (doesn't affect instruction)
    base.append(f"\n[Context] Subset: {group} | Topic ID: {topic_id} | Topic Size: {topic_size}")
    return "\n".join(base)


def get_prompt_builders(option: int) -> Tuple[Callable[[], str], Callable[..., str]]:
    if option == 1:
        return build_system_prompt_op1, format_user_prompt_op1
    if option == 2:
        return build_system_prompt_op2, format_user_prompt_op2
    if option == 3:
        return build_system_prompt_op3, format_user_prompt_op3
    raise ValueError("--prompt_option must be 1, 2, or 3")


# -----------------------
# Core builder
# -----------------------

def build_prompt_records_for_group(
    group: str,
    topics_path: str,
    outdir: str,
    max_snippets: int,
    max_snippet_chars: int,
    include_variant_note: bool,
    prompt_option: int,
) -> str:
    if not os.path.isfile(topics_path):
        log(f"  ✗ Missing topic artifact file: {topics_path}")
        return ""

    topics = read_jsonl(topics_path)

    out_path = os.path.join(outdir, f"prompts_{group}_op{prompt_option}.jsonl")
    rows: List[Dict[str, Any]] = []

    # Deterministic decoding
    decoding = {"temperature": 0, "top_p": 1, "max_new_tokens": 32}

    # JSON schema (informational)
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
                "description": "Short reason (≤ 10 words) why the label fits."
            }
        },
        "additionalProperties": False
    }

    build_system_prompt, format_user_prompt = get_prompt_builders(prompt_option)
    system_prompt = build_system_prompt()

    for rec in tqdm(topics, desc=f"[Group {group}] Building prompts (op{prompt_option})", leave=False):
        topic_id = int(rec.get("topic_id"))
        keywords: List[str] = list(rec.get("keywords", []))[:10]
        all_snips: List[str] = list(rec.get("representative_snippets", []))
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
                "prompt_option": prompt_option,
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
    parser = argparse.ArgumentParser(description="Step 9: Build strict, deterministic LLM prompts per subset (with variations).")
    parser.add_argument("--topics-dir", default="results/llm",
                        help="Directory containing topics_{A|B|C|D}.jsonl from Step 10.")
    parser.add_argument("--outdir", default="results/llm",
                        help="Directory to write prompts_{A|B|C|D}_op{1|2|3}.jsonl.")
    parser.add_argument("--only-groups", nargs="*", default=CANONICAL_GROUPS,
                        help="Subset of groups to process (default: A B C D).")
    parser.add_argument("--max-snippets", type=int, default=10,
                        help="Max representative snippets to include per topic.")
    parser.add_argument("--max-snippet-chars", type=int, default=300,
                        help="Max characters per snippet (hard cap).")
    parser.add_argument("--include-variant-note", type=int, default=0,
                        help="If 1, adds a note to prefer general, course-agnostic labels (op1 only).")
    parser.add_argument("--prompt_option", type=int, default=1, choices=[1, 2, 3],
                        help="Choose which prompt template to use: 1 (original), 2 (Variation A), 3 (Variation B).")
    args = parser.parse_args()

    log(f"Step 9 - Build LLM prompts (op{args.prompt_option}; deterministic, strict JSON schema)")
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
            prompt_option=args.prompt_option,
        )

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())