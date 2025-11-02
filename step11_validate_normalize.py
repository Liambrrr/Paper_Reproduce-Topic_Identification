"""
Step 11 - Output validation and normalization (Ollama)

For each (subset, model) raw output JSONL from Step 10:
  - Parse response.text into JSON (robust).
  - Validate the schema: {"label": str, "rationale": str}, no extra keys required.
  - If invalid, re-prompt once (same messages/decoding) via Ollama, then re-parse/validate.
  - Normalize the label minimally:
      * strip whitespace
      * lowercase
      * strip leading/trailing punctuation
      * enforce <= 4 words (keep first 4 tokens)
    (Do NOT stem/lemmatize the label.)
  - Write cleaned rows to labels_{group}_{model}_clean.jsonl.
  - Write unrecoverable rows to labels_{group}_{model}_rejects.jsonl.
  - Also write a tiny summary JSON with counts.

Input files (from Step 10):
  results/llm/prompts_{A|B|C|D}.jsonl
  results/llm/labels_{group}_{model}.jsonl

Output files:
  results/llm/labels_{group}_{model}_clean.jsonl
  results/llm/labels_{group}_{model}_rejects.jsonl
  results/llm/labels_{group}_{model}_summary.json

Usage example:
  python step11_validate_normalize.py \
  --in-dir results/llm \
  --out-dir results/llm \
  --groups A B C D \
  --models "meta-llama/llama-3.1-8b-instruct" "meta-llama/llama-3.3-70b-instruct" "qwen/qwen2.5-coder-7b-instruct" "qwen/qwen3-vl-30b-a3b-instruct" \
  --reprompt 0

Notes:
  - We rely on the 'meta.topic_id' to align a label row to its prompt.
  - We only reprompt a row once; if still invalid, we drop it (log to rejects).
  - Backend: Ollama only (http://127.0.0.1:11434 by default).
"""

import argparse
import json
import os
import re
import sys
import string
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import requests

CANONICAL_GROUPS = ["A", "B", "C", "D"]

PUNCT_STRIP = "".join(sorted(set(string.punctuation)))  # used for edge-strip


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


def sanitize_model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "-").replace(" ", "").lower()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse the model text into a JSON object.
    Robust to code fences and extra prose. Returns dict or None.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()

    # Remove code fences like ```json ... ```
    fence = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence.search(s)
    if m:
        s = m.group(1).strip()

    # If multiple braces present, slice from first '{' to last '}'.
    if "{" in s and "}" in s:
        s = s[s.find("{"): s.rfind("}") + 1]

    # Try to parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def is_valid_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "label" not in obj or "rationale" not in obj:
        return False
    if not isinstance(obj["label"], str):
        return False
    if not isinstance(obj["rationale"], str):
        return False
    return True


def normalize_label(raw_label: str, max_words: int = 4) -> str:
    """
    Minimal normalization:
      - trim whitespace
      - lowercase
      - strip leading/trailing punctuation
      - enforce <= max_words tokens (split on whitespace)
    """
    if not isinstance(raw_label, str):
        return ""
    s = raw_label.strip().lower()
    # strip edges punctuation repeatedly
    s = s.strip(PUNCT_STRIP)
    # collapse inner whitespace
    s = re.sub(r"\s+", " ", s)
    if not s:
        return s
    tokens = s.split()
    if len(tokens) > max_words:
        tokens = tokens[:max_words]
    return " ".join(tokens)


def call_ollama(api_base: str,
                model: str,
                messages: List[Dict[str, str]],
                temperature: float,
                top_p: float,
                max_new_tokens: int,
                timeout_s: int = 120) -> Optional[str]:
    """Return raw text (string) or None on hard failure."""
    url = api_base.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_new_tokens
        },
        "stream": False
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "message" in data and "content" in data["message"]:
            return (data["message"]["content"] or "").strip()
        return None
    except Exception:
        return None


def index_prompts_by_topic(prompts: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Build a map topic_id -> prompt record for quick lookup during re-prompt.
    """
    idx = {}
    for rec in prompts:
        meta = rec.get("meta", {})
        tid = meta.get("topic_id")
        if isinstance(tid, int):
            idx[tid] = rec
    return idx


def process_pair(group: str,
                 model: str,
                 in_dir: str,
                 out_dir: str,
                 api_base: str,
                 reprompt: bool,
                 sleep_s: float) -> Tuple[int, int, int]:
    """
    Return: (clean_count, reprompted_count, reject_count)
    """
    tag = sanitize_model_tag(model)
    raw_path = os.path.join(in_dir, f"labels_{group}_{tag}.jsonl")
    prompts_path = os.path.join(in_dir, f"prompts_{group}.jsonl")

    if not os.path.isfile(raw_path):
        log(f"[{group} | {model}] ✗ Missing raw file: {raw_path}")
        return (0, 0, 0)
    if not os.path.isfile(prompts_path):
        log(f"[{group} | {model}] ✗ Missing prompts file: {prompts_path}")
        return (0, 0, 0)

    raw_rows = read_jsonl(raw_path)
    prompts = read_jsonl(prompts_path)
    prompt_by_tid = index_prompts_by_topic(prompts)

    clean_rows: List[Dict[str, Any]] = []
    reject_rows: List[Dict[str, Any]] = []

    reprompted = 0

    for row in tqdm(raw_rows, desc=f"[{group} | {model}] validate", leave=False):
        meta = row.get("meta", {})
        tid = meta.get("topic_id")
        cleaned = None

        def try_parse_and_normalize(text: str) -> Optional[Dict[str, Any]]:
            obj = extract_json(text)
            if not obj or not is_valid_schema(obj):
                return None
            obj_norm = {
                "label": normalize_label(obj["label"], max_words=4),
                "rationale": obj["rationale"].strip(),
            }
            return obj_norm

        # 1) Try the original response
        resp = row.get("response", {})
        text = resp.get("text", "") if isinstance(resp, dict) else ""
        cleaned = try_parse_and_normalize(text)

        # 2) If invalid and reprompt is enabled, do one re-prompt
        if cleaned is None and reprompt:
            pr = prompt_by_tid.get(tid)
            if pr:
                messages = pr.get("messages", [])
                decoding = pr.get("decoding", {}) or {}
                temperature = float(decoding.get("temperature", 0.0))
                top_p = float(decoding.get("top_p", 1.0))
                max_new_tokens = int(decoding.get("max_new_tokens", 32))

                new_text = call_ollama(
                    api_base=api_base,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                if new_text is not None:
                    reprompted += 1
                    cleaned = try_parse_and_normalize(new_text)

        if cleaned is not None and cleaned["label"]:
            out_row = {
                "meta": meta,
                "clean": cleaned,
                "raw": row.get("response", None) or {"text": text},  # keep original response snapshot
            }
            clean_rows.append(out_row)
        else:
            reject_rows.append(row)

        if sleep_s > 0:
            import time as _t
            _t.sleep(sleep_s)

    # Write outputs
    clean_path = os.path.join(out_dir, f"labels_{group}_{tag}_clean.jsonl")
    reject_path = os.path.join(out_dir, f"labels_{group}_{tag}_rejects.jsonl")
    summary_path = os.path.join(out_dir, f"labels_{group}_{tag}_summary.json")

    write_jsonl(clean_path, clean_rows)
    write_jsonl(reject_path, reject_rows)

    summary = {
        "group": group,
        "model": model,
        "input_rows": len(raw_rows),
        "clean_rows": len(clean_rows),
        "reject_rows": len(reject_rows),
        "reprompted_once": reprompted,
        "in_dir": in_dir,
        "out_dir": out_dir
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"[{group} | {model}] ✓ clean={summary['clean_rows']} reject={summary['reject_rows']} reprompted={reprompted}")
    return (summary["clean_rows"], reprompted, summary["reject_rows"])


def main():
    ap = argparse.ArgumentParser(description="Step 11: Validate & normalize LLM outputs (Ollama).")
    ap.add_argument("--in-dir", default="results/llm", help="Directory with labels_{group}_{model}.jsonl and prompts_{group}.jsonl")
    ap.add_argument("--out-dir", default="results/llm", help="Directory to write *_clean.jsonl, *_rejects.jsonl, *_summary.json")
    ap.add_argument("--groups", nargs="*", default=CANONICAL_GROUPS, help="Groups to process (default: A B C D)")
    ap.add_argument("--models", nargs="+", required=True, help="Model identifiers used in Step 10 (same strings you passed to --models)")
    ap.add_argument("--api-base", default="http://127.0.0.1:11434", help="Ollama API base URL")
    ap.add_argument("--reprompt", type=int, default=1, help="If 1, re-prompt once on invalid output; if 0, never reprompt")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between rows")
    args = ap.parse_args()

    # Quick connectivity check (only if reprompt enabled)
    if args.reprompt:
        try:
            requests.get(args.api_base.rstrip("/") + "/api/tags", timeout=5).raise_for_status()
        except Exception as e:
            log(f"⚠ Reprompt enabled but Ollama not reachable at {args.api_base}: {e}")

    os.makedirs(args.out_dir, exist_ok=True)

    totals = {"clean": 0, "reprompted": 0, "rejects": 0}
    for g in args.groups:
        for m in args.models:
            c, r, x = process_pair(
                group=g,
                model=m,
                in_dir=args.in_dir,
                out_dir=args.out_dir,
                api_base=args.api_base,
                reprompt=bool(args.reprompt),
                sleep_s=args.sleep
            )
            totals["clean"] += c
            totals["reprompted"] += r
            totals["rejects"] += x

    log(f"\nDone. Totals: clean={totals['clean']} reprompted={totals['reprompted']} rejects={totals['rejects']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())