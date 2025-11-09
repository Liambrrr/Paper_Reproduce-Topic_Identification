#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 11 - Output validation and normalization (Ollama)

For each (subset, model) raw output JSONL from Step 10:
  - Parse response.text into JSON (robust).
  - Validate the schema: {"label": str, "rationale": str}, no extra keys required.
  - If invalid, re-prompt once (same messages/decoding) via Ollama, then re-parse/validate.
  - Normalize the label minimally (strip, lowercase, strip punctuation edges, <=4 words).
  - Write cleaned rows and rejects, plus a small summary JSON.

Inputs (from Step 10), with prompt variant:
  prompts_{A|B|C|D}_op{1|2|3}.jsonl
  labels_{group}_{model_tag}_op{1|2|3}.jsonl
  (Legacy fallback for op1: prompts_{group}.jsonl and labels_{group}_{model_tag}.jsonl)

Outputs (always include op{N} to disambiguate):
  labels_{group}_{model_tag}_op{N}_clean.jsonl
  labels_{group}_{model_tag}_op{N}_rejects.jsonl
  labels_{group}_{model_tag}_op{N}_summary.json

Usage:
  python step11_validate_normalize.py \
    --in-dir results/llm \
    --out-dir results/llm \
    --groups A B C D \
    --models "meta-llama/llama-3.1-8b-instruct" \
    --prompt_option 1 \
    --reprompt 0
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
PUNCT_STRIP = "".join(sorted(set(string.punctuation)))


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


def find_prompts_file(in_dir: str, group: str, prompt_option: int) -> Optional[str]:
    """
    For op1: prefer prompts_{G}_op1.jsonl; fallback to legacy prompts_{G}.jsonl if not found.
    For op2/op3: require prompts_{G}_op{N}.jsonl.
    """
    if prompt_option == 1:
        preferred = os.path.join(in_dir, f"prompts_{group}_op1.jsonl")
        legacy = os.path.join(in_dir, f"prompts_{group}.jsonl")
        if os.path.isfile(preferred):
            return preferred
        if os.path.isfile(legacy):
            return legacy
        return None
    else:
        path = os.path.join(in_dir, f"prompts_{group}_op{prompt_option}.jsonl")
        return path if os.path.isfile(path) else None


def find_labels_file(in_dir: str, group: str, model_tag: str, prompt_option: int) -> Optional[str]:
    primary = os.path.join(in_dir, f"labels_{group}_op{prompt_option}_{model_tag}.jsonl")
    if os.path.isfile(primary):
        return primary
    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()

    fence = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence.search(s)
    if m:
        s = m.group(1).strip()

    if "{" in s and "}" in s:
        s = s[s.find("{"): s.rfind("}") + 1]

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
    if not isinstance(raw_label, str):
        return ""
    s = raw_label.strip().lower()
    s = s.strip(PUNCT_STRIP)
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
                 sleep_s: float,
                 prompt_option: int) -> Tuple[int, int, int]:
    """
    Return: (clean_count, reprompted_count, reject_count)
    """
    model_tag = sanitize_model_tag(model)

    # Locate inputs
    raw_path = find_labels_file(in_dir, group, model_tag, prompt_option)
    prompts_path = find_prompts_file(in_dir, group, prompt_option)

    if not raw_path:
        log(f"[{group} | {model}] ✗ Missing labels file for op{prompt_option} in {in_dir}")
        return (0, 0, 0)
    if not prompts_path:
        log(f"[{group} | {model}] ✗ Missing prompts file for op{prompt_option} in {in_dir}")
        return (0, 0, 0)

    raw_rows = read_jsonl(raw_path)
    prompts = read_jsonl(prompts_path)
    prompt_by_tid = index_prompts_by_topic(prompts)

    clean_rows: List[Dict[str, Any]] = []
    reject_rows: List[Dict[str, Any]] = []

    reprompted = 0

    for row in tqdm(raw_rows, desc=f"[{group} | {model}] validate (op{prompt_option})", leave=False):
        meta = row.get("meta", {})
        tid = meta.get("topic_id")
        cleaned = None

        def try_parse_and_normalize(text: str) -> Optional[Dict[str, Any]]:
            obj = extract_json(text)
            if not obj or not is_valid_schema(obj):
                return None
            return {
                "label": normalize_label(obj["label"], max_words=4),
                "rationale": obj["rationale"].strip(),
            }

        # 1) Try original response
        resp = row.get("response", {})
        text = resp.get("text", "") if isinstance(resp, dict) else ""
        cleaned = try_parse_and_normalize(text)

        # 2) Optional single re-prompt (Ollama)
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
                "raw": row.get("response", None) or {"text": text},
            }
            clean_rows.append(out_row)
        else:
            reject_rows.append(row)

        if sleep_s > 0:
            import time as _t
            _t.sleep(sleep_s)

    # Always include op{N} in outputs
    base = os.path.join(out_dir, f"labels_{group}_{model_tag}_op{prompt_option}")
    clean_path = f"{base}_clean.jsonl"
    reject_path = f"{base}_rejects.jsonl"
    summary_path = f"{base}_summary.json"

    write_jsonl(clean_path, clean_rows)
    write_jsonl(reject_path, reject_rows)

    summary = {
        "group": group,
        "model": model,
        "prompt_option": prompt_option,
        "input_rows": len(raw_rows),
        "clean_rows": len(clean_rows),
        "reject_rows": len(reject_rows),
        "reprompted_once": reprompted,
        "in_dir": in_dir,
        "out_dir": out_dir,
        "labels_input": raw_path,
        "prompts_input": prompts_path,
        "clean_output": clean_path,
        "rejects_output": reject_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"[{group} | {model} | op{prompt_option}] ✓ clean={summary['clean_rows']} "
        f"reject={summary['reject_rows']} reprompted={reprompted}")
    return (summary["clean_rows"], reprompted, summary["reject_rows"])


def main():
    ap = argparse.ArgumentParser(description="Step 11: Validate & normalize LLM outputs (Ollama).")
    ap.add_argument("--in-dir", default="results/llm",
                    help="Directory with prompts_* and labels_* JSONL files")
    ap.add_argument("--out-dir", default="results/llm",
                    help="Directory to write *_clean.jsonl, *_rejects.jsonl, *_summary.json")
    ap.add_argument("--groups", nargs="*", default=CANONICAL_GROUPS,
                    help="Groups to process (default: A B C D)")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model identifiers as used in Step 10 (e.g., meta-llama/llama-3.1-8b-instruct)")
    ap.add_argument("--api-base", default="http://127.0.0.1:11434",
                    help="Ollama API base URL (only used if --reprompt 1)")
    ap.add_argument("--reprompt", type=int, default=1,
                    help="If 1, re-prompt once on invalid output; if 0, never reprompt")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Sleep seconds between rows")
    ap.add_argument("--prompt_option", type=int, default=1, choices=[1, 2, 3],
                    help="Prompt variant used in Step 10 (affects file names: *_op{N}.jsonl)")
    args = ap.parse_args()

    # Connectivity check for reprompt
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
                sleep_s=args.sleep,
                prompt_option=args.prompt_option,
            )
            totals["clean"] += c
            totals["reprompted"] += r
            totals["rejects"] += x

    log(f"\nDone (op{args.prompt_option}). Totals: clean={totals['clean']}, "
        f"reprompted={totals['reprompted']}, rejects={totals['rejects']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())