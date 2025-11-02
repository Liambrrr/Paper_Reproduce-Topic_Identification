#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 10 - Run LLMs for multiple times and store raw outputs

Reads prompts_{A|B|C|D}.jsonl from the previous step and, for each topic,
calls each specified model exactly once and writes the raw responses to
separate JSONL files per (subset, model).

Backends supported:
  --provider openai-compatible
      Uses HTTP to {--api-base}/v1/chat/completions with OpenAI-compatible schema
  --provider ollama
      Uses local Ollama at {--api-base or http://localhost:11434}/api/chat

Inputs (defaults):
  prompts directory: results/llm
    - prompts_A.jsonl
    - prompts_B.jsonl
    - prompts_C.jsonl
    - prompts_D.jsonl

Outputs:
  results/llm/labels_A_<model>.jsonl
  results/llm/labels_B_<model>.jsonl
  results/llm/labels_C_<model>.jsonl
  results/llm/labels_D_<model>.jsonl

Example usage:
  # Llama 3.3 70B (OpenAI-compatible endpoint, e.g., vLLM/Together)
  python step10_run_llms.py \
    --provider openai-compatible \
    --api-base https://api.your-endpoint.com \
    --api-key $YOUR_KEY \
    --models llama-3.3-70b-instruct

  # Qwen3 30B on Ollama
  python step10_run_llms.py \
    --provider ollama \
    --models qwen3:30b

  # Run all four requested models (names are examples; set to what your backend expects):
  python step10_run_llms.py \
    --provider openai-compatible \
    --api-base https://api.your-endpoint.com \
    --api-key $KEY \
    --models llama-3.1-8b llama-3.3-70b-instruct qwen2.5-7b qwen3-30b
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

import requests
from tqdm import tqdm

try:
    import openai
except Exception:
    openai = None


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


def sanitize_model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "-").replace(" ", "").lower()


def join_messages_for_tgi(messages: List[Dict[str, str]]) -> str:
    """Flatten system+user into a single prompt for TGI-like backends."""
    sys_msgs = [m["content"] for m in messages if m.get("role") == "system"]
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    sys_block = ("\n".join(sys_msgs)).strip()
    user_block = ("\n".join(user_msgs)).strip()
    prompt = ""
    if sys_block:
        prompt += f"[SYSTEM]\n{sys_block}\n\n"
    prompt += f"[USER]\n{user_block}"
    return prompt


# ---------------------------
# Backend callers
# ---------------------------

def call_openai_compatible(model: str,
                           api_key: Optional[str],
                           api_base: str,
                           messages: List[Dict[str, str]],
                           temperature: float,
                           top_p: float,
                           max_new_tokens: int) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    text = (choice["message"]["content"] or "").strip()
    return {
        "text": text,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "raw": data,
    }


def call_ollama(model: str,
                api_base: str,
                messages: List[Dict[str, str]],
                temperature: float,
                top_p: float,
                max_new_tokens: int) -> Dict[str, Any]:
    # Ollama chat API
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
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama returns the final message at the end of message list
    text = ""
    if "message" in data and "content" in data["message"]:
        text = (data["message"]["content"] or "").strip()
    return {
        "text": text,
        "finish_reason": data.get("done_reason"),
        "usage": None,
        "raw": data,
    }


def call_backend(provider: str,
                 model: str,
                 api_base: Optional[str],
                 api_key: Optional[str],
                 messages: List[Dict[str, str]],
                 temperature: float,
                 top_p: float,
                 max_new_tokens: int) -> Dict[str, Any]:
    provider = provider.lower()
    if provider == "openai-compatible":
        if not api_base:
            raise ValueError("--api-base required for openai-compatible provider")
        return call_openai_compatible(model, api_key, api_base, messages, temperature, top_p, max_new_tokens)
    elif provider == "ollama":
        api_base = api_base or "http://localhost:11434"
        return call_ollama(model, api_base, messages, temperature, top_p, max_new_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def main():
    p = argparse.ArgumentParser(description="Step 10: Run LLMs on prompts and store raw outputs per (subset, model).")
    p.add_argument("--prompts-dir", default="results/llm",
                   help="Directory containing prompts_{A|B|C|D}.jsonl.")
    p.add_argument("--outdir", default="results/llm",
                   help="Directory to write labels_{group}_{model}.jsonl.")
    p.add_argument("--only-groups", nargs="*", default=CANONICAL_GROUPS,
                   help="Which groups to process (default: A B C D).")
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more model identifiers (as your backend expects).")
    p.add_argument("--provider", choices=["openai", "openai-compatible", "ollama", "tgi"], required=True,
                   help="Backend provider to use.")
    p.add_argument("--api-base", default=None,
                   help="API base URL. Required for openai-compatible and tgi; optional for openai; defaults to localhost:11434 for ollama.")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="API key (env OPENAI_API_KEY used by default for OpenAI providers).")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Sleep seconds between calls to be polite with rate limits.")
    args = p.parse_args()

    log("Step 10 - Run LLMs and store raw outputs")
    os.makedirs(args.outdir, exist_ok=True)

    # Iterate groups
    for group in args.only_groups:
        prompts_path = os.path.join(args.prompts_dir, f"prompts_{group}.jsonl")
        if not os.path.isfile(prompts_path):
            log(f"[Group {group}] ✗ Missing prompts file: {prompts_path}")
            continue

        log(f"\n[Group {group}] Loading prompts: {prompts_path}")
        prompts = read_jsonl(prompts_path)
        if not prompts:
            log(f"[Group {group}] ⚠ No prompts in file.")
            continue

        # For each requested model, create its own output stream
        for model in args.models:
            tag = sanitize_model_tag(model)
            out_path = os.path.join(args.outdir, f"labels_{group}_{tag}.jsonl")
            log(f"[Group {group}] → Model: {model} | Output: {out_path}")

            written = 0
            out_f = open(out_path, "w", encoding="utf-8")
            try:
                for rec in tqdm(prompts, desc=f"[{group} | {model}] topics", leave=False):
                    meta = rec.get("meta", {})
                    messages = rec.get("messages", [])
                    decoding = rec.get("decoding", {})
                    temperature = float(decoding.get("temperature", 0.0))
                    top_p = float(decoding.get("top_p", 1.0))
                    max_new_tokens = int(decoding.get("max_new_tokens", 32))

                    # Fire the request
                    ts = int(time.time())
                    try:
                        resp = call_backend(
                            provider=args.provider,
                            model=model,
                            api_base=args.api_base,
                            api_key=args.api_key,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens
                        )

                        row = {
                            "meta": {
                                "group": meta.get("group"),
                                "topic_id": meta.get("topic_id"),
                                "topic_size": meta.get("topic_size"),
                                "model": model,
                                "provider": args.provider,
                                "timestamp": ts
                            },
                            "request": {
                                "messages": messages,
                                "decoding": decoding
                            },
                            "response": {
                                "text": resp.get("text", ""),
                                "finish_reason": resp.get("finish_reason"),
                                "usage": resp.get("usage")
                            }
                        }
                    except Exception as e:
                        # Store the error verbatim to keep raw logs complete
                        row = {
                            "meta": {
                                "group": meta.get("group"),
                                "topic_id": meta.get("topic_id"),
                                "topic_size": meta.get("topic_size"),
                                "model": model,
                                "provider": args.provider,
                                "timestamp": ts
                            },
                            "request": {
                                "messages": messages,
                                "decoding": decoding
                            },
                            "error": str(e)
                        }

                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

                    if args.sleep > 0:
                        time.sleep(args.sleep)
            finally:
                out_f.close()

            log(f"[Group {group}] ✓ Wrote {written} rows → {out_path}")

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())