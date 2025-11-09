"""
Step 13 - Compute average cosine similarity between the label and representative documents
(supports prompt options op1|op2|op3)

For each topic:
  - Embed the LLM label (from *_op{N}_clean.jsonl)
  - Embed the topic's representative snippets (from topics_{group}.jsonl)
  - Compute cosine similarity(label, each snippet)
  - Record mean / min / max (and count) per topic+model

Inputs (defaults):
  topics dir: results/llm
    - topics_A.jsonl, topics_B.jsonl, topics_C.jsonl, topics_D.jsonl
  labels dir: results/llm
    - labels_{GROUP}_{MODEL_TAG}_op{N}_clean.jsonl  (from Step 11)

Outputs:
  results/metrics/doc_sim_{GROUP}_{MODEL_TAG}_op{N}.jsonl

Usage:
  python step13_label_vs_docs.py \
      --topics-dir results/llm \
      --labels-dir results/llm \
      --outdir results/metrics \
      --only-groups A B C D \
      --batch-size 256 \
      --prompt_option 1
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
from tqdm import tqdm

# sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


CANONICAL_GROUPS = ["A", "B", "C", "D"]
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def load_topics(topics_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Return topic_id -> {"keywords": [...], "snippets": [...], "size": int}
    """
    out: Dict[int, Dict[str, Any]] = {}
    for rec in read_jsonl(topics_path):
        tid = int(rec.get("topic_id"))
        out[tid] = {
            "keywords": rec.get("keywords", []) or [],
            "snippets": rec.get("representative_snippets", []) or [],
            "size": int(rec.get("size", 0)),
        }
    return out


def embed_texts(model: "SentenceTransformer",
                texts: List[str],
                batch_size: int = 256) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)  # MiniLM-L6-v2 has 384 dims
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (d,) normalized
    b: (n, d) normalized
    returns (n,)
    """
    if b.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (b @ a.astype(np.float32))


def find_clean_label_files(labels_dir: str, group: str, prompt_option: int) -> List[Tuple[str, str]]:
    """
    Return list of (model_tag, path) for:
      labels_{GROUP}_{MODEL_TAG}_op{N}_clean.jsonl
    """
    pattern = os.path.join(labels_dir, f"labels_{group}_*_op{prompt_option}_clean.jsonl")
    paths = sorted(glob.glob(pattern))
    out: List[Tuple[str, str]] = []
    for p in paths:
        m = re.search(
            rf"labels_{group}_(.+?)_op{prompt_option}_clean\.jsonl$",
            os.path.basename(p)
        )
        if m:
            out.append((m.group(1), p))
    return out


def process_group(
    group: str,
    topics_dir: str,
    labels_dir: str,
    outdir: str,
    model: "SentenceTransformer",
    batch_size: int,
    prompt_option: int
) -> None:
    topics_path = os.path.join(topics_dir, f"topics_{group}.jsonl")
    if not os.path.isfile(topics_path):
        log(f"[Group {group}] ✗ Missing topics file: {topics_path}")
        return

    topics = load_topics(topics_path)
    if not topics:
        log(f"[Group {group}] ⚠ No topics in {topics_path}")
        return

    label_files = find_clean_label_files(labels_dir, group, prompt_option)
    if not label_files:
        log(f"[Group {group}] ⚠ No cleaned label files (op{prompt_option}) found in {labels_dir}")
        return

    log(f"[Group {group}] Topics loaded: {len(topics)}")
    for model_tag, clean_path in label_files:
        out_path = os.path.join(outdir, f"doc_sim_{group}_{model_tag}_op{prompt_option}.jsonl")

        rows_in = read_jsonl(clean_path)
        if not rows_in:
            log(f"[Group {group}] ⚠ Empty cleaned file: {clean_path}")
            write_jsonl(out_path, [])
            continue

        log(f"[Group {group}] → {model_tag} | inputs={len(rows_in)} | Output: {out_path}")

        # Build topic_id -> label text (use cleaned label)
        label_by_tid: Dict[int, str] = {}
        size_by_tid: Dict[int, int] = {}
        for r in rows_in:
            meta = r.get("meta", {})
            tid = meta.get("topic_id")
            if not isinstance(tid, int):
                continue
            label = ((r.get("clean") or {}).get("label") or "").strip()
            if not label:
                continue
            label_by_tid[tid] = label
            # try to retain topic size from meta; else fallback to topics file
            size_by_tid[tid] = int(meta.get("topic_size") or topics.get(tid, {}).get("size", 0))

        # Compute sims per topic
        out_rows: List[Dict[str, Any]] = []
        for tid, label in tqdm(label_by_tid.items(), desc=f"[{group} | {model_tag}] topics", leave=False):
            topic = topics.get(tid)
            if not topic:
                continue
            snippets: List[str] = topic.get("snippets", [])
            if not snippets:
                continue

            label_vec = embed_texts(model, [label], batch_size=batch_size)[0]   # (384,)
            snip_vecs = embed_texts(model, snippets, batch_size=batch_size)     # (k, 384)

            sims = cosine_sim(label_vec, snip_vecs)  # (k,)
            if sims.size == 0:
                continue

            out_rows.append({
                "meta": {
                    "group": group,
                    "topic_id": int(tid),
                    "topic_size": int(size_by_tid.get(tid, topic.get("size", 0))),
                    "model_tag": model_tag,
                    "prompt_option": prompt_option,
                },
                "label": label,
                "stats": {
                    "n_snippets": int(sims.size),
                    "mean": float(np.mean(sims)),
                    "min": float(np.min(sims)),
                    "max": float(np.max(sims)),
                }
            })

        write_jsonl(out_path, out_rows)
        log(f"[Group {group}] ✓ Wrote {len(out_rows)} rows → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Step 13: Cosine similarity between label and representative documents (with prompt options).")
    ap.add_argument("--topics-dir", default="results/llm", help="Directory with topics_{group}.jsonl")
    ap.add_argument("--labels-dir", default="results/llm", help="Directory with labels_{group}_{modeltag}_op{N}_clean.jsonl")
    ap.add_argument("--outdir", default="results/metrics", help="Output directory")
    ap.add_argument("--only-groups", nargs="*", default=CANONICAL_GROUPS, help="Groups to process")
    ap.add_argument("--batch-size", type=int, default=256, help="Embedder batch size")
    ap.add_argument("--prompt_option", type=int, choices=[1, 2, 3], default=1, help="Which prompt option (op1|op2|op3) to read and emit")
    args = ap.parse_args()

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")

    os.makedirs(args.outdir, exist_ok=True)
    log("Step 13 - Label vs representative documents (cosine similarity)")
    log(f"• Loading embedder: {EMBEDDER_NAME}")
    model = SentenceTransformer(EMBEDDER_NAME)

    for g in args.only_groups:
        process_group(
            group=g,
            topics_dir=args.topics_dir,
            labels_dir=args.labels_dir,
            outdir=args.outdir,
            model=model,
            batch_size=args.batch_size,
            prompt_option=args.prompt_option
        )

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())