"""
Step 14 - Compare the label with top-N keywords

For each topic (per group and per LLM model):
  (a) Embed the label and the top-N keywords (as phrases). Compute cosine(label, mean(keyword embeddings)).
  (b) Compute Jaccard between lowercased+lemmatized label tokens and lowercased keyword strings (phrases as-is).

Inputs (defaults):
  topics dir: results/llm
    - topics_A.jsonl, topics_B.jsonl, topics_C.jsonl, topics_D.jsonl
  labels dir: results/llm
   - labels_{GROUP}_{MODEL_TAG}_op{N}_clean.jsonl  (from Step 11)

Outputs:
  results/metrics/kw_sim_{GROUP}_{MODEL_TAG}_op{N}.jsonl

Usage:
  python step14_label_vs_keywords.py \
      --topics-dir results/llm \
      --labels-dir results/llm \
      --outdir results/metrics \
      --only-groups A B C D \
      --topn 10 \
      --batch-size 256 \
      --prompt_option 1
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, Any, List, Set

import numpy as np
from tqdm import tqdm

# Sentence embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Lightweight lemmatizer (best-effort)
_LEMM_WARNING_SHOWN = False
def _lemma_setup():
    """
    Try spaCy 'en_core_web_sm' first (best), else NLTK WordNet, else fallback to identity.
    We avoid downloading inside the script to keep it reproducible.
    """
    # spaCy
    try:
        import spacy  # noqa
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
            return ("spacy", nlp)
        except Exception:
            pass
    except Exception:
        pass

    # NLTK WordNet
    try:
        import nltk  # noqa
        from nltk.stem import WordNetLemmatizer  # noqa
        # Assume corpora installed; if not, we fallback.
        return ("nltk", WordNetLemmatizer())
    except Exception:
        pass

    return ("none", None)


_LEMMA_BACKEND, _LEMMA_OBJ = _lemma_setup()

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def lemmatize_tokens(text: str) -> List[str]:
    """Tokenize -> lowercase -> lemmatize (best-effort)."""
    global _LEMM_WARNING_SHOWN
    toks = TOKEN_RE.findall(text or "")
    if not toks:
        return []

    if _LEMMA_BACKEND == "spacy":
        doc = _LEMMA_OBJ(" ".join(toks).lower())
        return [t.lemma_ for t in doc if t.lemma_]

    if _LEMMA_BACKEND == "nltk":
        from nltk.stem import WordNetLemmatizer
        wnl: WordNetLemmatizer = _LEMMA_OBJ  # type: ignore
        return [wnl.lemmatize(t.lower()) for t in toks]

    # Fallback: lowercase only (warn once)
    if not _LEMM_WARNING_SHOWN:
        print("⚠ Lemmatizer not available; falling back to lowercase tokens only.", flush=True)
        _LEMM_WARNING_SHOWN = True
    return [t.lower() for t in toks]


EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
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


def load_topics(topics_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Returns: topic_id -> {"keywords": [...], "snippets": [...], "size": int}
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
        return np.zeros((0, 384), dtype=np.float32)  # MiniLM-L6-v2 dimension
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    a: (d,) normalized
    b: (d,) normalized
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def process_group(group: str,
                  topics_dir: str,
                  labels_dir: str,
                  outdir: str,
                  embedder: "SentenceTransformer",
                  topn: int,
                  batch_size: int,
                  prompt_option: int) -> None:
    topics_path = os.path.join(topics_dir, f"topics_{group}.jsonl")
    if not os.path.isfile(topics_path):
        log(f"[Group {group}] ✗ Missing topics file: {topics_path}")
        return

    topics = load_topics(topics_path)
    if not topics:
        log(f"[Group {group}] ⚠ No topics found in {topics_path}")
        return

    # All cleaned label files for this group
    clean_files = sorted(glob.glob(os.path.join(labels_dir, f"labels_{group}_*op{prompt_option}_clean.jsonl")))
    if not clean_files:
        log(f"[Group {group}] ⚠ No labels_{group}_*_op{prompt_option}_clean.jsonl files in {labels_dir}")
        return

    log(f"[Group {group}] topics={len(topics)} | models={len(clean_files)}")

    for clean_path in clean_files:
        tag = os.path.basename(clean_path)[len(f"labels_{group}_"):-len(f"_op{prompt_option}_clean.jsonl")]
        out_path = os.path.join(outdir, f"kw_sim_{group}_{tag}_op{prompt_option}.jsonl")

        rows_in = read_jsonl(clean_path)
        if not rows_in:
            write_jsonl(out_path, [])
            log(f"[Group {group}] → {tag}: empty input; wrote 0 rows.")
            continue

        # Build topic -> label
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
            size_by_tid[tid] = int(meta.get("topic_size") or topics.get(tid, {}).get("size", 0))

        out_rows: List[Dict[str, Any]] = []
        for tid, label in tqdm(label_by_tid.items(), desc=f"[{group} | {tag}] topics", leave=False):
            topic = topics.get(tid)
            if not topic:
                continue

            kws = (topic.get("keywords") or [])[:topn]
            kws_lower = [k.strip().lower() for k in kws if isinstance(k, str) and k.strip()]
            if not kws_lower:
                # no keywords → NaN/0
                out_rows.append({
                    "meta": {"group": group, "topic_id": int(tid), "topic_size": int(size_by_tid.get(tid, 0)), "model_tag": tag, "prompt_option": prompt_option},
                    "label": label,
                    "top_keywords": kws[:topn],
                    "metrics": {"embed_cosine": float("nan"), "jaccard": 0.0, "n_keywords": 0, "used_topn": int(topn)}
                })
                continue

            # (a) Embedding similarity
            label_vec = embed_texts(embedder, [label], batch_size=batch_size)[0]           # (384,)
            kw_vecs   = embed_texts(embedder, kws_lower, batch_size=batch_size)            # (n, 384)
            kw_mean   = np.mean(kw_vecs, axis=0)
            # normalize mean
            eps = 1e-9
            kw_mean = kw_mean / max(np.linalg.norm(kw_mean), eps)
            embed_sim = cosine(label_vec, kw_mean)

            # (b) Jaccard: label tokens (lemmatized) vs keyword phrases (lowercased strings)
            label_tok_set = set(lemmatize_tokens(label))
            kw_set = set(kws_lower)
            j = jaccard(label_tok_set, kw_set)

            out_rows.append({
                "meta": {
                    "group": group,
                    "topic_id": int(tid),
                    "topic_size": int(size_by_tid.get(tid, 0)),
                    "model_tag": tag,
                    "prompt_option": prompt_option
                },
                "label": label,
                "top_keywords": kws[:topn],
                "metrics": {
                    "embed_cosine": float(embed_sim),
                    "jaccard": float(j),
                    "n_keywords": int(len(kws_lower)),
                    "used_topn": int(topn)
                }
            })

        write_jsonl(out_path, out_rows)
        log(f"[Group {group}] ✓ {tag}: wrote {len(out_rows)} rows → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Step 14: Label vs top-N keywords (embedding cosine & Jaccard).")
    ap.add_argument("--topics-dir", default="results/llm", help="Directory with topics_{group}.jsonl")
    ap.add_argument("--labels-dir", default="results/llm", help="Directory with labels_{group}_{model}_clean.jsonl")
    ap.add_argument("--outdir", default="results/metrics", help="Output directory for kw_sim_*.jsonl")
    ap.add_argument("--only-groups", nargs="*", default=CANONICAL_GROUPS, help="Groups to process")
    ap.add_argument("--topn", type=int, default=10, help="Top-N keywords to use (default: 10)")
    ap.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    ap.add_argument("--prompt_option", type=int, choices=[1, 2, 3], default=1, help="Which prompt option (op1|op2|op3) to read and emit")
    args = ap.parse_args()

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")

    os.makedirs(args.outdir, exist_ok=True)
    log("Step 14 - Label vs keywords (cosine & Jaccard)")
    log(f"• Loading embedder: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    for g in args.only_groups:
        process_group(
            group=g,
            topics_dir=args.topics_dir,
            labels_dir=args.labels_dir,
            outdir=args.outdir,
            embedder=embedder,
            topn=args.topn,
            batch_size=args.batch_size,
            prompt_option = args.prompt_option
        )

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())