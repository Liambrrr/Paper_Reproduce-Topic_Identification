"""
Step 12 - Cosine similarity between label and topic centroid (supports prompt options op1|op2|op3)

For each subset (A/B/C/D):
  1) Load docs and the trained BERTopic model.
  2) Compute (or recompute) embeddings for all docs using 'sentence-transformers/all-MiniLM-L6-v2'.
  3) Build topic centroids by averaging doc embeddings per topic (skip topic -1).
  4) For each cleaned label (from Step 11 *_op{N}_clean.jsonl), embed the label text.
  5) Compute cosine(label, centroid[topic_id]) and write results.

Inputs (defaults):
  - groups_pkl: data/groups_preprocessed.pkl
  - model_template: models/bertopic_{group}
  - labels_dir: results/llm
      expects cleaned label files named like (from Step 11):
        labels_A_meta-llama_llama-3.1-8b-instruct_op1_clean.jsonl
        labels_B_meta-llama_llama-3.1-8b-instruct_op2_clean.jsonl
  - outdir: results/metrics

Outputs:
  - results/metrics/centroid_sim_{group}_{modeltag}_op{N}.jsonl
  - results/metrics/centroid_sim_op{N}_all.csv

Usage:
  python step12_label_vs_centroid.py \
    --groups-pkl data/groups_preprocessed.pkl \
    --model-template "models/bertopic_{group}" \
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
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

try:
    from bertopic import BERTopic
except Exception:
    BERTopic = None


# ------------------------
# Utilities
# ------------------------

CANONICAL_ORDER = ["A", "B", "C", "D"]
NAME_ALIASES = {
    "A": ["A", "All", "ALL", "all"],
    "B": ["B", "Education", "EDUCATION", "education"],
    "C": ["C", "Humanities", "HUMANITIES", "humanities"],
    "D": ["D", "Medicine", "MEDICINE", "medicine"],
}


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


def sanitize_tag(s: str) -> str:
    return s.replace("/", "_").replace(":", "-").replace(".", "-").replace(" ", "").lower()


def resolve_available_groups(obj_keys):
    obj_keys_set = set(map(str, obj_keys))
    mapping = {}
    for code, aliases in NAME_ALIASES.items():
        for alias in aliases:
            if alias in obj_keys_set:
                mapping[code] = alias
                break
    return mapping


def load_groups(groups_pkl: str):
    if not os.path.isfile(groups_pkl):
        raise FileNotFoundError(f"Groups pickle not found: {groups_pkl}")
    with open(groups_pkl, "rb") as f:
        obj = pickle.load(f)
    present = resolve_available_groups(obj.keys())
    if not present:
        raise KeyError(f"No recognized groups found in {groups_pkl}. Found keys: {list(obj.keys())}")
    for code, actual in present.items():
        df = obj[actual]
        if "cleaned_text" not in df.columns:
            raise KeyError(f"Group '{actual}' missing 'cleaned_text' column.")
    return obj, present


def try_load_model(model_template: str, group_code: str, actual_key: str):
    tried = []
    variants = []
    for t in (group_code, actual_key):
        variants.extend([t, t.lower(), t.upper(), t.capitalize()])
    seen = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    for v in variants:
        for candidate in (model_template.format(group=v), model_template.format(group=v) + ".pkl"):
            tried.append(candidate)
            if os.path.exists(candidate):
                log(f"  ↳ Loading BERTopic model: {candidate}")
                return BERTopic.load(candidate)
    raise FileNotFoundError(f"BERTopic model not found. Tried: {tried}")


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)
    if du == 0.0 or dv == 0.0:
        return float("nan")
    return float(np.dot(u, v) / (du * dv))


# ------------------------
# Core computation
# ------------------------

def compute_doc_embeddings(texts: List[str],
                           sbert: SentenceTransformer,
                           batch_size: int = 256) -> np.ndarray:
    emb = sbert.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return emb.astype(np.float32)


def topic_assignments(model: "BERTopic", docs: List[str]) -> pd.DataFrame:
    info = model.get_document_info(docs).copy()
    info["Topic"] = info["Topic"].astype(int)
    return info


def build_centroids(model: "BERTopic",
                    docs: List[str],
                    doc_emb: np.ndarray) -> Dict[int, np.ndarray]:
    info = topic_assignments(model, docs)
    centroids: Dict[int, np.ndarray] = {}
    for tid, grp in info.groupby("Topic"):
        if tid == -1:
            continue
        idx = grp.index.to_numpy()
        if idx.size == 0:
            continue
        centroid = doc_emb[idx].mean(axis=0)
        centroids[int(tid)] = centroid
    return centroids


def collect_clean_label_files(labels_dir: str, group: str, prompt_option: int) -> List[Tuple[str, str]]:
    """
    Find all files like: labels_{group}_{modeltag}_op{N}_clean.jsonl
    Returns list of (modeltag, path).
    """
    pattern = os.path.join(labels_dir, f"labels_{group}_*_op{prompt_option}_clean.jsonl")
    paths = sorted(glob.glob(pattern))
    results: List[Tuple[str, str]] = []
    for p in paths:
        m = re.search(
            rf"labels_{group}_(.+?)_op{prompt_option}_clean\.jsonl$",
            os.path.basename(p)
        )
        if m:
            results.append((m.group(1), p))
    return results


def main():
    ap = argparse.ArgumentParser(description="Step 12: cosine(label, topic-centroid) per topic & LLM (with prompt options).")
    ap.add_argument("--groups-pkl", default="data/groups_preprocessed.pkl",
                    help="Pickle with groups: dict[str->DataFrame] with 'cleaned_text'.")
    ap.add_argument("--model-template", default="models/bertopic_{group}",
                    help="Path template to BERTopic model per group (file or dir).")
    ap.add_argument("--labels-dir", default="results/llm",
                    help="Directory with labels_{group}_{modeltag}_op{N}_clean.jsonl (from Step 11).")
    ap.add_argument("--outdir", default="results/metrics",
                    help="Directory to write centroid_sim_*.jsonl and centroid_sim_op{N}_all.csv")
    ap.add_argument("--only-groups", nargs="*", default=CANONICAL_ORDER,
                    help="Which groups to process (default: A B C D).")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Embedding batch size for SentenceTransformer.")
    ap.add_argument("--sbert", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformer model id to embed docs & labels.")
    ap.add_argument("--prompt_option", type=int, choices=[1, 2, 3], default=1,
                    help="Which prompt option (op1/op2/op3) the cleaned label files correspond to.")
    args = ap.parse_args()

    if BERTopic is None:
        raise RuntimeError("bertopic is not installed. `pip install bertopic`")

    os.makedirs(args.outdir, exist_ok=True)

    log(f"Step 12 - Cosine similarity(label, topic-centroid)  (op{args.prompt_option})")

    # Load groups
    log(f"• Loading groups: {args.groups_pkl}")
    groups_obj, present = load_groups(args.groups_pkl)

    # Embedder
    log(f"• Loading embedder: {args.sbert}")
    sbert = SentenceTransformer(args.sbert)

    all_rows = []

    for code in [c for c in CANONICAL_ORDER if c in present and c in args.only_groups]:
        actual_key = present[code]
        log(f"\n[Group {code} | source key: {actual_key}]")

        df = groups_obj[actual_key].copy()
        df = df[~df["cleaned_text"].isna()]
        df = df[df["cleaned_text"].astype(str).str.strip().str.len() > 0]
        if df.empty:
            log("  ⚠ No non-empty cleaned_text rows; skipping.")
            continue
        docs: List[str] = df["cleaned_text"].astype(str).tolist()

        try:
            model = try_load_model(args.model_template, group_code=code, actual_key=actual_key)
        except Exception as e:
            log(f"  ✗ Could not load BERTopic model: {e}")
            continue

        log("  • Computing document embeddings (MiniLM-L6-v2)…")
        doc_emb = compute_doc_embeddings(docs, sbert, batch_size=args.batch_size)

        log("  • Building topic centroids…")
        centroids = build_centroids(model, docs, doc_emb)

        label_files = collect_clean_label_files(args.labels_dir, code, args.prompt_option)
        if not label_files:
            log(f"  ⚠ No cleaned label files found for group {code} with op{args.prompt_option} in {args.labels_dir}")
            continue

        topic_info = model.get_topic_info()
        size_map = {int(r["Topic"]): int(r["Count"]) for _, r in topic_info.iterrows()}

        for modeltag, path in label_files:
            out_path = os.path.join(args.outdir, f"centroid_sim_{code}_{modeltag}_op{args.prompt_option}.jsonl")
            log(f"  → Labels: {os.path.basename(path)}  |  Output: {os.path.basename(out_path)}")

            rows = read_jsonl(path)
            out_rows = []

            for r in tqdm(rows, desc=f"[{code} | {modeltag}] topics", leave=False):
                meta = r.get("meta", {})
                clean = r.get("clean", {})
                tid = int(meta.get("topic_id")) if meta.get("topic_id") is not None else None
                label = (clean.get("label") or "").strip()

                if tid is None or tid not in centroids or not label:
                    continue

                lab_emb = sbert.encode([label], convert_to_numpy=True, normalize_embeddings=False)[0].astype(np.float32)
                sim = cosine(lab_emb, centroids[tid])

                rec = {
                    "group": code,
                    "topic_id": tid,
                    "topic_size": int(size_map.get(tid, 0)),
                    "llm_model": meta.get("model"),
                    "llm_provider": meta.get("provider"),
                    "prompt_option": args.prompt_option,
                    "label": label,
                    "cosine": float(sim),
                }
                out_rows.append(rec)
                all_rows.append(rec)

            write_jsonl(out_path, out_rows)
            log(f"    ✓ Wrote {len(out_rows)} rows → {out_path}")

    if all_rows:
        csv_path = os.path.join(args.outdir, f"centroid_sim_op{args.prompt_option}_all.csv")
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        log(f"\n✓ Combined CSV written → {csv_path}")
    else:
        log("\n⚠ No rows produced. Ensure *_op{N}_clean.jsonl exist and topic IDs match your BERTopic models.")

    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())