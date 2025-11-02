"""
Step 8 - Export per-topic artifacts for LLM input.

For each subset (Group A: all urgent; B: Education; C: Humanities; D: Medicine), export a per-topic JSONL file
containing:
  - top-10 c-TF-IDF keywords (ordered),
  - representative documents/snippets (top-10 posts by topic probability; each truncated to <= 300 chars),
  - topic size (document count per topic).

Inputs:
  - Preprocessed groups pickle (from Step 2): data/groups_preprocessed.pkl
      Expected structure: { "A": pd.DataFrame, "B": pd.DataFrame, "C": pd.DataFrame, "D": pd.DataFrame }
      Each DF must include a 'cleaned_text' column.

  - One BERTopic model per group (from Step 4), e.g.:
      models/bertopic_A.pkl
      models/bertopic_B.pkl
      models/bertopic_C.pkl
      models/bertopic_D.pkl

Outputs:
  - results/llm/topics_A.jsonl
  - results/llm/topics_B.jsonl
  - results/llm/topics_C.jsonl
  - results/llm/topics_D.jsonl

Usage:
  python step8_export_per_topic_artifacts.py \
      --groups-pkl data/groups_preprocessed.pkl \
      --model-template "models/bertopic_{group}.pkl" \
      --outdir results/llm \
      --topn 10 \
      --max_snippet_chars 300

Notes:
  - Outlier topic -1 (if present) is skipped.
  - Snippets are chosen by highest probability for that topic.
  - Keywords come from BERTopic's c-TF-IDF (model.get_topic(topic_id)).
"""

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from bertopic import BERTopic
except Exception as e:
    BERTopic = None


def log(msg: str) -> None:
    print(msg, flush=True)


def truncate(text: str, max_len: int) -> str:
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"

# accept either short codes or long names.
CANONICAL_ORDER = ["A", "B", "C", "D"]
NAME_ALIASES = {
    "A": ["A", "All", "ALL", "all"],
    "B": ["B", "Education", "EDUCATION", "education"],
    "C": ["C", "Humanities", "HUMANITIES", "humanities"],
    "D": ["D", "Medicine", "MEDICINE", "medicine"],
}


def resolve_available_groups(obj_keys):
    """Return a dict canonical_code -> actual_key (from pickle), for all that exist."""
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
    # Build a map from canonical code -> actual key in pickle
    present = resolve_available_groups(obj.keys())
    if not present:
        raise KeyError(f"No recognized groups found in {groups_pkl}. Found keys: {list(obj.keys())}")

    # Validate DataFrames + cleaned_text
    for code, actual in present.items():
        df = obj[actual]
        if "cleaned_text" not in df.columns:
            raise KeyError(f"Group '{actual}' DataFrame missing 'cleaned_text' column.")
    return obj, present

def load_model_for_group(model_template: str, group_code: str, actual_key: str):
    """
    Try model_template with {group}=short code (A/B/C/D) first, then with the actual_key (e.g., 'All').
    """
    tried = []
    for token in (group_code, actual_key):
        path = model_template.format(group=token)
        tried.append(path)
        if os.path.isfile(path):
            log(f"  ↳ Loading BERTopic model: {path}")
            return BERTopic.load(path)
    raise FileNotFoundError(f"BERTopic model not found. Tried: {tried}")


def get_topic_keywords(model: "BERTopic", topic_id: int, topn: int) -> List[str]:
    """
    Returns ordered top-n keywords from c-TF-IDF for a given topic.
    model.get_topic(topic_id) -> List[Tuple[str, float]] or None
    """
    words_weights = model.get_topic(topic_id)
    if not words_weights:
        return []
    return [w for (w, _score) in words_weights[:topn]]


def get_topic_sizes(model: "BERTopic") -> Dict[int, int]:
    """
    Build a dict topic_id -> Count using model.get_topic_info()
    """
    info = model.get_topic_info()
    # info columns typically include: Topic, Count, Name, Representation, etc.
    sizes = {}
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        cnt = int(row["Count"])
        sizes[tid] = cnt
    return sizes


def get_representative_snippets(
    model: "BERTopic",
    docs: List[str],
    topic_id: int,
    topk: int,
    max_snippet_chars: int
) -> List[str]:
    """
    Rank documents by the model's probability for this topic and return topk text snippets (≤ max_snippet_chars).
    Uses model.get_document_info(docs), which returns per-doc Topic and Probability.
    """
    # get_document_info returns a DataFrame with columns: "Document", "Topic", "Probability", etc.
    doc_info = model.get_document_info(docs)

    # Strategy 1: If model assigns a single Topic/Probability per doc (typical),
    # we can take all docs where Topic == topic_id and sort by Probability desc.
    df = doc_info.copy()
    df = df[df["Topic"] == topic_id]
    if df.empty:
        return []

    df = df.sort_values("Probability", ascending=False)
    reps = []
    for _idx, row in df.head(topk).iterrows():
        snippet = row["Document"]
        # Ensure it's a string and truncate to max_snippet_chars
        snippet = "" if pd.isna(snippet) else str(snippet)
        reps.append(truncate(snippet, max_snippet_chars))
    return reps


def export_group_topics_jsonl(
    group: str,
    docs_df: pd.DataFrame,
    model: "BERTopic",
    outdir: str,
    topn_keywords: int,
    topk_snippets: int,
    max_snippet_chars: int
) -> str:
    """
    Exports one JSONL file for a group with records:
      {
        "topic_id": int,
        "keywords": [str, ...],  # top-10 ordered
        "representative_snippets": [str, ...],  # top-10 by probability (<=300 chars)
        "size": int
      }
    Skips topic -1 (outliers).
    """
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"topics_{group}.jsonl")
    docs = docs_df["cleaned_text"].astype(str).tolist()

    # Collect topic metadata
    sizes = get_topic_sizes(model)
    topic_ids = sorted([t for t in sizes.keys() if t != -1])  # skip outlier topic

    num_records = 0
    with open(outfile, "w", encoding="utf-8") as f:
        for tid in tqdm(topic_ids, desc=f"[Group {group}] Exporting topics", leave=False):
            keywords = get_topic_keywords(model, tid, topn_keywords)
            reps = get_representative_snippets(
                model=model,
                docs=docs,
                topic_id=tid,
                topk=topk_snippets,
                max_snippet_chars=max_snippet_chars
            )
            record = {
                "topic_id": int(tid),
                "keywords": keywords,
                "representative_snippets": reps,
                "size": int(sizes.get(tid, 0)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_records += 1

    log(f"  ✓ Wrote {num_records} topics to {outfile}")
    return outfile


def main():
    parser = argparse.ArgumentParser(description="Step 8: Export per-topic artifacts for LLM input (JSONL per subset).")
    parser.add_argument("--groups-pkl", default="data/groups_preprocessed.pkl",
                        help="Pickle file with preprocessed groups (dict of DataFrames with 'cleaned_text').")
    parser.add_argument("--model-template", default="models/bertopic_{group}",
                        help="Template path to BERTopic models per group, e.g., models/bertopic_{group}.pkl")
    parser.add_argument("--outdir", default="results/llm", help="Output directory for JSONL files.")
    parser.add_argument("--topn", type=int, default=10, help="Top-N c-TF-IDF keywords to export.")
    parser.add_argument("--topk-snippets", type=int, default=10, help="Top-K representative documents per topic.")
    parser.add_argument("--max-snippet-chars", type=int, default=300, help="Max chars for each snippet.")
    parser.add_argument("--only-groups", nargs="*", default=CANONICAL_ORDER,
                        help="Subset of groups to process (default: A B C D). Use codes like A B.")
    args = parser.parse_args()

    log("Step 8 - Export per-topic artifacts for LLM input")
    log(f"• Loading preprocessed groups: {args.groups_pkl}")
    groups_obj, present_map = load_groups(args.groups_pkl)

    # Determine which canonical codes to process (respect --only-groups if provided)
    requested = args.only_groups  # e.g., ["A","B","C","D"] by default
    to_process = [c for c in CANONICAL_ORDER if c in present_map and c in requested]

    if not to_process:
        log(f"⚠ No overlapping groups to process. Present: {present_map}, requested: {requested}")
        return 0

    for code in to_process:
        actual_key = present_map[code]
        log(f"\n[Group {code} | source key: {actual_key}]")
        df = groups_obj[actual_key].copy()
        df = df[~df["cleaned_text"].isna()]
        df = df[df["cleaned_text"].astype(str).str.strip().str.len() > 0]
        if df.empty:
            log(f"  ⚠ Skipping group {code}: no non-empty cleaned_text rows.")
            continue

        try:
            model = load_model_for_group(args.model_template, group_code=code, actual_key=actual_key)
        except Exception as e:
            log(f"  ✗ Failed to load model for group {code}: {e}")
            continue

        try:
            export_group_topics_jsonl(
                group=code,                 # filenames keep short code: topics_A.jsonl, etc.
                docs_df=df,
                model=model,
                outdir=args.outdir,
                topn_keywords=args.topn,
                topk_snippets=args.topk_snippets,
                max_snippet_chars=args.max_snippet_chars
            )
        except Exception as e:
            log(f"  ✗ Failed exporting topics for group {code}: {e}")
    
    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())