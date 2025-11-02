"""
Step 15 - Aggregate & summarize per model (A, B, C, D)

Reads metric JSONL files produced earlier:
  - Step 12: results/metrics/centroid_sim_{group}_{model_tag}.jsonl
      rows: {"meta": {"group","topic_id","model_tag",...}, "label": "...", "metrics": {"centroid_cosine": float}}
  - Step 13: results/metrics/doc_sim_{group}_{model_tag}.jsonl
      rows: {"meta": {...}, "label": "...", "doc_cosine": {"mean": float, "min": float, "max": float, "n_docs": int}}
  - Step 14: results/metrics/kw_sim_{group}_{model_tag}.jsonl
      rows: {"meta": {...}, "label": "...", "metrics": {"embed_cosine": float, "jaccard": float, ...}}

Produces a Markdown report with, per (group, model), distribution summaries across topics:
  - cosine(label, centroid)
  - mean cosine(label, representative docs)
  - cosine(label, mean(top-word embeddings))
  - Jaccard(label tokens, top-words)
Using summary stats: mean, std, median, IQR, min, max.

Usage:
  python step15_aggregate_report.py \
      --metrics-dir results/metrics \
      --out-md results/metrics/summary_step15.md \
      --only-groups A B C D
"""

import argparse
import glob
import json
import os
import statistics
from typing import Dict, List, Tuple, Optional
import math

CANONICAL_GROUPS = ["A", "B", "C", "D"]

def log(msg: str) -> None:
    print(msg, flush=True)

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows

def safe_stats(values: List[float]) -> Optional[Dict[str, float]]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    # std (sample if n>1 else 0.0)
    if n > 1:
        std = statistics.stdev(vals)
    else:
        std = 0.0
    median = statistics.median(vals)
    q1 = statistics.median(vals[: n // 2]) if n > 1 else vals[0]
    q3 = statistics.median(vals[(n + 1) // 2 :]) if n > 2 else vals[-1]
    iqr = q3 - q1
    vmin = vals[0]
    vmax = vals[-1]
    return {"mean": mean, "std": std, "median": median, "iqr": iqr, "min": vmin, "max": vmax, "n": n}

def fmt_stats(s: Optional[Dict[str, float]]) -> str:
    if not s:
        return "N/A"
    # compact human-readable format
    # mean ± std | median [IQR] | min–max (n)
    return f"{s['mean']:.3f} ± {s['std']:.3f} | {s['median']:.3f} [{s['iqr']:.3f}] | {s['min']:.3f}–{s['max']:.3f} (n={int(s['n'])})"

def file_tag_group(path: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a path like results/metrics/centroid_sim_A_llama-3.1-8b.jsonl
    and prefix 'centroid_sim_', returns ('A', 'llama-3.1-8b')
    """
    base = os.path.basename(path)
    if not base.startswith(prefix):
        return (None, None)
    body = base[len(prefix):]
    if not body.endswith(".jsonl"):
        return (None, None)
    body = body[:-len(".jsonl")]
    # Expect body like "{group}_{tag}"
    if "_" not in body:
        return (None, None)
    group = body.split("_", 1)[0]
    tag = body.split("_", 1)[1]
    return (group, tag)

def collect_step12(metrics_dir: str, only_groups: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Return map[group][model_tag] -> list of centroid_cosine across topics."""
    out: Dict[str, Dict[str, List[float]]] = {g: {} for g in only_groups}
    for path in glob.glob(os.path.join(metrics_dir, "centroid_sim_*.jsonl")):
        g, tag = file_tag_group(path, "centroid_sim_")
        if not g or g not in only_groups:
            continue
        rows = read_jsonl(path)
        vals: List[float] = []
        for r in rows:
            m = (r.get("metrics") or {})
            val = m.get("centroid_cosine")
            if isinstance(val, (int, float)) and not math.isnan(val):
                vals.append(float(val))
        if tag not in out[g]:
            out[g][tag] = []
        out[g][tag].extend(vals)
    return out

def collect_step13(metrics_dir: str, only_groups: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Return map[group][model_tag] -> list of doc_cosine.mean across topics."""
    out: Dict[str, Dict[str, List[float]]] = {g: {} for g in only_groups}
    for path in glob.glob(os.path.join(metrics_dir, "doc_sim_*.jsonl")):
        g, tag = file_tag_group(path, "doc_sim_")
        if not g or g not in only_groups:
            continue
        rows = read_jsonl(path)
        vals: List[float] = []
        for r in rows:
            d = (r.get("doc_cosine") or {})
            val = d.get("mean")
            if isinstance(val, (int, float)) and not math.isnan(val):
                vals.append(float(val))
        if tag not in out[g]:
            out[g][tag] = []
        out[g][tag].extend(vals)
    return out

def collect_step14(metrics_dir: str, only_groups: List[str]) -> Tuple[Dict[str, Dict[str, List[float]]],
                                                                       Dict[str, Dict[str, List[float]]]]:
    """Return:
        kw_embed[group][model_tag] -> list of embed_cosine
        kw_jaccard[group][model_tag] -> list of jaccard
    """
    kw_embed = {g: {} for g in only_groups}
    kw_jacc = {g: {} for g in only_groups}
    for path in glob.glob(os.path.join(metrics_dir, "kw_sim_*.jsonl")):
        g, tag = file_tag_group(path, "kw_sim_")
        if not g or g not in only_groups:
            continue
        rows = read_jsonl(path)
        embs: List[float] = []
        jacs: List[float] = []
        for r in rows:
            m = (r.get("metrics") or {})
            ev = m.get("embed_cosine")
            jv = m.get("jaccard")
            if isinstance(ev, (int, float)) and not math.isnan(ev):
                embs.append(float(ev))
            if isinstance(jv, (int, float)) and not math.isnan(jv):
                jacs.append(float(jv))
        if tag not in kw_embed[g]:
            kw_embed[g][tag] = []
        if tag not in kw_jacc[g]:
            kw_jacc[g][tag] = []
        kw_embed[g][tag].extend(embs)
        kw_jacc[g][tag].extend(jacs)
    return kw_embed, kw_jacc

def union_model_tags(*maps: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[str]]:
    """
    Given multiple group->tag->values maps, build a group->sorted list of all model tags present in any map.
    """
    groups = sorted({g for m in maps for g in m.keys()})
    out: Dict[str, List[str]] = {}
    for g in groups:
        tags = set()
        for m in maps:
            tags |= set(m.get(g, {}).keys())
        out[g] = sorted(tags)
    return out

def build_group_section(group: str,
                        tags: List[str],
                        centroid_map: Dict[str, Dict[str, List[float]]],
                        doc_map: Dict[str, Dict[str, List[float]]],
                        kw_embed_map: Dict[str, Dict[str, List[float]]],
                        kw_jacc_map: Dict[str, Dict[str, List[float]]]) -> str:
    lines: List[str] = []
    lines.append(f"## Group {group}")
    lines.append("")
    lines.append("| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |")
    lines.append("|---|---|---|---|---|")

    for tag in tags:
        s_centroid = safe_stats(centroid_map.get(group, {}).get(tag, []))
        s_doc = safe_stats(doc_map.get(group, {}).get(tag, []))
        s_kw_emb = safe_stats(kw_embed_map.get(group, {}).get(tag, []))
        s_kw_jac = safe_stats(kw_jacc_map.get(group, {}).get(tag, []))

        lines.append(
            f"| `{tag}` | {fmt_stats(s_centroid)} | {fmt_stats(s_doc)} | {fmt_stats(s_kw_emb)} | {fmt_stats(s_kw_jac)} |"
        )

    lines.append("")
    lines.append("> Format: mean ± std | median [IQR] | min–max (n)")
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Step 15: Aggregate & summarize per model for A/B/C/D.")
    ap.add_argument("--metrics-dir", default="results/metrics", help="Directory with *_sim_*.jsonl files.")
    ap.add_argument("--out-md", default="results/metrics/summary_step15.md", help="Output Markdown report path.")
    ap.add_argument("--only-groups", nargs="*", default=CANONICAL_GROUPS, help="Groups to include.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    log("Step 15 - Aggregate & summarize per model")
    log(f"• Metrics dir: {args.metrics_dir}")

    centroid_map = collect_step12(args.metrics_dir, args.only_groups)
    doc_map      = collect_step13(args.metrics_dir, args.only_groups)
    kw_embed_map, kw_jacc_map = collect_step14(args.metrics_dir, args.only_groups)

    # Decide which model tags to show per group (union across all metric maps)
    tag_union = union_model_tags(centroid_map, doc_map, kw_embed_map, kw_jacc_map)

    sections: List[str] = []
    sections.append("# Step 15: Model-wise Summary (per subset)")
    sections.append("")
    sections.append("_Per-metric distributions are computed across topics within each subset._")
    sections.append("")
    for g in args.only_groups:
        tags = tag_union.get(g, [])
        if not tags:
            sections.append(f"## Group {g}\n\n_No metrics found._\n")
            continue
        sections.append(
            build_group_section(g, tags, centroid_map, doc_map, kw_embed_map, kw_jacc_map)
        )

    md = "\n".join(sections)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    log(f"✓ Wrote report → {args.out_md}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())