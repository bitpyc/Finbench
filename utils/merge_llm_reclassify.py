#!/usr/bin/env python3
"""
Merge LLM reclassification caches by overriding specified tasks.

Use-case:
  - You have a base full cache (all tasks) in one root, e.g.:
      results/bizbench_run/llm_reclassify_mode/capability_difficulty_score_v1
  - You re-run one task (e.g., finer) with a better prompt in another root, e.g.:
      results/StructuredReasoning_run/llm_reclassify_mode/capability_difficulty_score_v2_finer_difficulty
  - You want a single "complete" cache root that is base + (override finer).

This script writes:
  <out_root>/<split>/classifications.jsonl
  <out_root>/<split>/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def _summarize(jsonl_path: Path) -> dict:
    cap_counts = Counter()
    score_hist = {str(i): 0 for i in range(11)}
    score_count = 0
    score_sum = 0.0
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    unparsed = 0

    for obj in _iter_jsonl(jsonl_path):
        parsed = obj.get("parsed") or {}
        cap = parsed.get("capability")
        if cap is not None:
            cap_counts[str(cap)] += 1
        s = parsed.get("difficulty_score")
        if s is None:
            unparsed += 1
            continue
        try:
            sf = float(s)
        except Exception:
            unparsed += 1
            continue
        if math.isnan(sf):
            unparsed += 1
            continue
        sf = 0.0 if sf < 0 else 10.0 if sf > 10 else sf
        score_count += 1
        score_sum += sf
        score_min = sf if score_min is None else min(score_min, sf)
        score_max = sf if score_max is None else max(score_max, sf)
        b = int(math.floor(sf))
        b = 0 if b < 0 else 10 if b > 10 else b
        score_hist[str(b)] += 1

    mean = (score_sum / score_count) if score_count else None
    return {
        "counts": {
            "capability": dict(cap_counts),
            "difficulty_score": {
                "count": score_count,
                "min": score_min,
                "max": score_max,
                "mean": mean,
                "hist_0_10": score_hist,
                "unparsed": unparsed,
            },
        }
    }


def _merge_split(
    base_jsonl: Path,
    overlay_jsonl: Optional[Path],
    override_tasks: List[str],
) -> List[dict]:
    override_set = set(override_tasks)
    merged: List[dict] = []

    # 1) base without override tasks
    for obj in _iter_jsonl(base_jsonl):
        task = obj.get("task")
        if task in override_set:
            continue
        merged.append(obj)

    # 2) overlay for override tasks (if provided)
    if overlay_jsonl and overlay_jsonl.exists():
        for obj in _iter_jsonl(overlay_jsonl):
            task = obj.get("task")
            if task in override_set:
                merged.append(obj)

    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", type=str, required=True, help="Root containing <split>/classifications.jsonl")
    ap.add_argument("--overlay_root", type=str, required=True, help="Root containing <split>/classifications.jsonl")
    ap.add_argument("--out_root", type=str, required=True, help="Output root to write merged cache")
    ap.add_argument("--splits", type=str, default="test,train,val", help="Comma-separated splits to process")
    ap.add_argument("--override_tasks", type=str, default="finer", help="Comma-separated task names to override")
    args = ap.parse_args()

    base_root = Path(args.base_root)
    overlay_root = Path(args.overlay_root)
    out_root = Path(args.out_root)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    override_tasks = [t.strip() for t in args.override_tasks.split(",") if t.strip()]

    for split in splits:
        base_jsonl = base_root / split / "classifications.jsonl"
        if not base_jsonl.exists():
            raise FileNotFoundError(f"Missing base jsonl: {base_jsonl}")

        overlay_jsonl = overlay_root / split / "classifications.jsonl"
        if not overlay_jsonl.exists():
            # Allow overlay missing non-test splits (e.g. you only re-ran finer test)
            overlay_jsonl = None

        merged = _merge_split(base_jsonl, overlay_jsonl, override_tasks)

        out_dir = out_root / split
        out_jsonl = out_dir / "classifications.jsonl"
        n = _write_jsonl(out_jsonl, merged)
        summ = _summarize(out_jsonl)
        summ.update(
            {
                "base_root": str(base_root),
                "overlay_root": str(overlay_root),
                "override_tasks": override_tasks,
                "split": split,
                "merged_lines": n,
                "output_jsonl": str(out_jsonl),
            }
        )
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)

        print(f"[merge] split={split} merged_lines={n} -> {out_dir}")


if __name__ == "__main__":
    main()




