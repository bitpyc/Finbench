#!/usr/bin/env python3
"""
Plot difficulty distribution (0-10 buckets) per capability from LLM reclassification jsonl.

Default input:
  results/bizbench_run/llm_reclassify_mode/capability_difficulty_score_v1/test/classifications.jsonl

Outputs (by default into --out_dir=plot/outputs):
  - capability_difficulty_hist__Information_Extraction.png
  - capability_difficulty_hist__Numerical_Calculation.png
  - capability_difficulty_hist__Domain_Knowledge.png
  - capability_difficulty_hist__Complex_Reasoning.png
  - capability_difficulty_hist__grid_2x2.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


CAP_ORDER = [
    "Information Extraction",
    "Numerical Calculation",
    "Domain Knowledge",
    "Complex Reasoning",
]


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def _load_histograms(jsonl_path: Path) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    Returns:
      - hists: capability -> [count bucket0..bucket10]
      - totals: capability -> total samples (with score parsed)
    """
    hists: Dict[str, List[int]] = defaultdict(lambda: [0] * 11)
    totals: Dict[str, int] = Counter()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parsed = obj.get("parsed") or {}
            cap = parsed.get("capability")
            score = parsed.get("difficulty_score")
            if cap is None or score is None:
                continue
            try:
                score_f = float(score)
            except Exception:
                continue
            if math.isnan(score_f):
                continue
            if score_f < 0:
                score_f = 0.0
            if score_f > 10:
                score_f = 10.0
            bucket = int(math.floor(score_f))
            bucket = 0 if bucket < 0 else 10 if bucket > 10 else bucket

            cap = str(cap).strip()
            hists[cap][bucket] += 1
            totals[cap] += 1

    return dict(hists), dict(totals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_jsonl",
        type=str,
        default="results/bizbench_run/llm_reclassify_mode/capability_difficulty_score_v1/test/classifications.jsonl",
    )
    ap.add_argument("--out_dir", type=str, default="plot/outputs")
    ap.add_argument("--title_prefix", type=str, default="Test difficulty distribution")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hists, totals = _load_histograms(input_path)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install it, e.g. `pip install matplotlib`.\n"
            f"Original import error: {e}"
        )

    buckets = list(range(11))

    # Per-capability plots (4 separate PNGs)
    for cap in CAP_ORDER:
        hist = hists.get(cap, [0] * 11)
        total = totals.get(cap, 0)

        fig = plt.figure(figsize=(8, 4.2), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(buckets, hist, color="#4C78A8")
        ax.set_xticks(buckets)
        ax.set_xlabel("Difficulty bucket (floor score)")
        ax.set_ylabel("Count")
        ax.set_title(f"{args.title_prefix}: {cap} (n={total})")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()

        out_path = out_dir / f"capability_difficulty_hist__{_slug(cap)}.png"
        fig.savefig(out_path)
        plt.close(fig)

    # 2x2 grid for convenience
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160, sharex=True, sharey=False)
    axes = axes.flatten()
    for i, cap in enumerate(CAP_ORDER):
        ax = axes[i]
        hist = hists.get(cap, [0] * 11)
        total = totals.get(cap, 0)
        ax.bar(buckets, hist, color="#4C78A8")
        ax.set_title(f"{cap} (n={total})")
        ax.set_xticks(buckets)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if i % 2 == 0:
            ax.set_ylabel("Count")
        if i >= 2:
            ax.set_xlabel("Difficulty bucket (floor score)")

    fig.suptitle(args.title_prefix, y=1.02)
    fig.tight_layout()
    grid_path = out_dir / "capability_difficulty_hist__grid_2x2.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


