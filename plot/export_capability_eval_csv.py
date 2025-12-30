#!/usr/bin/env python3
"""
Export capability difficulty-bucket accuracy into a CSV table.

Reads:
  results/StructuredReasoning_run/capability_eval_mode/<agent_method>/<mode>/<timestamp>/summary.json

Writes:
  A CSV shaped like:
    capability,difficulty,cot,self-refine,reflexion,dc,gepa,ace,amem

Notes on bucketing:
  - Information Extraction: not bucketed. We repeat the overall capability accuracy into easy/middle/hard rows.
  - Complex Reasoning: only middle/hard are meaningful. We leave easy empty.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CAPS = [
    "Information Extraction",
    "Numerical Calculation",
    "Domain Knowledge",
    "Complex Reasoning",
]

DIFFS = ["easy", "middle", "hard"]

# Display column -> folder name mapping
AGENT_COL_TO_DIR = {
    "cot": "cot",
    "self-refine": "self_refine",
    "reflexion": "reflexion",
    "dc": "dynamic_cheatsheet",
    "gepa": "gepa",
    "ace": "ace",
    "amem": "amem",
    "aoa": "aoa",
}


def _find_latest_summary(root: Path, agent_dir: str, mode: str) -> Optional[Path]:
    base = root / agent_dir / mode
    if not base.exists():
        return None
    # timestamp dirs: YYYYMMDD_HHMMSS
    candidates = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
    for d in candidates:
        s = d / "summary.json"
        if s.exists():
            return s
    return None


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cell_value(summary: dict, cap: str, diff: str) -> str:
    by_cap = summary.get("by_capability", {}) or {}
    by_cap_bucket = summary.get("by_capability_bucket", {}) or {}

    # capability overall accuracy
    cap_acc = None
    if cap in by_cap and isinstance(by_cap[cap], dict):
        cap_acc = by_cap[cap].get("accuracy", None)

    if cap == "Information Extraction":
        # Not bucketed: repeat same number to fill the 3 rows.
        return "" if cap_acc is None else f"{float(cap_acc):.4f}"

    if cap == "Complex Reasoning":
        # Only middle/hard meaningful, easy blank
        if diff == "easy":
            return ""
        cap_b = by_cap_bucket.get(cap, {}) or {}
        if diff in cap_b and isinstance(cap_b[diff], dict) and cap_b[diff].get("total", 0) > 0:
            return f"{float(cap_b[diff]['accuracy']):.4f}"
        return ""

    # Numerical Calculation / Domain Knowledge
    cap_b = by_cap_bucket.get(cap, {}) or {}
    if diff in cap_b and isinstance(cap_b[diff], dict) and cap_b[diff].get("total", 0) > 0:
        return f"{float(cap_b[diff]['accuracy']):.4f}"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cap_eval_root",
        type=str,
        default="results/StructuredReasoning_run/capability_eval_mode",
    )
    ap.add_argument("--mode", type=str, default="online")
    ap.add_argument(
        "--agents",
        type=str,
        default="cot,self-refine,reflexion,dc,gepa,ace,amem",
        help="Comma-separated display column names (subset of: cot,self-refine,reflexion,dc,gepa,ace,amem).",
    )
    ap.add_argument("--out_csv", type=str, default="plot/outputs_capability_eval/capability_eval_online.csv")
    args = ap.parse_args()

    root = Path(args.cap_eval_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    agent_cols: List[str] = [a.strip() for a in args.agents.split(",") if a.strip()]

    # Load summaries (latest per agent/mode)
    summaries: Dict[str, dict] = {}
    for col in agent_cols:
        agent_dir = AGENT_COL_TO_DIR.get(col)
        if not agent_dir:
            continue
        s_path = _find_latest_summary(root, agent_dir, args.mode)
        if s_path is None:
            continue
        summaries[col] = _load_summary(s_path)

    header = ["capability", "difficulty"] + agent_cols
    rows: List[List[str]] = []
    for cap in CAPS:
        for diff in DIFFS:
            row = [cap, diff]
            for col in agent_cols:
                summ = summaries.get(col)
                row.append("" if summ is None else _cell_value(summ, cap, diff))
            rows.append(row)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Wrote: {out_csv.resolve()}")


if __name__ == "__main__":
    main()





