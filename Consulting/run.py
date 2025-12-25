# Consulting/run.py

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from Agents.amem.agent import AMemAgent


def load_consulting_cases(data_path: str) -> List[Dict[str, Any]]:
    """
    根据当前 agsm_cases_all.json 的格式加载咨询案例。

    数据格式（以 agsm_cases_all.json 为例）:
    {
        "CABLE TELEVISION COMPANY": "CABLE TELEVISION COMPANY  ...  Page 21",
        "CHILLED BEVERAGES": "CHILLED BEVERAGES  ...  Page 22",
        ...
    }

    我们将其转换为统一的内部格式：
    [
        {
            "case_id": "case_0000",
            "case_text": "<完整案例文本>",
            "opening": "<案例开头一段或标题>",
            "interviewer_prompts": [],
            "meta": {"title": "<原始key标题>"}
        },
        ...
    ]
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Consulting data not found: {data_path}")

    samples: List[Dict[str, Any]] = []

    # 支持 jsonl / json，两种都兼容
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # 当前 agsm_cases_all.json 的情况：顶层是 {title: full_text, ...}
        if isinstance(obj, dict):
            for i, (title, full_text) in enumerate(obj.items()):
                text = full_text if isinstance(full_text, str) else str(full_text)

                # 简单取开头一段作为 opening（按两个换行分段）
                parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                opening = parts[0] if parts else title

                samples.append(
                    {
                        "case_id": f"case_{i:04d}",
                        "case_text": text,
                        "opening": opening,
                        "interviewer_prompts": [],  # 目前先留空，后续如需逐轮脚本可再扩展
                        "meta": {"title": title},
                    }
                )
        # 兼容老格式：顶层就是 list 或 {"cases": [...]} 的情况
        elif isinstance(obj, list):
            samples = obj
        else:
            samples = obj.get("cases", [])

    # 统一做一次标准化，确保字段齐全，避免 evaluate_consulting_set 出现缺字段
    normalized: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        cid = s.get("case_id") or f"case_{i:04d}"
        normalized.append(
            {
                "case_id": cid,
                "case_text": s.get("case_text", ""),
                "opening": s.get("opening", ""),
                "interviewer_prompts": s.get("interviewer_prompts", []),
                "meta": s.get("meta", {}),
            }
        )
    return normalized[:3]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to consulting cases (json/jsonl).",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        default="openai",
        help="API provider name (e.g. 'openai').",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="gpt-5",
        help="LLM model name for the candidate agent.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Root directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="eval_only",
        choices=["eval_only", "online"],
        help="Mode, kept consistent with other tasks.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=12,
        help="Maximum interview turns per case.",
    )
    parser.add_argument(
        "--consulting_judge",
        action="store_true",
        help="If set, run LLM judge to score each interview.",
    )
    args = parser.parse_args()

    # 1) 加载 / 标准化数据 -> test_samples
    test_samples = load_consulting_cases(args.data_path)

    # 2) 构造统一 AMemAgent（与 BeerGame / BizBench 一致）
    agent = AMemAgent(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        max_tokens=1024,
        agent_method="amem",
    )

    # 3) 组装 config，让 .run() 路由到 run_consulting
    config: Dict[str, Any] = {
        "task_name": "Consulting",          # 含 "consult"，会自动走 run_consulting
        "save_dir": args.save_dir,
        "consulting_max_turns": args.max_turns,
        "consulting_judge": args.consulting_judge,
        # 如有其它评测参数（如 judge 模型、prompt）也可以继续塞在 config 里
    }

    # 4) 统一入口：test_samples 由 run.py 提供，evaluate_consulting_set 不再读文件
    results = agent.run(
        mode=args.mode,
        test_samples=test_samples,
        data_processor=None,
        config=config,
    )

    print("=== Consulting evaluation finished ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
