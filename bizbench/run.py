#!/usr/bin/env python3
"""
Runner script for BizBench tasks supporting multiple agent methods.
"""
import os
import json
import argparse
from datetime import datetime

from .data_processor import DataProcessor
from Agents.ace import ACE
from Agents.cot import ChainOfThoughtAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ACE System - BizBench")

    parser.add_argument(
        "--agent_method",
        type=str,
        default="ace",
        choices=["ace", "cot"],
        help="Agent method to run. 'ace' enables the default ACE loop, 'cot' runs a lightweight chain-of-thought baseline.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=[
            "FinCode",
            "CodeFinQA",
            "CodeTAT-QA",
            "SEC-NUM",
            "TAT-QA",
            "ConvFinQA",
            "FinKnow",
            "FormulaEval",
            "finer",
            "formula",
        ],
        help="BizBench task to run",
    )
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Optional initial playbook")
    parser.add_argument("--config_path", type=str,
                        default="./bizbench/data/task_config.json",
                        help="Path to task config JSON")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")

    parser.add_argument("--api_provider", type=str, default="usd_guiji",
                        choices=["sambanova", "together", "openai", "usd_guiji"])
    parser.add_argument("--generator_model", type=str,
                        default="USD-guiji/deepseek-v3")
    parser.add_argument("--reflector_model", type=str,
                        default="USD-guiji/deepseek-v3")
    parser.add_argument("--curator_model", type=str,
                        default="USD-guiji/deepseek-v3")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=20)

    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold",
                        type=float, default=0.90)

    return parser.parse_args()


def load_data(data_path: str):
    """Load jsonl data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def preprocess_data(task_name: str, config: dict, mode: str):
    """Load and preprocess data according to mode."""
    processor = DataProcessor(task_name=task_name)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" not in config:
            raise ValueError(f"{mode} mode requires test_data in config.")

        test_samples = load_data(config["test_data"])
        test_samples = processor.process_task_data(test_samples)

        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")

    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

        print(
            f"Offline mode: Training on {len(train_samples)} examples, "
            f"validating on {len(val_samples)}, testing on {len(test_samples)}"
        )

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path: str):
    """Load initial playbook if path exists."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("BizBench Runner")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Agent Method: {args.agent_method.upper()}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'='*60}\n")

    with open(args.config_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name,
        task_config[args.task_name],
        args.mode,
    )

    initial_playbook = None
    if args.agent_method == "ace":
        initial_playbook = load_initial_playbook(args.initial_playbook_path)
        if initial_playbook:
            print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
        else:
            print("Using empty playbook as initial playbook\n")
    elif args.initial_playbook_path:
        print("Warning: --initial_playbook_path is ignored for non-ACE agent methods.\n")

    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "initial_playbook_path": args.initial_playbook_path,
        "use_bulletpoint_analyzer": args.use_bulletpoint_analyzer,
        "bulletpoint_analyzer_threshold": args.bulletpoint_analyzer_threshold,
        "api_provider": args.api_provider,
        "agent_method": args.agent_method,
    }

    if args.agent_method == "ace":
        ace_system = ACE(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.reflector_model,
            curator_model=args.curator_model,
            max_tokens=args.max_tokens,
            initial_playbook=initial_playbook,
            use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
            bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
            agent_method=args.agent_method,
        )

        ace_system.run(
            mode=args.mode,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )
    else:
        if args.mode not in ["online", "eval_only"]:
            raise ValueError(f"{args.agent_method.upper()} agent 当前只支持 online/eval_only 模式，收到 {args.mode}")

        cot_agent = ChainOfThoughtAgent(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            max_tokens=args.max_tokens,
            agent_method=args.agent_method,
        )
        cot_agent.run(
            mode=args.mode,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config,
        )


if __name__ == "__main__":
    main()


