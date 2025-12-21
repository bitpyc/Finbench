"""
Self-refine agent that mirrors the BizBench agent interface.

The agent performs an initial generation followed by iterative self-feedback
refinement steps, reusing the common evaluate_test_set utility for scoring.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from utils.tools import evaluate_test_set
from .generator import SelfRefineGenerator


class SelfRefineAgent:
    """
    Minimal self-refine agent that keeps parity with the ACE/COT interface.
    Only online/eval_only modes are supported because the workflow is
    generation-only (no training loop).
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        refine_rounds: int = 2,
        initial_temperature: float = 0.0,
        feedback_temperature: float = 0.2,
        agent_method: str = "self_refine",
    ):
        self.agent_method = agent_method
        self.max_tokens = max_tokens
        self.refine_rounds = refine_rounds
        self.generator = SelfRefineGenerator(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
            refine_rounds=refine_rounds,
            initial_temperature=initial_temperature,
            feedback_temperature=feedback_temperature,
        )

    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )

        if not test_samples:
            raise ValueError(f"{self.agent_method.upper()} agent requires test samples but none were provided.")

        save_dir = config.get("save_dir")
        if not save_dir:
            raise ValueError("Configuration missing 'save_dir' for SelfRefineAgent.")

        task_name = config.get("task_name", "unknown_task")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - SELF-REFINE EVALUATION")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Refinement rounds: {self.refine_rounds}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        results, error_log = evaluate_test_set(
            data_processor=data_processor,
            generator=self.generator,
            playbook="",
            test_samples=test_samples,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            max_workers=config.get("test_workers", 20),
            use_json_mode=config.get("json_mode", False),
        )

        with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({"test_results": results, "error_log": error_log}, f, indent=2)

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "self_refine_rounds": self.refine_rounds,
            }
        )

        with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Accuracy: {results.get('accuracy', 0.0):.3f}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results







