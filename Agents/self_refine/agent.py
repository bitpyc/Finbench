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

from utils.consulting_tools import evaluate_consulting_set
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
        self.generator_client = self.generator.init_role.client
        self.temperature = 0.7

    # ==========================================================
    # Consulting support: on_case_start / reply / on_case_end
    # ==========================================================

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        Lightweight JSON helper for consulting chat turns.
        """
        import json as _json

        resp = self.generator_client.chat.completions.create(
            model=self.generator.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 512),
        )
        text = (resp.choices[0].message.content or "").strip()

        if not text.startswith("{"):
            i = text.find("{")
            if i >= 0:
                text = text[i:]
        if not text.endswith("}"):
            j = text.rfind("}")
            if j >= 0:
                text = text[: j + 1]

        try:
            return _json.loads(text)
        except Exception:
            return {}

    def on_case_start(self, case_id: str) -> None:
        """
        Hook at the beginning of a consulting case. No cross-case memory kept.
        """
        self._current_case_id = case_id

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        Hook when a consulting case finishes. No persistence for baseline.
        """
        _ = (case_id, case_text, history)
        self._current_case_id = None

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """
        Consulting candidate reply interface (aligned with utils.consulting_tools).
        """
        turns = sum(1 for h in history if h.get("role") == "candidate")

        last_interviewer_msg = ""
        for h in reversed(history):
            if h.get("role") == "interviewer":
                last_interviewer_msg = h.get("content", "")
                break

        transcript_lines = [
            f"{h.get('role', 'unknown')}: {h.get('content', '')}"
            for h in history
        ]
        transcript_text = "\n".join(transcript_lines) or "[no previous dialogue]"

        system = (
            "You are the CANDIDATE in a consulting-style case interview.\n"
            "You only see the dialogue history, not the hidden case text.\n"
            "Act like a top-tier consulting candidate: structured, "
            "hypothesis-driven, quantitative when possible, clear and concise.\n\n"
            "Respond ONLY with what you would say next as the candidate.\n"
            'Wrap your answer in a JSON object of the form:\n'
            '  {\"reply\": \"<your answer>\"}\n'
            "Do not include any other fields."
        )

        user_parts = [
            f"Current case ID: {case_id}",
            "",
            "Dialogue so far (Interviewer / Candidate):",
            transcript_text,
            "",
            "Interviewer just said:",
            last_interviewer_msg or "[no interviewer message found]",
            "",
            "Now respond with your next candidate message, wrapped in JSON "
            'as {\"reply\": \"...\"}.',
        ]
        user_prompt = "\n".join(user_parts)

        data = self._call_llm_json(system=system, user=user_prompt)
        reply = data.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            reply = (
                "Let me structure the drivers, share a hypothesis, and outline the "
                "first analyses I’d run to validate it."
            )
        return reply.strip()

    # ==========================================================
    # Consulting: evaluation entry point
    # ==========================================================

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Consulting / case-interview evaluation entry for the self-refine baseline.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError("Consulting requires non-empty test_samples")

        save_dir = config.get("save_dir", "results")
        task_name = config.get("task_name", "Consulting")
        run_subdir = (
            f"{task_name}/{self.agent_method}/{mode}/"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - CONSULTING EVALUATION")
        print(f"{'='*60}")
        print(f"Cases: {len(test_samples)}")
        print(f"Save dir: {resolved_save_path}")
        print(f"{'='*60}\n")

        results, error_log = evaluate_consulting_set(
            agent=self,
            test_samples=test_samples,
            config=config,
            log_dir=resolved_save_path,
        )

        with open(
            os.path.join(resolved_save_path, "test_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {"test_results": results, "error_log": error_log},
                f,
                indent=2,
            )

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "self_refine_rounds": self.refine_rounds,
            }
        )
        with open(
            os.path.join(resolved_save_path, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_payload, f, indent=2)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - CONSULTING RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Num cases: {results.get('num_cases')}, "
              f"finished: {results.get('num_finished')}, "
              f"failed: {results.get('num_failed')}")
        print(f"Metrics: {results.get('metrics')}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    # ==========================================================
    # Unified run with consulting routing
    # ==========================================================

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

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", ""))).lower()
        if "consult" in task_name:
            return self.run_consulting(mode, test_samples, data_processor, config)

        save_dir = config.get("save_dir")
        if not save_dir:
            raise ValueError("Configuration missing 'save_dir' for SelfRefineAgent.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name or "unknown_task", self.agent_method, mode, timestamp)
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







