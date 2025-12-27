"""
Reflexion agent adapter for BizBench。

支持论文式“失败→反思→记忆→再用”流程：
- 默认窗口并行只读评估 + 窗口内串行写记忆（对齐 dynamic_cheatsheet 的性能折中）
- 可通过 strict_sequential 启用全串行，确保严格时序
记忆以 JSONL 存放于当前 run 目录，检索按 question+context 哈希精确匹配，不做相似度。
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from utils.consulting_tools import evaluate_consulting_set
from utils.tools import extract_answer
from .generator import ReflexionGenerator


class ReflexionAgent:
    """Lightweight Reflexion agent: only online / eval_only modes."""

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        reflexion_rounds: int = 2,
        initial_temperature: float = 0.0,
        reflect_temperature: float = 0.2,
        agent_method: str = "reflexion",
        strict_sequential: bool = True,
        memory_top_k: int = 3,
    ):
        self.agent_method = agent_method
        self.max_tokens = max_tokens
        self.reflexion_rounds = reflexion_rounds
        self.strict_sequential = strict_sequential
        self.memory_top_k = max(memory_top_k, 1)
        self.generator = ReflexionGenerator(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
            reflexion_rounds=reflexion_rounds,
            initial_temperature=initial_temperature,
            reflect_temperature=reflect_temperature,
        )
        self.generator_client = self.generator.init_role.client
        self.temperature = 0.7

        self.memory_path: Optional[str] = None
        self.memory: List[Dict[str, Any]] = []

    # ---------- memory utilities ----------
    @staticmethod
    def _hash_sample(question: str, context: str) -> str:
        digest = hashlib.sha256()
        digest.update((question or "").encode("utf-8"))
        digest.update(b"\n")
        digest.update((context or "").encode("utf-8"))
        return digest.hexdigest()

    def _load_memory(self, path: str) -> None:
        self.memory_path = path
        if not os.path.exists(path):
            self.memory = []
            return
        with open(path, "r", encoding="utf-8") as f:
            self.memory = [json.loads(line) for line in f if line.strip()]

    def _append_memory(self, entry: Dict[str, Any]) -> None:
        if not self.memory_path:
            return
        self.memory.append(entry)
        # 长记忆截断：仅保留最近 memory_top_k 条
        if len(self.memory) > self.memory_top_k:
            self.memory = self.memory[-self.memory_top_k :]
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _retrieve_reflections(self) -> List[str]:
        """
        按原始实现思路：不做相似度/哈希匹配，直接取最近的若干失败反思。
        保留 hash 字段仅用于记录，检索时不依赖它。
        """
        matched = [m for m in self.memory if not m.get("success")]
        matched = matched[-self.memory_top_k :]
        return [m.get("reflection", "") for m in matched if m.get("reflection")]

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
        Hook called at consulting case start. No cross-case memory kept.
        """
        self._current_case_id = case_id

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        Hook called when a consulting case finishes. No persistence for baseline.
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
                "Let me structure the issues, share my hypothesis, and outline the "
                "first analyses I'd run to validate it."
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
        Consulting / case-interview evaluation entry for the Reflexion baseline.
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
                "reflexion_rounds": self.reflexion_rounds,
                "reflexion_strict_sequential": self.strict_sequential,
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
            raise ValueError("Configuration missing 'save_dir' for ReflexionAgent.")

        task_name_safe = task_name or "unknown_task"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name_safe, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - REFLEXION EVALUATION")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Reflexion rounds: {self.reflexion_rounds}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        memory_path = os.path.join(resolved_save_path, "reflections.jsonl")
        self._load_memory(memory_path)

        test_workers = config.get("test_workers", 20)
        online_eval_frequency = config.get("online_eval_frequency", 15)
        json_mode = config.get("json_mode", False)

        results, error_log = self._run_with_memory(
            mode=mode,
            samples=test_samples,
            data_processor=data_processor,
            log_dir=log_dir,
            use_json_mode=json_mode,
            test_workers=test_workers,
            window_size=online_eval_frequency,
        )

        with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({"test_results": results, "error_log": error_log}, f, indent=2)

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": run_subdir,
                "resolved_save_path": resolved_save_path,
                "reflexion_rounds": self.reflexion_rounds,
                "reflexion_strict_sequential": self.strict_sequential,
                "reflexion_memory_path": memory_path,
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

    # ---------- core run with memory ----------
    def _evaluate_single(
        self,
        idx: int,
        sample: Dict[str, Any],
        log_dir: str,
        use_json_mode: bool,
        memory_snapshot: List[Dict[str, Any]],
        data_processor,
    ) -> Dict[str, Any]:
        # build prior reflections from snapshot (to keep read-only during parallel eval)
        question = sample.get("question", "")
        context = sample.get("context", "")
        target = sample.get("target", "")
        if memory_snapshot is self.memory:
            prior_refs = self._retrieve_reflections()
        else:
            prior_refs = [
                m.get("reflection", "")
                for m in memory_snapshot
                if not m.get("success")
            ][-self.memory_top_k :]

        response, _, meta = self.generator.generate(
            question=question,
            playbook="",
            context=context,
            reflection="(empty)",
            prior_reflections=prior_refs,
            use_json_mode=use_json_mode,
            call_id=f"reflexion_{idx}",
            log_dir=log_dir,
        )
        final_answer = extract_answer(response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return {
            "index": idx,
            "question": question,
            "context": context,
            "target": target,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "meta": meta,
            "raw_response": response,
        }

    def _run_window_parallel_eval(
        self,
        samples: List[Dict[str, Any]],
        start_idx: int,
        log_dir: str,
        use_json_mode: bool,
        test_workers: int,
        memory_snapshot: List[Dict[str, Any]],
        data_processor,
    ) -> List[Dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        records: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=test_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single,
                    start_idx + i,
                    sample,
                    log_dir,
                    use_json_mode,
                    memory_snapshot,
                    data_processor,
                ): i
                for i, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda x: x["index"])
        return records

    def _run_sequential(
        self,
        samples: List[Dict[str, Any]],
        start_idx: int,
        log_dir: str,
        use_json_mode: bool,
        data_processor,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for local_idx, sample in enumerate(samples):
            rec = self._evaluate_single(
                start_idx + local_idx,
                sample,
                log_dir,
                use_json_mode,
                self.memory,  # current memory (updated along the way)
                data_processor,
            )
            records.append(rec)
            self._maybe_update_memory(rec)
        return records

    def _maybe_update_memory(self, record: Dict[str, Any]) -> None:
        """If incorrect, append reflection entry to memory."""
        if record.get("is_correct"):
            # 可记录成功样本占位以便去重；目前仅跳过
            return
        reflections = record.get("meta", {}).get("reflections") or []
        if not reflections:
            return
        entry = {
            "hash": self._hash_sample(record.get("question", ""), record.get("context", "")),
            "question": record.get("question", ""),
            "context": record.get("context", ""),
            "reflection": reflections[-1],
            "timestamp": datetime.now().isoformat(),
            "success": False,
        }
        self._append_memory(entry)

    def _run_with_memory(
        self,
        mode: str,
        samples: List[Dict[str, Any]],
        data_processor,
        log_dir: str,
        use_json_mode: bool,
        test_workers: int,
        window_size: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.strict_sequential or mode == "eval_only":
            # eval_only 也走无窗口的严格序列，保证语义
            records = self._run_sequential(samples, 0, log_dir, use_json_mode, data_processor)
        else:
            total = len(samples)
            num_windows = (total + window_size - 1) // window_size
            records: List[Dict[str, Any]] = []
            for w in range(num_windows):
                start = w * window_size
                end = min((w + 1) * window_size, total)
                window_samples = samples[start:end]
                print(f"\n{'='*60}")
                print(f"WINDOW {w + 1}/{num_windows} | Samples {start} to {end - 1}")
                print(f"{'='*60}")

                memory_snapshot = list(self.memory)
                window_records = self._run_window_parallel_eval(
                    window_samples,
                    start,
                    log_dir,
                    use_json_mode,
                    test_workers,
                    memory_snapshot,
                    data_processor,
                )
                records.extend(window_records)

                # sequential memory update using ordered results
                for rec in window_records:
                    self._maybe_update_memory(rec)

        # aggregate metrics
        predictions = [r["final_answer"] for r in records]
        targets = [r["target"] for r in records]
        accuracy = data_processor.evaluate_accuracy(predictions, targets) if predictions and targets else 0.0
        errors = [
            {"index": r["index"], "prediction": r["final_answer"], "ground_truth": r["target"]}
            for r in records
            if not r.get("is_correct")
        ]
        results = {
            "accuracy": accuracy,
            "total": len(records),
            "correct": len(records) - len(errors),
            "samples": records,
        }
        error_log = {"errors": errors, "accuracy": accuracy}
        return results, error_log


