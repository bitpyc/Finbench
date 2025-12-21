from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core.language_model import DynamicCheatsheetLanguageModel
from .core.state import CheatsheetState


DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


@dataclass
class DynamicCheatsheetConfig:
    approach_name: str = "DynamicCheatsheet_Cumulative"
    generator_prompt_path: Path = DEFAULT_PROMPTS_DIR / "generator_prompt.txt"
    cheatsheet_prompt_path: Optional[Path] = DEFAULT_PROMPTS_DIR / "cheatsheet_cumulative.txt"
    temperature: float = 0.0
    max_num_rounds: int = 1
    retrieve_top_k: int = 3
    allow_code_execution: bool = True
    add_previous_answers: bool = True


class DynamicCheatsheetAgent:
    """
    BizBench-compatible Agent implementation of Dynamic Cheatsheet.
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        agent_method: str = "dynamic_cheatsheet",
        dc_config: Optional[DynamicCheatsheetConfig] = None,
    ):
        self.agent_method = agent_method
        self.max_tokens = max_tokens
        self.dc_config = dc_config or DynamicCheatsheetConfig()

        self.language_model = DynamicCheatsheetLanguageModel(
            api_provider=api_provider,
            model_name=generator_model,
            max_tokens=max_tokens,
            allow_code_execution=self.dc_config.allow_code_execution,
        )

        self.generator_prompt = _read_file(self.dc_config.generator_prompt_path)
        self.cheatsheet_prompt = (
            _read_file(self.dc_config.cheatsheet_prompt_path)
            if self.dc_config.cheatsheet_prompt_path
            else "(empty)"
        )
        self.retrieval_corpus: List[str] = []
        self.retrieval_embeddings: List[List[float]] = []
        self.retrieval_index: Dict[str, int] = {}
        self.retrieval_outputs: Dict[int, str] = {}

    def _prepare_dirs(self, task_name: str, mode: str, save_dir: str) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)
        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        cheatsheet_history_path = os.path.join(resolved_save_path, "cheatsheet_history.jsonl")

        return {
            "run_subdir": run_subdir,
            "resolved_save_path": resolved_save_path,
            "log_dir": log_dir,
            "cheatsheet_history_path": cheatsheet_history_path,
        }

    def _load_initial_cheatsheet(self, path: Optional[str]) -> str:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "(empty)"

    def _format_input(self, sample: Dict[str, Any], index: int, question_override: Optional[str] = None) -> str:
        raw_question = question_override if question_override is not None else sample.get("question", "")
        raw_context = sample.get("context", "")
        question = (raw_question or "").strip()
        context = (raw_context or "").strip()
        parts = [f"Question #{index + 1}:\n{question}"]
        if context:
            parts.append(f"\nContext:\n{context}")
        return "\n".join(parts).strip()

    def _select_prompts_for_task(self, task_name: str, base_generator_prompt: str, base_cheatsheet_prompt: str) -> Tuple[str, str]:
        """
        针对特定任务做 prompt 特化：
        - FormulaEval：只输出当前缺失的函数体，单一 python 代码块，无 JSON/解释/EXECUTE。
        """
        if task_name.lower() == "formulaeval":
            generator_prompt = (
                "You are a code completion assistant. The function/method signature is provided above.\n"
                "Fill in ONLY the missing body lines with correct indentation.\n"
                "Do NOT repeat the class definition or the signature.\n"
                "Do NOT return JSON or analysis; only return one ```python``` code block containing the body lines.\n"
                "If the signature is present, just give the indented body lines; if only the body is missing, fill it directly.\n"
                "Ensure the code is executable and self-contained (no external files/paths/imports beyond stdlib if needed).\n\n"
                "Question:\n[[QUESTION]]"
            )
            cheatsheet_prompt = base_cheatsheet_prompt  # 保持原有检索小抄，但输出格式受 generator_prompt 约束
            return generator_prompt, cheatsheet_prompt

        return base_generator_prompt, base_cheatsheet_prompt

    def _append_history(self, path: str, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _snapshot_state(self, state: CheatsheetState) -> CheatsheetState:
        """
        创建 cheatsheet 状态的浅拷贝，用于并行只读评估。
        """
        return CheatsheetState(
            cheatsheet_text=state.cheatsheet_text,
            previous_answers=list(state.previous_answers),
            input_history=list(state.input_history),
            output_history=list(state.output_history),
            embeddings=list(state.embeddings) if state.embeddings else None,
        )

    def _process_sample(
        self,
        sample: Dict[str, Any],
        idx: int,
        state: CheatsheetState,
        data_processor,
        paths: Dict[str, str],
        strip_task_instruction_fn,
        update_state: bool = True,
        log_history: bool = True,
        verbose: bool = False,
        call_prefix: Optional[str] = None,
        retrieval_corpus: Optional[List[str]] = None,
        retrieval_embeddings: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        统一的单样本处理逻辑，可用于训练（更新 state）或只读评估。
        """
        input_txt = self._format_input(sample, idx)
        stripped_question = strip_task_instruction_fn(sample.get("question", ""))
        input_txt_for_cheatsheet = self._format_input(sample, idx, question_override=stripped_question)
        target = sample.get("target")

        original_input_corpus = state.input_history + [input_txt]
        original_input_embeddings = None
        if retrieval_corpus and retrieval_embeddings:
            idx_in_retrieval = self.retrieval_index.get(input_txt)
            if idx_in_retrieval is None or idx_in_retrieval >= len(retrieval_embeddings):
                idx_in_retrieval = len(retrieval_embeddings) - 1
            permutation = [i for i in range(len(retrieval_corpus)) if i != idx_in_retrieval] + [idx_in_retrieval]
            prev_inputs = [retrieval_corpus[i] for i in permutation]
            prev_embeddings = [retrieval_embeddings[i] for i in permutation]
            original_input_corpus = prev_inputs
            original_input_embeddings = prev_embeddings
            generator_outputs_so_far = [self.retrieval_outputs.get(i, "") for i in permutation]
        else:
            generator_outputs_so_far = state.output_history

        output_dict = self.language_model.advanced_generate(
            approach_name=self.dc_config.approach_name,
            input_txt=input_txt,
            cheatsheet=state.cheatsheet_text,
            generator_template=self.generator_prompt_for_run,
            cheatsheet_template=self.cheatsheet_prompt_for_run,
            cheatsheet_question=input_txt_for_cheatsheet,
            temperature=self.dc_config.temperature,
            max_tokens=self.max_tokens,
            max_num_rounds=self.dc_config.max_num_rounds,
            allow_code_execution=self.dc_config.allow_code_execution,
            code_execution_flag="EXECUTE CODE!",
            add_previous_answers_to_cheatsheet=self.dc_config.add_previous_answers,
            original_input_corpus=original_input_corpus,
            original_input_embeddings=original_input_embeddings,
            generator_outputs_so_far=generator_outputs_so_far,
            retrieve_top_k=self.dc_config.retrieve_top_k,
            log_dir=paths["log_dir"],
            call_prefix=call_prefix or f"sample_{idx}",
        )

        final_answer = output_dict.get("final_answer")
        final_cheatsheet = output_dict.get("final_cheatsheet") or state.cheatsheet_text
        is_correct = data_processor.answer_is_correct(final_answer, target)

        if update_state:
            state.update_cheatsheet(final_cheatsheet)
            state.append_example(
                input_txt=input_txt,
                generator_output=output_dict.get("final_output", ""),
                generator_answer=final_answer or "",
            )
            if retrieval_corpus and retrieval_embeddings:
                idx_in_retrieval = self.retrieval_index.get(input_txt)
                if idx_in_retrieval is not None:
                    self.retrieval_outputs[idx_in_retrieval] = final_answer or ""

        if log_history and update_state:
            history_payload = {
                "index": idx,
                "input": input_txt,
                "steps": output_dict.get("steps"),
                "final_cheatsheet": final_cheatsheet,
            }
            self._append_history(paths["cheatsheet_history_path"], history_payload)

        if verbose:
            print(f"[Sample {idx + 1}] Correct: {is_correct} | Final Answer: {final_answer}")

        return {
            "index": idx,
            "input": input_txt,
            "target": target,
            "final_answer": final_answer,
            "final_output": output_dict.get("final_output"),
            "cheatsheet": final_cheatsheet,
            "is_correct": is_correct,
        }

    def _load_embedding_csv(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        corpus: List[str] = []
        embeddings: List[List[float]] = []
        index_map: Dict[str, int] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row.get("input", "")
                emb_json = row.get("embedding_json") or row.get("embedding")
                if not text or not emb_json:
                    continue
                try:
                    emb = json.loads(emb_json)
                except json.JSONDecodeError:
                    continue
                corpus.append(text)
                embeddings.append(emb)
                index_map[text] = i

        if not corpus or not embeddings:
            raise ValueError(f"No valid embeddings found in {path}")

        self.retrieval_corpus = corpus
        self.retrieval_embeddings = embeddings
        self.retrieval_index = index_map

    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError(f"{self.agent_method.upper()} agent requires test samples but none were provided.")

        save_dir = config.get("save_dir")
        if not save_dir:
            raise ValueError("Configuration missing 'save_dir' for DynamicCheatsheetAgent.")
        task_name = config.get("task_name", "unknown_task")

        paths = self._prepare_dirs(task_name, mode, save_dir)
        state = CheatsheetState(
            cheatsheet_text=self._load_initial_cheatsheet(config.get("initial_cheatsheet_path"))
        )

        retrieval_required = {
            "Dynamic_Retrieval",
            "DynamicCheatsheet_RetrievalSynthesis",
        }

        data_embedding_dir = config.get("data_embedding_dir", "bizbench/data/data_embedding")
        if self.dc_config.approach_name in retrieval_required:
            dataset_name = task_name.lower()
            embedding_path = os.path.join(data_embedding_dir, f"{dataset_name}_test_embeddings.csv")
            self._load_embedding_csv(embedding_path)

        def strip_task_instruction(question: str) -> str:
            try:
                from bizbench.data_processor import DataProcessor

                instruction_values = list(DataProcessor.TASK_INSTRUCTIONS.values()) + list(
                    DataProcessor.SPECIAL_FIN_INSTRUCTIONS.values()
                )
                for instr in instruction_values:
                    if instr and question.startswith(instr):
                        return question[len(instr):].lstrip()
            except Exception:
                return question
            return question

        # 根据任务选择合适的 prompt（FormulaEval 等任务使用特化提示）
        self.generator_prompt_for_run, self.cheatsheet_prompt_for_run = self._select_prompts_for_task(
            task_name, self.generator_prompt, self.cheatsheet_prompt
        )

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - DYNAMIC CHEATSHEET")
        print(f"{'='*60}")
        print(f"Samples: {len(test_samples)}")
        print(f"Approach: {self.dc_config.approach_name}")
        print(f"Log dir: {paths['log_dir']}")
        print(f"{'='*60}\n")

        test_workers = config.get("test_workers", 20)
        online_eval_frequency = config.get("online_eval_frequency", 15)

        def parallel_evaluate(
            samples: List[Dict[str, Any]],
            state_snapshot: CheatsheetState,
            start_idx: int = 0,
            prefix: str = "eval",
            retrieval_corpus: Optional[List[str]] = None,
            retrieval_embeddings: Optional[List[List[float]]] = None,
        ):
            results = []
            with ThreadPoolExecutor(max_workers=test_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_sample,
                        sample,
                        start_idx + idx,
                        state_snapshot,
                        data_processor,
                        paths,
                        strip_task_instruction,
                        False,  # update_state
                        False,  # log_history
                        False,  # verbose
                        f"{prefix}_{start_idx + idx}",
                        retrieval_corpus=retrieval_corpus,
                        retrieval_embeddings=retrieval_embeddings,
                    ): idx
                    for idx, sample in enumerate(samples)
                }
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda x: x["index"])
            return results

        sample_records: List[Dict[str, Any]] = []
        predictions: List[str] = []
        targets: List[Any] = []
        errors: List[Dict[str, Any]] = []

        retrieval_corpus = self.retrieval_corpus if self.retrieval_corpus else None
        retrieval_embeddings = self.retrieval_embeddings if self.retrieval_embeddings else None

        if mode == "eval_only":
            eval_records = parallel_evaluate(
                test_samples,
                self._snapshot_state(state),
                0,
                prefix="eval",
                retrieval_corpus=retrieval_corpus,
                retrieval_embeddings=retrieval_embeddings,
            )
            for rec in eval_records:
                sample_records.append(rec)
                predictions.append(rec["final_answer"])
                targets.append(rec["target"])
                if not rec["is_correct"]:
                    errors.append({"index": rec["index"], "prediction": rec["final_answer"], "ground_truth": rec["target"]})
        else:  # online
            total = len(test_samples)
            num_windows = (total + online_eval_frequency - 1) // online_eval_frequency
            print(f"Online windowed eval/train: total {total}, window size {online_eval_frequency}, windows {num_windows}")
            for window_idx in range(num_windows):
                start_idx = window_idx * online_eval_frequency
                end_idx = min((window_idx + 1) * online_eval_frequency, total)
                window_samples = test_samples[start_idx:end_idx]

                print(f"\n{'='*60}")
                print(f"WINDOW {window_idx + 1}/{num_windows} | Samples {start_idx} to {end_idx - 1}")
                print(f"{'='*60}")

                # 1) 并行只读评估（使用当前 cheatsheet 快照）
                snapshot = self._snapshot_state(state)
                window_records = parallel_evaluate(
                    window_samples,
                    snapshot,
                    start_idx,
                    prefix=f"eval_w{window_idx+1}",
                    retrieval_corpus=retrieval_corpus,
                    retrieval_embeddings=retrieval_embeddings,
                )
                window_correct = sum(1 for r in window_records if r["is_correct"])
                window_total = len(window_records)
                window_acc = window_correct / window_total if window_total else 0.0
                print(f"[Window {window_idx + 1}] Eval accuracy: {window_acc:.3f} ({window_correct}/{window_total})")

                sample_records.extend(window_records)
                predictions.extend([r["final_answer"] for r in window_records])
                targets.extend([r["target"] for r in window_records])
                errors.extend(
                    [{"index": r["index"], "prediction": r["final_answer"], "ground_truth": r["target"]} for r in window_records if not r["is_correct"]]
                )

                # 2) 串行训练/更新（保持顺序一致）
                print(f"[Window {window_idx + 1}] Start training/cheatsheet update (sequential)")
                for local_idx, sample in enumerate(window_samples):
                    global_idx = start_idx + local_idx
                    self._process_sample(
                        sample,
                        global_idx,
                        state,
                        data_processor,
                        paths,
                        strip_task_instruction,
                        update_state=True,
                        log_history=True,
                        verbose=True,
                        call_prefix=f"train_{global_idx}",
                        retrieval_corpus=retrieval_corpus,
                        retrieval_embeddings=retrieval_embeddings,
                    )

        accuracy = data_processor.evaluate_accuracy(predictions, targets) if predictions and targets else 0.0

        test_results = {
            "accuracy": accuracy,
            "total": len(sample_records),
            "correct": sum(1 for r in sample_records if r["is_correct"]),
            "samples": sample_records,
        }
        error_log = {"errors": errors, "accuracy": accuracy}

        with open(os.path.join(paths["resolved_save_path"], "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({"test_results": test_results, "error_log": error_log}, f, indent=2)

        config_payload = dict(config)
        config_payload.update(
            {
                "run_subdir": paths["run_subdir"],
                "resolved_save_path": paths["resolved_save_path"],
                "dynamic_cheatsheet": {
                    "approach_name": self.dc_config.approach_name,
                    "generator_prompt_path": str(self.dc_config.generator_prompt_path),
                    "cheatsheet_prompt_path": str(self.dc_config.cheatsheet_prompt_path)
                    if self.dc_config.cheatsheet_prompt_path
                    else None,
                },
            }
        )

        with open(os.path.join(paths["resolved_save_path"], "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2)

        final_cheatsheet_path = os.path.join(paths["resolved_save_path"], "final_cheatsheet.txt")
        with open(final_cheatsheet_path, "w", encoding="utf-8") as f:
            f.write(state.cheatsheet_text)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Results saved to: {paths['resolved_save_path']}")
        print(f"{'='*60}\n")

        return test_results

