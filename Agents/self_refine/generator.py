"""
Self-refine style generator，显式拆分 Init / Feedback / Iterate 三角色：
- Init 生成首答
- Feedback 生成自评与纠错提示
- Iterate 结合反馈重写
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import re

from utils.tools import initialize_clients
from utils.llm import timed_llm_call


class SelfRefineInit:
    """首轮生成角色"""

    def __init__(self, client, api_provider: str, model: str, max_tokens: int, temperature: float):
        self.client = client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(self, question: str, context: str) -> str:
        prompt_lines = [
            "You are a finance-domain reasoning assistant.",
            "First, produce your best answer with careful step-by-step reasoning.",
            "Return a JSON object with ONLY keys `reasoning` and `final_answer`.",
            "- `reasoning`: a concise but complete chain of thought (include code here if needed).",
            "- `final_answer`: strictly follow the question's required answer format (e.g., plain number, [[value]], or code+[[value]] as instructed). Do not add extra keys.",
            "",
            "Question (may include format instructions):",
            question,
            "",
            "Context:",
            context,
            "",
            "Respond strictly in JSON.",
        ]
        return "\n".join(prompt_lines)

    def generate(
        self,
        question: str,
        context: str,
        call_id: str,
        log_dir: Optional[str],
        use_json_mode: bool,
    ) -> Tuple[str, Dict]:
        prompt = self._build_prompt(question, context)
        return timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            prompt,
            role="self_refine_initial",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature,
        )


class SelfRefineFeedback:
    """自评/反馈角色"""

    def __init__(self, client, api_provider: str, model: str, max_tokens: int, temperature: float):
        self.client = client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(self, question: str, context: str, previous_response: str) -> str:
        prompt_lines = [
            "You are reviewing your previous answer for the finance question.",
            "Provide concise feedback: state if the answer is correct, list specific issues, and give short rewrite hints.",
            "Return a JSON object with keys `is_correct` (true/false) and `feedback` (concise bullet-style text).",
            "",
            "Question (may include format instructions):",
            question,
            "",
            "Context:",
            context,
            "",
            "Previous attempt:",
            previous_response,
            "",
            "Respond strictly in JSON.",
        ]
        return "\n".join(prompt_lines)

    def generate(
        self,
        question: str,
        context: str,
        previous_response: str,
        call_id: str,
        log_dir: Optional[str],
        use_json_mode: bool,
    ) -> Tuple[str, Dict]:
        prompt = self._build_prompt(question, context, previous_response)
        return timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            prompt,
            role="self_refine_feedback",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature,
        )


class SelfRefineIterate:
    """按反馈改写角色"""

    def __init__(self, client, api_provider: str, model: str, max_tokens: int, temperature: float):
        self.client = client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_prompt(self, question: str, context: str, previous_response: str, feedback: str) -> str:
        prompt_lines = [
            "You are revising your previous answer for the finance question using the provided feedback.",
            "Return a JSON object with ONLY keys `reasoning` and `final_answer`.",
            "- `reasoning`: a concise but complete chain of thought reflecting the fixes (code may be placed here).",
            "- `final_answer`: strictly follow the question's required answer format (e.g., plain number, [[value]], or code+[[value]] as instructed). Do not add extra keys.",
            "",
            "Question (may include format instructions):",
            question,
            "",
            "Context:",
            context,
            "",
            "Previous attempt:",
            previous_response,
            "",
            "Feedback to address:",
            feedback,
            "",
            "Provide the improved JSON answer:",
        ]
        return "\n".join(prompt_lines)

    def generate(
        self,
        question: str,
        context: str,
        previous_response: str,
        feedback: str,
        call_id: str,
        log_dir: Optional[str],
        use_json_mode: bool,
    ) -> Tuple[str, Dict]:
        prompt = self._build_prompt(question, context, previous_response, feedback)
        return timed_llm_call(
            self.client,
            self.api_provider,
            self.model,
            prompt,
            role="self_refine_iterate",
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=use_json_mode,
            temperature=self.temperature,
        )


class SelfRefineGenerator:
    """
    Self-refine 生成器：内部组合 Init / Feedback / Iterate 三角色，保持 BizBench
    的 generator.generate 接口。
    """

    def __init__(
        self,
        api_provider: str,
        model_name: str,
        max_tokens: int,
        refine_rounds: int = 2,
        initial_temperature: float = 0.0,
        feedback_temperature: float = 0.2,
    ):
        client, _, _ = initialize_clients(api_provider)
        self.api_provider = api_provider
        self.model = model_name
        self.max_tokens = max_tokens
        self.refine_rounds = max(refine_rounds, 0)
        self.initial_temperature = initial_temperature
        self.feedback_temperature = feedback_temperature

        # 三角色实例
        self.init_role = SelfRefineInit(
            client=client,
            api_provider=api_provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=initial_temperature,
        )
        self.feedback_role = SelfRefineFeedback(
            client=client,
            api_provider=api_provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=feedback_temperature,
        )
        self.iterate_role = SelfRefineIterate(
            client=client,
            api_provider=api_provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=feedback_temperature,
        )

    @staticmethod
    def _normalize(text: Optional[str], fallback: str) -> str:
        text = (text or "").strip()
        return text if text else fallback

    def generate(
        self,
        question: str,
        playbook: str = "",
        context: str = "",
        reflection: str = "(empty)",
        use_json_mode: bool = False,
        call_id: str = "self_refine",
        log_dir: Optional[str] = None,
    ) -> Tuple[str, List[str], dict]:
        question_text = self._normalize(question, "(Question is empty.)")
        context_text = self._normalize(context, "(No additional context provided.)")

        # Init
        init_response, init_info = self.init_role.generate(
            question=question_text,
            context=context_text,
            call_id=f"{call_id}_init",
            log_dir=log_dir,
            use_json_mode=use_json_mode,
        )

        best_response = init_response
        call_trace: List[Dict[str, Dict]] = [{"stage": "initial", "call_info": init_info}]

        # Feedback + Iterate loops
        for step in range(1, self.refine_rounds + 1):
            fb_response, fb_info = self.feedback_role.generate(
                question=question_text,
                context=context_text,
                previous_response=best_response,
                call_id=f"{call_id}_fb{step}",
                log_dir=log_dir,
                use_json_mode=use_json_mode,
            )
            call_trace.append({"stage": f"feedback_{step}", "call_info": fb_info})

            # 早停：简单关键字判定
            if re.search(r"\b(correct|no issues|looks good)\b", fb_response, re.IGNORECASE):
                break

            iter_response, iter_info = self.iterate_role.generate(
                question=question_text,
                context=context_text,
                previous_response=best_response,
                feedback=fb_response,
                call_id=f"{call_id}_iter{step}",
                log_dir=log_dir,
                use_json_mode=use_json_mode,
            )
            call_trace.append({"stage": f"iterate_{step}", "call_info": iter_info})
            best_response = iter_response

        return best_response, [], {"trace": call_trace}

