# model/a_mem/agent.py
"""
A-mem 版本的候选人，实现 CandidateAgent 接口。

只依赖 memory_layer.py（AgenticMemorySystem）和 case_judge 提供的工具。
"""

import os
from typing import List, Dict

from model.base import CandidateAgent
from memory_layer import AgenticMemorySystem
from case_dataset import CandidateStateView, history_to_transcript


CANDIDATE_SYSTEM_PROMPT = """
You are an LLM acting as the candidate in a financial consulting case interview. 
The interviewer will describe business situations and ask you questions based on an internal case text that you do not see directly. 
Your job is to think like a strong consulting candidate: clarify the objective, structure the problem, identify key drivers 
(market, customers, revenues, costs, operations, risks), and give clear, business-oriented reasoning and recommendations.

You may decide that you need additional information before giving a good answer. In that case, ask the interviewer a small number of targeted, 
high-leverage questions that are closely related to the case (for example about market size, demand trends, customer segments, pricing, 
cost structure, competitive landscape, or constraints). The interviewer may only reveal extra information when you ask the right kind of question, 
so avoid broad, unfocused questioning; focus on what is most critical to move your analysis forward. If you believe you already have enough information, 
do not ask more questions and instead proceed with your analysis and answer.

Base your reasoning only on what the interviewer has told you plus standard, reasonable business assumptions that you state explicitly. 
Do not invent concrete facts that contradict the case. Keep your tone professional and concise. 
In each turn, either (a) ask a few precise questions that would meaningfully advance the case, 
or (b) use the available information to walk through your structured thinking and provide a clear answer or next-step recommendation. 
Do not engage in any small talk or meta-discussion; everything you say should be directly related to solving the case.
"""


class AMemCandidateAgent(CandidateAgent):
    """
    使用 a-mem 的 AgenticMemorySystem 作为“长时记忆”的候选人实现。

    约束：
    - 在 reply() 过程中，永远不会看到当前 case 的 case_text；
      只能用对话历史 + 自己的长期记忆。
    - 在 on_case_end() 中，才把当前 case_text 写入长期记忆。
    """

    def __init__(
        self,
        name: str = "a-mem",
        model: str = "gpt-4o-mini",
        backend: str = "sglang",      # "openai" / "sglang" / "ollama"
        retrieve_k: int = 8,
        temperature: float = 0.5,
        api_base: str = "http://35.220.164.252:3888/v1/",
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        reset_memory_each_case: bool = False,
    ):
        super().__init__(name=name)
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.temperature = temperature
        self.api_base = api_base
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.reset_memory_each_case = reset_memory_each_case

        # 初始化 a-mem 记忆系统（签名与 memory_layer.py 中 AgenticMemorySystem 保持一致）
        self.memory_system = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend=self.backend,
            llm_model=self.model,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=self.api_base,
            sglang_host=self.sglang_host,
            sglang_port=self.sglang_port,
        )

    # ---------- 生命周期 hook ----------

    def on_case_start(self, case_id: str):
        """每道题开始前：按需重置记忆"""
        if self.reset_memory_each_case:
            self.memory_system = AgenticMemorySystem(
                model_name="all-MiniLM-L6-v2",
                llm_backend=self.backend,
                llm_model=self.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=self.api_base,
                sglang_host=self.sglang_host,
                sglang_port=self.sglang_port,
            )

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ):
        """
        面试结束后，把当前 case_text 写入长期记忆。
        """
        blocks = [b.strip() for b in case_text.split("\n\n") if b.strip()]
        for i, block in enumerate(blocks):
            content = f"[CASE {case_id} | CHUNK {i}]\n{block}"
            self.memory_system.add_note(content=content, time=None)
        self.memory_system.consolidate_memories()

    # ---------- 必须实现的接口：reply ----------

    def reply(
        self,
        state: CandidateStateView,
        interviewer_msg: str,
    ) -> str:
        """
        根据对话历史 + 面试官最新问题 + 当前长期记忆，生成下一句候选人回答。
        """
        # 1) 检索长期记忆（不会命中当前 case_text，因为只在 on_case_end 之后写入）
        try:
            retrieved_context = self.memory_system.find_related_memories_raw(
                interviewer_msg,
                k=self.retrieve_k,
            )
        except Exception:
            retrieved_context = ""

        # 2) 对话历史
        transcript = history_to_transcript(state.history)

        # 3) 拼 prompt，使用 memory_system 内部的 LLMController 生成回答
        user_prompt = (
            "You are the Candidate in a consulting-style case interview.\n\n"
            f"Current case ID: {state.case_id}\n\n"
            "Here is the dialogue so far between the Interviewer and the Candidate:\n"
            f"{transcript}\n\n"
            "Here is background information retrieved from your long-term memories:\n"
            f"{retrieved_context}\n\n"
            "Now, based on the interviewer's latest message and the retrieved background (if any), "
            "produce ONLY your next message as the Candidate. "
            "Do not include any out-of-character commentary."
        )

        # 注意：根据你 memory_layer.py 里 LLMController.get_completion 的实际签名调整参数
        reply_text = self.memory_system.llm_controller.get_completion(
            prompt=user_prompt,
            system_prompt=CANDIDATE_SYSTEM_PROMPT,
            json_mode=False,
            temperature=self.temperature,
            max_tokens=1024,
        )
        return reply_text.strip()
