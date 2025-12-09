# model/base.py
"""
候选人统一接口：所有待测模型都继承 CandidateAgent。
"""

from abc import ABC, abstractmethod
from typing import List, Dict

from case_dataset import CandidateStateView


class CandidateAgent(ABC):
    """
    抽象基类：所有 candidate 模型统一的接口。

    框架只强制实现一个方法：
        reply(state_view, interviewer_msg) -> str

    以及两个可选的生命周期 hook：
        on_case_start(case_id)
        on_case_end(case_id, case_text, history)
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reply(
        self,
        state: CandidateStateView,
        interviewer_msg: str,
    ) -> str:
        """
        输入:
          - state: 当前 case 的只读视图（不含 case_text）
          - interviewer_msg: 面试官上一轮的完整发言
        输出:
          - 候选人的下一句回答
        """
        raise NotImplementedError

    # 可选：每个 case 开始前
    def on_case_start(self, case_id: str):
        """每道题开始前调用一次"""
        pass

    # 可选：每个 case 结束后
    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ):
        """
        面试结束后调用。这里才允许看到 case_text。
        对于带长期记忆的模型，可以在这里将 case_text 写入长期记忆。
        """
        pass
