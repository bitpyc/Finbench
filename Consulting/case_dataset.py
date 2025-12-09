# case_dataset.py
"""
Casebook 面试数据集 + 固定面试官环境 + InterviewState 定义。
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from openai import OpenAI


# ============ 面试官配置 ============

INTERVIEWER_SYSTEM_PROMPT = """
You are an interviewer LLM conducting a consulting-style case interview with another LLM as the candidate. You are given three inputs in each turn: (1) these instructions, (2) the full text of a single case (including problem statement, background, any sections such as “To be divulged gradually”, “Further information”, “Additional information to be given to the candidate, only in response to being asked the appropriate questions”, “Interviewer notes”, “Suggested framework”, “Solution”, etc.), and (3) the chat history so far between you and the candidate. Your sole job is to generate the next interviewer message in the conversation.

Use the case statement and general background as information that the candidate is allowed to know. At the start of the case (when there is no prior chat), briefly present the situation and pose the main question from the case, then let the candidate drive the analysis with their questions and structure. In later turns, do not repeat the full case; instead, respond to what the candidate just said by answering relevant questions, asking focused follow-ups, and nudging them toward a structured, business-like analysis (for example clarifying objectives, markets, revenues, costs, customers, and constraints).

Treat any content under headings such as “To be divulged gradually”, “Further information”, or “Additional information to be given to the candidate, only in response to being asked the appropriate questions” as gated. Only reveal a specific gated fact when the candidate’s question clearly requests that type of information (for example, a question about market size, demand, costs, customer segments, geography, operations, or similar topics that match that fact). When they ask such a question, you may quote or closely paraphrase the corresponding gated text in your answer. Do not reveal other gated facts that have not been triggered yet, and never dump all gated information at once.

You must not invent or assume new facts that are not present in the case. If the candidate asks for information that is not in the case and not covered by any gated section, say that the case does not provide that detail and encourage them to proceed with reasonable assumptions or to explore another relevant dimension. You may use any “Interviewer notes”, “Suggested framework”, or “Solution” sections only as internal guidance to judge the candidate’s reasoning and to decide what to emphasize or probe; never reveal these sections directly, never say that you are showing them “the solution”, and avoid verbatim copying from these parts. If the candidate explicitly asks for high-level feedback or a summary of key drivers near the end, you may give concise, paraphrased feedback consistent with the solution.

To keep the interview efficient, it is acceptable to let the candidate ask clarifying questions in the early part of the case, but if the candidate keeps asking many questions without doing any analysis or the case has little extra information to reveal, you should gently prompt them to stop asking further questions and move on to structuring the problem and giving an answer.

You must only end the interview when you are truly finished and do not expect any further answers from the candidate. Concretely:
- Do NOT append any end marker in a message where you are asking a new question or inviting further analysis.
- First make sure the candidate has already provided a structured answer or recommendation that directly addresses the main objective of the case.
- If you want to end, send a final short message that may briefly summarize, give high-level feedback, or politely close the interview, but does not contain any open questions or prompts to continue.

Only in that final closing message, when you intend to terminate the interview and do not want the candidate to reply, you must end your reply with the exact token <<<INTERVIEW_OVER>>> as the very last characters of your message. Do not add any other explicit end markers such as “[End of case interview]”, and do not mention or explain this token; just append <<<INTERVIEW_OVER>>> at the end of your final interviewer message.

Keep your tone professional, concise, and focused on the case. Ask one or a small number of focused questions at a time. Do not engage in any small talk or meta-discussion. Use only the information in the case text and the chat history. Do not mention the existence of “case text”, “gated information”, “solutions”, or any internal labels in your messages; to the candidate, you are simply a human interviewer running a live case interview. Your output each turn should be just the next interviewer message addressed to the candidate, with no additional commentary.
"""

INTERVIEWER_MODEL = os.getenv("INTERVIEWER_MODEL", "gpt-5")
INTERVIEWER_END_TOKEN = "<<<INTERVIEW_OVER>>>"

_interviewer_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# =======================
# 评测用 Judge 模型配置
# =======================

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5")
_judge_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
)

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator of consulting-style case interviews. 
You will be given two inputs:
1) The full text of a single case from a casebook, including the problem statement, background, and any solution or teaching notes.
2) The complete interview transcript between an Interviewer and a Candidate, where the Candidate is trying to solve this case.

Your task is to evaluate ONLY the Candidate's performance in this interview. 
You should not assume that the Candidate has access to the full written solution; they are reasoning live based on what the interviewer reveals.

Evaluate the Candidate along the following dimensions, each on a 1–10 scale (1 = very poor, 10 = excellent):

- understanding: How well does the candidate understand and restate the objective and key question of the case?
- structuring: How clear, logical, and MECE is the candidate's problem structure and approach?
- analysis: How well does the candidate analyze drivers (market, customers, revenues, costs, operations, risks) and use information revealed in the interview?
- quant_reasoning: How reasonable and accurate is the candidate's quantitative or back-of-the-envelope reasoning (if any)?
- creativity: Does the candidate show insightful, non-trivial ideas or perspectives beyond the obvious?
- communication: Is the candidate's communication clear, concise, and business-like?
- overall: Your overall assessment of this candidate's performance on this case, considering all of the above.

When scoring, you may implicitly compare the Candidate to a strong human consulting candidate. 
Take into account both the content of their reasoning and their use of the information revealed by the interviewer. 
If the interviewer ended the interview early (few turns), you should still give scores but may comment on the limited evidence.

You must output a single JSON object with the following structure:

{
  "scores": {
    "understanding": <number 1-10>,
    "structuring": <number 1-10>,
    "analysis": <number 1-10>,
    "quant_reasoning": <number 1-10>,
    "creativity": <number 1-10>,
    "communication": <number 1-10>,
    "overall": <number 1-10>
  },
  "comment": "<a short paragraph (5-10 sentences) explaining your assessment and pointing out strengths and weaknesses>"
}

Do not include any additional fields. 
Do not wrap the JSON in backticks or any other formatting; return pure JSON.
"""


# ============ 数据结构 ============

@dataclass
class CaseItem:
    """单个 case 的基础信息：ID + 完整文本"""
    case_id: str
    case_text: str


@dataclass
class InterviewState:
    """
    一场面试的完整状态（用于保存 / 评测），内部可以包含 case_text。
    """
    case_id: str
    case_index: int
    case_text: str
    turns: int
    done: bool
    history: List[Dict[str, str]]  # [{"speaker": "interviewer"/"candidate", "text": "..."}]


@dataclass
class CandidateStateView:
    """
    暴露给面试者模型的只读视图：去掉 case_text，防止作弊。
    """
    case_id: str
    case_index: int
    turns: int
    done: bool
    history: List[Dict[str, str]]


def make_candidate_state_view(state: InterviewState) -> CandidateStateView:
    """从 InterviewState 构造面试者可见的视图（不含 case_text）"""
    return CandidateStateView(
        case_id=state.case_id,
        case_index=state.case_index,
        turns=state.turns,
        done=state.done,
        history=list(state.history),
    )


# ============ 工具函数 ============

def history_to_transcript(history: List[Dict[str, str]]) -> str:
    """把内部 history 转成简单对话文本"""
    lines = []
    for turn in history:
        speaker = "Interviewer" if turn["speaker"] == "interviewer" else "Candidate"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def call_judge_llm(
    case_text: str,
    history: List[Dict[str, str]],
    model: str = JUDGE_MODEL,
) -> Dict:
    """
    调用 LLM Judge，对一场面试给出 JSON 格式评分。
    输入: case 原始文案 + 完整 history
    输出: Python dict（解析后的 JSON），如果解析失败则返回一个退化结果
    """
    transcript = history_to_transcript(history)
    user_prompt = (
        "Below is a consulting-style case and the full interview transcript.\n\n"
        "===== CASE TEXT =====\n"
        f"{case_text}\n\n"
        "===== INTERVIEW TRANSCRIPT =====\n"
        f"{transcript}\n\n"
        "Please evaluate the candidate strictly following your instructions and output JSON only."
    )

    response = _interviewer_client.chat.completions.create(
        model=model,
        temperature=0.2,   # 评测尽量稳定
        max_tokens=512,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()

    # 解析 JSON，防止偶尔不合法
    try:
        result = json.loads(raw)
        if isinstance(result, dict) and "scores" in result:
            return result
    except Exception:
        pass

    # 如果解析失败，给一个兜底结构，方便下游代码不崩
    return {
        "scores": {
            "understanding": 0,
            "structuring": 0,
            "analysis": 0,
            "quant_reasoning": 0,
            "creativity": 0,
            "communication": 0,
            "overall": 0,
        },
        "comment": f"LLM judge output could not be parsed as valid JSON. Raw output was:\n{raw}",
    }


def call_interviewer_llm(
    system_prompt: str,
    user_content: str,
    model: str = INTERVIEWER_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """调用固定的面试官 LLM"""
    resp = _interviewer_client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content.strip()


def save_interview_states_to_json(states: List[InterviewState], filepath: str):
    """将多个 InterviewState 保存成一个 json 文件（列表形式）"""
    data = [asdict(s) for s in states]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_interview_states_from_json(filepath: str) -> List[InterviewState]:
    """从 json 文件中恢复多个 InterviewState"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [InterviewState(**d) for d in data]


# ============ Casebook 面试环境 ============

class CasebookInterviewDataset:
    """
    封装 casebook + 固定面试官的交互环境。

    - __len__     -> case 数量
    - iter_cases  -> (index, case_id)
    - start_interview(case_index) -> (InterviewState, interviewer_msg)
    - step(state, candidate_reply) -> (InterviewState, interviewer_msg 或 None)
    """

    def __init__(
        self,
        cases: List[CaseItem],
        interviewer_system_prompt: str = INTERVIEWER_SYSTEM_PROMPT,
        interviewer_model: str = INTERVIEWER_MODEL,
        max_turns: int = 10,
    ):
        self.cases = cases
        self.interviewer_system_prompt = interviewer_system_prompt
        self.interviewer_model = interviewer_model
        self.max_turns = max_turns

    # ---- 数据集接口 ----

    def __len__(self) -> int:
        return len(self.cases)

    def iter_cases(self):
        """遍历所有 case 的 (index, case_id)"""
        for i, c in enumerate(self.cases):
            yield i, c.case_id

    def get_case(self, index: int) -> CaseItem:
        return self.cases[index]

    # ---- 面试流程 ----

    def _build_interviewer_prompt(
        self,
        case: CaseItem,
        history: List[Dict[str, str]],
    ) -> str:
        transcript = history_to_transcript(history)
        return (
            "You are the interviewer in this case interview.\n\n"
            "CASE TEXT:\n"
            f"{case.case_text}\n\n"
            "CHAT HISTORY:\n"
            f"{transcript if transcript else '[no previous dialogue]'}\n\n"
            "Now, as the interviewer, produce ONLY your next message to the candidate, "
            "following your system instructions. Do not include any explanations about what you are doing."
        )

    def start_interview(self, case_index: int) -> Tuple[InterviewState, str]:
        """开始某一道 case 的面试"""
        case = self.cases[case_index]
        history: List[Dict[str, str]] = []

        user_prompt = self._build_interviewer_prompt(case, history)
        interviewer_msg = call_interviewer_llm(
            system_prompt=self.interviewer_system_prompt,
            user_content=user_prompt,
            model=self.interviewer_model,
        )
        history.append({"speaker": "interviewer", "text": interviewer_msg})

        state = InterviewState(
            case_id=case.case_id,
            case_index=case_index,
            case_text=case.case_text,
            turns=1,
            done=False,
            history=history,
        )

        if INTERVIEWER_END_TOKEN in interviewer_msg or state.turns >= self.max_turns:
            state.done = True

        return state, interviewer_msg

    def step(
        self,
        state: InterviewState,
        candidate_reply: str,
    ) -> Tuple[InterviewState, Optional[str]]:
        """
        单步对话：
          输入: 当前 state + 候选人回答
          输出: 更新后的 state + 下一句 interviewer_msg（结束后为 None）
        """
        if state.done:
            return state, None

        case = self.cases[state.case_index]

        # 记录候选人回答
        state.history.append({"speaker": "candidate", "text": candidate_reply})

        # 检查轮数
        if state.turns >= self.max_turns:
            state.done = True
            return state, None

        # 面试官下一句
        user_prompt = self._build_interviewer_prompt(case, state.history)
        interviewer_msg = call_interviewer_llm(
            system_prompt=self.interviewer_system_prompt,
            user_content=user_prompt,
            model=self.interviewer_model,
        )
        state.history.append({"speaker": "interviewer", "text": interviewer_msg})
        state.turns += 1

        if INTERVIEWER_END_TOKEN in interviewer_msg or state.turns >= self.max_turns:
            state.done = True

        return state, interviewer_msg


def evaluate_interview_with_llm(state: InterviewState) -> Dict:
    """
    使用 LLM Judge 对单个 InterviewState 进行评价打分。

    返回结构示例：
    {
      "case_id": "...",
      "case_index": 0,
      "scores": {...},
      "comment": "...",
    }
    """
    result = call_judge_llm(
        case_text=state.case_text,
        history=state.history,
    )
    # 附加 case 信息
    result_with_meta = {
        "case_id": state.case_id,
        "case_index": state.case_index,
        "scores": result.get("scores", {}),
        "comment": result.get("comment", ""),
    }
    return result_with_meta
