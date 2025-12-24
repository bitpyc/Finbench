# utils/consulting_tools.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# ============================================================
# 配置与特殊标记
# ============================================================

# 面试官结束本次面试的标记（尽量不与自然语言冲突）
INTERVIEW_END_TOKEN = "<|CASE_INTERVIEW_END|>"

# LLM 模型：默认固定为 gpt-5（可用环境变量覆盖）
CONSULTING_LLM_MODEL = os.getenv("CONSULTING_LLM_MODEL", "gpt-5")

OPENAI_BASE_URL = "http://35.220.164.252:3888/v1/"
API_KEY = os.getenv('OPENAI_API_KEY', '')

# 你可以用 consulting/case_dataset.py 中的系统 prompt 替换下面这两个
INTERVIEWER_SYSTEM_PROMPT = f"""
You are the INTERVIEWER in a financial consulting case interview.

You can see the full internal case description (case_text), but the candidate cannot.
Your goals:
- Introduce the case and clarify the client's objective.
- Ask focused, high-leverage questions to probe the candidate's problem structuring,
  quantitative reasoning, and business judgment.
- Reveal additional information from the case_text gradually, only when the candidate
  asks relevant questions or reaches the right line of reasoning.
- Keep the conversation professional and concise.

In each turn, you must do ONE of the following:
1) Ask 1-3 short, precise questions that meaningfully move the case forward; or
2) Conclude the interview with a short wrap-up if you believe you have seen enough.

If you decide to END the interview, you MUST:
- include the EXACT token {INTERVIEW_END_TOKEN} in your reply,
- then add a short closing remark on the next line.

Never use {INTERVIEW_END_TOKEN} unless you truly want to end the interview.
""".strip()

INTERVIEWER_STYLE_HINT = """
Guidelines:
- Start by briefly restating the situation and objective in your own words.
- Early in the interview, test whether the candidate can structure the problem.
- Later, push on key drivers, simple quantitative checks, and trade-offs.
- When you decide to end, use the end token and then say something like:
  "<|CASE_INTERVIEW_END|> Thank you, let's stop here."
""".strip()

JUDGE_SYSTEM_PROMPT = """
You are an expert consulting interviewer evaluating a case interview performance.

You will receive:
- The full case text (case_text); and
- The full dialogue transcript between interviewer and candidate (transcript_text).

Your task:
1. Carefully read the transcript and infer the quality of the candidate's performance.
2. Evaluate along several dimensions:
   - structure: ability to structure the problem and use clear frameworks;
   - quant: comfort with numbers, sanity checks, and quantitative insight;
   - business_sense: understanding of business drivers, risks, and recommendations;
   - communication: clarity, conciseness, and professionalism;
   - overall: your overall holistic judgment.

3. Output a JSON object with this schema:
   {
     "structure": float,        # 0-10
     "quant": float,            # 0-10
     "business_sense": float,   # 0-10
     "communication": float,    # 0-10
     "overall": float,          # 0-10 (not just an average; your global judgment)
     "feedback": string         # short textual feedback / summary (5-8 sentences)
   }

Be calibrated across cases: a score of 5 means "average candidate", 8 means "very strong",
and 9-10 should be reserved for truly exceptional performance.
Return ONLY a valid JSON object.
""".strip()


# ============================================================
# 基础 LLM 封装
# ============================================================

def _build_openai_client() -> OpenAI:
    if OPENAI_BASE_URL:
        return OpenAI(api_key=API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI()


def _chat_text(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    resp = client.chat.completions.create(
        model=CONSULTING_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _chat_json(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=CONSULTING_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception:
        return {
            "structure": 0.0,
            "quant": 0.0,
            "business_sense": 0.0,
            "communication": 0.0,
            "overall": 0.0,
            "feedback": f"JSON parse error. Raw content: {content[:500]}",
        }


# ============================================================
# 面试官（interviewer）& 单 case 运行
# ============================================================

def _build_interviewer_messages(
    case_text: str,
    opening: str,
    history: List[Dict[str, str]],
    remaining_turns: int,
) -> List[Dict[str, str]]:
    # history 是 [{role: interviewer/candidate, content: str}, ...]
    history_lines = [
        f"{h.get('role', 'unknown').upper()}: {h.get('content', '')}"
        for h in history
    ]
    history_block = "\n".join(history_lines) if history_lines else "(no dialogue yet)"

    user_content = (
        "You are running a consulting case interview.\n\n"
        "=== CASE TEXT (only you can see) ===\n"
        f"{case_text}\n\n"
        "=== OPENING / INITIAL DESCRIPTION ===\n"
        f"{opening}\n\n"
        "=== DIALOGUE SO FAR ===\n"
        f"{history_block}\n\n"
        "Now produce your NEXT TURN as the interviewer.\n"
        f"- Maximum remaining turns (including this one) is {remaining_turns}.\n"
        f"- If this is the first turn, briefly introduce the case and ask the candidate "
        f"to structure the problem.\n"
        f"- If you believe you have seen enough and want to end the interview, "
        f"you MUST include the exact token {INTERVIEW_END_TOKEN} in your reply, "
        f"then give a short closing remark.\n"
        "Do NOT mention that you can see the full case text."
    )

    return [
        {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
        {"role": "system", "content": INTERVIEWER_STYLE_HINT},
        {"role": "user", "content": user_content},
    ]


def _run_single_interview(
    client: OpenAI,
    agent: Any,
    case: Dict[str, Any],
    max_turns: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    单个 case 的完整面试流程：
    - LLM 担任 interviewer（可访问 case_text）；
    - Agent 担任 candidate（只能看到对话 history）；
    - 返回 (history, judge_scores)：
        history: [{role: "interviewer"/"candidate", content: str}, ...]
        judge_scores: evaluator LLM 输出的 JSON。
    """
    case_id = str(case.get("case_id", "unknown_case"))
    case_text = case.get("case_text", "")
    opening = case.get("opening", "") or "The interviewer briefly describes the client's situation and objective."

    # 通知 Agent：case 开始
    if hasattr(agent, "on_case_start"):
        agent.on_case_start(case_id)

    history: List[Dict[str, str]] = []
    turns_used = 0
    ended_by_interviewer = False

    while turns_used < max_turns:
        remaining = max_turns - turns_used

        # interviewer 说话
        interviewer_msgs = _build_interviewer_messages(
            case_text=case_text,
            opening=opening,
            history=history,
            remaining_turns=remaining,
        )
        interviewer_reply = _chat_text(client, interviewer_msgs, temperature=0.3)
        if not interviewer_reply:
            interviewer_reply = f"{INTERVIEW_END_TOKEN} Thank you, let's stop here."

        history.append({"role": "interviewer", "content": interviewer_reply})
        turns_used += 1

        # 检查是否结束
        if INTERVIEW_END_TOKEN in interviewer_reply:
            ended_by_interviewer = True
            break

        if turns_used >= max_turns:
            break

        # candidate 回复：接口为 reply(case_id, history)
        if hasattr(agent, "reply"):
            candidate_reply = agent.reply(case_id, history)
        else:
            candidate_reply = "(Agent.reply() not implemented.)"
        history.append({"role": "candidate", "content": candidate_reply})
        turns_used += 1

    # case 结束：通知 Agent 做总结/记忆更新
    if hasattr(agent, "on_case_end"):
        try:
            agent.on_case_end(case_id, case_text, history)
        except TypeError:
            # 如果签名不完全匹配，退回只传 case_id
            agent.on_case_end(case_id)

    # 为了日志可读性，把结束 token 从文本中拿掉（只保留在元信息里）
    cleaned_history: List[Dict[str, str]] = []
    for turn in history:
        content = turn.get("content", "")
        if INTERVIEW_END_TOKEN in content:
            content = content.replace(INTERVIEW_END_TOKEN, "").strip()
        cleaned_history.append({**turn, "content": content})
    history = cleaned_history

    # 评测者：对整段对话打分
    transcript_lines = [
        f"{h.get('role', 'unknown').upper()}: {h.get('content', '')}"
        for h in history
    ]
    transcript_text = "\n".join(transcript_lines)

    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "You are now given a consulting case and the full dialogue between interviewer and candidate.\n\n"
                "=== CASE TEXT ===\n"
                f"{case_text}\n\n"
                "=== DIALOGUE TRANSCRIPT ===\n"
                f"{transcript_text}\n\n"
                "Evaluate the candidate according to the instructions."
            ),
        },
    ]
    judge_scores = _chat_json(client, judge_messages, temperature=0.1)
    judge_scores.setdefault("ended_by_interviewer", ended_by_interviewer)
    judge_scores.setdefault("turns_used", turns_used)

    return history, judge_scores


# ============================================================
# 评测主入口：evaluate_consulting_set
# ============================================================

def evaluate_consulting_set(
    *,
    agent: Any,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    log_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Consulting 数据集评测入口（与 BizBench / BeerGame 接口形式保持一致）：

    - 由 AMemAgent.run_consulting(...) 调用；
    - 数据集/划分名称从 config 中读取（可选）；
    - 只在 log_dir 下写：
        - consulting_interview_states.json ：所有 case 的对话 + 打分；
    - 返回：
        results: 全局指标（不含 per-case transcript）
        error_log: 失败 case 信息（目前如果运行中抛异常会记录）
    """
    os.makedirs(log_dir, exist_ok=True)
    client = _build_openai_client()

    dataset_name = config.get("dataset_name", config.get("task_name", "Consulting"))
    split_name = config.get("split_name", "test")
    max_turns = int(config.get("consulting_max_turns", 12))

    interview_states: List[Dict[str, Any]] = []
    per_case_scores: List[Dict[str, Any]] = []
    error_log: Dict[str, Any] = {"failed_cases": []}

    for idx, case in enumerate(test_samples):
        case_id = str(case.get("case_id", f"case_{idx:04d}"))
        print(f"[Consulting] Running case {idx+1}/{len(test_samples)}: {case_id}")

        try:
            history, judge_scores = _run_single_interview(
                client=client,
                agent=agent,
                case=case,
                max_turns=max_turns,
            )
        except Exception as e:
            error_log["failed_cases"].append(
                {"case_index": idx, "case_id": case_id, "error": repr(e)}
            )
            continue

        interview_states.append(
            {
                "case_index": idx,
                "case_id": case_id,
                "meta": case.get("meta", {}),
                "history": history,
                "judge_scores": judge_scores,
            }
        )
        per_case_scores.append({"case_id": case_id, **judge_scores})

    # 聚合统计
    def _avg(field: str) -> float:
        vals = []
        for s in per_case_scores:
            v = s.get(field)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "split": split_name,
        "num_cases": len(test_samples),
        "num_finished": len(per_case_scores),
        "num_failed": len(error_log["failed_cases"]),
        "metrics": {
            "structure": _avg("structure"),
            "quant": _avg("quant"),
            "business_sense": _avg("business_sense"),
            "communication": _avg("communication"),
            "overall": _avg("overall"),
        },
    }

    # 写全局对话记录文件
    states_path = Path(log_dir) / "consulting_interview_states.json"
    with states_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "split": split_name,
                "num_cases": len(test_samples),
                "cases": interview_states,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 返回给 AMemAgent.run_consulting，由它来写 test_results.json / run_config.json
    results = summary
    return results, error_log
