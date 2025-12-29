# utils/consulting_tools.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# ============================================================
# 配置与特殊标记
# ============================================================

# 面试官结束本次面试的标记（尽量不与自然语言冲突）
INTERVIEW_END_TOKEN = "<|CASE_INTERVIEW_END|>"

# LLM 模型：默认固定为 gpt-5（可用环境变量覆盖）
CONSULTING_LLM_MODEL = os.getenv("CONSULTING_LLM_MODEL", "deepseek-v3")

OPENAI_BASE_URL = "http://35.220.164.252:3888/v1/"
API_KEY = os.getenv('OPENAI_API_KEY', '')

# 你可以用 consulting/case_dataset.py 中的系统 prompt 替换下面这两个
INTERVIEWER_SYSTEM_PROMPT = f"""
You are an interviewer LLM conducting a consulting-style case interview with another LLM as the candidate. You are given three inputs in each turn: (1) these instructions, (2) the full text of a single case (including problem statement, background, any sections such as “To be divulged gradually”, “Further information”, “Additional information to be given to the candidate, only in response to being asked the appropriate questions”, “Interviewer notes”, “Suggested framework”, “Solution”, etc.), and (3) the chat history so far between you and the candidate. Your sole job is to generate the next interviewer message in the conversation.

Use the case statement and general background as information that the candidate is allowed to know. At the start of the case (when there is no prior chat), briefly present the situation and pose the main question from the case, then let the candidate drive the analysis with their questions and structure. In later turns, do not repeat the full case; instead, respond to what the candidate just said by answering relevant questions, asking focused follow-ups, and nudging them toward a structured, business-like analysis (for example clarifying objectives, markets, revenues, costs, customers, and constraints).

Treat any content under headings such as “To be divulged gradually”, “Further information”, or “Additional information to be given to the candidate, only in response to being asked the appropriate questions” as gated. Only reveal a specific gated fact when the candidate’s question clearly requests that type of information (for example, a question about market size, demand, costs, customer segments, geography, operations, or similar topics that match that fact). When they ask such a question, you may quote or closely paraphrase the corresponding gated text in your answer. Do not reveal other gated facts that have not been triggered yet, and never dump all gated information at once.

You must not invent or assume new facts that are not present in the case. If the candidate asks for information that is not in the case and not covered by any gated section, say that the case does not provide that detail and encourage them to proceed with reasonable assumptions or to explore another relevant dimension. You may use any “Interviewer notes”, “Suggested framework”, or “Solution” sections only as internal guidance to judge the candidate’s reasoning and to decide what to emphasize or probe; never reveal these sections directly, never say that you are showing them “the solution”, and avoid verbatim copying from these parts. If the candidate explicitly asks for high-level feedback or a summary of key drivers near the end, you may give concise, paraphrased feedback consistent with the solution.

To keep the interview efficient, it is acceptable to let the candidate ask clarifying questions in the early part of the case, but if the candidate keeps asking many questions without doing any analysis or the case has little extra information to reveal, you should gently prompt them to stop asking further questions and move on to structuring the problem and giving an answer.

You must only end the interview when you are truly finished and do not expect any further answers from the candidate. Concretely:
- Do NOT append any end marker in a message where you are asking a new question or inviting further analysis.
- First make sure the candidate has already provided a structured answer or recommendation that directly addresses the main objective of the case.
- If you want to end, send a final short message that may briefly summarize, give high-level feedback, or politely close the interview, but does not contain any open questions or prompts to continue.

Only in that final closing message, when you intend to terminate the interview and do not want the candidate to reply, you must end your reply with the exact token {INTERVIEW_END_TOKEN} as the very last characters of your message. Do not add any other explicit end markers such as “[End of case interview]”, and do not mention or explain this token; just append {INTERVIEW_END_TOKEN} at the end of your final interviewer message.

Keep your tone professional, concise, and focused on the case. Ask one or a small number of focused questions at a time. Do not engage in any small talk or meta-discussion. Use only the information in the case text and the chat history. Do not mention the existence of “case text”, “gated information”, “solutions”, or any internal labels in your messages; to the candidate, you are simply a human interviewer running a live case interview. Your output each turn should be just the next interviewer message addressed to the candidate, with no additional commentary.
"""

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
     "feedback": string         # short textual feedback / summary (within 400 words)
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
    max_tokens: int = 8192,
) -> str:
    resp = client.chat.completions.create(
        model=CONSULTING_LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort="low",
    )
    return resp.choices[0].message.content or ""


def _chat_json(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 8192,
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
        response_cleaned = content.strip()
        # Try to find JSON content if wrapped in other text
        if not response_cleaned.startswith('{'):
            start_idx = response_cleaned.find('{')
            if start_idx != -1:
                response_cleaned = response_cleaned[start_idx:]
        if not response_cleaned.endswith('}'):
            end_idx = response_cleaned.rfind('}')
            if end_idx != -1:
                response_cleaned = response_cleaned[:end_idx + 1]
        return json.loads(response_cleaned)
    except Exception:
        return {
            "structure": 0.0,
            "quant": 0.0,
            "business_sense": 0.0,
            "communication": 0.0,
            "overall": 0.0,
            "feedback": f"JSON parse error. Raw content: {content[:5000]}",
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
    # {"role": "system", "content": INTERVIEWER_STYLE_HINT},
    return [
        {"role": "system", "content": INTERVIEWER_SYSTEM_PROMPT},
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

# ============================================================
# Consulting: agent-side helpers (model-agnostic prompt/query) + run helpers
# ============================================================

def consulting_build_candidate_query(*, case_id: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Build a compact decision state for the candidate agent:
    - query: for memory retrieval
    - turns: number of candidate turns so far
    - last_interviewer_msg: last interviewer utterance
    - transcript_text: plain text transcript (role: content)
    """
    turns = sum(1 for h in history if h.get("role") == "candidate")

    last_interviewer_msg = ""
    for h in reversed(history):
        if h.get("role") == "interviewer":
            last_interviewer_msg = h.get("content", "") or ""
            break

    transcript_lines = [f"{h.get('role','unknown')}: {h.get('content','')}" for h in history]
    transcript_text = "\n".join(transcript_lines) if transcript_lines else "[no previous dialogue]"

    query = (
        f"consulting case={case_id} "
        f"candidate_turns={turns} "
        f"last_interviewer={last_interviewer_msg}"
    )

    return {
        "case_id": case_id,
        "turns": turns,
        "last_interviewer_msg": last_interviewer_msg,
        "transcript_text": transcript_text,
        "query": query,
    }


_CANDIDATE_SYSTEM_PROMPT = (
    "You are the CANDIDATE in a consulting-style case interview.\n"
    "You DO NOT see the internal case text, only the dialogue history.\n"
    "Behave like a top-tier consulting candidate:\n"
    "- structure the problem explicitly when appropriate,\n"
    "- reason in a hypothesis-driven way,\n"
    "- use simple quantitative checks when possible,\n"
    "- communicate clearly and concisely.\n\n"
    "Respond ONLY with what you would say next as the candidate.\n"
    "Wrap your answer in a JSON object of the form:\n"
    "  {\"reply\": \"<your answer>\"}\n"
    "Do not include any other fields."
)

def consulting_render_candidate_prompt(
    *,
    case_id: str,
    state: Dict[str, Any],
    retrieved: str = "",
) -> Tuple[str, str]:
    """
    Render model-agnostic (system, user) prompts for candidate reply generation.
    """
    transcript_text = state.get("transcript_text", "") or "[no previous dialogue]"
    last_interviewer_msg = state.get("last_interviewer_msg", "") or ""

    user_parts = [
        f"Current case ID: {case_id}",
        "",
        "Dialogue so far (Interviewer / Candidate):",
        transcript_text,
        "",
        "Interviewer just said:",
        last_interviewer_msg or "[no interviewer message found]",
    ]
    if retrieved:
        user_parts += [
            "",
            "Some of your previous notes or remembered information:",
            retrieved,
        ]
    user_parts += [
        "",
        'Now respond with your next candidate message, wrapped in JSON as {"reply": "..."}.',
    ]
    return _CANDIDATE_SYSTEM_PROMPT, "\n".join(user_parts)


def consulting_extract_candidate_reply(data: Dict[str, Any]) -> str:
    """
    Extract candidate reply from LLM JSON. Provide a robust fallback.
    """
    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        reply = (
            "Let me first structure the key issues and then outline a "
            "hypothesis-driven approach to address the client's problem."
        )
    return reply.strip()


def consulting_format_memory_note(*, case_id: str, state: Dict[str, Any], reply: str) -> str:
    turns = int(state.get("turns", 0) or 0)
    last_interviewer_msg = state.get("last_interviewer_msg", "") or ""
    return (
        f"[CONSULTING][case_id={case_id}][turn={turns}] "
        f"Interviewer: {last_interviewer_msg}\n"
        f"Candidate: {reply}"
    )


def consulting_prepare_run(
    *,
    mode: str,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    allowed_modes: set,
    agent_method: str,
) -> Dict[str, Any]:
    """
    Prepare run directories and lightweight run context for consulting evaluation.
    """
    if mode not in allowed_modes:
        raise ValueError(f"{agent_method.upper()} agent only supports modes {allowed_modes}, got '{mode}'")
    if not test_samples:
        raise ValueError("Consulting requires non-empty test_samples")

    save_dir = config.get("save_dir", "results")
    task_name = config.get("task_name", "Consulting")

    run_subdir = f"{task_name}/{agent_method}/{mode}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    resolved_save_path = os.path.join(save_dir, run_subdir)
    os.makedirs(resolved_save_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{agent_method.upper()} - CONSULTING EVALUATION")
    print(f"{'='*60}")
    print(f"Cases: {len(test_samples)}")
    print(f"Save dir: {resolved_save_path}")
    print(f"{'='*60}\n")

    return {
        "task_name": task_name,
        "save_dir": save_dir,
        "run_subdir": run_subdir,
        "resolved_save_path": resolved_save_path,
    }


def consulting_evaluate_run(
    *,
    agent: Any,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute consulting evaluation.
    """
    return evaluate_consulting_set(
        agent=agent,
        test_samples=test_samples,
        config=config,
        log_dir=ctx["resolved_save_path"],
    )


def consulting_save_run(
    *,
    results: Dict[str, Any],
    error_log: Dict[str, Any],
    config: Dict[str, Any],
    ctx: Dict[str, Any],
) -> None:
    """
    Save run artifacts (test_results.json, run_config.json) in a consistent format.
    """
    resolved_save_path = ctx["resolved_save_path"]

    with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({"test_results": results, "error_log": error_log}, f, indent=2, ensure_ascii=False)

    config_payload = dict(config)
    config_payload["run_subdir"] = ctx["run_subdir"]
    config_payload["resolved_save_path"] = resolved_save_path

    with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"{ctx.get('task_name','Consulting')} - RUN COMPLETE")
    print(f"{'='*60}")
    print(
        f"Num cases: {results.get('num_cases')}, "
        f"finished: {results.get('num_finished')}, "
        f"failed: {results.get('num_failed')}"
    )
    print(f"Metrics: {results.get('metrics')}")
    print(f"Results saved to: {resolved_save_path}")
    print(f"{'='*60}\n")
