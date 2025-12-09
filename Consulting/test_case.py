# test_case.py
"""
测试入口：在 casebook 数据集上同时评测多个候选人模型（面试者）。
目前先挂 A-mem，一个框架支持后续继续加别的 CandidateAgent。
"""

import os
import json
import argparse
from typing import List

from case_dataset import (
    CaseItem,
    CasebookInterviewDataset,
    InterviewState,
    save_interview_states_to_json,
    make_candidate_state_view,
    load_interview_states_from_json,
    evaluate_interview_with_llm
)
from model.base import CandidateAgent
from model.a_mem.agent import AMemCandidateAgent


def evaluate_saved_runs(states_path: str, output_path: str):
    """
    对已经保存好的 InterviewState 列表，用 LLM Judge 逐场进行评价，并保存评分结果。
    """
    states = load_interview_states_from_json(states_path)
    results = []

    for state in states:
        res = evaluate_interview_with_llm(state)
        results.append(res)
        print(f"[Judge] case={res['case_id']} overall={res['scores'].get('overall', 0)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"All judge results saved to: {output_path}")


def load_casebook_from_json(path: str) -> List[CaseItem]:
    """
    假设 casebook json 结构：
        {
          "CABLE TELEVISION COMPANY": "full case text ...",
          "CHILLED BEVERAGES": "full case text ..."
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases: List[CaseItem] = []
    for case_id, case_text in data.items():
        cases.append(CaseItem(case_id=case_id, case_text=case_text))
    return cases


def run_benchmark_for_agent(
    agent: CandidateAgent,
    dataset: CasebookInterviewDataset,
    output_states_path: str,
):
    """
    用某个 CandidateAgent 在整个 casebook 上跑一遍面试，
    把每个 case 对应的 InterviewState 保存成一个 json 文件。
    """
    all_states: List[InterviewState] = []

    for idx, case_id in dataset.iter_cases():
        print(f"\n=== [{agent.name}] Start interview for case: {case_id} ===")

        agent.on_case_start(case_id)

        state, interviewer_msg = dataset.start_interview(idx)
        print("[Interviewer]:", interviewer_msg)

        while not state.done:
            # 给 agent 的是不含 case_text 的视图
            view = make_candidate_state_view(state)

            candidate_reply = agent.reply(
                state=view,
                interviewer_msg=interviewer_msg,
            )
            print(f"[{agent.name}]:", candidate_reply)

            state, interviewer_msg = dataset.step(state, candidate_reply)
            if interviewer_msg is not None:
                print("[Interviewer]:", interviewer_msg)

        print(f"=== [{agent.name}] Interview finished for case: {case_id}, turns: {state.turns} ===")
        all_states.append(state)

        # case 结束后，才把 case_text 暴露给 agent（例如 A-mem 用来写入长期记忆）
        agent.on_case_end(
            case_id=state.case_id,
            case_text=state.case_text,
            history=state.history,
        )

    save_interview_states_to_json(all_states, output_states_path)
    print(f"\n[{agent.name}] All interview states saved to: {output_states_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run candidate agents on casebook dataset.")
    parser.add_argument("--casebook_json", type=str, required=True,
                        help="Path to casebook json, key=case_id, value=full case text.")
    parser.add_argument("--output_dir", type=str, default="casebook_runs",
                        help="Directory to save all agents' interview states.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model name used by A-mem.")
    parser.add_argument("--backend", type=str, default="sglang",
                        help="LLM backend for A-mem: openai / sglang / ollama.")
    parser.add_argument("--api_base", type=str, default="http://35.220.164.252:3888/v1/",
                        help="Base URL for LLM API.")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="Host for sglang backend.")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="Port for sglang backend.")
    parser.add_argument("--retrieve_k", type=int, default=8,
                        help="Top-k memories to retrieve per turn for A-mem.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for candidate LLM.")
    parser.add_argument("--max_turns", type=int, default=10,
                        help="Max interviewer-candidate turns per case.")
    parser.add_argument("--reset_memory_each_case", action="store_true",
                        help="If set, reset A-mem memory for each case.")
    parser.add_argument("--judge_after_run", action="store_true",
                        help="If set, run LLM judge on the generated interviews immediately.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 加载 casebook
    cases = load_casebook_from_json(args.casebook_json)
    dataset = CasebookInterviewDataset(cases, max_turns=args.max_turns)

    # 2) 构造所有待测模型（后续可以继续加其他 CandidateAgent）
    agents: List[CandidateAgent] = []

    amem_agent = AMemCandidateAgent(
        name="a-mem",
        model=args.model,
        backend=args.backend,
        retrieve_k=args.retrieve_k,
        temperature=args.temperature,
        api_base=args.api_base,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        reset_memory_each_case=args.reset_memory_each_case,
    )
    agents.append(amem_agent)

    # TODO: 示例：你可以后续加一个纯 GPT baseline
    # from model.gpt_baseline.agent import GPTBaselineAgent
    # agents.append(GPTBaselineAgent(...))

    # 3) 逐个 agent 跑 benchmark
    for agent in agents:
        output_path = os.path.join(args.output_dir, f"{agent.name}_interviews.json")
        run_benchmark_for_agent(agent, dataset, output_path)

        if args.judge_after_run:
            judge_output = os.path.join(args.output_dir, f"{agent.name}_judge_scores.json")
            evaluate_saved_runs(output_path, judge_output)