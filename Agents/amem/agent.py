from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from utils.tools import initialize_clients, evaluate_test_set
from utils.seriousgame_tools import evaluate_beergame_set
from utils.consulting_tools import evaluate_consulting_set
from .generator import AMemBizBenchGenerator
from .memory_layer import AMemMemory

# ======================================================================
# SimpleMemory: 通用占位记忆模块，将来可以换成 memory_layer
# ======================================================================

class _SimpleMemory:
    """
    Lightweight A-mem-style memory bank.

    - 存储若干条文本 note（步经验、episode 总结等）
    - 用简单 token overlap 做检索，接口与真正记忆系统对齐：
        - add(content, meta)
        - retrieve(query, k) -> str
    """

    def __init__(self, max_notes: int = 2000):
        self.max_notes = max_notes
        self.notes: List[Dict[str, Any]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def add(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.notes.append({"content": content, "meta": meta or {}})
        if len(self.notes) > self.max_notes:
            self.notes = self.notes[-self.max_notes :]

    def retrieve(self, query: str, k: int = 6) -> str:
        q = set(self._tokenize(query))
        if not q or not self.notes:
            return ""
        scored: List[Tuple[float, str]] = []
        for n in self.notes[-self.max_notes :]:
            t = set(self._tokenize(n["content"]))
            score = len(q & t) / max(1, len(q))
            if score > 0:
                scored.append((score, n["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return "\n".join([s for _, s in scored[:k]])


# ======================================================================
# 统一 A-mem Agent：BizBench + BeerGame (+Consulting 预留)
# ======================================================================

class AMemAgent:
    """
    统一的 A-mem Agent。

    - BizBench: 静态 QA 数据集，经由 AMemBizBenchGenerator + evaluate_test_set
    - BeerGame: 严肃游戏，通过 MCP + evaluate_beergame_set
    - Consulting: 预留 run_consulting 分支，后续接入 casebench

    入口：
        run(mode, test_samples, data_processor, config)

      * 由 task_name 自动路由：
          - task_name 包含 'beer' / 'seriousgame' → run_beergame(...)
          - task_name 包含 'consult' → run_consulting(...)
          - 其他 → run_bizbench(...)
    """

    SUPPORTED_MODES = {"online", "eval_only"}

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int = 512,
        agent_method: str = "amem",
        *,
        temperature: float = 0.2,
        retrieve_k: int = 4,
        max_order_qty: int = 5000,
        note_every_n_steps: int = 2,
        summarize_every_n_steps: int = 4,
    ):
        self.api_provider = api_provider
        self.generator_model = generator_model
        self.max_tokens = max_tokens
        self.agent_method = agent_method

        self.temperature = float(temperature)
        self.retrieve_k = int(retrieve_k)
        self.max_order_qty = int(max_order_qty)
        self.note_every_n_steps = max(1, int(note_every_n_steps))
        self.summarize_every_n_steps = max(1, int(summarize_every_n_steps))

        # 通用 client：BeerGame & BizBench 共享
        self.generator_client, _, _ = initialize_clients(api_provider)

        # 记忆占位，将来可替换为真正 A-mem memory_layer
        self.memory = AMemMemory(
            embedding_model="all-MiniLM-L6-v2",
            evo_threshold=100,
            default_k=self.retrieve_k,
            llm_model=generator_model,  # 或固定 "gpt-5"
            llm_backend="openai",
        )

        # 轻量节流，避免 rate_limit_check_failed
        self.min_interval_sec = float(os.getenv("MIN_REQUEST_INTERVAL_SEC", "1.2"))
        self._last_call_ts = 0.0

        # BizBench generator: 单独放在 generator.py
        self.bizbench_generator = AMemBizBenchGenerator(
            client=self.generator_client,
            api_provider=self.api_provider,
            model=self.generator_model,
            max_tokens=self.max_tokens,
            memory=self.memory,
            retrieve_k=self.retrieve_k,
            temperature=self.temperature,
        )

    # ==================================================================
    # 公共工具：节流 + LLM JSON 调用
    # ==================================================================

    def _throttle(self) -> None:
        now = time.time()
        wait = self.min_interval_sec - (now - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def _call_llm_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        通用 JSON 调用：
        - 提示模型输出 JSON
        - 收到文本后尽量截取 {...} 再解析
        """
        import json as _json

        self._throttle()
        resp = self.generator_client.chat.completions.create(
            model=self.generator_model,
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


    # ==================================================================
    # Consulting：Agent 提供给评测框架的三个接口
    # ==================================================================

    def on_case_start(self, case_id: str) -> None:
        """
        一场咨询 Case 开始时调用。
        当前策略：每个 case 重置一次 memory，将记忆视为短期工作记忆。
        后续如果你想做“跨 case 长期记忆”，可以改成只清理部分记忆或用分桶。
        """
        # 简单做法：彻底清空，再新建一个 SimpleMemory
        pass

    def on_case_end(
        self,
        case_id: str,
        case_text: str,
        history: List[Dict[str, str]],
    ) -> None:
        """
        一场咨询 Case 结束时调用。
        - 当前实现：把完整 case 文本 + 对话 transcript 存成一条 note 写入 memory。
        - 将来你可以在这里做 meta-learning，总结策略等。
        """
        transcript = "\n".join(
            f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"
            for turn in history
        )
        note = (
            f"[CONSULTING][case_id={case_id}] CASE_TEXT:\n{case_text}\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
        self.memory.add(note, meta={"case_id": case_id, "type": "case_summary"})

    def reply(self, case_id: str, history: List[Dict[str, str]]) -> str:
        """
        Consulting 数据集候选人回复接口（与 utils.consulting_tools 调用方式对齐）。

        参数
        ----
        case_id : str
            当前 case 的标识符，由 consulting_tools 传入。
        history : List[Dict[str, str]]
            对话历史，每个元素形如：
                {"role": "interviewer" / "candidate", "content": "<文本>"}

        返回
        ----
        reply : str
            候选人下一轮的自然语言回答。
        """
        # 统计当前已经说过多少轮 candidate 回复（方便做记忆 meta）
        turns = sum(1 for h in history if h.get("role") == "candidate")

        # 找到最近一轮 interviewer 说的话
        last_interviewer_msg = ""
        for h in reversed(history):
            if h.get("role") == "interviewer":
                last_interviewer_msg = h.get("content", "")
                break

        # 构造 transcript（仅作为上下文，不落盘）
        transcript_lines = [
            f"{h.get('role', 'unknown')}: {h.get('content', '')}"
            for h in history
        ]
        transcript_text = "\n".join(transcript_lines) or "[no previous dialogue]"

        # 用 case_id + 最近 interviewer 提问 在记忆中检索相关 note
        query = (
            f"consulting case={case_id} "
            f"candidate_turns={turns} "
            f"last_interviewer={last_interviewer_msg}"
        )
        retrieved = self.memory.retrieve(query, k=self.retrieve_k)

        # System prompt：候选人的角色设定
        system = (
            "You are the CANDIDATE in a consulting-style case interview.\n"
            "You DO NOT see the internal case text, only the dialogue history.\n"
            "Behave like a top-tier consulting candidate:\n"
            "- structure the problem explicitly when appropriate,\n"
            "- reason in a hypothesis-driven way,\n"
            "- use simple quantitative checks when possible,\n"
            "- communicate clearly and concisely.\n\n"
            "Respond ONLY with what you would say next as the candidate.\n"
            'Wrap your answer in a JSON object of the form:\n'
            '  {\"reply\": \"<your answer>\"}\n'
            "Do not include any other fields."
        )

        # User prompt：提供 case_id、对话历史和记忆检索结果
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
            "Now respond with your next candidate message, wrapped in JSON "
            'as {\"reply\": \"...\"}.',
        ]
        user_prompt = "\n".join(user_parts)

        # 调用 LLM（复用已有的 JSON 调用工具）
        data = self._call_llm_json(system=system, user=user_prompt)
        reply = data.get("reply")

        # 兜底：如果 JSON 解析失败或者没有 reply 字段，用一个保守默认回答
        if not isinstance(reply, str) or not reply.strip():
            reply = (
                "Let me first structure the key issues and then outline a "
                "hypothesis-driven approach to address the client's problem."
            )
        reply = reply.strip()

        # 把这一轮对话写入记忆，用于后续检索
        self.memory.add(
            content=(
                f"[CONSULTING][case_id={case_id}][turn={turns}] "
                f"Interviewer: {last_interviewer_msg}\n"
                f"Candidate: {reply}"
            ),
            meta={"case_id": case_id, "turn": turns},
        )

        return reply


    # ==================================================================
    # BeerGame 专用：policy_fn + 运行函数
    # ==================================================================

    def _build_beergame_query(self, obs: Dict[str, Any]) -> str:
        return (
            f"role={obs.get('role')} week={obs.get('week')} "
            f"inv={obs.get('inventory')} bo={obs.get('backorder')} "
            f"in_order={obs.get('incoming_order')} supply={obs.get('supply_line')} "
            f"last_order={obs.get('last_order')}"
        )

    def _base_rule_order(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        简单规则基线，用作 fallback：

        order ≈ incoming_order + 0.5 * (target_inventory - (inventory - backorder))
        """
        incoming = int(obs.get("incoming_order", 0) or 0)
        inv = int(obs.get("inventory", 0) or 0)
        bo = int(obs.get("backorder", 0) or 0)

        target_inv = int(ctx.get("target_inventory", 20))
        adj = target_inv - (inv - bo)
        order_qty = incoming + max(0, int(0.5 * adj))
        return max(0, min(self.max_order_qty, order_qty))

    async def _decide_edt_settings(
        self,
        obs: Dict[str, Any],
        step_index: int,
        history: List[Dict[str, Any]],
        scenario_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        EDT 动作决策函数，完全仿照 _decide_order_qty 的风格：
        - 输入：当前 obs、step_index、历史若干步（可以截断）、场景 meta
        - 输出：用于 step-edt-env 的 settings dict（遵守 A0~A3 动作空间）

        返回示例：
          {}                                     # A0：不调整
          {"smEDT": {...fixed_cost...}}         # A1
          {"smEDT": {...revenue_risk_level...}} # A2
          {"smEDT": {...both...}}               # A3
        """
        # 压缩最近历史，防止 prompt 爆炸
        recent_history = history[-5:]

        # 简单把 flat_metrics 展开成可读字符串
        flat_metrics = obs.get("flat_metrics", {})
        metrics_str = "\n".join(f"- {k}: {v}" for k, v in flat_metrics.items())

        # 构造提示词：约束动作空间为 A0~A3，并要求返回 JSON
        user_prompt = (
            "你正在控制一个企业的经营策略（EDT 严肃游戏）。\n"
            f"当前是第 {step_index} 步。\n\n"
            "当前观测到的关键指标 (flat_metrics)：\n"
            f"{metrics_str or '(无显式指标)'}\n\n"
            "你可以在每一步选择以下几类动作（只能选一种，动作之外不要调整其他参数）：\n"
            "A0) 不动作：settings = {}\n"
            "A1) 只调整 fixed_cost：\n"
            "    settings = {\"smEDT\": {\"interactive\": {\"properties\": {\"fixed_cost\": {\"type\":\"Double\",\"value\": NEW}}}}}\n"
            "A2) 只调整 revenue_risk_level：\n"
            "    settings = {\"smEDT\": {\"interactive\": {\"properties\": {\"revenue_risk_level\": {\"type\":\"Double\",\"value\": NEW}}}}}\n"
            "A3) 同时调整二者：\n"
            "    settings = {\"smEDT\": {\"interactive\": {\"properties\": {\"fixed_cost\": {\"type\":\"Double\",\"value\": NEW1}, "
            "\"revenue_risk_level\": {\"type\":\"Double\",\"value\": NEW2}}}}}\n\n"
            "目标：在整个 episode 中尽量提升 accumulated_earnings，同时避免 cash 长期为负、profit_margin 长期过低。\n\n"
            "请你在综合当前指标与最近几步历史后，选择一个动作 A0~A3，并给出相应的 NEW / NEW1 / NEW2 数值。\n"
            "注意：数值不要变化过大（例如每步变化不超过 30%），以免系统震荡。\n\n"
            "输出必须是 JSON 对象，字段格式为：\n"
            "{\n"
            '  \"action\": \"A0\" | \"A1\" | \"A2\" | \"A3\",\n'
            '  \"fixed_cost\": <number or null>,\n'
            '  \"revenue_risk_level\": <number or null>\n'
            "}\n"
            "如果选择 A0，则 fixed_cost 和 revenue_risk_level 可以为 null。\n"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个审慎的企业财务与运营决策助手，负责在严肃游戏环境中调整参数以优化长期收益。"
                    "务必遵守动作空间约束，不要引入未定义的参数或结构。"
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=256,
        )

        try:
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
        except Exception:
            # 出错时退化为“不动作”
            return {}

        action = str(parsed.get("action", "A0")).upper()
        fixed_cost = parsed.get("fixed_cost", None)
        revenue_risk_level = parsed.get("revenue_risk_level", None)

        # 保守裁剪：避免极端数值（具体范围你可以按 EDT 场景再调）
        def _clip(v, lo, hi):
            try:
                v = float(v)
            except Exception:
                return None
            return max(lo, min(hi, v))

        if fixed_cost is not None:
            fixed_cost = _clip(fixed_cost, 0, 1e6)
        if revenue_risk_level is not None:
            revenue_risk_level = _clip(revenue_risk_level, 0.0, 1.0)

        # 根据 A0~A3 构造最终 settings（完全对齐 edt_mcp_server_local 的注释）
        if action == "A1" and fixed_cost is not None:
            return {
                "smEDT": {
                    "interactive": {
                        "properties": {
                            "fixed_cost": {"type": "Double", "value": fixed_cost}
                        }
                    }
                }
            }

        if action == "A2" and revenue_risk_level is not None:
            return {
                "smEDT": {
                    "interactive": {
                        "properties": {
                            "revenue_risk_level": {"type": "Double", "value": revenue_risk_level}
                        }
                    }
                }
            }

        if action == "A3" and fixed_cost is not None and revenue_risk_level is not None:
            return {
                "smEDT": {
                    "interactive": {
                        "properties": {
                            "fixed_cost": {"type": "Double", "value": fixed_cost},
                            "revenue_risk_level": {"type": "Double", "value": revenue_risk_level},
                        }
                    }
                }
            }

        # 其它情况统一退化为 A0：不动作
        return {}

    def _decide_order_qty(self, obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
        """
        BeerGame 单步决策：
        1) 构造 query，从记忆检索 note；
        2) LLM 输出 JSON {"order_qty": int, "note": str}；
        3) 解析失败时回退规则基线；
        4) 将本步写入记忆。
        """
        role = str(ctx.get("role", obs.get("role", "retailer")))
        week = int(obs.get("week", 0) or 0)

        query = self._build_beergame_query(obs)
        retrieved = self.memory.retrieve(query, k=self.retrieve_k)

        base_order = self._base_rule_order(obs, ctx)

        system = (
            "You are playing one role in the Beer Distribution Game.\n"
            "Your goal is to minimize long-run total cost (inventory + backlog).\n"
            "You must choose an integer 'order_qty' for the current week.\n"
            "You may also provide a short 'note' with your reasoning.\n"
            'Respond strictly in JSON: {"order_qty": <int>, "note": "<str>"}'
        )

        lines = [
            f"Role: {role}",
            "Current observation (for this role only):",
            json.dumps(obs, ensure_ascii=False),
            "",
            f"A simple baseline suggests order_qty ≈ {base_order}.",
        ]
        if retrieved:
            lines += [
                "",
                "Relevant notes from previous episodes/steps:",
                retrieved,
            ]
        user = "\n".join(lines)

        try:
            js = self._call_llm_json(system=system, user=user)
            order_qty = js.get("order_qty", base_order)
            note = js.get("note", "")
            try:
                order_qty = int(order_qty)
            except Exception:
                order_qty = base_order
        except Exception:
            order_qty = base_order
            note = ""

        order_qty = max(0, min(self.max_order_qty, int(order_qty)))

        if note:
            self.memory.add(
                content=f"[BeerGame] role={role} week={week} obs={obs} order={order_qty} note={note}",
                meta={"role": role, "week": week},
            )

        return order_qty

    def run_edt(
            self,
            test_samples: List[Dict[str, Any]],
            output_dir: str,
            mcp_server_cmd: Optional[List[str]] = None,
            max_steps: int = 96,
            verbose: bool = True,
            **kwargs,
    ):
        """
        EDT 评测入口。

        参数设计尽量与 BeerGame 保持一致：
          - test_samples: 由 EDT 的 data processor 生成的 [{\"scenario_id\", \"config\"}, ...]
          - output_dir: 日志/结果输出目录
          - mcp_server_cmd: 启动 edt_mcp_server_local.py 的命令，例如：
              [\"python\", \"SeriousGame/edt_mcp_server_local.py\"]
          - max_steps: 每个 episode 最多步数（默认 96，与 EDT_FIXED_CONFIG 对齐）
        """
        if mcp_server_cmd is None:
            # 默认在项目根目录下按相对路径启动 EDT MCP 服务器
            mcp_server_cmd = ["python", "SeriousGame/edt_mcp_server_local.py"]

        os.makedirs(output_dir, exist_ok=True)

        global_stats, detail_log = evaluate_edt_set(
            agent=self,
            server_cmd=mcp_server_cmd,
            test_samples=test_samples,
            max_steps=max_steps,
            log_dir=output_dir,
            log_prefix="edt",
            verbose=verbose,
        )

        # 你可以像 BeerGame 一样返回 (global_stats, detail_log)
        return global_stats, detail_log

    def run_beergame(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        严肃游戏 / BeerGame 专用入口。
        SeriousGame/run_beergame.py 可以直接调用这个函数，或者通过 run() 自动路由。
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError("BeerGame requires non-empty test_samples")

        save_dir = config.get("save_dir", "results")
        task_name = config.get("task_name", "BeerGame")
        run_subdir = (
            f"{task_name}/{self.agent_method}/{mode}/"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - BEERGAME EVALUATION")
        print(f"{'='*60}")
        print(f"Episodes: {len(test_samples)}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        # 允许在 config["beergame"] / 顶层 config["mcp"] 中配置 MCP 相关参数
        beergame_cfg = dict(config.get("beergame", {}))
        if "mcp" not in beergame_cfg and "mcp" in config:
            beergame_cfg["mcp"] = config["mcp"]
        if "toolmap" not in beergame_cfg and "toolmap" in config:
            beergame_cfg["toolmap"] = config["toolmap"]
        if "mcp_episode_config_key" not in beergame_cfg and "mcp_episode_config_key" in config:
            beergame_cfg["mcp_episode_config_key"] = config["mcp_episode_config_key"]

        def policy_fn(obs: Dict[str, Any], ctx: Dict[str, Any]) -> int:
            # 将 scenario-level target_inventory 透传到 ctx
            scenario_id = ctx.get("scenario_id")
            scenario_cfg = None
            for s in test_samples:
                if str(s.get("scenario_id")) == str(scenario_id):
                    scenario_cfg = s.get("config", {})
                    break
            if scenario_cfg and "target_inventory" in scenario_cfg:
                ctx = dict(ctx)
                ctx["target_inventory"] = scenario_cfg["target_inventory"]
            return self._decide_order_qty(obs, ctx)

        results, error_log = evaluate_beergame_set(
            test_samples=test_samples,
            policy_fn=policy_fn,
            config=beergame_cfg,
            log_dir=log_dir,
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
                ensure_ascii=False,
            )

        config_payload = dict(config)
        config_payload["run_subdir"] = run_subdir
        config_payload["resolved_save_path"] = resolved_save_path

        with open(
            os.path.join(resolved_save_path, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - BEERGAME RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Avg cost: {results.get('avg_total_cost_controlled')}")
        print(f"Avg bullwhip: {results.get('avg_bullwhip_controlled')}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    # ==================================================================
    # BizBench 专用：静态 QA 数据集
    # ==================================================================

    def run_bizbench(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        BizBench 评测入口。由 bizbench/run.py 调用，或者由 run() 自动路由。
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"{self.agent_method.upper()} agent only supports modes {self.SUPPORTED_MODES}, got '{mode}'"
            )
        if not test_samples:
            raise ValueError("BizBench requires non-empty test_samples")
        if data_processor is None:
            raise ValueError("BizBench tasks require a non-None data_processor")

        task_name = config.get("task_name", getattr(data_processor, "task_name", "BizBench"))
        save_dir = config.get("save_dir", "results")

        run_subdir = (
            f"{task_name}/{self.agent_method}/{mode}/"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        resolved_save_path = os.path.join(save_dir, run_subdir)
        os.makedirs(resolved_save_path, exist_ok=True)

        log_dir = os.path.join(resolved_save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - BIZBENCH EVALUATION")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        print(f"Samples: {len(test_samples)}")
        print(f"Log dir: {log_dir}")
        print(f"{'='*60}\n")

        results, error_log = evaluate_test_set(
            data_processor=data_processor,
            generator=self.bizbench_generator,
            playbook="",  # 当前版本不使用 playbook
            test_samples=test_samples,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            max_workers=config.get("test_workers", 20),
            use_json_mode=config.get("json_mode", False),
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
                ensure_ascii=False,
            )

        config_payload = dict(config)
        config_payload["run_subdir"] = run_subdir
        config_payload["resolved_save_path"] = resolved_save_path

        with open(
            os.path.join(resolved_save_path, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"{self.agent_method.upper()} - BIZBENCH RUN COMPLETE")
        print(f"{'='*60}")
        if "accuracy" in results:
            print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"Results saved to: {resolved_save_path}")
        print(f"{'='*60}\n")

        return results

    # ==================================================================
    # Consulting：完整评测入口
    # ==================================================================

    def run_consulting(
        self,
        mode: str,
        test_samples: List[Dict[str, Any]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Consulting / case interview 数据集入口。

        - 与 BizBench / BeerGame 一致：数据由 run.py 读好传进来；
        - 本函数负责组织保存路径 + 调用 evaluate_consulting_set；
        - 最终仍由本函数写 test_results.json / run_config.json。
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

        # 写 test_results.json（全局指标 + 错误日志）
        with open(
            os.path.join(resolved_save_path, "test_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {"test_results": results, "error_log": error_log},
                f,
                indent=2,
                ensure_ascii=False,
            )

        # 写 run_config.json（保持与其他数据集一致）
        config_payload = dict(config)
        config_payload["run_subdir"] = run_subdir
        config_payload["resolved_save_path"] = resolved_save_path
        with open(
            os.path.join(resolved_save_path, "run_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config_payload, f, indent=2, ensure_ascii=False)

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

    # ==================================================================
    # 统一入口：三个数据集共用 .run()
    # ==================================================================

    def run(
        self,
        mode: str,
        test_samples: Optional[List[Dict[str, Any]]],
        data_processor: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        统一入口，由 task_name 自动选择具体数据集的运行函数。

        - BeerGame: task_name 中包含 'beer' 或 'seriousgame'
        - Consulting: task_name 中包含 'consult'
        - BizBench: 其他情况默认按 BizBench 处理
        """
        if test_samples is None:
            raise ValueError("test_samples must not be None")

        task_name = str(config.get("task_name", getattr(data_processor, "task_name", "")))
        name_lower = task_name.lower()

        if ("beer" in name_lower) or ("seriousgame" in name_lower):
            return self.run_beergame(mode, test_samples, data_processor, config)
        if "consult" in name_lower:
            return self.run_consulting(mode, test_samples, data_processor, config)
        if "edt" in name_lower:
            # EDT 走和 beergame 平行的一套逻辑
            return self.run_edt(test_samples=test_samples, output_dir=output_dir, **kwargs)

        # 默认走 BizBench
        return self.run_bizbench(mode, test_samples, data_processor, config)
