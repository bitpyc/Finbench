#!/usr/bin/env python3
"""
SeriousGame evaluation utilities.

Design goal:
- Mirror BizBench's `utils.tools.evaluate_test_set` structure, but for interactive episode-based environments.
- Keep agent implementations independent: evaluator only requires a policy function `policy_fn(obs, ctx)->int`.
- Do NOT require ACE's Generator/timed_llm_call pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Prefer OpenAI Agents SDK MCP wrapper if available (already used elsewhere in this repo)
try:
    from agents.mcp import MCPServerStdio  # type: ignore
except Exception:  # pragma: no cover
    MCPServerStdio = None  # type: ignore


@dataclass
class EpisodeOutcome:
    scenario_id: str
    episode_id: str
    metrics: Dict[str, Any]
    trace: Optional[List[Dict[str, Any]]] = None


def _safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    # best-effort: extract first {...}
    if not t.startswith("{"):
        s = t.find("{")
        if s >= 0:
            t = t[s:]
    if not t.endswith("}"):
        e = t.rfind("}")
        if e >= 0:
            t = t[: e + 1]
    try:
        return json.loads(t)
    except Exception:
        return {}


async def _mcp_call(server: Any, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call MCP tool and parse JSON payload.

    The reference beergame MCP server used in this repo returns:
      {"content":[{"type":"text","text":"<json>"}], "isError": bool}
    """
    res = await server.call_tool(tool_name, arguments)
    content = res.get("content") if isinstance(res, dict) else getattr(res, "content", None)
    is_error = bool(res.get("isError", False)) if isinstance(res, dict) else bool(getattr(res, "isError", False))

    if not content:
        if is_error:
            raise RuntimeError(f"MCP tool {tool_name} returned error with empty payload.")
        return {}

    first = content[0]
    text = first.get("text", "") if isinstance(first, dict) else getattr(first, "text", "")
    payload = _safe_json_loads(text)

    # 加这一行：把 raw 响应完整打出来
    print(f"[DEBUG] MCP {tool_name} raw response: {res}", flush=True)

    if is_error or payload.get("error"):
        raise RuntimeError(f"MCP tool {tool_name} error: {payload}")

    return payload


def _compute_bullwhip_from_trace(trace: List[Dict[str, Any]], role: str) -> Optional[float]:
    """
    bullwhip := Var(orders) / Var(demand)
    Demand comes from trace[i]["demand"] (if present).
    Orders come from trace[i]["stages"][role]["outgoing_order"] (if present).
    """
    if not trace or len(trace) < 3:
        return None

    demands: List[float] = []
    orders: List[float] = []
    for row in trace:
        d = row.get("demand")
        st = (row.get("stages") or {}).get(role, {})
        o = st.get("outgoing_order")
        if isinstance(d, (int, float)) and isinstance(o, (int, float)):
            demands.append(float(d))
            orders.append(float(o))

    if len(demands) < 3:
        return None

    def var(xs: List[float]) -> float:
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    vd = var(demands)
    vo = var(orders)
    if vd <= 1e-12:
        return None
    return vo / vd


async def _run_one_episode(
    server: Any,
    scenario: Dict[str, Any],
    policy_fn: Callable[[Dict[str, Any], Dict[str, Any]], int],
    log_dir: str,
    toolmap: Dict[str, str],
    mcp_episode_config_key: str = "config",
) -> EpisodeOutcome:
    scenario_id = str(scenario.get("scenario_id", "unknown"))
    episode_cfg = scenario.get("config", scenario)

    new_tool = toolmap["new_episode"]
    step_tool = toolmap["step"]
    metrics_tool = toolmap["metrics"]
    trace_tool = toolmap.get("trace")
    close_tool = toolmap["close"]

    new_payload = {mcp_episode_config_key: episode_cfg} if mcp_episode_config_key else episode_cfg

    # 1) 初始化：这里要读 env_id，而不是 episode_id
    init_out = await _mcp_call(server, new_tool, new_payload)
    env_id = str(init_out.get("env_id", ""))          # ✅ 使用 env_id
    obs = init_out.get("obs", {}) or {}
    role = str((init_out.get("config") or {}).get("controlled_role", episode_cfg.get("controlled_role", "retailer")))

    steps: List[Dict[str, Any]] = []
    done = False
    while not done:
        order_qty = int(policy_fn(obs, {"scenario_id": scenario_id, "episode_id": env_id, "role": role}))
        if order_qty < 0:
            order_qty = 0

        # 2) step 调用：参数名也要改成 env_id
        step_out = await _mcp_call(server, step_tool, {"env_id": env_id, "order_qty": order_qty})
        next_obs = step_out.get("obs", {}) or {}
        done = bool(step_out.get("done", False))

        steps.append({
            "obs": obs,
            "action": {"order_qty": order_qty},
            "step_out": step_out,
        })
        obs = next_obs

        if len(steps) > int(episode_cfg.get("horizon_weeks", 52)) + 10:
            break

    # 3) metrics 调用：同样用 env_id
    metrics = await _mcp_call(server, metrics_tool, {"env_id": env_id})

    trace: Optional[List[Dict[str, Any]]] = None
    if trace_tool:
        try:
            # 目前你没有 trace 工具，这里通常不会被调用，如果将来实现了，也要按 env_id 传
            trace_pack = await _mcp_call(server, trace_tool, {"env_id": env_id})
            trace = trace_pack.get("trace")
        except Exception:
            trace = None

    # 4) 关闭环境：同样用 env_id
    try:
        await _mcp_call(server, close_tool, {"env_id": env_id})
    except Exception:
        pass

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"episode_{scenario_id}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scenario_id": scenario_id,
            "episode_id": env_id,   # 这里可以继续叫 episode_id，只是保存用
            "role": role,
            "episode_config": episode_cfg,
            "steps": steps,
            "metrics": metrics,
            "trace": trace,
        }, f, indent=2, ensure_ascii=False)

    return EpisodeOutcome(scenario_id=scenario_id, episode_id=env_id, metrics=metrics, trace=trace)


def evaluate_beergame_set(
    *,
    test_samples: List[Dict[str, Any]],
    policy_fn: Callable[[Dict[str, Any], Dict[str, Any]], int],
    config: Dict[str, Any],
    log_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate a set of BeerGame scenarios (each scenario = one episode).

    Returns (final_results, error_log) similar to utils.tools.evaluate_test_set.
    """
    if MCPServerStdio is None:
        raise RuntimeError("agents.mcp.MCPServerStdio not available. Please install/enable OpenAI Agents SDK.")

    # MCP params
    mcp = config.get("mcp", {}) or {}
    cmd = mcp.get("command", "python")
    args = mcp.get("args")
    if not args:
        raise ValueError("Missing config['mcp']['args'] for BeerGame MCP server. Example: {'command':'python','args':['SeriousGame/beergame_mcp_server.py']}")

    timeout_sec = float(mcp.get("timeout_sec", 60))
    toolmap = config.get("toolmap", {}) or {
        "new_episode": "beergame_new_episode",
        "step": "beergame_step",
        "metrics": "beergame_get_metrics",
        "trace": "beergame_get_trace",
        "close": "beergame_close_episode",
    }
    mcp_episode_config_key = str(config.get("mcp_episode_config_key", "config"))

    os.makedirs(log_dir, exist_ok=True)

    async def _run_all() -> List[EpisodeOutcome]:
        outcomes: List[EpisodeOutcome] = []
        async with MCPServerStdio(
            name="beergame-mcp",
            params={"command": cmd, "args": args},
            client_session_timeout_seconds=timeout_sec,
            max_retry_attempts=int(mcp.get("max_retry_attempts", 2)),
            retry_backoff_seconds_base=float(mcp.get("retry_backoff_seconds_base", 1.0)),
            cache_tools_list=bool(mcp.get("cache_tools_list", False)),
        ) as server:
            for scenario in test_samples:
                outcomes.append(await _run_one_episode(
                    server=server,
                    scenario=scenario,
                    policy_fn=policy_fn,
                    log_dir=os.path.join(log_dir, "detailed_episode_logs"),
                    toolmap=toolmap,
                    mcp_episode_config_key=mcp_episode_config_key,
                ))
        return outcomes

    # Run episodes (serial for determinism; you can parallelize later if needed)
    outcomes = asyncio.run(_run_all())

    # Aggregate metrics
    # We try common keys from server metrics; if absent, we still return raw metrics list.
    def collect(key: str) -> List[float]:
        vals: List[float] = []
        for o in outcomes:
            v = o.metrics.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    # Controlled-role metrics (server-dependent keys)
    total_costs = collect("total_cost_controlled") or collect("total_cost")  # fallback
    backlog = collect("total_backlog_controlled") or collect("total_backlog")
    inv = collect("total_inventory_controlled") or collect("total_inventory")

    # bullwhip computed from trace if available
    bullwhips: List[float] = []
    for o in outcomes:
        trace = o.trace
        role = str((o.metrics.get("controlled_role") or "")).strip() or "retailer"
        if trace:
            bw = _compute_bullwhip_from_trace(trace, role=role)
            if isinstance(bw, float):
                bullwhips.append(bw)

    def avg_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not xs:
            return None, None
        if len(xs) == 1:
            return xs[0], 0.0
        return float(statistics.mean(xs)), float(statistics.pstdev(xs))

    avg_cost, std_cost = avg_std(total_costs)
    avg_bw, std_bw = avg_std(bullwhips)
    avg_bo, std_bo = avg_std(backlog)
    avg_inv, std_inv = avg_std(inv)

    final_results: Dict[str, Any] = {
        "episodes": len(outcomes),
        "avg_total_cost_controlled": avg_cost,
        "std_total_cost_controlled": std_cost,
        "avg_bullwhip_controlled": avg_bw,
        "std_bullwhip_controlled": std_bw,
        "avg_total_backlog_controlled": avg_bo,
        "std_total_backlog_controlled": std_bo,
        "avg_total_inventory_controlled": avg_inv,
        "std_total_inventory_controlled": std_inv,
        "episode_metrics": [
            {"scenario_id": o.scenario_id, "episode_id": o.episode_id, **o.metrics} for o in outcomes
        ],
    }

    error_log: Dict[str, Any] = {
        "episodes": len(outcomes),
        "errors": [],
    }

    # Save a concise rollup for convenience
    with open(os.path.join(log_dir, "beergame_rollup.json"), "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    return final_results, error_log
