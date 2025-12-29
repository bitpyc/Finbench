#!/usr/bin/env python3
"""
SeriousGame evaluation utilities.

Design goal:
- Mirror BizBench's `utils.tools.evaluate_test_set` structure, but for interactive episode-based environments.
- Keep agent implementations independent: evaluator only requires a policy function `policy_fn(obs, ctx)->int`.
- Do NOT require ACE's Generator/timed_llm_call pipeline.
"""

from __future__ import annotations

import inspect
import asyncio
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import uuid
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
    # print(f"[DEBUG] MCP {tool_name} raw response: {res}", flush=True)

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

# ============================
# EDT evaluation (scenario-level)
# ============================

def _pick_free_port(host: str = "127.0.0.1") -> int:
    """Bind to port 0 to obtain a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _start_bptk_server(
    *,
    python_exe: str,
    bptk_script: str,
    repo_root: str,
    host: str,
    port: int,
    bearer_token: str = "",
) -> subprocess.Popen:
    cmd = [python_exe, bptk_script, "--repo-root", repo_root, "--host", host, "--port", str(port)]
    if bearer_token:
        cmd += ["--bearer-token", bearer_token]
    env = os.environ.copy()
    # ensure consistent auth passthrough
    if bearer_token:
        env["BPTK_BEARER_TOKEN"] = bearer_token
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


def _wait_for_bptk(base_url: str, timeout_sec: float) -> bool:
    import requests

    start = time.time()
    while time.time() - start < float(timeout_sec):
        try:
            r = requests.get(f"{base_url}/scenarios", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _stop_process(proc: Optional[subprocess.Popen]):
    if not proc:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


async def _run_one_edt_episode_scenario_level(
    *,
    agent: Any,
    sample: Dict[str, Any],
    edt_cfg: Dict[str, Any],
    log_dir: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """Run a single scenario-level EDT episode.

    Flow:
    1) Ask the agent for a minimal schema based on the template scenario summary.
    2) Materialize a temporary BPTK repo root + generated scenario.
    3) Start BPTK server (pointing at temp root) on a free port.
    4) Start EDT MCP server against that base_url.
    5) Run `max_steps` by calling step-edt-env with empty settings.
    6) Collect metrics from last obs.flat_metrics.
    """
    scenario_id = str(sample.get("scenario_id", "interactive"))
    base_cfg = dict(sample.get("config", {}) or {})

    # ---- config extraction
    bptk = dict(edt_cfg.get("bptk", {}) or {})
    template_repo_root = str(bptk.get("template_repo_root", "") or "")
    bptk_script = str(bptk.get("script", "") or "")
    bptk_host = str(bptk.get("host", "127.0.0.1") or "127.0.0.1")
    bearer_token = str(bptk.get("bearer_token", "") or "")
    start_timeout = float(bptk.get("start_timeout_sec", 60.0) or 60.0)

    base_scenario = dict(edt_cfg.get("base_scenario", {}) or {})
    scenario_manager = str(base_scenario.get("scenario_manager", "smEDT"))
    scenario_file = str(base_scenario.get("scenario_file", "interactive.json"))
    base_scenario_key = str(base_scenario.get("scenario_key", "interactive"))

    if not template_repo_root:
        raise ValueError("EDT requires --bptk_repo_root (template repo root containing scenarios/ and simulation_models/)")
    if not bptk_script:
        raise ValueError("EDT requires bptk script path (e.g. SeriousGame/run_bptk_server.py)")

    # ---- generate scenario from agent schema
    from SeriousGame.edt_scenario_generator import (
        generate_temp_repo_from_schema,
        load_scenario_document,
        extract_sm_document,
        extract_scenario,
        summarize_base_scenario,
    )

    # summarize template for the agent
    doc = load_scenario_document(template_repo_root, scenario_file)
    sm_doc = extract_sm_document(doc, scenario_manager)
    base_scen = extract_scenario(sm_doc, base_scenario_key)
    base_summary = summarize_base_scenario(base_scen)

    schema_call = agent._decide_edt_scenario_schema(
        base_summary=base_summary,
        scenario_meta={
            "scenario_id": scenario_id,
            "scenario_manager": scenario_manager,
            "scenario_file": scenario_file,
            "base_scenario_key": base_scenario_key,
        },
    )
    schema = await schema_call if inspect.isawaitable(schema_call) else schema_call

    # unique scenario key (BPTK uses this as the scenario name)
    uid = uuid.uuid4().hex[:10]
    new_scenario_key = f"{base_scenario_key}__{uid}"

    gen = generate_temp_repo_from_schema(
        template_repo_root=template_repo_root,
        scenario_manager=scenario_manager,
        scenario_file=scenario_file,
        base_scenario_key=base_scenario_key,
        schema=schema,
        new_scenario_key=new_scenario_key,
        out_scenario_file=f"{new_scenario_key}.json",
        keep_temp=bool(edt_cfg.get("keep_temp_bptk_repo", False)),
    )

    temp_repo_root = gen["temp_repo_root"]
    max_steps = int(edt_cfg.get("max_steps", base_cfg.get("max_steps", 96)))

    # override config to point to generated scenario
    episode_cfg = dict(base_cfg)
    episode_cfg["scenario_managers"] = [scenario_manager]
    episode_cfg["scenarios"] = [new_scenario_key]
    episode_cfg["max_steps"] = max_steps

    # ---- start bptk server
    port = _pick_free_port(bptk_host)
    base_url = f"http://{bptk_host}:{port}"

    bptk_proc: Optional[subprocess.Popen] = None
    outcome: Dict[str, Any] = {}
    try:
        if verbose:
            print(f"[EDT] scenario_id={scenario_id} → start BPTK @ {base_url}")
        bptk_proc = _start_bptk_server(
            python_exe=sys.executable,
            bptk_script=bptk_script,
            repo_root=temp_repo_root,
            host=bptk_host,
            port=port,
            bearer_token=bearer_token,
        )

        if not _wait_for_bptk(base_url, start_timeout):
            # attach logs if available
            out, err = (bptk_proc.communicate(timeout=1) if bptk_proc else (b"", b""))
            raise RuntimeError(
                "BPTK server did not become ready in time. "
                f"stdout={out[-2000:].decode(errors='ignore')} stderr={err[-2000:].decode(errors='ignore')}"
            )

        # ---- run MCP episode
        mcp_cfg = dict(edt_cfg.get("mcp", {}) or {})
        mcp_cmd = str(mcp_cfg.get("command", sys.executable))
        mcp_args = list(mcp_cfg.get("args", []))
        if not mcp_args:
            raise ValueError("EDT requires edt.mcp.args to contain the EDT MCP server script")

        # base_url is per-episode
        mcp_args = mcp_args + ["--base-url", base_url]

        async with MCPServerStdio(
            name=f"edt-{scenario_id}-{uid}",
            params={"command": mcp_cmd, "args": mcp_args},
        ) as server:
            if verbose:
                print(f"[EDT] init-edt-env scenario={new_scenario_key} steps={max_steps}")

            init_payload = {"config": episode_cfg}
            init_out = await _mcp_call(server, "init-edt-env", init_payload)
            env_id = init_out["env_id"]

            obs = init_out.get("obs", {}) or {}
            done = False
            step_idx = 0
            total_reward = 0.0
            trajectory: List[Dict[str, Any]] = []

            while not done and step_idx < max_steps:
                step_out = await _mcp_call(server, "step-edt-env", {"env_id": env_id, "settings": {}})
                obs = step_out.get("obs", {}) or {}
                reward = float(step_out.get("reward", 0.0) or 0.0)
                done = bool(step_out.get("done", False))
                total_reward += reward

                if edt_cfg.get("save_trajectory", False):
                    trajectory.append({"step": step_idx, "obs": obs, "reward": reward})
                step_idx += 1

            last_obs = obs
            flat = (last_obs.get("flat_metrics") or {}).copy()
            metrics: Dict[str, float] = {}
            for k, v in flat.items():
                try:
                    metrics[k] = float(v)
                except Exception:
                    continue
            metrics.setdefault("total_reward", float(total_reward))

            # best-effort close
            try:
                await _mcp_call(server, "close-edt-env", {"env_id": env_id})
            except Exception:
                pass

            outcome = {
                "scenario_id": scenario_id,
                "generated": {
                    "scenario_key": new_scenario_key,
                    "scenario_file": gen.get("scenario_file"),
                    "scenario_path": gen.get("scenario_path"),
                    "temp_repo_root": temp_repo_root,
                },
                "schema": gen.get("schema"),
                "base_summary": gen.get("base_summary"),
                "config": episode_cfg,
                "steps": step_idx,
                "total_reward": float(total_reward),
                "metrics": metrics,
            }
            if trajectory:
                outcome["trajectory"] = trajectory

            # per-episode log
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                ep_path = os.path.join(log_dir, f"edt_episode_{scenario_id}_{uid}.json")
                with open(ep_path, "w", encoding="utf-8") as f:
                    json.dump(outcome, f, ensure_ascii=False, indent=2)

            return outcome

    finally:
        _stop_process(bptk_proc)
        if not bool(edt_cfg.get("keep_temp_bptk_repo", False)):
            try:
                shutil.rmtree(temp_repo_root, ignore_errors=True)
            except Exception:
                pass


def evaluate_edt_set(
    *,
    agent: Any,
    test_samples: List[Dict[str, Any]],
    edt_cfg: Dict[str, Any],
    log_dir: Optional[str] = None,
    log_prefix: str = "edt",
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Scenario-level EDT evaluation.

    Returns:
      - global_stats: per-metric mean over episodes
      - detail_log: {"episodes": [...], "errors": [...]}
    """

    async def _run_all() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        outcomes: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for sample in test_samples:
            try:
                out = await _run_one_edt_episode_scenario_level(
                    agent=agent,
                    sample=sample,
                    edt_cfg=edt_cfg,
                    log_dir=log_dir,
                    verbose=verbose,
                )
                outcomes.append(out)
            except Exception as e:
                sid = str(sample.get("scenario_id", "interactive"))
                err = {"scenario_id": sid, "error": repr(e)}
                errors.append(err)
                if verbose:
                    print(f"[EDT][ERROR] scenario_id={sid} err={e}")
        return outcomes, errors

    outcomes, errors = asyncio.run(_run_all())

    metric_lists: Dict[str, List[float]] = {}
    for ep in outcomes:
        m = ep.get("metrics", {}) or {}
        for k, v in m.items():
            if isinstance(v, (int, float)):
                metric_lists.setdefault(k, []).append(float(v))

    global_stats: Dict[str, float] = {}
    for k, lst in metric_lists.items():
        if lst:
            global_stats[k] = float(sum(lst) / len(lst))

    detail_log: Dict[str, Any] = {"episodes": outcomes, "errors": errors}

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"{log_prefix}_episodes.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detail_log, f, ensure_ascii=False, indent=2)
        agg_path = os.path.join(log_dir, f"{log_prefix}_metrics.json")
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)

    return global_stats, detail_log



# ============================
# BeerGame run helpers (model-agnostic): Prepare / Evaluate / Save
# ============================

def beergame_prepare_run(
    *,
    mode: str,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    allowed_modes: Optional[set] = None,
    agent_method: str = "agent",
) -> Dict[str, Any]:
    """Prepare run directories and a minimal context for BeerGame evaluation.

    Returns ctx with keys:
      - run_subdir
      - resolved_save_path
      - log_dir
      - scenario_index: {scenario_id -> scenario_config}
    """
    if allowed_modes is not None and mode not in allowed_modes:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {sorted(list(allowed_modes))}")
    if not test_samples:
        raise ValueError("BeerGame requires non-empty test_samples")

    save_dir = str(config.get("save_dir", "results"))
    task_name = str(config.get("task_name", "BeerGame"))

    run_subdir = f"{task_name}/{agent_method}/{mode}/{time.strftime('%Y%m%d_%H%M%S')}"
    resolved_save_path = os.path.join(save_dir, run_subdir)
    os.makedirs(resolved_save_path, exist_ok=True)

    log_dir = os.path.join(resolved_save_path, "detailed_logs")
    os.makedirs(log_dir, exist_ok=True)

    scenario_index: Dict[str, Dict[str, Any]] = {}
    for s in test_samples:
        sid = str(s.get("scenario_id", ""))
        if sid:
            scenario_index[sid] = dict(s.get("config", {}) or {})

    print(f"\n{'='*60}")
    print("BEERGAME EVALUATION")
    print(f"{'='*60}")
    print(f"Episodes: {len(test_samples)}")
    print(f"Log dir: {log_dir}")
    print(f"{'='*60}\n")

    return {
        "run_subdir": run_subdir,
        "resolved_save_path": resolved_save_path,
        "log_dir": log_dir,
        "scenario_index": scenario_index,
    }


def beergame_evaluate_run(
    *,
    agent: Any,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Tuple[Dict[str, Any], Any]:
    """Run BeerGame evaluation. Model-agnostic except for agent decision hook."""
    beergame_cfg = dict(config.get("beergame", {}) or {})

    # Inherit MCP/tool routing config from top-level if not explicitly provided in beergame_cfg
    if "mcp" not in beergame_cfg and "mcp" in config:
        beergame_cfg["mcp"] = config["mcp"]
    if "toolmap" not in beergame_cfg and "toolmap" in config:
        beergame_cfg["toolmap"] = config["toolmap"]
    if "mcp_episode_config_key" not in beergame_cfg and "mcp_episode_config_key" in config:
        beergame_cfg["mcp_episode_config_key"] = config["mcp_episode_config_key"]

    scenario_index = ctx.get("scenario_index") or {}

    def policy_fn(obs: Dict[str, Any], step_ctx: Dict[str, Any]) -> int:
        scenario_id = str(step_ctx.get("scenario_id", ""))
        scenario_cfg = scenario_index.get(scenario_id, {}) if isinstance(scenario_index, dict) else {}
        if scenario_cfg and "target_inventory" in scenario_cfg:
            step_ctx = dict(step_ctx)
            step_ctx["target_inventory"] = scenario_cfg["target_inventory"]
        return int(agent._decide_order_qty(obs, step_ctx))

    results, error_log = evaluate_beergame_set(
        test_samples=test_samples,
        policy_fn=policy_fn,
        config=beergame_cfg,
        log_dir=ctx.get("log_dir"),
    )
    return results, error_log


def beergame_save_run(
    *,
    results: Dict[str, Any],
    error_log: Any,
    config: Dict[str, Any],
    ctx: Dict[str, Any],
) -> None:
    """Save test_results.json and run_config.json. Model-agnostic."""
    resolved_save_path = str(ctx["resolved_save_path"])

    with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({"test_results": results, "error_log": error_log}, f, indent=2, ensure_ascii=False)

    config_payload = dict(config)
    config_payload["run_subdir"] = ctx.get("run_subdir")
    config_payload["resolved_save_path"] = resolved_save_path
    with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("BEERGAME RUN COMPLETE")
    print(f"{'='*60}")
    if isinstance(results, dict):
        print(f"Avg cost: {results.get('avg_total_cost_controlled')}")
        print(f"Avg bullwhip: {results.get('avg_bullwhip_controlled')}")
    print(f"Results saved to: {resolved_save_path}")
    print(f"{'='*60}\n")


# ============================
# BeerGame decision helpers (task-specific, model-agnostic)
# ============================

def beergame_build_query(obs: Dict[str, Any]) -> str:
    return (
        f"role={obs.get('role')} week={obs.get('week')} "
        f"inv={obs.get('inventory')} bo={obs.get('backorder')} "
        f"in_order={obs.get('incoming_order')} supply={obs.get('supply_line')} "
        f"last_order={obs.get('last_order')}"
    )


def beergame_base_rule_order(
    *,
    obs: Dict[str, Any],
    ctx: Dict[str, Any],
    max_order_qty: int,
    default_target_inventory: int = 20,
) -> int:
    """Simple fallback baseline:
    order ≈ incoming_order + 0.5 * (target_inventory - (inventory - backorder))
    """
    incoming = int(obs.get("incoming_order", 0) or 0)
    inv = int(obs.get("inventory", 0) or 0)
    bo = int(obs.get("backorder", 0) or 0)
    target_inv = int(ctx.get("target_inventory", default_target_inventory))
    adj = target_inv - (inv - bo)
    order_qty = incoming + max(0, int(0.5 * adj))
    return max(0, min(int(max_order_qty), int(order_qty)))


def beergame_render_prompt(
    *,
    role: str,
    obs: Dict[str, Any],
    retrieved: str,
    base_order: int,
) -> Tuple[str, str]:
    """Render BeerGame system/user prompts (task-specific, reusable across models)."""
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
        f"A simple baseline suggests order_qty ≈ {int(base_order)}.",
    ]
    if retrieved:
        lines += [
            "",
            "Relevant notes from previous episodes/steps:",
            retrieved,
        ]
    user = "\n".join(lines)
    return system, user


def beergame_extract_order_and_note(
    *,
    js: Dict[str, Any],
    base_order: int,
    max_order_qty: int,
) -> Tuple[int, str]:
    """Parse and clip model output."""
    order_qty = js.get("order_qty", base_order) if isinstance(js, dict) else base_order
    note = js.get("note", "") if isinstance(js, dict) else ""
    try:
        order_qty = int(order_qty)
    except Exception:
        order_qty = int(base_order)
    order_qty = max(0, min(int(max_order_qty), int(order_qty)))
    note = note.strip() if isinstance(note, str) else ""
    return int(order_qty), note


def beergame_format_memory_note(
    *,
    role: str,
    week: int,
    obs: Dict[str, Any],
    order_qty: int,
    note: str,
) -> str:
    return (
        f"[BeerGame] role={role} week={int(week)} "
        f"obs={obs} order={int(order_qty)} note={note}"
    )



# ============================
# EDT run helpers (model-agnostic): Prepare / Evaluate / Save
# ============================

def edt_prepare_run(
    *,
    mode: str,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    allowed_modes: Optional[set] = None,
) -> Dict[str, str]:
    """Prepare run directories and a minimal context for EDT evaluation.

    All logic here is model-agnostic.
    Returns ctx with keys: run_subdir, resolved_save_path, log_dir.
    """
    if allowed_modes is not None and mode not in allowed_modes:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {sorted(list(allowed_modes))}")

    if not test_samples:
        raise ValueError("EDT requires non-empty test_samples")

    save_dir = str(config.get("save_dir", "results"))
    task_name = str(config.get("task_name", "SeriousGame/EDT"))

    run_subdir = f"{task_name}/{mode}/{time.strftime('%Y%m%d_%H%M%S')}"
    resolved_save_path = os.path.join(save_dir, run_subdir)
    os.makedirs(resolved_save_path, exist_ok=True)

    log_dir = os.path.join(resolved_save_path, "detailed_logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("EDT EVALUATION")
    print(f"{'='*60}")
    print(f"Episodes: {len(test_samples)}")
    print(f"Log dir: {log_dir}")
    print(f"{'='*60}\n")

    return {
        "run_subdir": run_subdir,
        "resolved_save_path": resolved_save_path,
        "log_dir": log_dir,
    }


def edt_evaluate_run(
    *,
    agent: Any,
    test_samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    ctx: Dict[str, str],
) -> Tuple[Dict[str, Any], Any]:
    """Run EDT evaluation (scenario-level). Model-agnostic except for agent decision hooks."""
    edt_cfg = dict(config.get("edt", {}) or {})

    global_stats, detail_log = evaluate_edt_set(
        agent=agent,
        test_samples=test_samples,
        edt_cfg=edt_cfg,
        log_dir=ctx.get("log_dir"),
        log_prefix="edt",
        verbose=True,
    )

    # Keep output structure consistent with other tasks in this repo.
    results: Dict[str, Any] = global_stats
    error_log = (detail_log or {}).get("errors", [])

    return results, error_log


def edt_save_run(
    *,
    results: Dict[str, Any],
    error_log: Any,
    config: Dict[str, Any],
    ctx: Dict[str, str],
) -> None:
    """Save test_results.json and run_config.json. Model-agnostic."""
    resolved_save_path = ctx["resolved_save_path"]

    with open(os.path.join(resolved_save_path, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({"test_results": results, "error_log": error_log}, f, indent=2, ensure_ascii=False)

    config_payload = dict(config)
    config_payload["run_subdir"] = ctx.get("run_subdir")
    config_payload["resolved_save_path"] = resolved_save_path
    with open(os.path.join(resolved_save_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("EDT RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Aggregated EDT metrics: {results}")
    print(f"Results saved to: {resolved_save_path}")
    print(f"{'='*60}\n")

# ========== EDT: shared (model-agnostic) helpers ==========

from typing import Any, Dict, List, Tuple, Optional


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _clip_float(v: Any, lo: float, hi: float, default: float) -> float:
    x = _safe_float(v, default)
    return float(lo) if x < lo else float(hi) if x > hi else float(x)


def _clip_int(v: Any, lo: int, hi: int, default: int) -> int:
    x = _safe_int(v, default)
    return int(lo) if x < lo else int(hi) if x > hi else int(x)


def build_edt_decision_context(
    *,
    base_summary: Dict[str, Any],
    scenario_meta: Optional[Dict[str, Any]] = None,
    max_steps_hint: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Produce a robust, model-agnostic decision context for EDT prompting + normalization.
    Key feature: if horizon/timeline in base_summary is invalid, infer from project deadlines.
    """
    scenario_meta = scenario_meta or {}
    projects = list(base_summary.get("projects", []) or [])

    dt = _safe_float(base_summary.get("dt", 0.25), 0.25)
    if dt <= 0:
        dt = 0.25

    starttime = _safe_float(base_summary.get("starttime", 0.0), 0.0)
    stoptime = _safe_float(base_summary.get("stoptime", starttime), starttime)

    # fallback derive timeline if invalid
    if stoptime <= starttime and projects:
        starts, deadlines = [], []
        for p in projects:
            st = p.get("start_time", None)
            dl = p.get("deadline", None)
            if isinstance(st, (int, float)): starts.append(float(st))
            if isinstance(dl, (int, float)): deadlines.append(float(dl))
        if starts and deadlines:
            starttime = min(starts)
            stoptime = max(deadlines)

    # horizon fallback
    horizon_steps = _safe_int(base_summary.get("horizon_steps", 0), 0)
    if horizon_steps <= 0:
        horizon_steps = max(1, int(round((stoptime - starttime) / dt))) if stoptime > starttime else max(1, int(max_steps_hint or 96))

    if max_steps_hint is not None:
        horizon_steps = min(horizon_steps, int(max_steps_hint))

    cmax = _safe_int(base_summary.get("consultants_max", 0), 0)
    default_r = _safe_float(base_summary.get("revenue_risk_level", 1.0), 1.0)

    # compute default per-project steps + keep rich fields for prompting
    proj_ctx: List[Dict[str, Any]] = []
    default_P: List[Any] = []
    for i, p in enumerate(projects):
        p_start_time = p.get("start_time", starttime)
        p_deadline = p.get("deadline", stoptime)

        s_step = int(round((float(p_start_time) - starttime) / dt)) if isinstance(p_start_time, (int, float)) else 0
        d_step = int(round((float(p_deadline) - starttime) / dt)) if isinstance(p_deadline, (int, float)) else horizon_steps

        s_step = max(0, min(horizon_steps, s_step))
        d_step = max(0, min(horizon_steps, d_step))
        if d_step < s_step:
            d_step = s_step

        default_P.append([1, s_step, d_step])

        proj_ctx.append({
            "index": i,
            "name": str(p.get("name", f"Project {i+1}")),
            "required_consultants": _safe_int(p.get("consultants", 1), 1),
            "contracted_effort": _safe_float(p.get("contracted_effort", 0.0), 0.0),
            "billing_rate": _safe_float(p.get("billing_rate", 0.0), 0.0),
            "contracted_probability": _clip_float(p.get("contracted_probability", 1.0), 0.0, 1.0, 1.0),
            "extension_probability": _clip_float(p.get("extension_probability", 0.0), 0.0, 1.0, 0.0),
            "follow_on_probability": _clip_float(p.get("follow_on_probability", 0.0), 0.0, 1.0, 0.0),
            "default_start_step": s_step,
            "default_deadline_step": d_step,
        })

    return {
        "scenario_id": str(scenario_meta.get("scenario_id", "interactive")),
        "dt": dt,
        "starttime": starttime,
        "stoptime": stoptime,
        "horizon_steps": horizon_steps,
        "C_max": cmax,
        "R_default": _clip_float(default_r, 0.0, 1.0, 1.0),
        "projects": proj_ctx,
        "default_P": default_P,
        # optional cost info if upstream later adds them (only displayed if present)
        "fixed_cost": base_summary.get("fixed_cost", None),
        "consultant_salary": base_summary.get("consultant_salary", None),
        "consultant_workplace_cost": base_summary.get("consultant_workplace_cost", None),
    }


def render_edt_prompt(ctx: Dict[str, Any]) -> Tuple[str, str]:
    """Create model-agnostic EDT system/user prompts from ctx."""
    system = "你是一个企业运营经理，目标是在仿真中最大化利润。只输出 JSON，不要解释，不要 markdown。"

    cost_lines = []
    if ctx.get("fixed_cost") is not None:
        cost_lines.append(f"- fixed_cost/step: {ctx['fixed_cost']}")
    if ctx.get("consultant_salary") is not None:
        cost_lines.append(f"- consultant salary/step: {ctx['consultant_salary']}")
    if ctx.get("consultant_workplace_cost") is not None:
        cost_lines.append(f"- consultant workplace_cost/step: {ctx['consultant_workplace_cost']}")
    cost_block = "\n".join(cost_lines) if cost_lines else "- 成本包含固定成本与顾问成本（数值可能不在此处给出），它们会持续侵蚀利润。\n"

    proj_blocks = []
    for p in ctx["projects"]:
        proj_blocks.append(
            f"{p['index']}: {p['name']}\n"
            f"  - required_consultants: {p['required_consultants']}\n"
            f"  - contracted_effort: {p['contracted_effort']}\n"
            f"  - billing_rate(不可控): {p['billing_rate']}\n"
            f"  - contracted_p: {p['contracted_probability']}\n"
            f"  - extension_p: {p['extension_probability']}\n"
            f"  - follow_on_p: {p['follow_on_probability']}\n"
            f"  - default_start_step: {p['default_start_step']}\n"
            f"  - default_deadline_step: {p['default_deadline_step']}"
        )

    user = (
        "EDT（Enterprise Digital Twin）场景级决策（一次性）。你只在仿真开始前决策一次，中途不再干预。\n\n"
        "目标：最大化终局 accumulated_earnings（累计净利润≈累计收入-累计支出）。\n"
        "直觉：执行项目会带来收入；维持顾问与公司运营会产生持续成本；R 会影响延期/后续等风险事件触发，从而改变收入/成本轨迹。\n\n"
        f"时间轴：starttime={ctx['starttime']}, stoptime={ctx['stoptime']}, dt={ctx['dt']}, horizon_steps={ctx['horizon_steps']}。\n"
        f"资源上限：C_max={ctx['C_max']}；模板 R_default={ctx['R_default']}。\n\n"
        f"成本信息：\n{cost_block}\n\n"
        "项目列表（保持索引顺序，不要重排）：\n"
        + "\n\n".join(proj_blocks)
        + "\n\n"
        "你只能输出最小 schema：\n"
        "- C: 顾问人数（0..C_max）\n"
        "- R: revenue_risk_level（0..1）\n"
        "- P: 长度=N_projects；P[i]=0 表示禁用；或 [1,start_step,deadline_step] 启用并设置排期（步索引）。\n\n"
        "硬约束：\n"
        "- 至少启用 1 个项目（不要输出全 0）。\n"
        "- 只要启用任何项目，则 C 必须 >=1。\n"
        "- 0 <= start_step <= deadline_step <= horizon_steps。\n"
        "- 不允许修改 salary、fixed_cost、billing_rate 等结构性参数。\n\n"
        "只返回 JSON：\n"
        '{ "C": <int>, "R": <float>, "P": [ ... ] }'
    )
    return system, user


def normalize_edt_schema(raw: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Clip/repair schema and enforce non-degenerate constraints."""
    n = len(ctx["projects"])
    horizon = int(ctx["horizon_steps"])
    cmax = int(ctx["C_max"])
    default_P = list(ctx["default_P"])

    out = {}
    out["C"] = _clip_int(raw.get("C", cmax), 0, max(0, cmax), cmax)
    out["R"] = _clip_float(raw.get("R", ctx["R_default"]), 0.0, 1.0, ctx["R_default"])

    P = raw.get("P", None)
    if not isinstance(P, list) or len(P) != n:
        P = default_P

    P_norm: List[Any] = []
    for i, dec in enumerate(P):
        if dec == 0 or dec is False or dec is None:
            P_norm.append(0)
            continue
        if isinstance(dec, list) and len(dec) == 3 and _safe_int(dec[0], 0) == 1:
            s = _clip_int(dec[1], 0, horizon, default_P[i][1])
            d = _clip_int(dec[2], 0, horizon, default_P[i][2])
            if d < s:
                d = s
            P_norm.append([1, s, d])
        else:
            P_norm.append(default_P[i])

    # enforce at least one enabled
    if all(x == 0 for x in P_norm):
        P_norm = default_P[:]

    # enforce C>=1 if any enabled
    if any(isinstance(x, list) and x and x[0] == 1 for x in P_norm) and out["C"] == 0:
        out["C"] = 1 if cmax > 0 else 0

    out["P"] = P_norm
    return out
