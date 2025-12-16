#!/usr/bin/env python3
"""
gym-anytrading MCP server

工具列表：
1. init-stocks-env
   初始化 stocks-v0 环境，可以指定 csv_path / frame_bound / window_size
2. step-stocks-env
   执行动作（0/1/2），返回 obs/reward/done/info
3. get-stocks-state
   返回当前观测、总收益、持仓等信息

运行方式（本地调试）：
    python server.py
然后在 MCP 客户端里配置这个 server 作为 stdio transport 即可。
"""

from __future__ import annotations

from typing import Any, Optional

import asyncio
import json

import numpy as np
import pandas as pd
import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import STOCKS_GOOGL  # 内置示例数据 :contentReference[oaicite:0]{index=0}

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions


# ========== 全局环境状态 ==========
CURRENT_ENV: Optional[gym.Env] = None
LAST_OBS: Optional[np.ndarray] = None

server = Server("gym-anytrading-mcp")


def _json_safe(obj: Any) -> Any:
    """
    尽量把对象转换成可以 JSON 序列化的结构：
    - dict / list / tuple / 基本类型递归处理
    - numpy / pandas 转成 Python 标量 / list / dict
    - 其它自定义对象用 str(obj)
    """
    # 基本类型，直接返回
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy 标量
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas 类型
    try:
        import pandas as pd
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
    except Exception:
        pass

    # 字典：递归处理
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # list / tuple：递归处理
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    # 其他自定义对象（例如 Positions）：退化成字符串表示
    return str(obj)


# ========== 工具列表 ==========
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    声明当前 MCP server 提供的工具。
    """
    return [
        types.Tool(
            name="init-stocks-env",
            description=(
                "Initialize a gym-anytrading 'stocks-v0' environment. "
                "If csv_path is not provided, use built-in STOCKS_GOOGL dataset."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {
                        "type": "string",
                        "description": "Optional. Local CSV file path with OHLCV data.",
                    },
                    "frame_bound_start": {
                        "type": "integer",
                        "description": "Start index for frame_bound (inclusive).",
                        "default": 50,
                    },
                    "frame_bound_end": {
                        "type": "integer",
                        "description": "End index for frame_bound (exclusive).",
                        "default": 300,
                    },
                    "window_size": {
                        "type": "integer",
                        "description": "Rolling window size.",
                        "default": 10,
                        "minimum": 1,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="step-stocks-env",
            description=(
                "Take one step in the 'stocks-v0' environment with the given action. "
                "Actions are typically: 0 = hold, 1 = buy, 2 = sell."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "integer",
                        "description": "Discrete action index.",
                        "minimum": 0,
                    },
                },
                "required": ["action"],
            },
        ),
        types.Tool(
            name="get-stocks-state",
            description=(
                "Get current observation and trading stats (total_profit, position, step index, etc.) "
                "from the running 'stocks-v0' environment."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    ]


# ========== 工具实现 ==========
def _ensure_env_initialized() -> gym.Env:
    """
    确保环境已初始化，否则抛错。
    """
    global CURRENT_ENV
    if CURRENT_ENV is None:
        raise RuntimeError("Environment is not initialized. Call 'init-stocks-env' first.")
    return CURRENT_ENV


def _obs_to_list(obs: Any) -> Any:
    """
    将 numpy 类型安全转为 Python 基本类型，便于 JSON 序列化。
    """
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    if isinstance(obs, (np.floating, np.integer)):
        return obs.item()
    return obs


def _safe_reset(env: gym.Env):
    """
    兼容 gym / gymnasium 的 reset 返回格式。
    """
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs, info = result, {}
    return obs, info


def _safe_step(env: gym.Env, action: int):
    """
    兼容不同版本 step 返回 (obs, reward, done, info) 或 (obs, reward, terminated, truncated, info)
    """
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
    elif isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
    else:
        raise RuntimeError(f"Unexpected step() return format: {type(result)} length {len(result)}")
    return obs, float(reward), bool(done), info


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    统一的工具调用入口，根据 name 分发逻辑。
    返回内容用 TextContent 包一层 JSON，方便 LLM 解析。
    """
    if arguments is None:
        arguments = {}

    try:
        if name == "init-stocks-env":
            result = await _tool_init_stocks_env(arguments)
        elif name == "step-stocks-env":
            result = await _tool_step_stocks_env(arguments)
        elif name == "get-stocks-state":
            result = await _tool_get_stocks_state(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        # 统一返回 JSON 文本，方便 LLM 反序列化
        return [
            types.TextContent(
                type="text",
                text=json.dumps(_json_safe(result), ensure_ascii=False),
            )
        ]

    except Exception as e:
        # 将错误也包装成 text，避免直接抛给客户端导致 trace 污染
        error_payload = {
            "error": str(e),
            "tool": name,
        }
        return [
            types.TextContent(
                type="text",
                text=json.dumps(error_payload, ensure_ascii=False),
            )
        ]


# ========== 具体工具实现 ==========
async def _tool_init_stocks_env(args: dict) -> dict[str, Any]:
    """
    初始化 stocks-v0 环境。
    """
    global CURRENT_ENV, LAST_OBS

    csv_path = args.get("csv_path")
    frame_start = int(args.get("frame_bound_start", 50))
    frame_end = int(args.get("frame_bound_end", 300))
    window_size = int(args.get("window_size", 10))

    # 加载数据
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        # gym-anytrading 自带的 Google 股票数据集 :contentReference[oaicite:1]{index=1}
        df = STOCKS_GOOGL.copy()

    # 创建环境
    env = gym.make(
        "stocks-v0",
        df=df,
        frame_bound=(frame_start, frame_end),
        window_size=window_size,
    )
    CURRENT_ENV = env

    # reset 环境
    obs, info = _safe_reset(env)
    LAST_OBS = obs

    # 返回一些基础信息，方便 LLM 做动作编码
    result = {
        "status": "ok",
        "env_id": "stocks-v0",
        "frame_bound": [frame_start, frame_end],
        "window_size": window_size,
        "action_space": {
            "type": "discrete",
            "n": int(getattr(env.action_space, "n", 3)),
            "semantic": {
                "0": "hold",
                "1": "buy",
                "2": "sell",
            },
        },
        "observation": _obs_to_list(obs),
        "observation_shape": getattr(getattr(env, "observation_space", None), "shape", None),
        "info": info,
    }
    return result


async def _tool_step_stocks_env(args: dict) -> dict[str, Any]:
    """
    执行动作，推进一步。
    """
    global CURRENT_ENV, LAST_OBS

    env = _ensure_env_initialized()

    if "action" not in args:
        raise ValueError("Missing required field 'action'.")

    action = int(args["action"])
    obs, reward, done, info = _safe_step(env, action)
    LAST_OBS = obs

    # 一些 gym-anytrading 内部状态字段（非官方 API，但实践中常用） :contentReference[oaicite:2]{index=2}
    total_profit = float(getattr(env, "_total_profit", 0.0))
    step_index = int(getattr(env, "_current_step", 0))
    position = int(getattr(env, "_position", 0))  # 0 无仓，1 多仓

    result = {
        "status": "ok",
        "action_taken": action,
        "observation": _obs_to_list(obs),
        "reward": reward,
        "done": done,
        "info": info,
        "env_stats": {
            "total_profit": total_profit,
            "current_step": step_index,
            "position": position,
        },
    }
    return result


async def _tool_get_stocks_state(args: dict) -> dict[str, Any]:
    """
    返回当前环境状态，不推进时间。
    """
    global CURRENT_ENV, LAST_OBS

    env = _ensure_env_initialized()

    # 仍然是非官方内部字段，只作为实验用
    total_profit = float(getattr(env, "_total_profit", 0.0))
    step_index = int(getattr(env, "_current_step", 0))
    position = int(getattr(env, "_position", 0))

    result = {
        "status": "ok",
        "observation": _obs_to_list(LAST_OBS),
        "env_stats": {
            "total_profit": total_profit,
            "current_step": step_index,
            "position": position,
        },
    }
    return result


# ========== 主入口 ==========
async def main() -> None:
    """
    通过 stdin/stdout 运行 MCP server。
    """
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gym-anytrading-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
