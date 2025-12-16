import asyncio
import os
import time
import random
from dotenv import load_dotenv

from openai import AsyncOpenAI
from openai import (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
)

from agents import (
    Agent,
    Runner,
    RunConfig,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio

load_dotenv()

API_KEY = "sk-UrncoIcMYsnMGJZwQy0VOxmQC2OZCrLkLPCzL5eSMnI1cRGz"
BASE_URL = "http://35.220.164.252:3888/v1/"  # 建议排查时先不设置，直连官方更稳

# ---- OpenAI 降速/重试参数（可通过环境变量覆盖） ----
MIN_REQUEST_INTERVAL_SEC = float(os.getenv("MIN_REQUEST_INTERVAL_SEC", "1.5"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "8"))
BACKOFF_BASE_SEC = float(os.getenv("OPENAI_BACKOFF_BASE_SEC", "1.0"))
BACKOFF_CAP_SEC = float(os.getenv("OPENAI_BACKOFF_CAP_SEC", "20.0"))

# ---- MCP 超时（关键）：默认 5 秒，建议提高 ----
MCP_TIMEOUT_SEC = float(os.getenv("MCP_TIMEOUT_SEC", "30"))


def patch_openai_client_with_throttle_and_retry(client: AsyncOpenAI) -> None:
    """对 OpenAI 请求做节流 + 指数退避重试，降低 rate_limit_check_failed 概率。"""
    lock = asyncio.Lock()
    last_call_ts = 0.0

    async def _throttle():
        nonlocal last_call_ts
        async with lock:
            now = time.monotonic()
            wait = MIN_REQUEST_INTERVAL_SEC - (now - last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            last_call_ts = time.monotonic()

    async def _call_with_retry(orig, *args, **kwargs):
        for attempt in range(OPENAI_MAX_RETRIES + 1):
            await _throttle()
            try:
                return await orig(*args, **kwargs)
            except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError):
                if attempt >= OPENAI_MAX_RETRIES:
                    raise
                backoff = min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * (2 ** attempt))
                jitter = random.uniform(0, 0.25 * backoff)
                await asyncio.sleep(backoff + jitter)

    orig_chat_create = client.chat.completions.create

    async def chat_create_wrapped(*args, **kwargs):
        return await _call_with_retry(orig_chat_create, *args, **kwargs)

    client.chat.completions.create = chat_create_wrapped

    if hasattr(client, "responses"):
        orig_resp_create = client.responses.create

        async def responses_create_wrapped(*args, **kwargs):
            return await _call_with_retry(orig_resp_create, *args, **kwargs)

        client.responses.create = responses_create_wrapped


async def main():
    if not API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    patch_openai_client_with_throttle_and_retry(client)

    set_default_openai_client(client)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)

    # 注意：这里 args 一定要指向“Beer Game 的 server 文件”
    # 强烈建议用 server_beergame.py，避免和股票 server.py 冲突
    async with MCPServerStdio(
        name="beergame-mcp",
        params={"command": "python", "args": ["beergame_mcp_server.py"]},
        client_session_timeout_seconds=MCP_TIMEOUT_SEC,  # 默认是 5 秒，这里拉高 :contentReference[oaicite:1]{index=1}
        max_retry_attempts=2,
        retry_backoff_seconds_base=1.0,
        cache_tools_list=False,
    ) as beergame_server:

        agent = Agent(
            name="BeerGameTester",
            instructions=(
                "你是 Beer Distribution Game（啤酒分销游戏）的决策智能体，只扮演 retailer。"
                "你可以使用 MCP 工具：init-beer-env, step-beer-env, get-beer-state, get-beer-metrics, close-beer-env。"
                "每一步（step）代表 1 周，你需要输出 order_qty（非负整数，下单给上游）。"
                "目标：最小化累计成本（库存成本+缺货成本），同时避免长期 backorder。"
                "请先 init-beer-env，然后循环 step-beer-env 直到 done=True，最后调用 get-beer-metrics 汇报结果与策略总结。"
            ),
            model=os.getenv("OPENAI_MODEL", "gpt-5"),
            mcp_servers=[beergame_server],
        )

        run_config = RunConfig()  # 52+ 次工具调用建议至少 120~200

        result = await Runner.run(
            agent,
            "请用默认参数跑完整个 episode（12 周），最后输出 total_cost_controlled、bullwhip_controlled 以及你的策略总结。",
            run_config=run_config,
            max_turns=200
        )

        print("\n===== Agent Final Output =====")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
