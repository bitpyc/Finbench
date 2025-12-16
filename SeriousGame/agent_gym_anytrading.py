import asyncio
import os
from dotenv import load_dotenv

from openai import AsyncOpenAI
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

async def main():
    # 1. 统一使用一个 AsyncOpenAI client（含 api_key 和 base_url）
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url="http://35.220.164.252:3888/v1/",  # 如果你有自定义 base_url，在这里改
    )
    # 告诉 Agents SDK：所有模型调用都用这个 client
    set_default_openai_client(client)

    # 2. 将默认 API 切成 Chat Completions，避开 Responses + reasoning 的 rs_* 机制
    set_default_openai_api("chat_completions")  # :contentReference[oaicite:6]{index=6}

    # 3. 暂时关闭 tracing，避免额外的 trace 调用和相关错误
    set_tracing_disabled(True)

    # 4. 启动你的 MCP server（gym-anytrading）
    async with MCPServerStdio(
        name="gym-anytrading-mcp",
        params={"command": "python", "args": ["server.py"]},
    ) as trading_server:

        agent = Agent(
            name="TradingTester",
            instructions=(
                "你是一个在 gym-anytrading 股票环境中做交易实验的智能体。"
                "可以使用 MCP 工具：init-stocks-env, step-stocks-env, get-stocks-state。"
                "目标是在整个 episode 中尽量提升 total_profit，并解释你的交易决策。"
            ),
            model="gpt-5",   # 或你有权限的 GPT-5 型号
            mcp_servers=[trading_server],
        )

        run_config = RunConfig()  # 简单场景下可以不特殊配置

        result = await Runner.run(
            agent,
            "先初始化环境，然后连续交易 50 步，最后报告 total_profit 和你的策略总结。",
            run_config=run_config,
            max_turns=200
        )

        print("\n===== Agent Final Output =====")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
