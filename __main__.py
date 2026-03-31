#!/usr/bin/env python3
"""
Nano-OpenCode CLI 入口

使用方法:
    python -m nano_opencode "帮我分析这个项目"
    python -m nano_opencode --agent plan "制定重构计划"
"""

import asyncio
import argparse
import os
import sys
from typing import Optional

from . import AgentRunner, LLMConfig, LoopState


async def main_async(args):
    """异步主函数"""
    # 配置 LLM
    llm_config = LLMConfig(
        model=args.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=args.temperature
    )
    
    # 创建运行器
    runner = AgentRunner(
        workspace=args.workspace,
        llm_config=llm_config,
        storage_dir=args.storage
    )
    
    # 定义回调
    output_text = []
    
    async def on_text(text: str):
        """文本输出回调"""
        print(text, end="", flush=True)
        output_text.append(text)
    
    async def on_tool_start(tool_name: str, arguments: dict):
        """工具开始回调"""
        print(f"\n[Tool] {tool_name}({arguments})...", flush=True)
    
    async def on_tool_end(tool_name: str, result):
        """工具结束回调"""
        if result.success:
            print(f"[Tool] {tool_name} completed", flush=True)
        else:
            print(f"[Tool] {tool_name} failed: {result.error}", flush=True)
    
    async def on_step(step: int):
        """步骤回调"""
        if args.verbose:
            print(f"\n[Step {step}]", flush=True)
    
    # 运行
    print(f"[Agent: {args.agent}] {args.prompt}\n", flush=True)
    
    result = await runner.run(
        prompt=args.prompt,
        agent_name=args.agent,
        callbacks={
            "on_text": on_text,
            "on_tool_start": on_tool_start,
            "on_tool_end": on_tool_end,
            "on_step": on_step
        }
    )
    
    # 输出结果
    print("\n")
    
    if result.error:
        print(f"[Error] {result.error}")
        return 1
    
    if result.state == LoopState.STOP:
        print("[Done]")
    
    if args.verbose:
        print(f"[Stats] Steps: {result.tool_calls} tool calls")
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Nano-OpenCode: 简化版 AI 编程助手"
    )
    
    parser.add_argument(
        "prompt",
        help="用户提示"
    )
    
    parser.add_argument(
        "--agent", "-a",
        default="build",
        choices=["build", "plan", "general", "explore"],
        help="使用的 Agent (default: build)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4",
        help="使用的模型 (default: gpt-4)"
    )
    
    parser.add_argument(
        "--workspace", "-w",
        default=".",
        help="工作目录 (default: .)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="温度参数 (default: 0.7)"
    )
    
    parser.add_argument(
        "--storage", "-s",
        default=None,
        help="会话存储目录"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # 运行
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
