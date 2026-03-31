#!/usr/bin/env python3
"""
Nano-OpenCode CLI 入口

使用方法:
    uv run main.py "帮我分析这个项目"
    uv run main.py --agent plan "制定重构计划"
    uv run main.py --provider ollama --model qwen2.5:7b "帮我分析"
"""

import asyncio
import argparse
import os
import sys
from typing import Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import AgentConfig, AgentMode, agent_registry
from tools import tool_registry
from permission import PermissionAction, PermissionRule
from llm import LLMConfig, OllamaConfig
from loop import AgentLoop, AgentRunner, LoopState
from message import generate_id


async def main_async(args):
    """异步主函数"""
    # 配置 LLM
    if args.provider == "ollama":
        llm_config = LLMConfig(
            provider="ollama",
            model=args.model,
            temperature=args.temperature,
        )
        ollama_config = OllamaConfig(
            base_url=args.ollama_url,
            model=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
    else:
        llm_config = LLMConfig(
            provider=args.provider,
            model=args.model,
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            temperature=args.temperature
        )
        ollama_config = None
    
    # 注册默认 Agent
    _register_default_agents()
    
    # 创建运行器
    runner = AgentRunner(
        workspace=args.workspace,
        llm_config=llm_config,
        ollama_config=ollama_config,
        storage_dir=args.storage
    )
    
    # 定义回调
    output_text = []
    
    async def on_text(text: str):
        """文本输出回调"""
        print(text, end="", flush=True)
        output_text.append(text)
    
    async def on_reasoning(text: str):
        """推理输出回调"""
        if args.verbose:
            print(f"\n[思考] {text}", end="", flush=True)
    
    async def on_tool_start(tool_name: str, arguments: dict):
        """工具开始回调"""
        print(f"\n[工具] {tool_name}({arguments})...", flush=True)
    
    async def on_tool_end(tool_name: str, result):
        """工具结束回调"""
        if result.success:
            print(f"[工具] {tool_name} 完成. output: {result.output[:10]}", flush=True)
        else:
            print(f"[工具] {tool_name} 失败: {result.error}", flush=True)
    
    async def on_step(step: int):
        """步骤回调"""
        if args.verbose:
            print(f"\n[步骤 {step}]", flush=True)
    
    # 运行
    print(f"[Agent: {args.agent}] {args.prompt}\n", flush=True)
    
    result = await runner.run(
        prompt=args.prompt,
        agent_name=args.agent,
        callbacks={
            "on_text": on_text,
            "on_reasoning": on_reasoning,
            "on_tool_start": on_tool_start,
            "on_tool_end": on_tool_end,
            "on_step": on_step
        }
    )
    
    # 输出结果
    print("\n")
    
    if result.error:
        print(f"[错误] {result.error}")
        return 1
    
    if result.state == LoopState.STOP:
        print("[完成]")
    
    if args.verbose:
        print(f"[统计] 步数: {result.tool_calls} 次工具调用")
    
    return 0


def _register_default_agents():
    """注册默认 Agent"""
    # build agent
    build = AgentConfig(
        name="build",
        description="默认 Agent，执行所有操作",
        mode=AgentMode.PRIMARY,
        permission_rules=[
            PermissionRule(permission="*", pattern="*", action=PermissionAction.ALLOW),
        ],
    )
    agent_registry.register(build)
    
    # plan agent
    plan = AgentConfig(
        name="plan",
        description="计划模式，只读分析",
        mode=AgentMode.PRIMARY,
        permission_rules=[
            PermissionRule(permission="read", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="glob", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="grep", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="write", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="edit", pattern="*", action=PermissionAction.DENY),
        ],
    )
    agent_registry.register(plan)
    
    # explore subagent
    from task_tool import create_explore_agent, create_general_agent
    agent_registry.register(create_explore_agent())
    agent_registry.register(create_general_agent())


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
        choices=["build", "plan", "explore", "general"],
        help="使用的 Agent (default: build)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "anthropic", "ollama"],
        help="LLM 提供商 (default: openai)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o",
        help="使用的模型 (default: gpt-4o, ollama: qwen2.5:7b)"
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
    
    # Ollama 特定参数
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API 地址 (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=131072,
        help="Ollama 上下文长度 (default: 131072)"
    )
    
    parser.add_argument(
        "--num-predict",
        type=int,
        default=8192,
        help="Ollama 最大生成 token 数 (default: 8192)"
    )
    
    args = parser.parse_args()
    
    # 设置默认模型
    if args.provider == "ollama" and args.model == "gpt-4o":
        args.model = "glm-4.7-flash"
    
    # 检查 API Key (Ollama 不需要)
    if args.provider != "ollama":
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            print("错误: 请设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY 环境变量")
            sys.exit(1)
    
    # 运行
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
