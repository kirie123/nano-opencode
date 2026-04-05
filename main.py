#!/usr/bin/env python3
"""
Nano-OpenCode CLI 入口

使用方法:
    uv run main.py "帮我分析这个项目"
    uv run main.py --agent plan "制定重构计划"
    uv run main.py --provider ollama --model qwen2.5:7b "帮我分析"
    uv run main.py -i                    # 交互模式
    uv run main.py -i --provider ollama  # 交互模式 + Ollama
"""

import asyncio
import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import AgentConfig, AgentMode, agent_registry
from tools.tools import tool_registry
from permission.permission import PermissionAction, PermissionRule
from llm.llm import LLMConfig, OllamaConfig
from loop import AgentLoop, AgentRunner, LoopState
from core.message import generate_id


async def main_async(args):
    """异步主函数"""
    llm_config, ollama_config = _create_llm_config(args)
    _register_default_agents()
    
    runner = AgentRunner(
        workspace=args.workspace,
        llm_config=llm_config,
        ollama_config=ollama_config,
        storage_dir=args.storage
    )
    
    if args.interactive:
        return await _run_interactive(args, runner)
    else:
        return await _run_single(args, runner)


def _create_llm_config(args):
    """创建 LLM 配置"""
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
    
    return llm_config, ollama_config


async def _run_single(args, runner):
    """单次执行模式"""
    callbacks = _create_callbacks(args)
    
    print(f"[Agent: {args.agent}] {args.prompt}\n", flush=True)
    
    result, cache_info = await runner.run(
        prompt=args.prompt,
        agent_name=args.agent,
        callbacks=callbacks
    )
    
    print("\n")
    
    if result.error:
        print(f"[错误] {result.error}")
        return 1
    
    if result.state == LoopState.STOP:
        print("[完成]")
    
    if args.verbose:
        print(f"[统计] 步数: {result.tool_calls} 次工具调用")
        if cache_info:
            stats = cache_info.get("cache_stats", {})
            print(f"[缓存] Prompt命中: {stats.get('prompt_cache_hits', 0)}, 压缩命中: {stats.get('compaction_cache_hits', 0)}")
    
    return 0


async def _run_interactive(args, runner):
    """交互模式"""
    print("=" * 50)
    print("Nano-OpenCode 交互模式")
    print(f"Provider: {args.provider}, Model: {args.model}")
    print("输入 'exit' 或 'quit' 退出")
    print("=" * 50)
    print()
    
    callbacks = _create_callbacks(args)
    
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break
        
        if not prompt:
            continue
        
        if prompt.lower() in ("exit", "quit", "q"):
            print("再见!")
            break
        
        if prompt.startswith("/"):
            await _handle_command(prompt, args, runner)
            continue
        
        print(f"\n[Agent: {args.agent}]\n", flush=True)
        
        result, cache_info = await runner.run(
            prompt=prompt,
            agent_name=args.agent,
            callbacks=callbacks
        )
        
        print("\n")
        
        if result.error:
            print(f"[错误] {result.error}")
        
        if result.state == LoopState.STOP:
            if args.verbose:
                print(f"[统计] 步数: {result.tool_calls} 次工具调用")
                if cache_info:
                    stats = cache_info.get("cache_stats", {})
                    print(f"[缓存] Prompt命中: {stats.get('prompt_cache_hits', 0)}, 压缩命中: {stats.get('compaction_cache_hits', 0)}")
    
    return 0


async def _handle_command(command: str, args, runner):
    """处理斜杠命令"""
    cmd = command.lower().strip()
    
    if cmd in ("/help", "/h", "/?"):
        print("""
可用命令:
  /help, /h, /?    显示帮助
  /clear           清除会话历史
  /agent <name>    切换 Agent (build, plan, explore, general, analyze)
  /model <name>    切换模型
  /exit, /quit     退出程序
""")
    elif cmd == "/clear":
        runner.session = None
        print("[会话已清除]")
    elif cmd.startswith("/agent "):
        agent_name = command[7:].strip()
        if agent_name in ["build", "plan", "explore", "general", "analyze"]:
            args.agent = agent_name
            print(f"[已切换到 Agent: {agent_name}]")
        else:
            print(f"[未知 Agent: {agent_name}]")
    elif cmd.startswith("/model "):
        model_name = command[7:].strip()
        args.model = model_name
        print(f"[已切换到模型: {model_name}]")
    elif cmd in ("/exit", "/quit"):
        print("再见!")
        sys.exit(0)
    else:
        print(f"[未知命令: {command}]")


def _create_callbacks(args):
    """创建回调函数"""
    async def on_text(text: str):
        print(text, end="", flush=True)
    
    async def on_reasoning(text: str):
        if args.verbose:
            print(f"\n[思考] {text}", end="", flush=True)
    
    async def on_tool_start(tool_name: str, arguments: dict):
        print(f"\n[工具] {tool_name}({arguments})...", flush=True)
    
    async def on_tool_end(tool_name: str, result):
        if result.success:
            print(f"[工具] {tool_name} 完成. output: {result.output[:10]}", flush=True)
        else:
            print(f"[工具] {tool_name} 失败: {result.error}", flush=True)
    
    async def on_step(step: int):
        if args.verbose:
            print(f"\n[步骤 {step}]", flush=True)
    
    return {
        "on_text": on_text,
        "on_reasoning": on_reasoning,
        "on_tool_start": on_tool_start,
        "on_tool_end": on_tool_end,
        "on_step": on_step
    }


def _register_default_agents():
    """注册默认 Agent"""
    build = AgentConfig(
        name="build",
        description="默认 Agent，执行所有操作",
        mode=AgentMode.PRIMARY,
        permission_rules=[
            PermissionRule(permission="*", pattern="*", action=PermissionAction.ALLOW),
        ],
    )
    agent_registry.register(build)
    
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
    
    from tools.task_tool import create_explore_agent, create_general_agent, create_analyze_agent
    agent_registry.register(create_explore_agent())
    agent_registry.register(create_general_agent())
    agent_registry.register(create_analyze_agent())


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Nano-OpenCode: 简化版 AI 编程助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run main.py "帮我分析这个项目"
  uv run main.py -i --provider ollama
  uv run main.py --agent plan "制定重构计划"
  uv run main.py -p ollama -m qwen2.5:7b "帮我分析"
"""
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="用户提示 (交互模式下可选)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="启动交互模式"
    )
    
    parser.add_argument(
        "--agent", "-a",
        default="build",
        choices=["build", "plan", "explore", "general", "analyze"],
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
    
    if not args.interactive and not args.prompt:
        parser.error("需要提供 prompt 或使用 -i 进入交互模式")
    
    if args.provider == "ollama" and args.model == "gpt-4o":
        args.model = "glm-4.7-flash"
    
    if args.provider != "ollama":
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            print("错误: 请设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY 环境变量")
            sys.exit(1)
    
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
