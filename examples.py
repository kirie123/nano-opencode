"""
使用示例：展示如何使用 Nano-OpenCode
"""

import asyncio
import os
from nano_opencode import (
    AgentRunner,
    AgentConfig,
    AgentMode,
    PermissionRule,
    LLMConfig,
    tool_registry,
    Tool,
    ToolResult,
    ToolContext
)


async def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===\n")
    
    # 创建运行器
    runner = AgentRunner(
        workspace=".",
        llm_config=LLMConfig(
            model="gpt-4",
            temperature=0.7
        )
    )
    
    # 运行 Agent
    result = await runner.run(
        prompt="列出当前目录下的 Python 文件",
        agent_name="build"
    )
    
    print(f"\n结果状态: {result.state}")
    print(f"工具调用次数: {result.tool_calls}")


async def example_with_callbacks():
    """带回调的使用示例"""
    print("\n=== 带回调的使用示例 ===\n")
    
    runner = AgentRunner(
        workspace=".",
        llm_config=LLMConfig(model="gpt-4")
    )
    
    # 定义回调
    async def on_text(text: str):
        print(text, end="", flush=True)
    
    async def on_tool_start(tool_name: str, arguments: dict):
        print(f"\n>>> 执行工具: {tool_name}")
        print(f">>> 参数: {arguments}")
    
    async def on_tool_end(tool_name: str, result: ToolResult):
        print(f">>> 结果: {result.output[:100]}..." if len(result.output) > 100 else f">>> 结果: {result.output}")
    
    result = await runner.run(
        prompt="搜索当前目录下包含 'import' 的 Python 文件",
        agent_name="explore",  # 使用只读的 explore agent
        callbacks={
            "on_text": on_text,
            "on_tool_start": on_tool_start,
            "on_tool_end": on_tool_end
        }
    )
    
    print(f"\n完成! 状态: {result.state}")


async def example_custom_agent():
    """自定义 Agent 示例"""
    print("\n=== 自定义 Agent 示例 ===\n")
    
    from nano_opencode import agent_registry, AgentConfig, AgentMode, PermissionRule
    
    # 创建自定义 Agent
    custom_agent = AgentConfig(
        name="reviewer",
        description="代码审查 Agent，只读权限",
        mode=AgentMode.PRIMARY,
        permission_rules=[
            PermissionRule("read", "*", "allow"),
            PermissionRule("grep", "*", "allow"),
            PermissionRule("glob", "*", "allow"),
            PermissionRule("*", "*", "deny"),  # 其他全部禁止
        ],
        temperature=0.3,
        prompt="你是一个代码审查专家。请仔细分析代码，找出潜在的问题和改进建议。"
    )
    
    # 注册 Agent
    agent_registry.register(custom_agent)
    
    print(f"已注册 Agent: {custom_agent.name}")
    print(f"权限规则: {len(custom_agent.permission_rules)} 条")


async def example_custom_tool():
    """自定义工具示例"""
    print("\n=== 自定义工具示例 ===\n")
    
    from nano_opencode import Tool, ToolResult, ToolContext, ToolParameter, tool_registry
    
    # 创建自定义工具
    class WeatherTool(Tool):
        """Get current weather for a location"""
        
        def _init_params(self):
            self._params = [
                ToolParameter("location", "string", "City name"),
                ToolParameter("unit", "string", "Temperature unit (celsius/fahrenheit)", False, default="celsius")
            ]
        
        async def execute(self, args: dict, context: ToolContext) -> ToolResult:
            location = args.get("location", "Unknown")
            unit = args.get("unit", "celsius")
            
            # 模拟天气数据
            weather_data = {
                "location": location,
                "temperature": 22 if unit == "celsius" else 72,
                "condition": "Sunny",
                "humidity": 45
            }
            
            return ToolResult(
                output=f"Weather in {location}: {weather_data['temperature']}°{unit[0].upper()}, {weather_data['condition']}",
                title=f"Weather: {location}",
                metadata=weather_data
            )
    
    # 注册工具
    weather_tool = WeatherTool()
    tool_registry.register(weather_tool)
    
    print(f"已注册工具: {weather_tool.id}")
    print(f"工具 Schema: {weather_tool.get_schema()}")


async def example_session_management():
    """会话管理示例"""
    print("\n=== 会话管理示例 ===\n")
    
    from nano_opencode import SessionManager, Session, Message, MessageRole, Part, PartType
    
    # 创建会话管理器
    manager = SessionManager(storage_dir="./sessions")
    
    # 创建会话
    session = manager.create_session(
        title="示例会话",
        workspace="."
    )
    
    print(f"创建会话: {session.id}")
    
    # 添加消息
    user_msg = session.create_user_message("你好，请帮我分析代码")
    print(f"添加用户消息: {user_msg.id}")
    
    assistant_msg = session.create_assistant_message()
    assistant_msg.add_text("好的，我来帮你分析代码。")
    print(f"添加助手消息: {assistant_msg.id}")
    
    # 列出会话
    sessions = manager.list_sessions()
    print(f"当前会话数: {len(sessions)}")


async def example_permission_system():
    """权限系统示例"""
    print("\n=== 权限系统示例 ===\n")
    
    from nano_opencode import PermissionEvaluator, PermissionRule, PermissionAction
    
    # 创建权限评估器
    evaluator = PermissionEvaluator([
        PermissionRule("*", "*", PermissionAction.ALLOW),
        PermissionRule("bash", "*", PermissionAction.ASK),
        PermissionRule("write", "*.env", PermissionAction.DENY),
    ])
    
    # 测试权限
    tests = [
        ("read", "test.py"),
        ("bash", "ls -la"),
        ("write", ".env"),
        ("write", "main.py"),
    ]
    
    for permission, pattern in tests:
        rule = evaluator.evaluate(permission, pattern)
        print(f"{permission}:{pattern} -> {rule.action.value}")


async def main():
    """运行所有示例"""
    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: OPENAI_API_KEY 未设置，跳过需要 LLM 的示例")
        print("请设置环境变量: export OPENAI_API_KEY=your-key\n")
        
        # 只运行不需要 LLM 的示例
        await example_custom_agent()
        await example_custom_tool()
        await example_session_management()
        await example_permission_system()
        return
    
    # 运行所有示例
    await example_basic_usage()
    await example_with_callbacks()
    await example_custom_agent()
    await example_custom_tool()
    await example_session_management()
    await example_permission_system()


if __name__ == "__main__":
    asyncio.run(main())
