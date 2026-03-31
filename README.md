# Nano-OpenCode

简化版 OpenCode 实现，展示了 OpenCode 的核心设计理念。

## 核心设计

### 1. Agent 系统

Agent 定义了 AI 的行为、权限和配置：

```python
from nano_opencode import AgentConfig, AgentMode, PermissionRule

agent = AgentConfig(
    name="build",
    description="默认 Agent，拥有所有工具权限",
    mode=AgentMode.PRIMARY,
    permission_rules=[
        PermissionRule("*", "*", "allow"),
        PermissionRule("bash", "*", "ask"),  # shell 需询问
    ]
)
```

### 2. 工具系统

每个工具遵循统一接口：

```python
from nano_opencode import Tool, ToolResult, ToolContext

class MyTool(Tool):
    """工具描述"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("input", "string", "输入参数"),
        ]
    
    async def execute(self, args: dict, context: ToolContext) -> ToolResult:
        return ToolResult(
            output="结果",
            title="执行标题",
            metadata={}
        )
```

### 3. 权限系统

三级权限控制：`allow`、`deny`、`ask`

```python
from nano_opencode import PermissionEvaluator, PermissionAction

evaluator = PermissionEvaluator([
    PermissionRule("read", "*", PermissionAction.ALLOW),
    PermissionRule("write", "*.env", PermissionAction.DENY),
    PermissionRule("bash", "*", PermissionAction.ASK),
])
```

### 4. Agent Loop

核心的"思考-行动-观察"循环：

```
while True:
    1. 获取历史消息
    2. 调用 LLM
    3. 处理响应：
       - 文本 → 输出
       - 工具调用 → 执行 → 添加结果 → 继续
       - 完成 → 退出
```

## 安装

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install aiohttp pypdf
```

## 使用方法

### CLI 方式

```bash
# 使用 uv run 启动
uv run main.py "帮我分析这个项目"
uv run main.py --agent plan "制定重构计划"
uv run main.py --model gpt-4 --verbose "搜索代码"

# 使用 Ollama
uv run main.py --provider ollama "帮我分析这个项目"
uv run main.py -p ollama -m qwen2.5:7b "搜索代码"
```

### 交互模式

```bash
# 启动交互模式
uv run main.py -i
uv run main.py -i --provider ollama

# 交互模式命令
>>> /help           # 显示帮助
>>> /clear          # 清除会话历史
>>> /agent plan     # 切换 Agent
>>> /model qwen2.5  # 切换模型
>>> exit            # 退出
```

### 编程方式

```python
import asyncio
from nano_opencode import AgentRunner, LLMConfig

async def main():
    runner = AgentRunner(
        workspace=".",
        llm_config=LLMConfig(model="gpt-4")
    )
    
    result = await runner.run(
        prompt="帮我分析这个项目",
        agent_name="build"
    )
    
    print(f"状态: {result.state}")
    print(f"工具调用: {result.tool_calls}")

asyncio.run(main())
```

### 带回调

```python
async def on_text(text: str):
    print(text, end="")

async def on_tool_start(tool_name: str, args: dict):
    print(f"执行: {tool_name}")

result = await runner.run(
    prompt="搜索代码",
    callbacks={
        "on_text": on_text,
        "on_tool_start": on_tool_start
    }
)
```

## 文件结构

```
nano-opencode/
├── __init__.py        # 包入口
├── __main__.py        # CLI 入口
├── main.py            # 主程序入口
├── agent.py           # Agent 定义
├── tools.py           # 工具系统
├── permission.py      # 权限系统
├── session.py         # 会话管理
├── llm.py             # LLM 客户端 (支持 OpenAI/Anthropic/Ollama)
├── loop.py            # Agent 循环
├── stream.py          # 流式处理
├── message.py         # 消息模型 (Part 系统)
├── error.py           # 错误处理
├── compaction.py      # 上下文压缩
├── task_tool.py       # 子 Agent 调用
├── prompt_manager.py  # 提示词管理
├── prompt/            # 提示词目录
├── examples.py        # 使用示例
├── pyproject.toml     # 项目配置
└── README.md          # 文档
```

## 内置 Agent

| Agent | 模式 | 说明 |
|-------|------|------|
| `build` | primary | 默认 Agent，拥有所有权限 |
| `plan` | primary | 计划模式，禁止编辑 |
| `general` | subagent | 通用子 Agent，研究复杂问题 |
| `explore` | subagent | 代码探索，只读 |
| `analyze` | subagent | 数据分析，处理文档和研报 |

## LLM 提供商

| 提供商 | 参数 | 说明 |
|--------|------|------|
| OpenAI | `--provider openai` | 默认，需要 OPENAI_API_KEY |
| Anthropic | `--provider anthropic` | 需要 ANTHROPIC_API_KEY |
| Ollama | `--provider ollama` | 本地运行，无需 API Key |

### Ollama 配置

```bash
# 启动 Ollama 服务
ollama serve

# 拉取模型
ollama pull glm-4.7-flash

# 使用
uv run main.py --provider ollama "你的问题"
uv run main.py -p ollama -m qwen2.5:7b "你的问题"
```

## 内置工具

| 工具 | 说明 |
|------|------|
| `read` | 读取文件 |
| `write` | 写入文件 |
| `edit` | 编辑文件 |
| `glob` | 文件搜索 |
| `grep` | 内容搜索 |
| `bash` | Shell 执行 |
| `pdf` | 读取 PDF 文件 |
| `task` | 子 Agent 调用 |

## 与 OpenCode 的对应关系

| OpenCode (TypeScript) | Nano-OpenCode (Python) |
|----------------------|------------------------|
| `src/agent/agent.ts` | `agent.py` |
| `src/tool/*.ts` | `tools.py` |
| `src/permission/index.ts` | `permission.py` |
| `src/session/*.ts` | `session.py` |
| `src/session/llm.ts` | `llm.py` |
| `src/session/prompt.ts` | `loop.py` |

## 运行示例

```bash
# 运行示例（不需要 API Key）
python -m nano_opencode.examples
```

## 许可证

MIT
