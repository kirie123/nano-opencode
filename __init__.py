"""
Nano-OpenCode: 简化版 OpenCode 实现

这是一个完整的 Python 实现，展示了 OpenCode 的核心设计：
1. Agent 系统 - 定义 AI Agent 的行为和权限
2. 工具系统 - 文件操作、代码搜索、Shell 执行
3. 会话管理 - 持久化对话历史
4. LLM 集成 - OpenAI API 流式调用
5. Agent Loop - "思考-行动-观察"迭代模式
6. 流式处理 - text/reasoning/tool 完整支持
7. 错误处理 - 重试、回退策略
8. 智能压缩 - 上下文窗口管理
9. 子 Agent - TaskTool 完整实现

使用方法:
    from nano_opencode import AgentRunner
    
    runner = AgentRunner()
    result = await runner.run("帮我分析这个项目")
"""

from .agent import (
    AgentConfig,
    AgentMode,
    AgentRegistry,
    agent_registry,
    PermissionRule
)

from .tools import (
    Tool,
    ToolResult,
    ToolContext,
    ToolRegistry,
    tool_registry
)

from .permission import (
    PermissionAction,
    PermissionEvaluator,
    PermissionManager,
    permission_manager
)

from .session import (
    Session,
    SessionManager,
)

from .message import (
    Message,
    Part,
    TextPart,
    ReasoningPart,
    ToolPart,
    FilePart,
    ImagePart,
    PartType,
    MessageRole,
    ToolStatus,
    generate_id,
)

from .llm import (
    LLMClient,
    LLMConfig,
    StreamEvent
)

from .loop import (
    AgentLoop,
    AgentRunner,
    AgentLoopConfig,
    LoopState,
    LoopResult
)

from .prompt_manager import SystemPrompt

from .stream import (
    StreamProcessor,
    StreamEventType,
    StreamResult,
    MockStreamGenerator,
)

from .error import (
    ErrorHandler,
    RetryConfig,
    ErrorInfo,
    ErrorType,
    ErrorSeverity,
    AgentError,
    NetworkError,
    RateLimitError,
    TimeoutError,
    AuthError,
    ValidationError,
    ToolError,
    PermissionError,
    ContextOverflowError,
    RecoveryStrategy,
    ErrorAggregator,
)

from .compaction import (
    CompactionManager,
    CompactionConfig,
    CompactionStrategy,
    CompactionResult,
    TokenCounter,
    Compactor,
    TruncateCompactor,
    SummarizeCompactor,
    ImportanceCompactor,
)

from .task_tool import (
    TaskTool,
    SubtaskSession,
    SubtaskManager,
    subtask_manager,
    register_subagents,
    create_explore_agent,
    create_general_agent,
)

__all__ = [
    # Agent
    "AgentConfig",
    "AgentMode",
    "AgentRegistry",
    "agent_registry",
    "PermissionRule",
    
    # Tools
    "Tool",
    "ToolResult",
    "ToolContext",
    "ToolRegistry",
    "tool_registry",
    
    # Permission
    "PermissionAction",
    "PermissionEvaluator",
    "PermissionManager",
    "permission_manager",
    
    # Session
    "Session",
    "SessionManager",
    
    # Message
    "Message",
    "Part",
    "TextPart",
    "ReasoningPart",
    "ToolPart",
    "FilePart",
    "ImagePart",
    "PartType",
    "MessageRole",
    "ToolStatus",
    "generate_id",
    
    # LLM
    "LLMClient",
    "LLMConfig",
    "StreamEvent",
    
    # Loop
    "AgentLoop",
    "AgentRunner",
    "AgentLoopConfig",
    "LoopState",
    "LoopResult",
    
    # Prompt
    "SystemPrompt",
    
    # Stream
    "StreamProcessor",
    "StreamEventType",
    "StreamResult",
    "MockStreamGenerator",
    
    # Error
    "ErrorHandler",
    "RetryConfig",
    "ErrorInfo",
    "ErrorType",
    "ErrorSeverity",
    "AgentError",
    "NetworkError",
    "RateLimitError",
    "TimeoutError",
    "AuthError",
    "ValidationError",
    "ToolError",
    "PermissionError",
    "ContextOverflowError",
    "RecoveryStrategy",
    "ErrorAggregator",
    
    # Compaction
    "CompactionManager",
    "CompactionConfig",
    "CompactionStrategy",
    "CompactionResult",
    "TokenCounter",
    "Compactor",
    "TruncateCompactor",
    "SummarizeCompactor",
    "ImportanceCompactor",
    
    # TaskTool
    "TaskTool",
    "SubtaskSession",
    "SubtaskManager",
    "subtask_manager",
    "register_subagents",
    "create_explore_agent",
    "create_general_agent",
]

__version__ = "0.2.0"
