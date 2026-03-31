"""
Agent Loop: 核心 Agent 运行循环

这是 OpenCode 的核心，实现了"思考-行动-观察"的迭代模式：
1. 获取历史消息
2. 调用 LLM 生成响应
3. 如果有工具调用，执行工具
4. 将工具结果添加到历史，继续循环
5. 直到 LLM 返回最终答案

支持：
- 完整的流式处理（text/reasoning/tool）
- 错误处理（重试、回退）
- 智能摘要压缩
- 子 Agent 调用
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from agent import AgentConfig, AgentMode, agent_registry
from message import (
    Message, Part, TextPart, ReasoningPart, ToolPart, 
    MessageRole, ToolStatus, generate_id
)
from session import Session, SessionManager
from tools import Tool, ToolRegistry, ToolResult, ToolContext, tool_registry
from permission import PermissionEvaluator, PermissionAction, PermissionRule
from llm import LLMClient, LLMConfig, OllamaConfig
from prompt_manager import SystemPrompt
from stream import (
    StreamProcessor, StreamEvent, StreamEventType, 
    StreamResult, MockStreamGenerator
)
from error import (
    ErrorHandler, RetryConfig, ErrorInfo, ErrorType,
    AgentError, ToolError, ContextOverflowError
)
from compaction import (
    CompactionManager, CompactionConfig, CompactionStrategy,
    TokenCounter, CompactionResult
)
from task_tool import TaskTool, SubtaskManager, subtask_manager, register_subagents


class LoopState(Enum):
    CONTINUE = "continue"
    STOP = "stop"
    COMPACT = "compact"
    ERROR = "error"


@dataclass
class LoopResult:
    state: LoopState
    message: Optional[Message] = None
    error: Optional[str] = None
    tokens_used: int = 0
    tool_calls: int = 0
    cost: float = 0.0


@dataclass
class AgentLoopConfig:
    max_steps: int = 50
    max_tokens: int = 128000
    target_tokens: int = 100000
    doom_loop_threshold: int = 3
    enable_compaction: bool = True
    enable_retry: bool = True
    max_retries: int = 3


class AgentLoop:
    """
    Agent 主循环
    
    核心逻辑：
    while True:
        1. 获取消息历史
        2. 检查退出条件
        3. 调用 LLM（流式）
        4. 处理响应：
           - text → 输出给用户
           - reasoning → 思考过程
           - tool-call → 执行工具
           - finish → 检查是否退出
        5. 死循环检测
        6. 压缩检查
    """
    
    def __init__(
        self,
        agent: AgentConfig,
        llm_client: LLMClient = None,
        workspace: str = ".",
        session_id: str = None,
        config: AgentLoopConfig = None,
        session: Session = None,
    ):
        self.agent = agent
        self.llm = llm_client
        self.workspace = workspace
        self.config = config or AgentLoopConfig()
        
        if session:
            self.session = session
        else:
            self.session = Session(
                session_id=session_id or generate_id("session_"),
                workspace=workspace,
            )
        
        self.step = 0
        self.abort = False
        self.tool_call_history: List[Dict[str, Any]] = []
        self.consecutive_errors = 0
        
        self.error_handler = ErrorHandler(
            retry_config=RetryConfig(
                max_retries=self.config.max_retries,
            ),
            on_error=self._on_error,
            on_retry=self._on_retry,
        )
        
        self.compaction_manager = CompactionManager(
            config=CompactionConfig(
                max_tokens=self.config.max_tokens,
                target_tokens=self.config.target_tokens,
                strategy=CompactionStrategy.SUMMARIZE,
            ),
            llm_client=llm_client,
        )
        
        self.subtask_manager = subtask_manager
        
        self.on_text: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_reasoning: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_tool_start: Optional[Callable[[str, Dict], Awaitable[None]]] = None
        self.on_tool_end: Optional[Callable[[str, ToolResult], Awaitable[None]]] = None
        self.on_step: Optional[Callable[[int], Awaitable[None]]] = None
        self.on_error_event: Optional[Callable[[ErrorInfo], Awaitable[None]]] = None
        self.on_compaction: Optional[Callable[[CompactionResult], Awaitable[None]]] = None
    
    async def run(self, user_message: str = None) -> LoopResult:
        """运行 Agent 循环"""
        if user_message:
            self._create_user_message(user_message)
        
        while not self.abort:
            self.step += 1
            
            if self.on_step:
                await self.on_step(self.step)
            
            if self.step > self.config.max_steps:
                return LoopResult(
                    state=LoopState.STOP,
                    error=f"Max steps ({self.config.max_steps}) reached"
                )
            
            history = self.session.messages
            tools = self._get_available_tools()
            
            assistant_message = self._create_assistant_message()
            self.session.messages.append(assistant_message)
            
            try:
                result = await self._process_step(
                    history=history,
                    tools=tools,
                    assistant_message=assistant_message
                )
                
                if result.state == LoopState.STOP:
                    return result
                elif result.state == LoopState.COMPACT:
                    if self.config.enable_compaction:
                        await self._compact_history()
                    self.consecutive_errors = 0
                    continue
                elif result.state == LoopState.ERROR:
                    self.consecutive_errors += 1
                    if self.session.messages and self.session.messages[-1] == assistant_message:
                        self.session.messages.pop()
                    if self.consecutive_errors >= 3:
                        print(f"[ERROR] 连续错误 {self.consecutive_errors} 次，停止重试", flush=True)
                        return result
                    if self.config.enable_retry:
                        continue
                    return result
                
                self.consecutive_errors = 0
                    
            except Exception as e:
                self.consecutive_errors += 1
                if self.session.messages and self.session.messages[-1] == assistant_message:
                    self.session.messages.pop()
                if self.consecutive_errors >= 3:
                    print(f"[ERROR] 连续异常 {self.consecutive_errors} 次，停止重试: {e}", flush=True)
                    return LoopResult(
                        state=LoopState.ERROR,
                        error=str(e),
                    )
                error_info = self.error_handler._classify_error(e)
                if self.on_error_event:
                    await self.on_error_event(error_info)
                
                if error_info.recoverable and self.config.enable_retry:
                    continue
                
                return LoopResult(
                    state=LoopState.ERROR,
                    error=str(e),
                )
        
        return LoopResult(state=LoopState.STOP, error="Aborted")
    
    def _create_user_message(self, text: str) -> Message:
        """创建用户消息"""
        message = Message(
            id=generate_id("msg_"),
            session_id=self.session.id,
            role=MessageRole.USER,
            agent=self.agent.name,
        )
        message.add_text(text)
        self.session.messages.append(message)
        return message
    
    def _create_assistant_message(self) -> Message:
        """创建助手消息"""
        message = Message(
            id=generate_id("msg_"),
            session_id=self.session.id,
            role=MessageRole.ASSISTANT,
            agent=self.agent.name,
            model_id=self.llm.config.model if self.llm else None,
            provider_id=self.llm.config.provider if self.llm else None,
        )
        return message
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具"""
        all_tools = tool_registry.get_schemas()
        available = []
        
        evaluator = PermissionEvaluator(self.agent.permission_rules)
        
        for tool_schema in all_tools:
            tool_name = tool_schema["function"]["name"]
            result = evaluator.evaluate(
                permission=tool_name,
                pattern="*",
            )
            
            if result.action != PermissionAction.DENY:
                available.append(tool_schema)
        
        return available
    
    async def _process_step(
        self,
        history: List[Message],
        tools: List[Dict[str, Any]],
        assistant_message: Message
    ) -> LoopResult:
        """处理单步"""
        stream = self._create_llm_stream(history, tools, assistant_message)
        
        processor = StreamProcessor(
            session_id=self.session.id,
            message=assistant_message,
            on_event=self._on_stream_event,
            on_part_update=self._on_part_update,
        )
        
        result = await processor.process_stream(stream)
        
        if result.error:
            print(f"[ERROR] 流处理错误: {result.error}", flush=True)
            return LoopResult(
                state=LoopState.ERROR,
                error=str(result.error),
                message=assistant_message,
            )
        
        if result.blocked:
            return LoopResult(
                state=LoopState.STOP,
                message=assistant_message,
            )
        
        tool_parts = assistant_message.get_tool_parts()
        
        if tool_parts:
            tool_results = await self._execute_tool_calls(tool_parts, assistant_message)
            
            self._check_doom_loop(tool_parts)
            
            if result.needs_compaction or self._should_compact():
                return LoopResult(
                    state=LoopState.COMPACT,
                    message=assistant_message,
                    tool_calls=len(tool_results),
                )
            
            return LoopResult(
                state=LoopState.CONTINUE,
                message=assistant_message,
                tool_calls=len(tool_results),
            )
        
        finish_reason = assistant_message.finish_reason
        text_content = assistant_message.get_text_content()
        
        if finish_reason in ["stop", "end_turn", "complete"]:
            return LoopResult(
                state=LoopState.STOP,
                message=assistant_message,
            )
        
        if text_content:
            return LoopResult(
                state=LoopState.STOP,
                message=assistant_message,
            )
        
        if not tool_parts and not text_content:
            print(f"[警告] LLM 返回空响应，停止循环", flush=True)
            return LoopResult(
                state=LoopState.STOP,
                message=assistant_message,
                error="Empty response from LLM",
            )
        
        return LoopResult(
            state=LoopState.CONTINUE,
            message=assistant_message,
        )
    
    async def _create_llm_stream(
        self,
        history: List[Message],
        tools: List[Dict[str, Any]],
        assistant_message: Message
    ) -> AsyncGenerator[StreamEvent, None]:
        """创建 LLM 流"""
        if not self.llm:
            async for event in MockStreamGenerator.generate_text_stream(
                "LLM 客户端未配置，请先配置 LLM 客户端。"
            ):
                yield event
            return
        
        messages = self._build_llm_messages(history)
        system_prompt = self._build_system_prompt(tools)
        
        try:
            async for event in self.llm.chat_completion_stream(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            ):
                yield self._convert_llm_event(event)
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error=str(e),
            )
    
    def _build_llm_messages(self, history: List[Message]) -> List[Dict[str, Any]]:
        """构建 LLM 消息格式"""
        messages = []
        
        for msg in history:
            if msg.role == MessageRole.USER:
                content = msg.get_text_content()
                if content:
                    messages.append({"role": "user", "content": content})
            
            elif msg.role == MessageRole.ASSISTANT:
                content = msg.get_text_content()
                tool_calls = []
                
                for part in msg.parts:
                    if isinstance(part, ToolPart):
                        if part.status == ToolStatus.COMPLETED or part.status == ToolStatus.ERROR:
                            tool_calls.append({
                                "id": part.call_id,
                                "type": "function",
                                "function": {
                                    "name": part.tool,
                                    "arguments": part.input,
                                }
                            })
                
                msg_dict = {"role": "assistant"}
                if content:
                    msg_dict["content"] = content
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                
                if content or tool_calls:
                    messages.append(msg_dict)
                
                for part in msg.parts:
                    if isinstance(part, ToolPart):
                        if part.status == ToolStatus.COMPLETED:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": part.call_id,
                                "content": part.output or "",
                            })
                        elif part.status == ToolStatus.ERROR:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": part.call_id,
                                "content": f"Error: {part.error}",
                            })
        
        return messages
    
    def _build_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """构建系统提示"""
        return SystemPrompt.build(
            model=self.llm.config.model if self.llm else "unknown",
            workspace=self.workspace,
            agent=self.agent,
            tools=tools,
            custom_prompt=self.agent.get_prompt(),
        )
    
    def _convert_llm_event(self, event: Any) -> StreamEvent:
        """转换 LLM 事件为流事件"""
        if hasattr(event, 'type'):
            if event.type == "content":
                return StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text=event.content,
                )
            elif event.type == "reasoning":
                return StreamEvent(
                    type=StreamEventType.REASONING_DELTA,
                    text=event.content,
                )
            elif event.type == "tool_calls":
                if event.tool_calls:
                    tc = event.tool_calls[0]
                    return StreamEvent(
                        type=StreamEventType.TOOL_CALL,
                        tool_name=tc.name,
                        tool_call_id=tc.id or generate_id("call_"),
                        input=tc.arguments,
                    )
            elif event.type == "tool_call":
                return StreamEvent(
                    type=StreamEventType.TOOL_CALL,
                    tool_name=event.tool_name,
                    tool_call_id=event.tool_call_id,
                    input=event.arguments,
                )
            elif event.type == "done":
                return StreamEvent(
                    type=StreamEventType.FINISH_STEP,
                    finish_reason=event.finish_reason,
                    usage=event.usage,
                )
            elif event.type == "error":
                return StreamEvent(
                    type=StreamEventType.ERROR,
                    error=event.error,
                )
        
        return StreamEvent(type=StreamEventType.FINISH)
    
    async def _execute_tool_calls(
        self,
        tool_parts: List[ToolPart],
        assistant_message: Message
    ) -> List[ToolResult]:
        """执行工具调用"""
        results = []
        
        for part in tool_parts:
            if part.status != ToolStatus.RUNNING:
                continue
            
            tool_name = part.tool
            arguments = part.input
            
            if self.on_tool_start:
                await self.on_tool_start(tool_name, arguments)
            
            result = await self._execute_single_tool(tool_name, arguments, part.call_id)
            
            if result.success:
                assistant_message.update_tool_result(
                    part.call_id,
                    output=result.output,
                    title=result.metadata.get("title") if result.metadata else None,
                )
            else:
                assistant_message.update_tool_result(
                    part.call_id,
                    error=result.error,
                )
            
            if self.on_tool_end:
                await self.on_tool_end(tool_name, result)
            
            results.append(result)
        
        return results
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: str
    ) -> ToolResult:
        """执行单个工具"""
        evaluator = PermissionEvaluator(self.agent.permission_rules)
        permission_result = evaluator.evaluate(
            permission=tool_name,
            pattern="*",
        )
        
        if permission_result.action == PermissionAction.DENY:
            return ToolResult(
                output="",
                error=f"Permission denied for tool: {tool_name}",
                success=False,
            )
        
        tool = tool_registry.get(tool_name)
        
        if not tool:
            return ToolResult(
                output="",
                error=f"Unknown tool: {tool_name}",
                success=False,
            )
        
        context = ToolContext(
            session_id=self.session.id,
            message_id=call_id,
            agent=self.agent.name,
            cwd=self.workspace,
            workspace_root=self.workspace,
            messages=[],
            extra={"parent_session_id": self.session.id},
        )
        
        if tool_name == "task":
            tool = TaskTool(llm_client=self.llm)
        
        try:
            if self.config.enable_retry:
                result = await self.error_handler.execute_with_retry(
                    tool.execute,
                    arguments,
                    context,
                )
            else:
                result = await tool.execute(arguments, context)
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ToolResult(
                output="",
                error=str(e),
                success=False,
            )
    
    def _check_doom_loop(self, tool_parts: List[ToolPart]):
        """检测死循环"""
        for part in tool_parts:
            self.tool_call_history.append({
                "tool": part.tool,
                "arguments": part.input,
                "step": self.step,
            })
        
        recent = self.tool_call_history[-self.config.doom_loop_threshold:]
        
        if len(recent) == self.config.doom_loop_threshold:
            first = recent[0]
            if all(
                tc["tool"] == first["tool"] and
                tc["arguments"] == first["arguments"]
                for tc in recent
            ):
                print(f"[DEBUG] 检测到死循环，停止执行", flush=True)
                self.abort = True
    
    def _should_compact(self) -> bool:
        """检查是否需要压缩"""
        tokens = TokenCounter.estimate_messages(self.session.messages)
        return tokens > self.config.max_tokens * 0.9
    
    async def _compact_history(self):
        """压缩历史"""
        compacted, result = await self.compaction_manager.auto_compact_if_needed(
            self.session.messages
        )
        
        if result and self.on_compaction:
            await self.on_compaction(result)
        
        if compacted:
            self.session.messages = compacted
    
    async def _on_stream_event(self, event: StreamEvent):
        """流事件回调"""
        if event.type == StreamEventType.TEXT_DELTA and self.on_text:
            await self.on_text(event.text or "")
        elif event.type == StreamEventType.REASONING_DELTA and self.on_reasoning:
            await self.on_reasoning(event.text or "")
    
    async def _on_part_update(self, part: Part):
        """Part 更新回调"""
        pass
    
    def _on_error(self, error_info: ErrorInfo):
        """错误回调"""
        pass
    
    def _on_retry(self, error_info: ErrorInfo, attempt: int):
        """重试回调"""
        pass
    
    def cancel(self):
        """取消循环"""
        self.abort = True


class AgentRunner:
    """Agent 运行器"""
    
    def __init__(
        self,
        workspace: str = None,
        llm_config: LLMConfig = None,
        ollama_config: "OllamaConfig" = None,
        storage_dir: str = None
    ):
        self.workspace = workspace or "."
        self.llm_config = llm_config or LLMConfig()
        self.ollama_config = ollama_config
        self.storage_dir = storage_dir
        self.session_manager = SessionManager(storage_dir)
        
        register_subagents()
    
    async def run(
        self,
        prompt: str,
        agent_name: str = "build",
        session_id: str = None,
        callbacks: Dict[str, Callable] = None
    ) -> LoopResult:
        """运行 Agent"""
        session = None
        if session_id:
            session = self.session_manager.get_session(session_id)
        
        if not session:
            session = Session(
                session_id=session_id or generate_id("session_"),
                workspace=self.workspace,
            )
        
        agent = agent_registry.get(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")
        
        async with LLMClient(self.llm_config, self.ollama_config) as llm_client:
            loop = AgentLoop(
                agent=agent,
                llm_client=llm_client,
                workspace=self.workspace,
                session=session,
            )
            
            if callbacks:
                loop.on_text = callbacks.get("on_text")
                loop.on_reasoning = callbacks.get("on_reasoning")
                loop.on_tool_start = callbacks.get("on_tool_start")
                loop.on_tool_end = callbacks.get("on_tool_end")
                loop.on_step = callbacks.get("on_step")
                loop.on_error_event = callbacks.get("on_error")
                loop.on_compaction = callbacks.get("on_compaction")
            
            result = await loop.run(prompt)
        
        return result
