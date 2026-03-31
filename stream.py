"""
流式处理器

支持完整的流式处理：
- text: 文本内容流式输出
- reasoning: 推理过程流式输出
- tool: 工具调用流式处理
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union, Awaitable
from enum import Enum
from abc import ABC, abstractmethod

from message import (
    Message, Part, TextPart, ReasoningPart, ToolPart, 
    MessageRole, ToolStatus, generate_id
)


class StreamEventType(Enum):
    START = "start"
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_END = "reasoning_end"
    TOOL_INPUT_START = "tool_input_start"
    TOOL_INPUT_DELTA = "tool_input_delta"
    TOOL_INPUT_END = "tool_input_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    ERROR = "error"
    START_STEP = "start_step"
    FINISH_STEP = "finish_step"
    FINISH = "finish"


@dataclass
class StreamEvent:
    """流式事件"""
    type: StreamEventType
    id: Optional[str] = None
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[str] = None
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type.value}
        if self.id:
            result["id"] = self.id
        if self.text:
            result["text"] = self.text
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.input:
            result["input"] = self.input
        if self.output:
            result["output"] = self.output
        if self.error:
            result["error"] = self.error
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        if self.usage:
            result["usage"] = self.usage
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class StreamResult:
    """流式处理结果"""
    message: Message
    blocked: bool = False
    needs_compaction: bool = False
    error: Optional[Exception] = None


class StreamProcessor:
    """流式处理器"""
    
    def __init__(
        self,
        session_id: str,
        message: Message,
        on_event: Optional[Callable[[StreamEvent], Awaitable[None]]] = None,
        on_part_update: Optional[Callable[[Part], Awaitable[None]]] = None,
    ):
        self.session_id = session_id
        self.message = message
        self.on_event = on_event
        self.on_part_update = on_part_update
        
        self._current_text: Optional[TextPart] = None
        self._current_reasoning: Optional[ReasoningPart] = None
        self._pending_tools: Dict[str, ToolPart] = {}
        self._snapshot: Optional[str] = None
        self._step_count = 0
        self._blocked = False
        self._needs_compaction = False
    
    async def process_stream(
        self, 
        stream: AsyncGenerator[StreamEvent, None]
    ) -> StreamResult:
        """处理流式事件"""
        try:
            async for event in stream:
                await self._handle_event(event)
                
                if self._blocked:
                    break
            
            return StreamResult(
                message=self.message,
                blocked=self._blocked,
                needs_compaction=self._needs_compaction,
            )
        except Exception as e:
            return StreamResult(
                message=self.message,
                error=e,
            )
    
    async def _handle_event(self, event: StreamEvent):
        """处理单个事件"""
        if self.on_event:
            await self.on_event(event)
        
        handler = {
            StreamEventType.START: self._handle_start,
            StreamEventType.TEXT_START: self._handle_text_start,
            StreamEventType.TEXT_DELTA: self._handle_text_delta,
            StreamEventType.TEXT_END: self._handle_text_end,
            StreamEventType.REASONING_START: self._handle_reasoning_start,
            StreamEventType.REASONING_DELTA: self._handle_reasoning_delta,
            StreamEventType.REASONING_END: self._handle_reasoning_end,
            StreamEventType.TOOL_INPUT_START: self._handle_tool_input_start,
            StreamEventType.TOOL_INPUT_DELTA: self._handle_tool_input_delta,
            StreamEventType.TOOL_INPUT_END: self._handle_tool_input_end,
            StreamEventType.TOOL_CALL: self._handle_tool_call,
            StreamEventType.TOOL_RESULT: self._handle_tool_result,
            StreamEventType.TOOL_ERROR: self._handle_tool_error,
            StreamEventType.ERROR: self._handle_error,
            StreamEventType.START_STEP: self._handle_start_step,
            StreamEventType.FINISH_STEP: self._handle_finish_step,
            StreamEventType.FINISH: self._handle_finish,
        }.get(event.type)
        
        if handler:
            await handler(event)
    
    async def _handle_start(self, event: StreamEvent):
        """处理开始事件"""
        pass
    
    async def _handle_text_start(self, event: StreamEvent):
        """处理文本开始"""
        self._current_text = TextPart(
            id=event.id or generate_id("text_"),
            message_id=self.message.id,
            session_id=self.session_id,
            text="",
        )
        self.message.parts.append(self._current_text)
        
        if self.on_part_update:
            await self.on_part_update(self._current_text)
    
    async def _handle_text_delta(self, event: StreamEvent):
        """处理文本增量"""
        if not self._current_text:
            self._current_text = TextPart(
                id=generate_id("text_"),
                message_id=self.message.id,
                session_id=self.session_id,
                text="",
            )
            self.message.parts.append(self._current_text)
        
        if event.text:
            self._current_text.text += event.text
            
            if self.on_part_update:
                await self.on_part_update(self._current_text)
    
    async def _handle_text_end(self, event: StreamEvent):
        """处理文本结束"""
        if self._current_text:
            self._current_text.text = self._current_text.text.rstrip()
            self._current_text.time["end"] = int(time.time() * 1000)
            
            if self.on_part_update:
                await self.on_part_update(self._current_text)
            
            self._current_text = None
    
    async def _handle_reasoning_start(self, event: StreamEvent):
        """处理推理开始"""
        self._current_reasoning = ReasoningPart(
            id=event.id or generate_id("reasoning_"),
            message_id=self.message.id,
            session_id=self.session_id,
            text="",
        )
        self.message.parts.append(self._current_reasoning)
        
        if self.on_part_update:
            await self.on_part_update(self._current_reasoning)
    
    async def _handle_reasoning_delta(self, event: StreamEvent):
        """处理推理增量"""
        if not self._current_reasoning:
            self._current_reasoning = ReasoningPart(
                id=generate_id("reasoning_"),
                message_id=self.message.id,
                session_id=self.session_id,
                text="",
            )
            self.message.parts.append(self._current_reasoning)
        
        if event.text:
            self._current_reasoning.text += event.text
            
            if self.on_part_update:
                await self.on_part_update(self._current_reasoning)
    
    async def _handle_reasoning_end(self, event: StreamEvent):
        """处理推理结束"""
        if self._current_reasoning:
            self._current_reasoning.text = self._current_reasoning.text.rstrip()
            self._current_reasoning.time["end"] = int(time.time() * 1000)
            
            if self.on_part_update:
                await self.on_part_update(self._current_reasoning)
            
            self._current_reasoning = None
    
    async def _handle_tool_input_start(self, event: StreamEvent):
        """处理工具输入开始"""
        part = ToolPart(
            id=event.id or generate_id("tool_"),
            message_id=self.message.id,
            session_id=self.session_id,
            tool=event.tool_name or "",
            call_id=event.tool_call_id or generate_id("call_"),
            state={
                "status": ToolStatus.PENDING.value,
                "input": {},
                "raw": "",
            },
        )
        self._pending_tools[part.call_id] = part
        self.message.parts.append(part)
        
        if self.on_part_update:
            await self.on_part_update(part)
    
    async def _handle_tool_input_delta(self, event: StreamEvent):
        """处理工具输入增量"""
        if event.tool_call_id and event.tool_call_id in self._pending_tools:
            part = self._pending_tools[event.tool_call_id]
            part.state["raw"] += event.text or ""
    
    async def _handle_tool_input_end(self, event: StreamEvent):
        """处理工具输入结束"""
        pass
    
    async def _handle_tool_call(self, event: StreamEvent):
        """处理工具调用"""
        call_id = event.tool_call_id or generate_id("call_")
        
        if call_id not in self._pending_tools:
            part = ToolPart(
                id=generate_id("tool_"),
                message_id=self.message.id,
                session_id=self.session_id,
                tool=event.tool_name or "",
                call_id=call_id,
                state={
                    "status": ToolStatus.RUNNING.value,
                    "input": event.input or {},
                    "time": {"start": int(time.time() * 1000)},
                },
            )
            self._pending_tools[call_id] = part
            self.message.parts.append(part)
        else:
            part = self._pending_tools[call_id]
            part.tool = event.tool_name or part.tool
            part.state["status"] = ToolStatus.RUNNING.value
            part.state["input"] = event.input or {}
            part.state["time"] = {"start": int(time.time() * 1000)}
        
        if self.on_part_update:
            await self.on_part_update(part)
    
    async def _handle_tool_result(self, event: StreamEvent):
        """处理工具结果"""
        if event.tool_call_id and event.tool_call_id in self._pending_tools:
            part = self._pending_tools[event.tool_call_id]
            part.state["status"] = ToolStatus.COMPLETED.value
            part.state["output"] = event.output or ""
            part.state["time"]["end"] = int(time.time() * 1000)
            
            if event.metadata:
                part.metadata.update(event.metadata)
            
            if self.on_part_update:
                await self.on_part_update(part)
            
            del self._pending_tools[event.tool_call_id]
    
    async def _handle_tool_error(self, event: StreamEvent):
        """处理工具错误"""
        if event.tool_call_id and event.tool_call_id in self._pending_tools:
            part = self._pending_tools[event.tool_call_id]
            part.state["status"] = ToolStatus.ERROR.value
            part.state["error"] = event.error or "Unknown error"
            part.state["time"]["end"] = int(time.time() * 1000)
            
            if self.on_part_update:
                await self.on_part_update(part)
            
            del self._pending_tools[event.tool_call_id]
    
    async def _handle_error(self, event: StreamEvent):
        """处理错误"""
        raise Exception(event.error or "Stream error")
    
    async def _handle_start_step(self, event: StreamEvent):
        """处理步骤开始"""
        self._step_count += 1
        self._snapshot = event.metadata.get("snapshot")
    
    async def _handle_finish_step(self, event: StreamEvent):
        """处理步骤结束"""
        self.message.finish_reason = event.finish_reason
        
        if event.usage:
            self.message.tokens = event.usage
        
        if event.metadata:
            self.message.cost = event.metadata.get("cost", 0.0)
    
    async def _handle_finish(self, event: StreamEvent):
        """处理完成"""
        pass
    
    def set_blocked(self, blocked: bool = True):
        """设置阻塞状态"""
        self._blocked = blocked
    
    def set_needs_compaction(self, needs: bool = True):
        """设置需要压缩"""
        self._needs_compaction = needs


class MockStreamGenerator:
    """模拟流式生成器（用于测试）"""
    
    @staticmethod
    async def generate_text_stream(text: str) -> AsyncGenerator[StreamEvent, None]:
        """生成文本流"""
        yield StreamEvent(type=StreamEventType.START)
        yield StreamEvent(type=StreamEventType.TEXT_START, id="text_1")
        
        words = text.split()
        for i, word in enumerate(words):
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text=word + (" " if i < len(words) - 1 else ""),
            )
            await asyncio.sleep(0.05)
        
        yield StreamEvent(type=StreamEventType.TEXT_END)
        yield StreamEvent(
            type=StreamEventType.FINISH_STEP,
            finish_reason="stop",
            usage={"input": 100, "output": len(text.split())},
        )
        yield StreamEvent(type=StreamEventType.FINISH)
    
    @staticmethod
    async def generate_tool_stream(
        tool_name: str, 
        tool_input: Dict, 
        tool_output: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """生成工具调用流"""
        yield StreamEvent(type=StreamEventType.START)
        yield StreamEvent(
            type=StreamEventType.TOOL_INPUT_START,
            id="tool_1",
            tool_name=tool_name,
            tool_call_id="call_1",
        )
        
        input_str = json.dumps(tool_input)
        for i in range(0, len(input_str), 10):
            yield StreamEvent(
                type=StreamEventType.TOOL_INPUT_DELTA,
                tool_call_id="call_1",
                text=input_str[i:i+10],
            )
            await asyncio.sleep(0.02)
        
        yield StreamEvent(
            type=StreamEventType.TOOL_CALL,
            tool_name=tool_name,
            tool_call_id="call_1",
            input=tool_input,
        )
        
        yield StreamEvent(
            type=StreamEventType.TOOL_RESULT,
            tool_call_id="call_1",
            output=tool_output,
        )
        
        yield StreamEvent(
            type=StreamEventType.FINISH_STEP,
            finish_reason="tool-calls",
            usage={"input": 100, "output": 50},
        )
        yield StreamEvent(type=StreamEventType.FINISH)
    
    @staticmethod
    async def generate_reasoning_stream(
        reasoning: str, 
        text: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """生成推理+文本流"""
        yield StreamEvent(type=StreamEventType.START)
        
        yield StreamEvent(type=StreamEventType.REASONING_START, id="reasoning_1")
        for char in reasoning:
            yield StreamEvent(type=StreamEventType.REASONING_DELTA, text=char)
            await asyncio.sleep(0.01)
        yield StreamEvent(type=StreamEventType.REASONING_END)
        
        yield StreamEvent(type=StreamEventType.TEXT_START, id="text_1")
        for char in text:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=char)
            await asyncio.sleep(0.01)
        yield StreamEvent(type=StreamEventType.TEXT_END)
        
        yield StreamEvent(
            type=StreamEventType.FINISH_STEP,
            finish_reason="stop",
            usage={"input": 100, "output": 100, "reasoning": len(reasoning)},
        )
        yield StreamEvent(type=StreamEventType.FINISH)
