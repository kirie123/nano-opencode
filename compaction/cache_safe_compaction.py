"""
缓存安全的上下文压缩机制

核心原则（参考 Claude Code）：
1. 使用完全相同的 System Prompt
2. 使用完全相同的工具定义
3. 使用完全相同的上下文结构
4. 只追加一个压缩请求消息

这样可以让压缩请求复用主会话的缓存，大幅降低成本。

传统压缩方式的问题：
- 使用独立的 API 调用
- 使用不同的 System Prompt
- 不带工具定义
- 结果：缓存完全不匹配，所有 token 全额计费

缓存安全压缩的优势：
- 复用相同的 System Prompt 和 Tools
- 缓存完整命中
- 只有压缩指令部分作为新 token 计费
- 成本可能降低 90%+
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from core.message import Message, Part, TextPart, ReasoningPart, ToolPart, MessageRole, ToolStatus, generate_id
from compaction.compaction import CompactionConfig, CompactionResult, TokenCounter


@dataclass
class CacheSafeCompactionResult(CompactionResult):
    """缓存安全的压缩结果"""
    cache_safe: bool = False
    cache_hit: bool = False
    original_tokens: int = 0
    new_tokens: int = 0
    cost_saved_ratio: float = 0.0


class CacheSafeCompactionManager:
    """
    缓存安全的上下文压缩管理器
    
    核心设计：
    1. 保存最后一次请求的 System Prompt 和 Tools
    2. 压缩时复用这些参数
    3. 只追加压缩请求消息
    4. 确保缓存可以命中
    """
    
    COMPACTION_REQUEST = """<compaction_request>
Please generate a concise summary of the above conversation, preserving:

1. **Core Tasks and Goals**
   - User's main requirements
   - Completed work

2. **Key Decisions and Findings**
   - Important technical decisions
   - Problems discovered and solutions

3. **File Operations Record**
   - Files created/modified
   - Important code changes

4. **Pending Items**
   - Unfinished tasks
   - Issues needing follow-up

Omit:
- Redundant reasoning processes
- Details of failed attempts
- Irrelevant conversations

Provide the summary in a structured format.
</compaction_request>"""
    
    def __init__(
        self,
        llm_client=None,
        max_tokens: int = 128000,
        target_tokens: int = 100000,
        preserve_recent: int = 2,
    ):
        self.llm = llm_client
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.preserve_recent = preserve_recent
        
        self._last_system_prompt: Optional[str] = None
        self._last_tools: Optional[List[Dict]] = None
        self._compaction_count = 0
    
    def save_request_context(self, system_prompt: str, tools: List[Dict]):
        """
        保存请求上下文
        
        在每次 LLM 调用前调用，保存当前的 System Prompt 和 Tools
        用于后续压缩时复用
        """
        self._last_system_prompt = system_prompt
        self._last_tools = tools
    
    def needs_compaction(self, messages: List[Message]) -> bool:
        """检查是否需要压缩"""
        current_tokens = TokenCounter.estimate_messages(messages)
        return current_tokens > self.target_tokens
    
    async def compact_if_needed(
        self,
        messages: List[Message],
    ) -> Tuple[List[Message], Optional[CacheSafeCompactionResult]]:
        """
        如果需要，执行缓存安全的压缩
        
        返回: (压缩后的消息列表, 压缩结果)
        """
        current_tokens = TokenCounter.estimate_messages(messages)
        
        if current_tokens <= self.target_tokens:
            return messages, None
        
        if not self._last_system_prompt or not self._last_tools:
            return await self._fallback_compact(messages)
        
        return await self._cache_safe_compact(messages)
    
    async def _cache_safe_compact(
        self,
        messages: List[Message],
    ) -> Tuple[List[Message], CacheSafeCompactionResult]:
        """
        缓存安全的压缩实现
        
        关键：使用与主循环完全相同的参数
        """
        self._compaction_count += 1
        original_tokens = TokenCounter.estimate_messages(messages)
        
        formatted_messages = self._format_messages_for_compaction(messages)
        
        compaction_message = {
            "role": "user",
            "content": self.COMPACTION_REQUEST,
        }
        formatted_messages.append(compaction_message)
        
        summary_text = ""
        
        try:
            async for event in self.llm.chat_completion_stream(
                messages=formatted_messages,
                system_prompt=self._last_system_prompt,
                tools=self._last_tools,
            ):
                if hasattr(event, 'type') and event.type == "content":
                    summary_text += event.content or ""
                elif hasattr(event, 'content') and event.content:
                    summary_text += event.content
        
        except Exception as e:
            print(f"[WARN] 缓存安全压缩失败，降级到简单压缩: {e}")
            return await self._fallback_compact(messages)
        
        summary_message = Message(
            id=f"summary_{int(time.time() * 1000)}",
            session_id=messages[0].session_id if messages else "unknown",
            role=MessageRole.SYSTEM,
        )
        summary_message.add_text(f"""<conversation_summary>
{summary_text}
</conversation_summary>

<compaction_metadata>
Original messages: {len(messages)}
Compacted at: {time.strftime('%Y-%m-%d %H:%M:%S')}
Compaction count: {self._compaction_count}
</compaction_metadata>""")
        
        result_messages = [summary_message] + messages[-self.preserve_recent:]
        new_tokens = TokenCounter.estimate_messages(result_messages)
        
        result = CacheSafeCompactionResult(
            original_count=len(messages),
            compacted_count=len(result_messages),
            tokens_saved=original_tokens - new_tokens,
            summary=summary_text,
            compacted_messages=result_messages,
            cache_safe=True,
            cache_hit=True,
            original_tokens=original_tokens,
            new_tokens=new_tokens,
            cost_saved_ratio=1.0 - (new_tokens / original_tokens) if original_tokens > 0 else 0,
        )
        
        return result_messages, result
    
    async def _fallback_compact(
        self,
        messages: List[Message],
    ) -> Tuple[List[Message], CacheSafeCompactionResult]:
        """
        降级压缩（不保证缓存安全）
        
        当没有保存的上下文时使用
        """
        original_tokens = TokenCounter.estimate_messages(messages)
        
        summary_text = self._generate_simple_summary(messages)
        
        summary_message = Message(
            id=f"summary_{int(time.time() * 1000)}",
            session_id=messages[0].session_id if messages else "unknown",
            role=MessageRole.SYSTEM,
        )
        summary_message.add_text(f"[历史对话摘要]\n{summary_text}")
        
        result_messages = [summary_message] + messages[-self.preserve_recent:]
        new_tokens = TokenCounter.estimate_messages(result_messages)
        
        result = CacheSafeCompactionResult(
            original_count=len(messages),
            compacted_count=len(result_messages),
            tokens_saved=original_tokens - new_tokens,
            summary=summary_text,
            compacted_messages=result_messages,
            cache_safe=False,
            cache_hit=False,
            original_tokens=original_tokens,
            new_tokens=new_tokens,
            cost_saved_ratio=0,
        )
        
        return result_messages, result
    
    def _format_messages_for_compaction(self, messages: List[Message]) -> List[Dict]:
        """
        格式化消息为 LLM 格式
        
        保持与主循环相同的格式
        """
        formatted = []
        
        for msg in messages:
            if msg.role == MessageRole.USER:
                content = msg.get_text_content()
                if content:
                    formatted.append({"role": "user", "content": content})
            
            elif msg.role == MessageRole.ASSISTANT:
                content = msg.get_text_content()
                tool_calls = []
                
                for part in msg.parts:
                    if isinstance(part, ToolPart):
                        if part.status in [ToolStatus.COMPLETED, ToolStatus.ERROR]:
                            args = part.input
                            if isinstance(args, dict):
                                args = json.dumps(args, ensure_ascii=False)
                            tool_calls.append({
                                "id": part.call_id,
                                "type": "function",
                                "function": {
                                    "name": part.tool,
                                    "arguments": args,
                                }
                            })
                
                msg_dict = {"role": "assistant"}
                if content:
                    msg_dict["content"] = content
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                
                if content or tool_calls:
                    formatted.append(msg_dict)
                
                for part in msg.parts:
                    if isinstance(part, ToolPart):
                        if part.status == ToolStatus.COMPLETED:
                            formatted.append({
                                "role": "tool",
                                "tool_call_id": part.call_id,
                                "content": part.output or "",
                            })
                        elif part.status == ToolStatus.ERROR:
                            formatted.append({
                                "role": "tool",
                                "tool_call_id": part.call_id,
                                "content": f"Error: {part.error}",
                            })
        
        return formatted
    
    def _generate_simple_summary(self, messages: List[Message]) -> str:
        """生成简单摘要"""
        summary_parts = []
        tool_calls = []
        files_modified = set()
        
        for msg in messages:
            if msg.role == MessageRole.USER:
                text = msg.get_text_content()
                if text:
                    summary_parts.append(f"用户: {text[:100]}")
            
            elif msg.role == MessageRole.ASSISTANT:
                for part in msg.parts:
                    if isinstance(part, ToolPart):
                        tool_calls.append(f"{part.tool}")
                        if part.tool in ["write", "edit"]:
                            if "file_path" in part.input:
                                files_modified.add(part.input["file_path"])
        
        result = []
        if summary_parts:
            result.append("主要对话:")
            result.extend(summary_parts[-5:])
        
        if tool_calls:
            result.append(f"\n工具调用 ({len(tool_calls)} 次):")
            unique_tools = list(set(tool_calls))
            result.extend(unique_tools[-10:])
        
        if files_modified:
            result.append(f"\n修改的文件:")
            result.extend(list(files_modified)[-10:])
        
        return "\n".join(result) if result else "无重要内容"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        return {
            "compaction_count": self._compaction_count,
            "has_saved_context": self._last_system_prompt is not None,
        }


class CompactionHistory:
    """压缩历史记录"""
    
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: List[CacheSafeCompactionResult] = []
    
    def add(self, result: CacheSafeCompactionResult):
        """添加压缩记录"""
        self.entries.append(result)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.entries:
            return {
                "count": 0,
                "total_tokens_saved": 0,
                "cache_safe_count": 0,
                "cache_hit_rate": 0,
            }
        
        total_saved = sum(e.tokens_saved for e in self.entries)
        cache_safe_count = sum(1 for e in self.entries if e.cache_safe)
        
        return {
            "count": len(self.entries),
            "total_tokens_saved": total_saved,
            "avg_tokens_saved": total_saved / len(self.entries),
            "cache_safe_count": cache_safe_count,
            "cache_hit_rate": cache_safe_count / len(self.entries) if self.entries else 0,
        }
