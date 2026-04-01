"""
智能摘要压缩机制

支持：
- 上下文窗口管理
- 智能摘要生成
- 消息压缩策略
- 压缩历史追踪
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod

from core.message import Message, Part, TextPart, ReasoningPart, ToolPart, MessageRole


class CompactionStrategy(Enum):
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    SLIDING_WINDOW = "sliding_window"
    IMPORTANCE = "importance"


@dataclass
class CompactionConfig:
    """压缩配置"""
    max_tokens: int = 128000
    target_tokens: int = 100000
    min_messages: int = 4
    preserve_recent: int = 2
    strategy: CompactionStrategy = CompactionStrategy.SUMMARIZE
    summarize_tools: bool = True
    summarize_reasoning: bool = True


@dataclass
class CompactionResult:
    """压缩结果"""
    original_count: int
    compacted_count: int
    tokens_saved: int
    summary: Optional[str] = None
    compacted_messages: List[Message] = field(default_factory=list)


@dataclass
class MessageImportance:
    """消息重要性评分"""
    message: Message
    score: float
    reasons: List[str] = field(default_factory=list)


class TokenCounter:
    """Token 计数器"""
    
    @staticmethod
    def estimate(text: str) -> int:
        return len(text) // 4
    
    @staticmethod
    def estimate_message(message: Message) -> int:
        total = 0
        for part in message.parts:
            if isinstance(part, TextPart):
                total += TokenCounter.estimate(part.text)
            elif isinstance(part, ReasoningPart):
                total += TokenCounter.estimate(part.text)
            elif isinstance(part, ToolPart):
                total += TokenCounter.estimate(json.dumps(part.state))
        return total
    
    @staticmethod
    def estimate_messages(messages: List[Message]) -> int:
        return sum(TokenCounter.estimate_message(m) for m in messages)


class Compactor(ABC):
    """压缩器基类"""
    
    @abstractmethod
    async def compact(self, messages: List[Message], config: CompactionConfig) -> CompactionResult:
        pass


class TruncateCompactor(Compactor):
    """截断压缩器"""
    
    async def compact(self, messages: List[Message], config: CompactionConfig) -> CompactionResult:
        if len(messages) <= config.min_messages:
            return CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                tokens_saved=0,
            )
        
        total_tokens = TokenCounter.estimate_messages(messages)
        if total_tokens <= config.target_tokens:
            return CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                tokens_saved=0,
            )
        
        preserve_count = config.preserve_recent + 1
        preserved = messages[-preserve_count:]
        to_compact = messages[:-preserve_count]
        
        target_tokens = config.target_tokens - TokenCounter.estimate_messages(preserved)
        compacted = []
        current_tokens = 0
        
        for msg in reversed(to_compact):
            msg_tokens = TokenCounter.estimate_message(msg)
            if current_tokens + msg_tokens <= target_tokens:
                compacted.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        result_messages = compacted + preserved
        
        return CompactionResult(
            original_count=len(messages),
            compacted_count=len(result_messages),
            tokens_saved=total_tokens - TokenCounter.estimate_messages(result_messages),
            compacted_messages=result_messages,
        )


class SummarizeCompactor(Compactor):
    """摘要压缩器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def compact(self, messages: List[Message], config: CompactionConfig) -> CompactionResult:
        if len(messages) <= config.min_messages:
            return CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                tokens_saved=0,
            )
        
        total_tokens = TokenCounter.estimate_messages(messages)
        if total_tokens <= config.target_tokens:
            return CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                tokens_saved=0,
            )
        
        preserve_count = config.preserve_recent + 1
        preserved = messages[-preserve_count:]
        to_summarize = messages[:-preserve_count]
        
        summary = await self._generate_summary(to_summarize, config)
        
        summary_message = Message(
            id=f"summary_{int(time.time() * 1000)}",
            session_id=preserved[0].session_id if preserved else "unknown",
            role=MessageRole.SYSTEM,
            parts=[TextPart(
                id="summary_text",
                message_id=f"summary_{int(time.time() * 1000)}",
                session_id=preserved[0].session_id if preserved else "unknown",
                text=f"[历史对话摘要]\n{summary}",
            )],
            metadata={"compaction": True, "compacted_count": len(to_summarize)},
        )
        
        result_messages = [summary_message] + preserved
        new_tokens = TokenCounter.estimate_messages(result_messages)
        
        return CompactionResult(
            original_count=len(messages),
            compacted_count=len(result_messages),
            tokens_saved=total_tokens - new_tokens,
            summary=summary,
            compacted_messages=result_messages,
        )
    
    async def _generate_summary(self, messages: List[Message], config: CompactionConfig) -> str:
        if not self.llm_client:
            return self._generate_simple_summary(messages, config)
        
        summary_prompt = self._build_summary_prompt(messages)
        
        try:
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.get("content", self._generate_simple_summary(messages, config))
        except Exception:
            return self._generate_simple_summary(messages, config)
    
    def _build_summary_prompt(self, messages: List[Message]) -> str:
        content_parts = []
        
        for msg in messages:
            role = msg.role.value
            text = msg.get_text_content()
            tools = msg.get_tool_parts()
            
            if tools:
                tool_info = []
                for tool in tools:
                    tool_info.append(f"  - {tool.tool}: {json.dumps(tool.input, ensure_ascii=False)[:100]}")
                content_parts.append(f"[{role}] {text[:200]}\n工具调用:\n" + "\n".join(tool_info))
            else:
                content_parts.append(f"[{role}] {text[:300]}")
        
        return f"""请为以下对话生成一个简洁的摘要，保留关键信息、决策和上下文：

{chr(10).join(content_parts)}

摘要要求：
1. 保留重要的决策和结论
2. 记录关键的文件操作和代码变更
3. 保留未完成的任务和待办事项
4. 省略无关的细节和闲聊
5. 使用简洁的中文表达

摘要："""
    
    def _generate_simple_summary(self, messages: List[Message], config: CompactionConfig) -> str:
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
                        tool_calls.append(f"{part.tool}({json.dumps(part.input, ensure_ascii=False)[:50]})")
                        if part.tool in ["write", "edit"]:
                            if "file_path" in part.input:
                                files_modified.add(part.input["file_path"])
        
        result = []
        if summary_parts:
            result.append("主要对话:")
            result.extend(summary_parts[-5:])
        
        if tool_calls:
            result.append(f"\n工具调用 ({len(tool_calls)} 次):")
            result.extend(tool_calls[-10:])
        
        if files_modified:
            result.append(f"\n修改的文件:")
            result.extend(list(files_modified)[-10:])
        
        return "\n".join(result) if result else "无重要内容"


class ImportanceCompactor(Compactor):
    """重要性压缩器"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    async def compact(self, messages: List[Message], config: CompactionConfig) -> CompactionResult:
        if len(messages) <= config.min_messages:
            return CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                tokens_saved=0,
            )
        
        total_tokens = TokenCounter.estimate_messages(messages)
        if total_tokens <= config.target_tokens:
            return CompactionResult(
                original_count=len(messages),
                compacted_count=len(messages),
                tokens_saved=0,
            )
        
        preserve_count = config.preserve_recent + 1
        preserved = messages[-preserve_count:]
        to_compact = messages[:-preserve_count]
        
        scored = [self._score_importance(msg) for msg in to_compact]
        scored.sort(key=lambda x: x.score, reverse=True)
        
        target_tokens = config.target_tokens - TokenCounter.estimate_messages(preserved)
        selected = []
        current_tokens = 0
        
        for item in scored:
            msg_tokens = TokenCounter.estimate_message(item.message)
            if current_tokens + msg_tokens <= target_tokens:
                selected.append(item.message)
                current_tokens += msg_tokens
        
        selected.sort(key=lambda m: messages.index(m))
        result_messages = selected + preserved
        
        return CompactionResult(
            original_count=len(messages),
            compacted_count=len(result_messages),
            tokens_saved=total_tokens - TokenCounter.estimate_messages(result_messages),
            compacted_messages=result_messages,
        )
    
    def _score_importance(self, message: Message) -> MessageImportance:
        score = 0.0
        reasons = []
        
        if message.role == MessageRole.USER:
            score += 10.0
            reasons.append("用户消息")
        
        text = message.get_text_content()
        
        keywords = ["错误", "error", "问题", "bug", "重要", "important", "关键", "critical"]
        for kw in keywords:
            if kw in text.lower():
                score += 5.0
                reasons.append(f"包含关键词: {kw}")
        
        for part in message.parts:
            if isinstance(part, ToolPart):
                if part.tool in ["write", "edit", "bash"]:
                    score += 8.0
                    reasons.append(f"重要工具: {part.tool}")
                elif part.tool in ["read", "glob", "grep"]:
                    score += 2.0
                    reasons.append(f"查询工具: {part.tool}")
                
                if part.state.get("error"):
                    score += 5.0
                    reasons.append("包含错误信息")
        
        reasoning = message.get_reasoning_content()
        if reasoning:
            score += 3.0
            reasons.append("包含推理过程")
        
        return MessageImportance(message=message, score=score, reasons=reasons)


class CompactionManager:
    """压缩管理器"""
    
    def __init__(
        self,
        config: CompactionConfig = None,
        llm_client=None,
    ):
        self.config = config or CompactionConfig()
        self.llm_client = llm_client
        
        self.compactors = {
            CompactionStrategy.TRUNCATE: TruncateCompactor(),
            CompactionStrategy.SUMMARIZE: SummarizeCompactor(llm_client),
            CompactionStrategy.IMPORTANCE: ImportanceCompactor(llm_client),
        }
    
    def is_overflow(self, messages: List[Message]) -> bool:
        """检查是否需要压缩"""
        tokens = TokenCounter.estimate_messages(messages)
        return tokens > self.config.max_tokens
    
    async def compact(self, messages: List[Message]) -> CompactionResult:
        """执行压缩"""
        compactor = self.compactors.get(self.config.strategy)
        if not compactor:
            compactor = self.compactors[CompactionStrategy.SUMMARIZE]
        
        return await compactor.compact(messages, self.config)
    
    async def auto_compact_if_needed(self, messages: List[Message]) -> Tuple[List[Message], Optional[CompactionResult]]:
        """自动压缩（如果需要）"""
        if not self.is_overflow(messages):
            return messages, None
        
        result = await self.compact(messages)
        return result.compacted_messages, result


class CompactionHistory:
    """压缩历史"""
    
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.entries: List[CompactionResult] = []
    
    def add(self, result: CompactionResult):
        """添加压缩记录"""
        self.entries.append(result)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.entries:
            return {"count": 0}
        
        total_saved = sum(e.tokens_saved for e in self.entries)
        avg_saved = total_saved / len(self.entries)
        
        return {
            "count": len(self.entries),
            "total_tokens_saved": total_saved,
            "avg_tokens_saved": avg_saved,
            "last_compaction": self.entries[-1].to_dict() if hasattr(self.entries[-1], 'to_dict') else None,
        }
