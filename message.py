"""
消息模型：完整的 Part 系统

支持多种 Part 类型：
- TextPart: 文本内容
- ReasoningPart: 推理过程（思考）
- ToolPart: 工具调用和结果
- FilePart: 文件附件
- ImagePart: 图片附件
- StepPart: 步骤标记（开始/结束）
- PatchPart: 代码变更补丁
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal, Union
from enum import Enum
from abc import ABC, abstractmethod


class PartType(Enum):
    TEXT = "text"
    REASONING = "reasoning"
    TOOL = "tool"
    FILE = "file"
    IMAGE = "image"
    STEP_START = "step_start"
    STEP_FINISH = "step_finish"
    PATCH = "patch"


class ToolStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Part(ABC):
    """Part 基类"""
    id: str
    type: PartType
    message_id: str
    session_id: str
    time: Dict[str, int] = field(default_factory=lambda: {"start": int(time.time() * 1000)})
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Part":
        part_type = PartType(data["type"])
        if part_type == PartType.TEXT:
            return TextPart.from_dict(data)
        elif part_type == PartType.REASONING:
            return ReasoningPart.from_dict(data)
        elif part_type == PartType.TOOL:
            return ToolPart.from_dict(data)
        elif part_type == PartType.FILE:
            return FilePart.from_dict(data)
        elif part_type == PartType.IMAGE:
            return ImagePart.from_dict(data)
        elif part_type == PartType.STEP_START:
            return StepStartPart.from_dict(data)
        elif part_type == PartType.STEP_FINISH:
            return StepFinishPart.from_dict(data)
        elif part_type == PartType.PATCH:
            return PatchPart.from_dict(data)
        raise ValueError(f"Unknown part type: {part_type}")


@dataclass
class TextPart(Part):
    """文本内容"""
    text: str = ""
    
    def __init__(self, id: str, message_id: str, session_id: str, text: str = "", **kwargs):
        super().__init__(id, PartType.TEXT, message_id, session_id, **kwargs)
        self.text = text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "text": self.text,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextPart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            text=data.get("text", ""),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReasoningPart(Part):
    """推理过程（思考）"""
    text: str = ""
    
    def __init__(self, id: str, message_id: str, session_id: str, text: str = "", **kwargs):
        super().__init__(id, PartType.REASONING, message_id, session_id, **kwargs)
        self.text = text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "text": self.text,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningPart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            text=data.get("text", ""),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ToolPart(Part):
    """工具调用和结果"""
    tool: str = ""
    call_id: str = ""
    state: Dict[str, Any] = field(default_factory=lambda: {"status": ToolStatus.PENDING.value})
    
    def __init__(
        self, 
        id: str, 
        message_id: str, 
        session_id: str, 
        tool: str = "", 
        call_id: str = "",
        state: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(id, PartType.TOOL, message_id, session_id, **kwargs)
        self.tool = tool
        self.call_id = call_id
        self.state = state or {"status": ToolStatus.PENDING.value}
    
    @property
    def status(self) -> ToolStatus:
        return ToolStatus(self.state.get("status", ToolStatus.PENDING.value))
    
    @property
    def input(self) -> Dict[str, Any]:
        return self.state.get("input", {})
    
    @property
    def output(self) -> str:
        return self.state.get("output", "")
    
    @property
    def error(self) -> Optional[str]:
        return self.state.get("error")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "tool": self.tool,
            "call_id": self.call_id,
            "state": self.state,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolPart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            tool=data.get("tool", ""),
            call_id=data.get("call_id", ""),
            state=data.get("state", {"status": ToolStatus.PENDING.value}),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FilePart(Part):
    """文件附件"""
    url: str = ""
    filename: str = ""
    mime: str = "text/plain"
    
    def __init__(self, id: str, message_id: str, session_id: str, url: str = "", filename: str = "", mime: str = "text/plain", **kwargs):
        super().__init__(id, PartType.FILE, message_id, session_id, **kwargs)
        self.url = url
        self.filename = filename
        self.mime = mime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "url": self.url,
            "filename": self.filename,
            "mime": self.mime,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilePart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            url=data.get("url", ""),
            filename=data.get("filename", ""),
            mime=data.get("mime", "text/plain"),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ImagePart(Part):
    """图片附件"""
    url: str = ""
    mime: str = "image/png"
    
    def __init__(self, id: str, message_id: str, session_id: str, url: str = "", mime: str = "image/png", **kwargs):
        super().__init__(id, PartType.IMAGE, message_id, session_id, **kwargs)
        self.url = url
        self.mime = mime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "url": self.url,
            "mime": self.mime,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImagePart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            url=data.get("url", ""),
            mime=data.get("mime", "image/png"),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StepStartPart(Part):
    """步骤开始标记"""
    snapshot: Optional[str] = None
    
    def __init__(self, id: str, message_id: str, session_id: str, snapshot: str = None, **kwargs):
        super().__init__(id, PartType.STEP_START, message_id, session_id, **kwargs)
        self.snapshot = snapshot
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "snapshot": self.snapshot,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepStartPart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            snapshot=data.get("snapshot"),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StepFinishPart(Part):
    """步骤结束标记"""
    reason: str = ""
    tokens: Dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    snapshot: Optional[str] = None
    
    def __init__(self, id: str, message_id: str, session_id: str, reason: str = "", tokens: Dict = None, cost: float = 0.0, snapshot: str = None, **kwargs):
        super().__init__(id, PartType.STEP_FINISH, message_id, session_id, **kwargs)
        self.reason = reason
        self.tokens = tokens or {}
        self.cost = cost
        self.snapshot = snapshot
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "reason": self.reason,
            "tokens": self.tokens,
            "cost": self.cost,
            "snapshot": self.snapshot,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepFinishPart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            reason=data.get("reason", ""),
            tokens=data.get("tokens", {}),
            cost=data.get("cost", 0.0),
            snapshot=data.get("snapshot"),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PatchPart(Part):
    """代码变更补丁"""
    hash: str = ""
    files: List[Dict[str, Any]] = field(default_factory=list)
    
    def __init__(self, id: str, message_id: str, session_id: str, hash: str = "", files: List = None, **kwargs):
        super().__init__(id, PartType.PATCH, message_id, session_id, **kwargs)
        self.hash = hash
        self.files = files or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "hash": self.hash,
            "files": self.files,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchPart":
        return cls(
            id=data["id"],
            message_id=data["message_id"],
            session_id=data["session_id"],
            hash=data.get("hash", ""),
            files=data.get("files", []),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """消息"""
    id: str
    session_id: str
    role: MessageRole
    parts: List[Part] = field(default_factory=list)
    parent_id: Optional[str] = None
    agent: Optional[str] = None
    model_id: Optional[str] = None
    provider_id: Optional[str] = None
    finish_reason: Optional[str] = None
    cost: float = 0.0
    tokens: Dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0, "reasoning": 0})
    time: Dict[str, int] = field(default_factory=lambda: {"created": int(time.time() * 1000)})
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_text(self, text: str, part_id: str = None) -> TextPart:
        part = TextPart(
            id=part_id or f"{self.id}_text_{len(self.parts)}",
            message_id=self.id,
            session_id=self.session_id,
            text=text,
        )
        self.parts.append(part)
        return part
    
    def add_reasoning(self, text: str, part_id: str = None) -> ReasoningPart:
        part = ReasoningPart(
            id=part_id or f"{self.id}_reasoning_{len(self.parts)}",
            message_id=self.id,
            session_id=self.session_id,
            text=text,
        )
        self.parts.append(part)
        return part
    
    def add_tool_call(self, tool: str, call_id: str, input: Dict = None) -> ToolPart:
        part = ToolPart(
            id=f"{self.id}_tool_{len(self.parts)}",
            message_id=self.id,
            session_id=self.session_id,
            tool=tool,
            call_id=call_id,
            state={"status": ToolStatus.PENDING.value, "input": input or {}},
        )
        self.parts.append(part)
        return part
    
    def update_tool_result(self, call_id: str, output: str = None, error: str = None, title: str = None):
        for part in self.parts:
            if isinstance(part, ToolPart) and part.call_id == call_id:
                if error:
                    part.state["status"] = ToolStatus.ERROR.value
                    part.state["error"] = error
                else:
                    part.state["status"] = ToolStatus.COMPLETED.value
                    part.state["output"] = output
                    part.state["time"]["end"] = int(time.time() * 1000)
                if title:
                    part.state["title"] = title
                break
    
    def get_text_content(self) -> str:
        texts = []
        for part in self.parts:
            if isinstance(part, TextPart):
                texts.append(part.text)
        return "\n".join(texts)
    
    def get_reasoning_content(self) -> str:
        texts = []
        for part in self.parts:
            if isinstance(part, ReasoningPart):
                texts.append(part.text)
        return "\n".join(texts)
    
    def get_tool_parts(self) -> List[ToolPart]:
        return [p for p in self.parts if isinstance(p, ToolPart)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value,
            "parts": [p.to_dict() for p in self.parts],
            "parent_id": self.parent_id,
            "agent": self.agent,
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "finish_reason": self.finish_reason,
            "cost": self.cost,
            "tokens": self.tokens,
            "time": self.time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        parts = [Part.from_dict(p) for p in data.get("parts", [])]
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            role=MessageRole(data["role"]),
            parts=parts,
            parent_id=data.get("parent_id"),
            agent=data.get("agent"),
            model_id=data.get("model_id"),
            provider_id=data.get("provider_id"),
            finish_reason=data.get("finish_reason"),
            cost=data.get("cost", 0.0),
            tokens=data.get("tokens", {}),
            time=data.get("time", {}),
            metadata=data.get("metadata", {}),
        )


def generate_id(prefix: str = "") -> str:
    import uuid
    return f"{prefix}{uuid.uuid4().hex[:16]}" if prefix else uuid.uuid4().hex[:16]
