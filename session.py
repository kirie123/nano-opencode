"""
会话管理：处理对话历史、消息存储和会话状态
"""

import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import json
import os


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class PartType(Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE = "file"
    REASONING = "reasoning"


@dataclass
class Part:
    """消息部分"""
    type: PartType
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Part':
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            type=PartType(data["type"]),
            content=data.get("content", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class Message:
    """消息"""
    role: MessageRole
    parts: List[Part] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    agent: Optional[str] = None  # 使用哪个 Agent
    model: Optional[str] = None  # 使用哪个模型
    parent_id: Optional[str] = None  # 父消息 ID（用于多轮对话）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "parts": [p.to_dict() for p in self.parts],
            "timestamp": self.timestamp,
            "agent": self.agent,
            "model": self.model,
            "parent_id": self.parent_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            role=MessageRole(data["role"]),
            parts=[Part.from_dict(p) for p in data.get("parts", [])],
            timestamp=data.get("timestamp", time.time()),
            agent=data.get("agent"),
            model=data.get("model"),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {})
        )
    
    def get_text_content(self) -> str:
        """获取文本内容"""
        texts = []
        for part in self.parts:
            if part.type == PartType.TEXT:
                texts.append(part.content)
        return "\n".join(texts)
    
    def add_text(self, text: str, metadata: Dict[str, Any] = None):
        """添加文本部分"""
        self.parts.append(Part(
            type=PartType.TEXT,
            content=text,
            metadata=metadata or {}
        ))
    
    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """添加工具调用部分"""
        self.parts.append(Part(
            type=PartType.TOOL_CALL,
            content=json.dumps({"tool": tool_name, "arguments": arguments}),
            metadata={"tool": tool_name, "arguments": arguments}
        ))
    
    def add_tool_result(self, tool_call_id: str, result: str):
        """添加工具结果部分"""
        self.parts.append(Part(
            type=PartType.TOOL_RESULT,
            content=result,
            metadata={"tool_call_id": tool_call_id}
        ))


class Session:
    """会话"""
    
    def __init__(self, 
                 session_id: str = None,
                 title: str = None,
                 workspace: str = None,
                 parent_id: str = None):
        self.id = session_id or str(uuid.uuid4())
        self.title = title or f"Session {self.id[:8]}"
        self.workspace = workspace or os.getcwd()
        self.parent_id = parent_id  # 父会话 ID（用于子 Agent）
        self.messages: List[Message] = []
        self.created_at = time.time()
        self.updated_at = time.time()
        self.metadata: Dict[str, Any] = {}
        self.current_agent: Optional[str] = None
        self.permissions: List[Dict[str, Any]] = []
    
    def add_message(self, message: Message) -> Message:
        """添加消息"""
        message.timestamp = time.time()
        self.messages.append(message)
        self.updated_at = time.time()
        
        # 自动更新当前 Agent
        if message.agent:
            self.current_agent = message.agent
        
        return message
    
    def create_user_message(self, text: str, agent: str = None, 
                           files: List[Dict[str, Any]] = None) -> Message:
        """创建用户消息"""
        message = Message(
            role=MessageRole.USER,
            agent=agent or self.current_agent,
            parent_id=self.messages[-1].id if self.messages else None
        )
        
        # 添加文本内容
        if text:
            message.add_text(text)
        
        # 添加文件
        if files:
            for file_info in files:
                message.parts.append(Part(
                    type=PartType.FILE,
                    content=file_info.get("content", ""),
                    metadata={
                        "filename": file_info.get("filename"),
                        "mime_type": file_info.get("mime_type")
                    }
                ))
        
        return self.add_message(message)
    
    def create_assistant_message(self, agent: str = None,
                                 parent_id: str = None) -> Message:
        """创建助手消息（空，等待填充内容）"""
        message = Message(
            role=MessageRole.ASSISTANT,
            agent=agent or self.current_agent,
            parent_id=parent_id or (self.messages[-1].id if self.messages else None)
        )
        return self.add_message(message)
    
    def get_history(self, limit: int = None) -> List[Message]:
        """获取历史消息"""
        messages = self.messages
        if limit:
            messages = messages[-limit:]
        return messages
    
    def get_message_chain(self, message_id: str) -> List[Message]:
        """获取消息链（从根消息到指定消息）"""
        chain = []
        current_id = message_id
        
        # 构建消息映射
        msg_map = {m.id: m for m in self.messages}
        
        while current_id and current_id in msg_map:
            msg = msg_map[current_id]
            chain.append(msg)
            current_id = msg.parent_id
        
        return list(reversed(chain))
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "workspace": self.workspace,
            "parent_id": self.parent_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "current_agent": self.current_agent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """从字典反序列化"""
        session = cls(
            session_id=data.get("id"),
            title=data.get("title"),
            workspace=data.get("workspace"),
            parent_id=data.get("parent_id")
        )
        session.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        session.created_at = data.get("created_at", time.time())
        session.updated_at = data.get("updated_at", time.time())
        session.metadata = data.get("metadata", {})
        session.current_agent = data.get("current_agent")
        return session


class SessionManager:
    """会话管理器"""
    
    def __init__(self, storage_dir: str = None):
        self.sessions: Dict[str, Session] = {}
        self.storage_dir = storage_dir
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            self._load_sessions()
    
    def create_session(self, title: str = None, 
                      workspace: str = None,
                      parent_id: str = None) -> Session:
        """创建新会话"""
        session = Session(
            title=title,
            workspace=workspace,
            parent_id=parent_id
        )
        self.sessions[session.id] = session
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[Session]:
        """列出所有会话"""
        return sorted(
            self.sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True
        )
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.storage_dir:
                filepath = os.path.join(self.storage_dir, f"{session_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
            return True
        return False
    
    def _save_session(self, session: Session):
        """保存会话到磁盘"""
        if not self.storage_dir:
            return
        
        filepath = os.path.join(self.storage_dir, f"{session.id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving session: {e}")
    
    def _load_sessions(self):
        """从磁盘加载会话"""
        if not self.storage_dir:
            return
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    session = Session.from_dict(data)
                    self.sessions[session.id] = session
                except Exception as e:
                    print(f"Error loading session {filename}: {e}")


# 全局会话管理器实例
session_manager = SessionManager()
