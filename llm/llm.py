"""
LLM 客户端：支持 OpenAI、Anthropic、Ollama 等多种 API
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass, field
import aiohttp


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    
    def __post_init__(self):
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class OllamaConfig:
    """Ollama 配置"""
    base_url: str = "http://localhost:11434"
    model: str = "glm-4.7-flash"
    temperature: float = 0.7
    num_ctx: int = 131072
    num_predict: int = 8192
    timeout_sec: int = 600
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.2
    disable_thinking: bool = True
    stop_sequences: tuple = field(default_factory=lambda: ("</think",))


@dataclass
class Message:
    """消息"""
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    type: str
    function: Dict[str, Any]
    
    @property
    def name(self) -> str:
        return self.function.get("name", "")
    
    @property
    def arguments(self) -> Dict[str, Any]:
        try:
            return json.loads(self.function.get("arguments", "{}"))
        except json.JSONDecodeError:
            return {}


@dataclass
class StreamEvent:
    """流式事件"""
    type: str  # "content", "reasoning", "tool_call", "tool_calls", "done", "error"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_sec)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @classmethod
    def from_config(cls, config_dict: dict) -> "OllamaClient":
        """从配置字典创建客户端"""
        return cls(OllamaConfig(**config_dict))
    
    async def chat_completion_stream(
        self,
        messages: List[Union[Message, Dict]],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """流式聊天完成"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_sec)
            )
        
        formatted_messages = self._format_messages(messages, system_prompt)
        
        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "stream": True,
            "think": not self.config.disable_thinking,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
                "stop": list(self.config.stop_sequences),
            },
        }
        
        if tools:
            body["tools"] = self._convert_tools(tools)
        
        url = f"{self.config.base_url}/api/chat"
        
        try:
            async with self.session.post(url, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield StreamEvent(type="error", error=f"Ollama error {response.status}: {error_text}")
                    return
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        event = self._parse_ollama_chunk(data)
                        if event:
                            yield event
                    except json.JSONDecodeError:
                        continue
                
                yield StreamEvent(type="done")
                
        except Exception as e:
            yield StreamEvent(type="error", error=str(e))
    
    async def chat_completion(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """非流式聊天完成"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_sec)
            )
        
        formatted_messages = self._format_messages(messages, system_prompt)
        
        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "stream": False,
            "think": not self.config.disable_thinking,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
            },
        }
        
        if tools:
            body["tools"] = self._convert_tools(tools)
        
        url = f"{self.config.base_url}/api/chat"
        
        try:
            async with self.session.post(url, json=body) as response:
                result = await response.json()
                if response.status != 200:
                    raise Exception(f"Ollama error {response.status}: {result}")
                return result
        except Exception as e:
            raise Exception(f"Ollama request failed: {str(e)}")
    
    def _format_messages(
        self,
        messages: List[Union[Message, Dict]],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """格式化消息"""
        formatted = []
        
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append(msg)
            else:
                formatted.append(msg.to_dict())
        
        return formatted
    
    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """转换工具格式为 Ollama 格式"""
        ollama_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                })
        return ollama_tools
    
    def _parse_ollama_chunk(self, data: Dict[str, Any]) -> Optional[StreamEvent]:
        """解析 Ollama 响应块"""
        message = data.get("message", {})
        role = message.get("role", "")
        content = message.get("content", "")
        thinking = message.get("thinking", "")
        tool_calls = message.get("tool_calls", [])
        
        done = data.get("done", False)
        
        if thinking:
            return StreamEvent(type="reasoning", content=thinking)
        
        if tool_calls:
            calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                calls.append(ToolCall(
                    id=tc.get("id", ""),
                    type="function",
                    function={
                        "name": func.get("name", ""),
                        "arguments": json.dumps(func.get("arguments", {})),
                    }
                ))
            return StreamEvent(type="tool_calls", tool_calls=calls)
        
        if content:
            return StreamEvent(type="content", content=content)
        
        if done:
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            return StreamEvent(
                type="done",
                finish_reason="stop",
                usage={
                    "input": prompt_eval_count,
                    "output": eval_count,
                }
            )
        
        return None
    
    async def list_models(self) -> List[str]:
        """列出可用模型"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.config.base_url}/api/tags"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [m["name"] for m in data.get("models", [])]
                return []
        except Exception:
            return []


class LLMClient:
    """统一 LLM 客户端"""
    
    def __init__(self, config: LLMConfig = None, ollama_config: OllamaConfig = None):
        self.config = config or LLMConfig()
        self.ollama_config = ollama_config
        self.session: Optional[aiohttp.ClientSession] = None
        self._ollama_client: Optional[OllamaClient] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self._ollama_client:
            await self._ollama_client.__aexit__(exc_type, exc_val, exc_tb)
    
    @property
    def ollama_client(self) -> OllamaClient:
        """获取 Ollama 客户端"""
        if not self._ollama_client:
            self._ollama_client = OllamaClient(self.ollama_config)
        return self._ollama_client
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
    
    def _get_api_base(self) -> str:
        """获取 API 基础 URL"""
        if self.config.api_base:
            return self.config.api_base
        if self.config.provider == "anthropic":
            return "https://api.anthropic.com/v1"
        return "https://api.openai.com/v1"
    
    async def chat_completion_stream(
        self,
        messages: List[Union[Message, Dict]],
        tools: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """流式聊天完成"""
        if self.config.provider == "ollama":
            async for event in self.ollama_client.chat_completion_stream(
                messages, tools, system_prompt
            ):
                yield event
            return
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        for m in messages:
            if isinstance(m, dict):
                formatted_messages.append(m)
            else:
                formatted_messages.append(m.to_dict())
        
        payload = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True
        }
        
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        url = f"{self._get_api_base()}/chat/completions"
        
        try:
            async with self.session.post(
                url,
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield StreamEvent(
                        type="error",
                        error=f"API error {response.status}: {error_text}"
                    )
                    return
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            chunk = json.loads(data)
                            event = self._parse_chunk(chunk)
                            if event:
                                yield event
                        except json.JSONDecodeError:
                            continue
                
                yield StreamEvent(type="done")
                
        except Exception as e:
            yield StreamEvent(type="error", error=str(e))
    
    async def chat_completion(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        system_prompt: Optional[str] = None,
    ) -> Union[AsyncIterator[StreamEvent], Dict[str, Any]]:
        """聊天完成 API"""
        if stream:
            return self.chat_completion_stream(messages, tools, system_prompt)
        
        if self.config.provider == "ollama":
            return await self.ollama_client.chat_completion(messages, tools, system_prompt)
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        formatted_messages.extend([m.to_dict() for m in messages])
        
        payload = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False
        }
        
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        url = f"{self._get_api_base()}/chat/completions"
        
        try:
            async with self.session.post(
                url,
                headers=self._get_headers(),
                json=payload
            ) as response:
                result = await response.json()
                if response.status != 200:
                    raise Exception(f"API error {response.status}: {result}")
                return result
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def _parse_chunk(self, chunk: Dict[str, Any]) -> Optional[StreamEvent]:
        """解析 SSE 数据块"""
        choices = chunk.get("choices", [])
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")
        
        if "tool_calls" in delta:
            tool_calls = []
            for tc in delta["tool_calls"]:
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=tc.get("function", {})
                ))
            return StreamEvent(type="tool_calls", tool_calls=tool_calls)
        
        if "content" in delta and delta["content"]:
            return StreamEvent(
                type="content",
                content=delta["content"],
                finish_reason=finish_reason
            )
        
        if finish_reason:
            usage = chunk.get("usage", {})
            return StreamEvent(
                type="done",
                finish_reason=finish_reason,
                usage=usage,
            )
        
        return None


class SessionStore:
    """会话存储"""
    
    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
    
    def save(self, session: "Session"):
        """保存会话"""
        if not self.storage_dir:
            return
        
        filepath = os.path.join(self.storage_dir, f"{session.id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load(self, session_id: str) -> Optional["Session"]:
        """加载会话"""
        if not self.storage_dir:
            return None
        
        filepath = os.path.join(self.storage_dir, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Session.from_dict(data)
    
    def list_sessions(self) -> List[str]:
        """列出所有会话 ID"""
        if not self.storage_dir:
            return []
        
        sessions = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                sessions.append(filename[:-5])
        return sessions
