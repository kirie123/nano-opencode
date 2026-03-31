"""
错误处理系统

支持：
- 自动重试（可配置重试次数和延迟）
- 错误回退策略
- 错误分类和处理
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
from enum import Enum
from abc import ABC, abstractmethod
import traceback


class ErrorType(Enum):
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTH = "auth"
    VALIDATION = "validation"
    TOOL = "tool"
    PERMISSION = "permission"
    CONTEXT_OVERFLOW = "context_overflow"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """错误信息"""
    type: ErrorType
    severity: ErrorSeverity
    message: str
    original_error: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recoverable: bool = True
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "retry_count": self.retry_count,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exception(type(self.original_error), self.original_error, self.original_error.__traceback__) if self.original_error else None,
        }


class AgentError(Exception):
    """Agent 错误基类"""
    
    def __init__(
        self, 
        message: str, 
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        context: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.info = ErrorInfo(
            type=error_type,
            severity=severity,
            message=message,
            original_error=self,
            context=context or {},
            recoverable=recoverable,
        )


class NetworkError(AgentError):
    """网络错误"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorType.NETWORK, ErrorSeverity.MEDIUM, True, context)


class RateLimitError(AgentError):
    """速率限制错误"""
    def __init__(self, message: str, retry_after: int = 60, context: Dict = None):
        ctx = context or {}
        ctx["retry_after"] = retry_after
        super().__init__(message, ErrorType.RATE_LIMIT, ErrorSeverity.MEDIUM, True, ctx)


class TimeoutError(AgentError):
    """超时错误"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorType.TIMEOUT, ErrorSeverity.MEDIUM, True, context)


class AuthError(AgentError):
    """认证错误"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorType.AUTH, ErrorSeverity.HIGH, False, context)


class ValidationError(AgentError):
    """验证错误"""
    def __init__(self, message: str, context: Dict = None):
        super().__init__(message, ErrorType.VALIDATION, ErrorSeverity.LOW, False, context)


class ToolError(AgentError):
    """工具执行错误"""
    def __init__(self, message: str, tool_name: str = None, context: Dict = None):
        ctx = context or {}
        if tool_name:
            ctx["tool_name"] = tool_name
        super().__init__(message, ErrorType.TOOL, ErrorSeverity.MEDIUM, True, ctx)


class PermissionError(AgentError):
    """权限错误"""
    def __init__(self, message: str, permission: str = None, context: Dict = None):
        ctx = context or {}
        if permission:
            ctx["permission"] = permission
        super().__init__(message, ErrorType.PERMISSION, ErrorSeverity.HIGH, False, ctx)


class ContextOverflowError(AgentError):
    """上下文溢出错误"""
    def __init__(self, message: str, tokens: int = 0, limit: int = 0, context: Dict = None):
        ctx = context or {}
        ctx["tokens"] = tokens
        ctx["limit"] = limit
        super().__init__(message, ErrorType.CONTEXT_OVERFLOW, ErrorSeverity.HIGH, True, ctx)


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.NETWORK,
        ErrorType.RATE_LIMIT,
        ErrorType.TIMEOUT,
        ErrorType.TOOL,
    ])


@dataclass
class RetryState:
    """重试状态"""
    attempt: int = 0
    last_error: Optional[ErrorInfo] = None
    total_delay: float = 0.0
    start_time: int = field(default_factory=lambda: int(time.time() * 1000))


T = TypeVar('T')


class ErrorHandler:
    """错误处理器"""
    
    def __init__(
        self,
        retry_config: RetryConfig = None,
        on_error: Optional[Callable[[ErrorInfo], None]] = None,
        on_retry: Optional[Callable[[ErrorInfo, int], None]] = None,
        on_fallback: Optional[Callable[[ErrorInfo, Any], None]] = None,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.on_error = on_error
        self.on_retry = on_retry
        self.on_fallback = on_fallback
        self._fallback_handlers: Dict[ErrorType, Callable] = {}
    
    def register_fallback(self, error_type: ErrorType, handler: Callable):
        """注册回退处理器"""
        self._fallback_handlers[error_type] = handler
    
    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """带重试的执行"""
        state = RetryState()
        
        while True:
            try:
                return await self._call_func(func, *args, **kwargs)
            except Exception as e:
                error_info = self._classify_error(e)
                error_info.retry_count = state.attempt
                
                if self.on_error:
                    self.on_error(error_info)
                
                if not self._should_retry(error_info, state):
                    if error_info.recoverable and error_info.type in self._fallback_handlers:
                        fallback_result = await self._execute_fallback(error_info, *args, **kwargs)
                        if fallback_result is not None:
                            return fallback_result
                    raise
                
                state.attempt += 1
                state.last_error = error_info
                
                delay = self._calculate_delay(state.attempt, error_info)
                state.total_delay += delay
                
                if self.on_retry:
                    self.on_retry(error_info, state.attempt)
                
                await asyncio.sleep(delay)
    
    async def _call_func(self, func: Callable, *args, **kwargs):
        """调用函数"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _classify_error(self, error: Exception) -> ErrorInfo:
        """分类错误"""
        if isinstance(error, AgentError):
            return error.info
        
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            return ErrorInfo(
                type=ErrorType.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                original_error=error,
                recoverable=True,
            )
        
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorInfo(
                type=ErrorType.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                original_error=error,
                recoverable=True,
            )
        
        if "network" in error_str or "connection" in error_str or "econnrefused" in error_str:
            return ErrorInfo(
                type=ErrorType.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                original_error=error,
                recoverable=True,
            )
        
        if "auth" in error_str or "unauthorized" in error_str or "401" in error_str or "403" in error_str:
            return ErrorInfo(
                type=ErrorType.AUTH,
                severity=ErrorSeverity.HIGH,
                message=str(error),
                original_error=error,
                recoverable=False,
            )
        
        if "context" in error_str or "token" in error_str or "length" in error_str:
            return ErrorInfo(
                type=ErrorType.CONTEXT_OVERFLOW,
                severity=ErrorSeverity.HIGH,
                message=str(error),
                original_error=error,
                recoverable=True,
            )
        
        return ErrorInfo(
            type=ErrorType.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            message=str(error),
            original_error=error,
            recoverable=True,
        )
    
    def _should_retry(self, error_info: ErrorInfo, state: RetryState) -> bool:
        """判断是否应该重试"""
        if not error_info.recoverable:
            return False
        
        if state.attempt >= self.retry_config.max_retries:
            return False
        
        if error_info.type not in self.retry_config.retryable_errors:
            return False
        
        return True
    
    def _calculate_delay(self, attempt: int, error_info: ErrorInfo) -> float:
        """计算重试延迟"""
        if error_info.type == ErrorType.RATE_LIMIT:
            retry_after = error_info.context.get("retry_after", 60)
            return min(retry_after, self.retry_config.max_delay)
        
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** (attempt - 1))
        delay = min(delay, self.retry_config.max_delay)
        
        if self.retry_config.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay
    
    async def _execute_fallback(self, error_info: ErrorInfo, *args, **kwargs):
        """执行回退策略"""
        handler = self._fallback_handlers.get(error_info.type)
        if not handler:
            return None
        
        try:
            result = await self._call_func(handler, error_info, *args, **kwargs)
            if self.on_fallback:
                self.on_fallback(error_info, result)
            return result
        except Exception:
            return None


class RecoveryStrategy(ABC):
    """恢复策略基类"""
    
    @abstractmethod
    async def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Any:
        pass


class ContextCompactionStrategy(RecoveryStrategy):
    """上下文压缩恢复策略"""
    
    def __init__(self, compactor):
        self.compactor = compactor
    
    async def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Any:
        if error_info.type != ErrorType.CONTEXT_OVERFLOW:
            return None
        
        messages = context.get("messages", [])
        if not messages:
            return None
        
        compacted = await self.compactor.compact(messages)
        return {"messages": compacted, "compacted": True}


class ToolFallbackStrategy(RecoveryStrategy):
    """工具回退策略"""
    
    def __init__(self, fallback_tools: Dict[str, str]):
        self.fallback_tools = fallback_tools
    
    async def recover(self, error_info: ErrorInfo, context: Dict[str, Any]) -> Any:
        if error_info.type != ErrorType.TOOL:
            return None
        
        tool_name = context.get("tool_name")
        fallback = self.fallback_tools.get(tool_name)
        
        if fallback:
            return {"use_tool": fallback}


class ErrorAggregator:
    """错误聚合器"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.errors: List[ErrorInfo] = []
    
    def add(self, error_info: ErrorInfo):
        """添加错误"""
        self.errors.append(error_info)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        if not self.errors:
            return 0.0
        return len(self.errors) / self.window_size
    
    def get_error_types(self) -> Dict[ErrorType, int]:
        """获取错误类型统计"""
        counts = {}
        for error in self.errors:
            counts[error.type] = counts.get(error.type, 0) + 1
        return counts
    
    def should_circuit_break(self, threshold: float = 0.5) -> bool:
        """判断是否应该熔断"""
        return self.get_error_rate() >= threshold
    
    def clear(self):
        """清空错误记录"""
        self.errors = []
