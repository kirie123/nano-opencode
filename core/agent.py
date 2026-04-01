"""
Agent 系统：定义 AI Agent 的行为、权限和配置
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from pathlib import Path

from permission.permission import PermissionRule, PermissionAction


class AgentMode(Enum):
    PRIMARY = "primary"    # 主 Agent，用户直接交互
    SUBAGENT = "subagent"  # 子 Agent，被主 Agent 调用
    ALL = "all"           # 既是主又是子


@dataclass
class AgentConfig:
    """Agent 配置"""
    name: str
    description: str
    mode: AgentMode
    permission_rules: List[PermissionRule] = field(default_factory=list)
    model: Optional[str] = None  # 指定模型
    temperature: float = 0.7
    max_steps: int = 100
    prompt: Optional[str] = None  # 自定义系统提示
    prompt_file: Optional[str] = None  # 自定义提示词文件路径
    native: bool = True  # 是否内置
    hidden: bool = False  # 是否隐藏
    
    def get_prompt(self) -> Optional[str]:
        """获取提示词（优先从文件加载）"""
        if self.prompt_file:
            path = Path(self.prompt_file)
            if not path.is_absolute():
                path = Path.cwd() / self.prompt_file
            if path.exists():
                return path.read_text(encoding="utf-8")
        return self.prompt


class AgentRegistry:
    """Agent 注册表"""
    
    def __init__(self):
        self._agents: Dict[str, AgentConfig] = {}
        self._init_builtin_agents()
    
    def _init_builtin_agents(self):
        """初始化内置 Agent"""
        # Build Agent - 默认主 Agent
        self.register(AgentConfig(
            name="build",
            description="The default agent. Executes tools based on configured permissions.",
            mode=AgentMode.PRIMARY,
            permission_rules=[
                PermissionRule(permission="*", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="doom_loop", pattern="*", action=PermissionAction.ASK),
            ],
            native=True
        ))
        
        # Plan Agent - 计划模式
        self.register(AgentConfig(
            name="plan",
            description="Plan mode. Disallows all edit tools.",
            mode=AgentMode.PRIMARY,
            permission_rules=[
                PermissionRule(permission="*", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="edit", pattern="*", action=PermissionAction.DENY),
                PermissionRule(permission="write", pattern="*", action=PermissionAction.DENY),
                PermissionRule(permission="bash", pattern="*", action=PermissionAction.ASK),
            ],
            native=True
        ))
        
        # General Subagent - 通用子 Agent
        self.register(AgentConfig(
            name="general",
            description="General-purpose agent for researching complex questions",
            mode=AgentMode.SUBAGENT,
            permission_rules=[
                PermissionRule(permission="*", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="todoread", pattern="*", action=PermissionAction.DENY),
                PermissionRule(permission="todowrite", pattern="*", action=PermissionAction.DENY),
            ],
            native=True
        ))
        
        # Explore Subagent - 代码库探索
        self.register(AgentConfig(
            name="explore",
            description="Fast agent specialized for exploring codebases",
            mode=AgentMode.SUBAGENT,
            permission_rules=[
                PermissionRule(permission="read", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="grep", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="glob", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="list", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="bash", pattern="*", action=PermissionAction.ALLOW),
                PermissionRule(permission="*", pattern="*", action=PermissionAction.DENY),
            ],
            native=True
        ))
    
    def register(self, config: AgentConfig):
        """注册 Agent"""
        self._agents[config.name] = config
    
    def get(self, name: str) -> Optional[AgentConfig]:
        """获取 Agent 配置"""
        return self._agents.get(name)
    
    def list(self, mode: Optional[AgentMode] = None) -> List[AgentConfig]:
        """列出所有 Agent"""
        agents = list(self._agents.values())
        if mode:
            agents = [a for a in agents if a.mode == mode]
        return [a for a in agents if not a.hidden]
    
    def get_default(self) -> AgentConfig:
        """获取默认 Agent"""
        primary_agents = self.list(AgentMode.PRIMARY)
        if primary_agents:
            return primary_agents[0]
        raise ValueError("No primary agent found")


# 全局注册表实例
agent_registry = AgentRegistry()
