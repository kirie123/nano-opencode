"""
TaskTool - 完整的子 Agent 调用实现

支持：
- 子会话创建和管理
- 权限继承和限制
- 子 Agent 循环执行
- 任务恢复
"""

import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from message import Message, MessageRole, generate_id
from tools import Tool, ToolContext, ToolResult
from agent import AgentConfig, AgentMode, agent_registry
from permission import PermissionEvaluator, PermissionAction
from session import Session, SessionManager


TASK_TOOL_DESCRIPTION = """
委派任务给专门的子 Agent 执行。

子 Agent 是专门化的 Agent，可以并行执行特定类型的任务。使用子 Agent 可以：
1. 并行执行多个独立任务
2. 使用专门化的工具集
3. 隔离任务执行环境

可用的子 Agent：
{agents}

使用场景：
- 需要并行执行多个独立任务时
- 需要使用特定权限的工具时
- 需要隔离任务执行环境时

注意事项：
- 子 Agent 有独立的权限限制
- 子 Agent 不能调用其他子 Agent（除非明确允许）
- 子 Agent 的结果会返回给主 Agent
"""


@dataclass
class TaskParameters:
    """任务参数"""
    description: str
    prompt: str
    subagent_type: str
    task_id: Optional[str] = None
    command: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "description": self.description,
            "prompt": self.prompt,
            "subagent_type": self.subagent_type,
        }
        if self.task_id:
            result["task_id"] = self.task_id
        if self.command:
            result["command"] = self.command
        return result


@dataclass
class SubtaskSession:
    """子任务会话"""
    id: str
    parent_session_id: str
    agent_name: str
    description: str
    status: str = "pending"
    messages: List[Message] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    time: Dict[str, int] = field(default_factory=lambda: {"created": int(time.time() * 1000)})
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_session_id": self.parent_session_id,
            "agent_name": self.agent_name,
            "description": self.description,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "time": self.time,
            "metadata": self.metadata,
        }


class SubtaskManager:
    """子任务管理器"""
    
    def __init__(self):
        self._subtasks: Dict[str, SubtaskSession] = {}
    
    def create(
        self,
        parent_session_id: str,
        agent_name: str,
        description: str,
    ) -> SubtaskSession:
        """创建子任务会话"""
        subtask_id = generate_id("subtask_")
        subtask = SubtaskSession(
            id=subtask_id,
            parent_session_id=parent_session_id,
            agent_name=agent_name,
            description=description,
        )
        self._subtasks[subtask_id] = subtask
        return subtask
    
    def get(self, subtask_id: str) -> Optional[SubtaskSession]:
        """获取子任务"""
        return self._subtasks.get(subtask_id)
    
    def update(self, subtask: SubtaskSession):
        """更新子任务"""
        self._subtasks[subtask.id] = subtask
    
    def list_by_parent(self, parent_session_id: str) -> List[SubtaskSession]:
        """列出父会话的所有子任务"""
        return [s for s in self._subtasks.values() if s.parent_session_id == parent_session_id]


subtask_manager = SubtaskManager()


class TaskTool(Tool):
    """任务工具 - 调用子 Agent"""
    
    name = "task"
    
    def __init__(self, llm_client=None, session_manager=None):
        self.llm_client = llm_client
        self.session_manager = session_manager or SessionManager()
        self._params = []
    
    def _init_params(self):
        pass
    
    @property
    def description(self) -> str:
        agents = agent_registry.list()
        subagents = [a for a in agents if a.mode == AgentMode.SUBAGENT]
        
        agent_list = "\n".join([
            f"- {a.name}: {a.description}"
            for a in subagents
        ])
        
        return TASK_TOOL_DESCRIPTION.format(agents=agent_list)
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "任务简短描述（3-5个词）",
                },
                "prompt": {
                    "type": "string",
                    "description": "子 Agent 需要执行的具体任务",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "要使用的子 Agent 类型",
                    "enum": [a.name for a in agent_registry.list() if a.mode == AgentMode.SUBAGENT],
                },
                "task_id": {
                    "type": "string",
                    "description": "用于恢复之前任务的 task_id（可选）",
                },
                "command": {
                    "type": "string",
                    "description": "触发此任务的命令（可选）",
                },
            },
            "required": ["description", "prompt", "subagent_type"],
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """获取 JSON Schema 格式的工具定义"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        """执行任务"""
        params = TaskParameters(
            description=args.get("description", ""),
            prompt=args.get("prompt", ""),
            subagent_type=args.get("subagent_type", ""),
            task_id=args.get("task_id"),
            command=args.get("command"),
        )
        
        if not params.description or not params.prompt or not params.subagent_type:
            return ToolResult(
                output="错误：缺少必要参数。需要 description、prompt 和 subagent_type。",
                error="Missing required parameters",
            )
        
        if context.extra and context.extra.get("bypass_agent_check"):
            pass
        else:
            agent_config = agent_registry.get(context.agent) if context.agent else None
            rules = agent_config.permission_rules if agent_config else []
            evaluator = PermissionEvaluator(rules)
            permission_result = evaluator.evaluate(
                permission="task",
                pattern=params.subagent_type,
            )
            
            if permission_result.action == PermissionAction.DENY:
                return ToolResult(
                    output=f"错误：没有权限调用子 Agent '{params.subagent_type}'",
                    error="Permission denied",
                )
            
            if permission_result.action == PermissionAction.ASK:
                if context.ask_permission:
                    approved = await context.ask_permission(
                        permission="task",
                        patterns=[params.subagent_type],
                        metadata={
                            "description": params.description,
                            "subagent_type": params.subagent_type,
                        },
                    )
                    if not approved:
                        return ToolResult(
                            output=f"用户拒绝了调用子 Agent '{params.subagent_type}' 的请求",
                            error="Permission rejected",
                        )
        
        agent = agent_registry.get(params.subagent_type)
        if not agent:
            return ToolResult(
                output=f"错误：未知的子 Agent 类型 '{params.subagent_type}'",
                error="Unknown agent type",
            )
        
        if agent.mode != AgentMode.SUBAGENT:
            return ToolResult(
                output=f"错误：'{params.subagent_type}' 不是子 Agent，不能通过 task 工具调用",
                error="Not a subagent",
            )
        
        subtask = None
        if params.task_id:
            subtask = subtask_manager.get(params.task_id)
        
        if not subtask:
            subtask = subtask_manager.create(
                parent_session_id=context.session_id,
                agent_name=agent.name,
                description=params.description,
            )
        
        subtask.status = "running"
        subtask_manager.update(subtask)
        
        try:
            result = await self._run_subagent(
                agent=agent,
                prompt=params.prompt,
                subtask=subtask,
                context=context,
            )
            
            subtask.status = "completed"
            subtask.result = result
            subtask.time["completed"] = int(time.time() * 1000)
            subtask_manager.update(subtask)
            
            output = self._format_result(subtask, result)
            
            return ToolResult(
                output=output,
                metadata={
                    "task_id": subtask.id,
                    "agent": agent.name,
                    "description": params.description,
                },
            )
            
        except Exception as e:
            subtask.status = "error"
            subtask.error = str(e)
            subtask.time["error"] = int(time.time() * 1000)
            subtask_manager.update(subtask)
            
            return ToolResult(
                output=f"子 Agent 执行失败: {str(e)}",
                error=str(e),
            )
    
    async def _run_subagent(
        self,
        agent: AgentConfig,
        prompt: str,
        subtask: SubtaskSession,
        context: ToolContext,
    ) -> str:
        """运行子 Agent"""
        from loop import AgentLoop
        
        has_task_permission = any(
            r.permission == "task" for r in agent.permission_rules
        )
        has_todowrite_permission = any(
            r.permission == "todowrite" for r in agent.permission_rules
        )
        
        sub_permission_rules = list(agent.permission_rules)
        
        if not has_todowrite_permission:
            from permission import PermissionRule, PermissionAction
            sub_permission_rules.append(PermissionRule(
                permission="todowrite",
                pattern="*",
                action=PermissionAction.DENY,
            ))
        
        if not has_task_permission:
            from permission import PermissionRule, PermissionAction
            sub_permission_rules.append(PermissionRule(
                permission="task",
                pattern="*",
                action=PermissionAction.DENY,
            ))
        
        sub_agent = AgentConfig(
            name=agent.name,
            description=agent.description,
            mode=agent.mode,
            permission_rules=sub_permission_rules,
            model=agent.model,
            temperature=agent.temperature,
            max_steps=agent.max_steps,
            prompt=agent.prompt,
        )
        
        loop = AgentLoop(
            agent=sub_agent,
            llm_client=self.llm_client,
            workspace=context.workspace,
            session_id=subtask.id,
        )
        
        result = await loop.run(prompt)
        
        return result.message.get_text_content() if result and result.message else ""
    
    def _format_result(self, subtask: SubtaskSession, result: str) -> str:
        """格式化结果"""
        lines = [
            f"task_id: {subtask.id} (可用于恢复此任务)",
            "",
            "<task_result>",
            result,
            "</task_result>",
        ]
        return "\n".join(lines)


def create_explore_agent() -> AgentConfig:
    """创建探索子 Agent"""
    from permission import PermissionRule, PermissionAction
    
    return AgentConfig(
        name="explore",
        description="快速探索代码库的专用 Agent。用于查找文件模式、搜索代码关键词或回答代码库相关问题。",
        mode=AgentMode.SUBAGENT,
        permission_rules=[
            PermissionRule(permission="read", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="glob", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="grep", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="bash", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="webfetch", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="websearch", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="task", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="todowrite", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="write", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="edit", pattern="*", action=PermissionAction.DENY),
        ],
        prompt="""你是代码库探索专家。你的任务是快速、高效地探索代码库并回答问题。

探索策略：
1. 首先使用 glob 了解项目结构
2. 使用 grep 搜索关键代码
3. 使用 read 深入查看具体文件
4. 综合信息给出清晰的答案

探索深度：
- quick: 基础搜索，快速回答
- medium: 中等深度，适度探索
- very thorough: 全面分析，多角度探索

请根据用户需求选择合适的探索深度，并给出准确的答案。""",
        max_steps=50,
    )


def create_general_agent() -> AgentConfig:
    """创建通用子 Agent"""
    from permission import PermissionRule, PermissionAction
    
    return AgentConfig(
        name="general",
        description="通用研究 Agent，用于研究复杂问题和执行多步骤任务。可以并行执行多个工作单元。",
        mode=AgentMode.SUBAGENT,
        permission_rules=[
            PermissionRule(permission="*", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="task", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="todowrite", pattern="*", action=PermissionAction.DENY),
        ],
        prompt="""你是通用研究助手。你的任务是研究复杂问题并执行多步骤任务。

工作方式：
1. 分析任务需求
2. 制定执行计划
3. 并行执行独立任务
4. 综合结果给出答案

你可以使用大部分工具，但不能：
- 调用其他子 Agent
- 修改待办事项列表

请高效地完成任务并给出全面的答案。""",
        max_steps=100,
    )


def create_analyze_agent() -> AgentConfig:
    """创建分析子 Agent"""
    from permission import PermissionRule, PermissionAction
    
    return AgentConfig(
        name="analyze",
        description="数据分析 Agent，用于分析文档、研报、数据文件等。擅长提取关键信息、总结分析和生成报告。",
        mode=AgentMode.SUBAGENT,
        permission_rules=[
            PermissionRule(permission="read", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="glob", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="grep", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="bash", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="pdf", pattern="*", action=PermissionAction.ALLOW),
            PermissionRule(permission="task", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="todowrite", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="write", pattern="*", action=PermissionAction.DENY),
            PermissionRule(permission="edit", pattern="*", action=PermissionAction.DENY),
        ],
        prompt="""你是数据分析专家。你的任务是分析文档、研报、数据文件并提取关键信息。

分析策略：
1. 首先使用 glob 了解文件结构
2. 使用 pdf 工具读取 PDF 文件
3. 使用 read 工具读取其他文件
4. 提取关键信息：日期、主题、评级、推荐标的等
5. 综合分析并给出结构化报告

输出格式：
- 使用清晰的标题和列表
- 按重要性排序信息
- 给出具体的投资建议和风险提示

请仔细分析并给出专业的结论。""",
        max_steps=100,
    )


def register_subagents():
    """注册子 Agent"""
    agent_registry.register(create_explore_agent())
    agent_registry.register(create_general_agent())
    agent_registry.register(create_analyze_agent())
