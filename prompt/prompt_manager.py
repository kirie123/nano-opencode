"""
系统提示词管理

根据不同模型选择合适的提示词，并动态注入环境信息。
"""

import os
import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentConfig
    from .llm import LLMConfig

PROMPT_DIR = Path(__file__).parent / "prompt"


class SystemPrompt:
    """系统提示词管理器"""
    
    _cache: dict = {}
    
    @classmethod
    def _load_prompt(cls, name: str) -> str:
        """加载提示词文件"""
        if name in cls._cache:
            return cls._cache[name]
        
        path = PROMPT_DIR / f"{name}.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            cls._cache[name] = content
            return content
        
        return ""
    
    @classmethod
    def for_model(cls, model: str) -> List[str]:
        """
        根据模型选择提示词
        
        参数:
            model: 模型 ID（如 "gpt-4", "claude-3-opus", "gemini-pro"）
        
        返回:
            提示词内容列表
        """
        model_lower = model.lower()
        
        # GPT-4 / o1 / o3 高级模型
        if any(x in model_lower for x in ["gpt-4", "o1-", "o3-"]):
            return [cls._load_prompt("beast")]
        
        # GPT 系列
        if "gpt" in model_lower:
            if "codex" in model_lower:
                return [cls._load_prompt("codex")]
            return [cls._load_prompt("gpt")]
        
        # Gemini 系列
        if "gemini" in model_lower:
            return [cls._load_prompt("gemini")]
        
        # Claude 系列
        if "claude" in model_lower:
            return [cls._load_prompt("anthropic")]
        
        # Trinity
        if "trinity" in model_lower:
            return [cls._load_prompt("trinity")]
        
        # 默认
        return [cls._load_prompt("default")]
    
    @classmethod
    def environment(cls, workspace: str, model: str) -> str:
        """
        生成环境信息
        
        参数:
            workspace: 工作目录
            model: 模型 ID
        
        返回:
            环境信息文本
        """
        import platform
        
        lines = [
            f"你正在运行模型：{model}",
            "以下是关于你运行环境的有用信息：",
            "<env>",
            f"  工作目录: {workspace}",
            f"  工作区根目录: {workspace}",
            f"  是否为 git 仓库: {cls._is_git_repo(workspace)}",
            f"  平台: {platform.system()}",
            f"  今日日期: {time.strftime('%Y-%m-%d')}",
            "</env>",
        ]
        
        return "\n".join(lines)
    
    @classmethod
    def max_steps_reminder(cls, current_step: int, max_steps: int) -> str:
        """
        生成最大步数提醒
        
        参数:
            current_step: 当前步数
            max_steps: 最大步数
        
        返回:
            提醒文本
        """
        return f"""
关键 - 已达到最大步数

此任务允许的最大步数已接近。当前步数：{current_step}/{max_steps}。

严格要求：
1. 如果接近最大步数，请尽快总结工作并给出下一步建议
2. 如果已完成主要任务，请简洁地总结结果
3. 如果需要更多步数，请明确说明还需要做什么

此约束覆盖所有其他指令。
"""
    
    @classmethod
    def plan_mode(cls) -> str:
        """获取计划模式提示词"""
        return cls._load_prompt("plan")
    
    @classmethod
    def build(
        cls,
        model: str,
        workspace: str,
        agent: "AgentConfig",
        tools: List[dict] = None,
        custom_prompt: str = None
    ) -> str:
        """
        构建完整的系统提示词
        
        参数:
            model: 模型 ID
            workspace: 工作目录
            agent: Agent 配置
            tools: 可用工具列表
            custom_prompt: 自定义提示词（覆盖默认）
        
        返回:
            完整的系统提示词
        """
        sections = []
        
        # 1. 基础提示词（根据模型选择）
        base_prompts = cls.for_model(model)
        for prompt in base_prompts:
            if prompt:
                sections.append(prompt)
        
        # 2. 环境信息
        sections.append(cls.environment(workspace, model))
        
        # 3. Agent 角色定义
        if agent.description:
            sections.append(f"""
# 当前角色

你是 {agent.name}。
{agent.description}
""")
        
        # 4. 可用工具
        if tools:
            tool_descriptions = []
            for tool_schema in tools:
                func = tool_schema.get("function", {})
                name = func.get("name", "")
                desc = func.get("description", "")
                params = func.get("parameters", {})
                
                tool_descriptions.append(f"- {name}: {desc}")
                
                # 添加参数说明
                properties = params.get("properties", {})
                required = params.get("required", [])
                if properties:
                    for param_name, param_info in properties.items():
                        req_marker = "（必需）" if param_name in required else "（可选）"
                        param_desc = param_info.get("description", "")
                        tool_descriptions.append(f"  - {param_name}{req_marker}: {param_desc}")
            
            sections.append(f"""
# 可用工具

{chr(10).join(tool_descriptions)}

使用工具时，请提供适当的参数。
""")
        
        # 5. 自定义提示词
        if custom_prompt:
            sections.append(custom_prompt)
        elif agent.prompt:
            sections.append(agent.prompt)
        
        return "\n\n".join(sections)
    
    @staticmethod
    def _is_git_repo(path: str) -> str:
        """检查是否为 git 仓库"""
        git_dir = os.path.join(path, ".git")
        return "是" if os.path.exists(git_dir) else "否"
