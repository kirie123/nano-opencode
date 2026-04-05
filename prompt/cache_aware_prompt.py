"""
缓存感知的 Prompt 构建器

核心设计原则（参考 Claude Code）：
1. 静态内容在前，动态内容在后
2. 使用边界标记分隔静态/动态部分
3. 工具定义保持稳定，不中途增删
4. 动态信息通过消息注入，而非修改 System Prompt

缓存原理：
- Prompt Caching 是前缀匹配机制
- 只要前缀一致，模型可复用已计算的状态
- 任何前缀变化都会导致缓存完全失效
"""

import hashlib
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


PROMPT_DIR = Path(__file__).parent / "prompt"


@dataclass
class PromptSection:
    """Prompt 区段"""
    content: str
    is_static: bool = True
    order: int = 0
    name: str = ""


@dataclass
class CacheInfo:
    """缓存信息"""
    prompt_hash: str
    tools_hash: str
    static_tokens: int
    dynamic_tokens: int
    cache_eligible: bool


class CacheAwarePromptBuilder:
    """
    缓存感知的 Prompt 构建器
    
    结构：
    [静态内容] + [工具定义] + [边界标记] + [动态内容]
    
    静态内容：永远不变，最高缓存命中率
    工具定义：保持稳定，不中途增删
    动态内容：每次请求可能变化，不缓存
    """
    
    DYNAMIC_BOUNDARY = "\n\n<SYSTEM_PROMPT_DYNAMIC_BOUNDARY>\n\n"
    
    def __init__(self):
        self._static_sections: List[PromptSection] = []
        self._dynamic_sections: List[PromptSection] = []
        self._last_prompt_hash: Optional[str] = None
        self._last_tools_hash: Optional[str] = None
        self._last_full_prompt: Optional[str] = None
        self._last_tools: Optional[List[Dict]] = None
        self._prompt_cache: Dict[str, str] = {}
    
    def add_static(self, content: str, order: int = 0, name: str = ""):
        """
        添加静态内容（可缓存）
        
        静态内容应该：
        - 永远不变
        - 放在最前面
        - 包含核心身份、行为约束等
        """
        if not content.strip():
            return
        
        self._static_sections.append(PromptSection(
            content=content.strip(),
            is_static=True,
            order=order,
            name=name,
        ))
        self._static_sections.sort(key=lambda x: (x.order, x.name))
    
    def add_dynamic(self, content: str, order: int = 0, name: str = ""):
        """
        添加动态内容（不缓存）
        
        动态内容：
        - 可能每次请求都变化
        - 放在边界标记之后
        - 包含时间、状态等
        """
        if not content.strip():
            return
        
        self._dynamic_sections.append(PromptSection(
            content=content.strip(),
            is_static=False,
            order=order,
            name=name,
        ))
        self._dynamic_sections.sort(key=lambda x: (x.order, x.name))
    
    def clear_dynamic(self):
        """清空动态内容"""
        self._dynamic_sections = []
    
    def build(self, tools: List[Dict] = None) -> str:
        """
        构建完整的 System Prompt
        
        结构：
        1. 静态内容（按 order 排序）
        2. 工具定义
        3. 边界标记
        4. 动态内容（按 order 排序）
        """
        parts = []
        
        for section in self._static_sections:
            parts.append(section.content)
        
        if tools:
            parts.append(self._format_tools(tools))
        
        parts.append(self.DYNAMIC_BOUNDARY)
        
        for section in self._dynamic_sections:
            parts.append(section.content)
        
        full_prompt = "\n\n".join(parts)
        
        self._last_full_prompt = full_prompt
        self._last_tools = tools
        self._last_prompt_hash = self._compute_hash(full_prompt)
        if tools:
            self._last_tools_hash = self._compute_hash(str(tools))
        
        return full_prompt
    
    def _format_tools(self, tools: List[Dict]) -> str:
        """
        格式化工具定义
        
        工具定义应该保持稳定，格式一致
        """
        lines = ["<available_tools>"]
        
        sorted_tools = sorted(tools, key=lambda t: t.get("function", {}).get("name", ""))
        
        for tool in sorted_tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            
            lines.append(f"<tool name=\"{name}\">")
            lines.append(f"<description>{desc}</description>")
            
            properties = params.get("properties", {})
            required = params.get("required", [])
            
            if properties:
                lines.append("<parameters>")
                for param_name in sorted(properties.keys()):
                    param_info = properties[param_name]
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "any")
                    is_required = param_name in required
                    req_marker = "required" if is_required else "optional"
                    lines.append(f"  <param name=\"{param_name}\" type=\"{param_type}\" {req_marker}>")
                    lines.append(f"    {param_desc}")
                    lines.append(f"  </param>")
                lines.append("</parameters>")
            
            lines.append("</tool>")
        
        lines.append("</available_tools>")
        return "\n".join(lines)
    
    def _compute_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_cache_key(self) -> Tuple[Optional[str], Optional[str]]:
        """获取缓存键"""
        return (self._last_prompt_hash, self._last_tools_hash)
    
    def get_last_prompt(self) -> Optional[str]:
        """获取最后一次构建的 Prompt"""
        return self._last_full_prompt
    
    def get_last_tools(self) -> Optional[List[Dict]]:
        """获取最后一次使用的工具定义"""
        return self._last_tools
    
    def get_cache_info(self) -> CacheInfo:
        """获取缓存信息"""
        static_content = "\n\n".join(s.content for s in self._static_sections)
        dynamic_content = "\n\n".join(s.content for s in self._dynamic_sections)
        
        return CacheInfo(
            prompt_hash=self._last_prompt_hash or "",
            tools_hash=self._last_tools_hash or "",
            static_tokens=len(static_content) // 4,
            dynamic_tokens=len(dynamic_content) // 4,
            cache_eligible=len(self._static_sections) > 0,
        )
    
    @staticmethod
    def create_dynamic_injection(info_type: str, content: str) -> str:
        """
        创建动态信息注入消息
        
        不修改 System Prompt，而是生成一个消息注入
        这样可以保持缓存前缀不变
        """
        return f"""<system-reminder type="{info_type}">
{content}
</system-reminder>"""
    
    @staticmethod
    def create_step_reminder(current_step: int, max_steps: int) -> str:
        """创建步数提醒注入"""
        return CacheAwarePromptBuilder.create_dynamic_injection(
            "step_reminder",
            f"当前步骤: {current_step}/{max_steps}\n请在达到最大步数前完成任务。"
        )
    
    @staticmethod
    def create_time_injection() -> str:
        """创建时间注入"""
        return CacheAwarePromptBuilder.create_dynamic_injection(
            "time_update",
            f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    @staticmethod
    def create_state_injection(state: str, reason: str = "") -> str:
        """创建状态注入"""
        content = f"当前状态: {state}"
        if reason:
            content += f"\n原因: {reason}"
        return CacheAwarePromptBuilder.create_dynamic_injection("state_change", content)


class SystemPromptBuilder:
    """
    系统提示词构建器
    
    整合缓存感知构建器，提供完整的 System Prompt 构建功能
    """
    
    _prompt_cache: Dict[str, str] = {}
    
    @classmethod
    def _load_prompt_file(cls, name: str) -> str:
        """加载提示词文件"""
        if name in cls._prompt_cache:
            return cls._prompt_cache[name]
        
        path = PROMPT_DIR / f"{name}.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            cls._prompt_cache[name] = content
            return content
        
        return ""
    
    @classmethod
    def build_cache_aware(
        cls,
        model: str,
        workspace: str,
        agent: Any,
        tools: List[Dict] = None,
        custom_prompt: str = None,
    ) -> Tuple[CacheAwarePromptBuilder, str]:
        """
        构建缓存感知的 System Prompt
        
        返回: (builder, full_prompt)
        """
        builder = CacheAwarePromptBuilder()
        
        model_lower = model.lower()
        if any(x in model_lower for x in ["gpt-4", "o1-", "o3-"]):
            base_prompt = cls._load_prompt_file("beast")
        elif "gpt" in model_lower:
            base_prompt = cls._load_prompt_file("gpt")
        elif "gemini" in model_lower:
            base_prompt = cls._load_prompt_file("gemini")
        elif "claude" in model_lower:
            base_prompt = cls._load_prompt_file("anthropic")
        else:
            base_prompt = cls._load_prompt_file("default")
        
        if base_prompt:
            builder.add_static(base_prompt, order=0, name="base")
        
        if agent and agent.description:
            agent_section = f"""<agent_role>
Name: {agent.name}
Description: {agent.description}
</agent_role>"""
            builder.add_static(agent_section, order=1, name="agent_role")
        
        env_section = cls._build_environment_static(workspace, model)
        builder.add_static(env_section, order=2, name="environment")
        
        rules_section = cls._build_rules_section()
        builder.add_static(rules_section, order=3, name="rules")
        
        if custom_prompt:
            builder.add_dynamic(custom_prompt, order=0, name="custom")
        elif agent and agent.prompt:
            builder.add_dynamic(agent.prompt, order=0, name="agent_prompt")
        
        full_prompt = builder.build(tools)
        
        return builder, full_prompt
    
    @classmethod
    def _build_environment_static(cls, workspace: str, model: str) -> str:
        """构建静态环境信息（不包含时间等动态内容）"""
        import os
        
        git_repo = "是" if os.path.exists(os.path.join(workspace, ".git")) else "否"
        
        return f"""<environment>
Working directory: {workspace}
Workspace root: {workspace}
Is git repository: {git_repo}
Platform: {platform.system()}
Model: {model}
</environment>"""
    
    @classmethod
    def _build_rules_section(cls) -> str:
        """构建核心规则部分"""
        return """<core_rules>
1. Understand the task before acting
2. Use tools appropriately and efficiently
3. Provide clear explanations for your actions
4. Ask for clarification when needed
5. Verify your work before declaring completion
6. Follow the permission system constraints
</core_rules>"""
    
    @classmethod
    def build_dynamic_time_message(cls) -> str:
        """构建动态时间消息（用于注入，不修改 System Prompt）"""
        return CacheAwarePromptBuilder.create_dynamic_injection(
            "time",
            f"Current date: {time.strftime('%Y-%m-%d')}\nCurrent time: {time.strftime('%H:%M:%S')}"
        )
    
    @classmethod
    def build_compaction_request_message(cls) -> str:
        """构建压缩请求消息"""
        return """<compaction_request>
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
</compaction_request>"""
