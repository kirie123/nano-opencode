"""
权限系统：控制工具的访问权限
"""

import fnmatch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class PermissionAction(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionRule:
    """权限规则"""
    permission: str  # 工具名或权限类型
    pattern: str = "*"   # 匹配模式
    action: PermissionAction = PermissionAction.ALLOW
    
    def matches(self, permission: str, pattern: str = "*") -> bool:
        """检查是否匹配"""
        perm_match = fnmatch.fnmatch(permission, self.permission)
        pattern_match = fnmatch.fnmatch(pattern, self.pattern)
        return perm_match and pattern_match


class PermissionEvaluator:
    """权限评估器"""
    
    def __init__(self, rules: List[PermissionRule] = None):
        self.rules: List[PermissionRule] = rules or []
    
    def evaluate(self, permission: str, pattern: str = "*") -> PermissionRule:
        """
        评估权限
        规则按优先级：后面的规则覆盖前面的
        """
        # 默认规则
        default_rule = PermissionRule("*", "*", PermissionAction.ASK)
        
        # 按顺序匹配，后面的覆盖前面的
        matched_rule = default_rule
        for rule in self.rules:
            if rule.matches(permission, pattern):
                matched_rule = rule
        
        return matched_rule
    
    def can_execute(self, permission: str, pattern: str = "*") -> tuple[bool, bool]:
        """
        检查是否可以执行
        返回: (can_execute, needs_ask)
        """
        rule = self.evaluate(permission, pattern)
        if rule.action == PermissionAction.DENY:
            return False, False
        if rule.action == PermissionAction.ASK:
            return False, True
        return True, False
    
    def add_rule(self, rule: PermissionRule):
        """添加规则"""
        self.rules.append(rule)
    
    def merge(self, other: 'PermissionEvaluator') -> 'PermissionEvaluator':
        """合并两个权限评估器"""
        merged = PermissionEvaluator(self.rules.copy())
        merged.rules.extend(other.rules)
        return merged


class PermissionManager:
    """权限管理器"""
    
    def __init__(self):
        self._session_permissions: Dict[str, PermissionEvaluator] = {}
        self._user_defaults = self._create_default_permissions()
    
    def _create_default_permissions(self) -> PermissionEvaluator:
        """创建默认权限"""
        evaluator = PermissionEvaluator()
        
        # 默认允许大部分工具
        evaluator.add_rule(PermissionRule("*", "*", PermissionAction.ALLOW))
        
        # 敏感操作需要询问
        evaluator.add_rule(PermissionRule("bash", "*", PermissionAction.ASK))
        evaluator.add_rule(PermissionRule("write", "*.env", PermissionAction.ASK))
        evaluator.add_rule(PermissionRule("write", "*.key", PermissionAction.ASK))
        
        # 死循环检测
        evaluator.add_rule(PermissionRule("doom_loop", "*", PermissionAction.ASK))
        
        return evaluator
    
    def get_evaluator(self, session_id: str, agent_permissions: List[PermissionRule] = None) -> PermissionEvaluator:
        """获取会话的权限评估器"""
        if session_id not in self._session_permissions:
            # 合并默认权限和 Agent 特定权限
            base = self._user_defaults
            if agent_permissions:
                agent_eval = PermissionEvaluator(agent_permissions)
                base = base.merge(agent_eval)
            self._session_permissions[session_id] = base
        
        return self._session_permissions[session_id]
    
    def clear_session(self, session_id: str):
        """清除会话权限"""
        if session_id in self._session_permissions:
            del self._session_permissions[session_id]


# 全局权限管理器
permission_manager = PermissionManager()
