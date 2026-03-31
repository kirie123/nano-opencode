"""
工具系统：定义和执行各种工具
"""

import os
import re
import json
import subprocess
import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from pathlib import Path
import asyncio


@dataclass
class ToolResult:
    """工具执行结果"""
    output: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


class Tool(ABC):
    """工具基类"""
    
    def __init__(self):
        self.id = self.__class__.__name__.lower().replace('tool', '')
        self._params: List[ToolParameter] = []
        self._init_params()
    
    @abstractmethod
    def _init_params(self):
        """初始化参数定义"""
        pass
    
    @abstractmethod
    async def execute(self, args: Dict[str, Any], context: 'ToolContext') -> ToolResult:
        """执行工具"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取 JSON Schema 格式的工具定义"""
        properties = {}
        required = []
        
        for param in self._params:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self._get_description(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def _get_description(self) -> str:
        """获取工具描述"""
        return self.__doc__ or f"Execute {self.id} tool"
    
    def _validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        """验证参数"""
        for param in self._params:
            if param.required and param.name not in args:
                return f"Missing required parameter: {param.name}"
            if param.name in args:
                value = args[param.name]
                if param.enum and value not in param.enum:
                    return f"Invalid value for {param.name}: must be one of {param.enum}"
        return None


@dataclass
class ToolContext:
    """工具执行上下文"""
    session_id: str
    message_id: str
    agent: str
    cwd: str  # 当前工作目录
    workspace_root: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    on_metadata: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    on_ask: Optional[Callable[[str, Dict[str, Any]], Awaitable[bool]]] = None


class ReadTool(Tool):
    """Read the contents of a file"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("file_path", "string", "The absolute path to the file to read"),
            ToolParameter("offset", "number", "Line number to start reading from (1-indexed)", False, default=1),
            ToolParameter("limit", "number", "Maximum number of lines to read", False, default=None),
        ]
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        error = self._validate_args(args)
        if error:
            return ToolResult("", error=error, success=False)
        
        file_path = args["file_path"]
        offset = args.get("offset", 1) - 1  # 转为 0-indexed
        limit = args.get("limit")
        
        try:
            if not os.path.exists(file_path):
                return ToolResult("", error=f"File not found: {file_path}", success=False)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 应用 offset 和 limit
            end = offset + limit if limit else len(lines)
            selected_lines = lines[offset:end]
            
            # 添加行号
            result_lines = []
            for i, line in enumerate(selected_lines, start=offset+1):
                result_lines.append(f"{i:4} │ {line.rstrip()}")
            
            content = '\n'.join(result_lines)
            
            return ToolResult(
                output=content,
                title=f"Read {file_path} ({len(selected_lines)} lines)",
                metadata={
                    "file_path": file_path,
                    "total_lines": len(lines),
                    "read_lines": len(selected_lines)
                }
            )
            
        except Exception as e:
            return ToolResult("", error=str(e), success=False)


class WriteTool(Tool):
    """Create or overwrite a file with content"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("file_path", "string", "The absolute path to the file to write"),
            ToolParameter("content", "string", "The content to write to the file"),
        ]
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        error = self._validate_args(args)
        if error:
            return ToolResult("", error=error, success=False)
        
        file_path = args["file_path"]
        content = args["content"]
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(
                output=f"Successfully wrote {len(content)} characters to {file_path}",
                title=f"Write {file_path}",
                metadata={
                    "file_path": file_path,
                    "content_length": len(content)
                }
            )
            
        except Exception as e:
            return ToolResult("", error=str(e), success=False)


class PDFTool(Tool):
    """Read and extract text content from a PDF file"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("file_path", "string", "The absolute path to the PDF file to read"),
            ToolParameter("start_page", "number", "Page number to start reading from (1-indexed)", False, default=1),
            ToolParameter("end_page", "number", "Page number to end reading at (inclusive)", False, default=None),
        ]
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        error = self._validate_args(args)
        if error:
            return ToolResult("", error=error, success=False)
        
        file_path = args["file_path"]
        start_page = args.get("start_page", 1) - 1
        end_page = args.get("end_page")
        
        try:
            if not os.path.exists(file_path):
                return ToolResult("", error=f"File not found: {file_path}", success=False)
            
            if not file_path.lower().endswith('.pdf'):
                return ToolResult("", error=f"File is not a PDF: {file_path}", success=False)
            
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            if start_page >= total_pages:
                return ToolResult("", error=f"Start page {start_page + 1} exceeds total pages {total_pages}", success=False)
            
            actual_end = min(end_page, total_pages) if end_page else total_pages
            
            extracted_text = []
            for page_num in range(start_page, actual_end):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{text}")
            
            content = '\n\n'.join(extracted_text)
            
            if not content.strip():
                content = "[PDF contains no extractable text - may be scanned images]"
            
            return ToolResult(
                output=content,
                title=f"Read PDF {file_path} ({actual_end - start_page} pages)",
                metadata={
                    "file_path": file_path,
                    "total_pages": total_pages,
                    "pages_read": actual_end - start_page,
                    "start_page": start_page + 1,
                    "end_page": actual_end,
                }
            )
            
        except ImportError:
            return ToolResult("", error="pypdf not installed. Run: pip install pypdf", success=False)
        except Exception as e:
            return ToolResult("", error=f"Failed to read PDF: {str(e)}", success=False)


class EditTool(Tool):
    """Edit specific lines in a file using SEARCH/REPLACE blocks"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("file_path", "string", "The absolute path to the file to edit"),
            ToolParameter("old_string", "string", "The exact string to search for"),
            ToolParameter("new_string", "string", "The replacement string"),
        ]
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        error = self._validate_args(args)
        if error:
            return ToolResult("", error=error, success=False)
        
        file_path = args["file_path"]
        old_string = args["old_string"]
        new_string = args["new_string"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_string not in content:
                return ToolResult(
                    "", 
                    error=f"Could not find the specified string in {file_path}",
                    success=False
                )
            
            # 替换第一次出现
            new_content = content.replace(old_string, new_string, 1)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # 生成差异摘要
            old_lines = old_string.split('\n')
            new_lines = new_string.split('\n')
            
            return ToolResult(
                output=f"Successfully edited {file_path}",
                title=f"Edit {file_path}",
                metadata={
                    "file_path": file_path,
                    "old_lines": len(old_lines),
                    "new_lines": len(new_lines)
                }
            )
            
        except Exception as e:
            return ToolResult("", error=str(e), success=False)


class GlobTool(Tool):
    """Find files matching a pattern using glob"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("pattern", "string", "The glob pattern to match (e.g., '**/*.py')"),
            ToolParameter("path", "string", "The directory to search in", False, default="."),
            ToolParameter("limit", "number", "Maximum number of results", False, default=100),
        ]
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        error = self._validate_args(args)
        if error:
            return ToolResult("", error=error, success=False)
        
        pattern = args["pattern"]
        path = args.get("path", ".")
        limit = args.get("limit", 100)
        
        try:
            import glob as glob_module
            
            search_path = os.path.join(path, pattern)
            matches = glob_module.glob(search_path, recursive=True)
            
            # 限制结果数量
            matches = matches[:limit]
            
            # 过滤掉目录
            files = [m for m in matches if os.path.isfile(m)]
            
            output = f"Found {len(files)} files:\n" + "\n".join(files)
            
            return ToolResult(
                output=output,
                title=f"Glob: {pattern}",
                metadata={
                    "pattern": pattern,
                    "path": path,
                    "total": len(files)
                }
            )
            
        except Exception as e:
            return ToolResult("", error=str(e), success=False)


class GrepTool(Tool):
    """Search for patterns in file contents"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("pattern", "string", "The regex pattern to search for"),
            ToolParameter("path", "string", "The directory or file to search in", False, default="."),
            ToolParameter("glob", "string", "File pattern to limit search (e.g., '*.py')", False),
            ToolParameter("limit", "number", "Maximum number of results", False, default=50),
        ]
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        error = self._validate_args(args)
        if error:
            return ToolResult("", error=error, success=False)
        
        pattern = args["pattern"]
        path = args.get("path", ".")
        glob_pattern = args.get("glob")
        limit = args.get("limit", 50)
        
        try:
            import re
            
            results = []
            regex = re.compile(pattern, re.IGNORECASE)
            
            if os.path.isfile(path):
                # 搜索单个文件
                files = [path]
            else:
                # 递归搜索目录
                files = []
                for root, _, filenames in os.walk(path):
                    for filename in filenames:
                        if glob_pattern and not fnmatch.fnmatch(filename, glob_pattern):
                            continue
                        files.append(os.path.join(root, filename))
            
            # 搜索文件内容
            for filepath in files:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_no, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append({
                                    'file': filepath,
                                    'line': line_no,
                                    'content': line.strip()[:100]
                                })
                                if len(results) >= limit:
                                    break
                            if len(results) >= limit:
                                break
                except Exception:
                    continue
                if len(results) >= limit:
                    break
            
            # 格式化输出
            output_lines = [f"Found {len(results)} matches for '{pattern}':"]
            for r in results:
                output_lines.append(f"{r['file']}:{r['line']}: {r['content']}")
            
            return ToolResult(
                output="\n".join(output_lines),
                title=f"Grep: {pattern}",
                metadata={
                    "pattern": pattern,
                    "matches": len(results)
                }
            )
            
        except Exception as e:
            return ToolResult("", error=str(e), success=False)


class BashTool(Tool):
    """Execute bash commands"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("command", "string", "The bash command to execute"),
            ToolParameter("timeout", "number", "Timeout in seconds", False, default=30),
            ToolParameter("cwd", "string", "Working directory", False),
        ]
    
    def _convert_to_windows_command(self, command: str) -> str:
        """将 Linux 命令转换为 Windows PowerShell 命令"""
        if command.strip() == 'ls' or command.strip() == 'ls -la' or command.strip() == 'ls -l':
            return 'dir'
        
        if command.strip() == 'pwd':
            return 'cd'
        
        replacements = [
            ('ls -la', 'dir'),
            ('ls -l', 'dir'),
            ('ls ', 'dir '),
            ('ls$', 'dir'),
            ('pwd', 'cd'),
            ('cat ', 'type '),
            ('rm ', 'del '),
            ('rm -rf ', 'rmdir /s /q '),
            ('mkdir -p ', 'mkdir '),
            ('touch ', 'echo $null >> '),
            ('clear', 'cls'),
            ('which ', 'where '),
            ('grep ', 'findstr '),
            ('find .', 'dir /s /b'),
            ('chmod', 'icacls'),
            ('chown', 'icacls'),
            ('head -n ', 'select -First '),
            ('tail -n ', 'select -Last '),
            (' | head ', ' | select -First '),
            (' | tail ', ' | select -Last '),
            ('2>/dev/null', '2>$null'),
            ('/dev/null', '$null'),
        ]
        
        result = command
        for linux_cmd, windows_cmd in replacements:
            result = result.replace(linux_cmd, windows_cmd)
        
        return result
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        error = self._validate_args(args)
        if error:
            return ToolResult("", error=error, success=False)
        
        command = args["command"]
        timeout = args.get("timeout", 30)
        cwd = args.get("cwd") or context.cwd
        
        if os.name == 'nt':
            command = self._convert_to_windows_command(command)
        
        try:
            if os.name == 'nt':
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    "",
                    error=f"Command timed out after {timeout} seconds",
                    success=False
                )
            
            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')
            
            if error_output:
                output += f"\n[stderr]:\n{error_output}"
            
            return ToolResult(
                output=output,
                title=f"$ {command[:50]}{'...' if len(command) > 50 else ''}",
                metadata={
                    "command": command,
                    "exit_code": process.returncode,
                    "cwd": cwd
                }
            )
            
        except Exception as e:
            return ToolResult("", error=str(e), success=False)


class TaskTool(Tool):
    """Dispatch a task to a subagent"""
    
    def _init_params(self):
        self._params = [
            ToolParameter("description", "string", "A short (3-5 words) description of the task"),
            ToolParameter("prompt", "string", "The detailed task for the subagent"),
            ToolParameter("subagent_type", "string", "The subagent to use (general, explore, analyze)"),
            ToolParameter("task_id", "string", "Task ID for resuming (optional)", False),
        ]
    
    async def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        # 子任务执行 - 在实际实现中会创建新的 Agent 会话
        # 这里返回一个占位结果
        description = args.get("description", "Task")
        prompt = args.get("prompt", "")
        subagent_type = args.get("subagent_type", "general")
        
        return ToolResult(
            output=f"[Task dispatched to {subagent_type} subagent]\nDescription: {description}\n\nThis is a placeholder. In full implementation, this would spawn a new agent session.",
            title=description,
            metadata={
                "subagent": subagent_type,
                "prompt_preview": prompt[:100] if prompt else ""
            }
        )


# 工具注册表
class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._init_builtin_tools()
    
    def _init_builtin_tools(self):
        """初始化内置工具"""
        tools = [
            ReadTool(),
            WriteTool(),
            EditTool(),
            GlobTool(),
            GrepTool(),
            BashTool(),
            PDFTool(),
            TaskTool(),
        ]
        for tool in tools:
            self.register(tool)
    
    def register(self, tool: Tool):
        """注册工具"""
        self._tools[tool.id] = tool
    
    def get(self, tool_id: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(tool_id)
    
    def list(self) -> List[Tool]:
        """列出所有工具"""
        return list(self._tools.values())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的 JSON Schema"""
        return [tool.get_schema() for tool in self._tools.values()]


# 全局工具注册表
tool_registry = ToolRegistry()
