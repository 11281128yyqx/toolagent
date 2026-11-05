"""ToolAgent —— 基于 Function Calling 的任务型 AI 助手（工具注册 / 参数校验 / 异常降级）。"""

from .agent import Agent
from .llm import DeepSeekClient, MockLLM, build_llm
from .registry import ToolExecError, ToolParamError, ToolRegistry, Tool
from .tools import registry

__all__ = [
    "Agent",
    "Tool",
    "ToolRegistry",
    "ToolParamError",
    "ToolExecError",
    "DeepSeekClient",
    "MockLLM",
    "build_llm",
    "registry",
]

__version__ = "1.0.0"
