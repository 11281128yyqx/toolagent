"""Agent 主循环：模型决策 → 工具执行 → 结果回灌，直到产出最终回答。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry

SYSTEM_PROMPT = (
    "你是一个工具型助手。需要实时数据或计算时，优先调用已注册的工具，"
    "不要凭空编造数字。工具返回错误时，向用户说明原因并给出替代做法。"
)


class Agent:
    """把 LLM、工具注册表和多轮对话状态串起来。"""

    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        max_steps: int = 6,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self.trace: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ 主流程
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self.trace = []

        for step in range(1, self.max_steps + 1):
            try:
                message = self.llm.chat(self.messages, tools=self.registry.specs())
            except Exception as exc:  # noqa: BLE001 - 模型不可用时整体降级
                reply = f"模型服务暂时不可用（{exc}）。已保留本轮上下文，稍后可以重试。"
                self.messages.append({"role": "assistant", "content": reply})
                return reply

            tool_calls = message.get("tool_calls") or []

            # 没有工具调用 = 模型已经能回答
            if not tool_calls:
                content = message.get("content") or ""
                self.messages.append({"role": "assistant", "content": content})
                return content

            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                raw_args = function.get("arguments", "{}")
                outcome = self.registry.execute(name, raw_args)

                self.trace.append(
                    {
                        "step": step,
                        "tool": name,
                        "arguments": raw_args,
                        "ok": outcome["ok"],
                        "detail": outcome.get("result") or outcome.get("error"),
                    }
                )

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": name,
                        "content": json.dumps({"ok": outcome["ok"], "tool": name, **outcome}, ensure_ascii=False),
                    }
                )

        # 步数耗尽：不再让模型循环，直接兜底
        reply = "这个问题需要多次工具调用仍未收敛，我先停下来，建议把问题拆成更具体的一步。"
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------ 辅助
    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.trace = []

    @property
    def tool_names(self) -> List[str]:
        return [spec["function"]["name"] for spec in self.registry.specs()]

    def last_tool_used(self) -> Optional[str]:
        return self.trace[-1]["tool"] if self.trace else None
