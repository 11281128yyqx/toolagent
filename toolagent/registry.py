"""工具注册中心：声明式注册 + JSON Schema 参数校验 + 异常降级。

设计要点
--------
1. 工具用装饰器注册，自动生成 OpenAI / DeepSeek 兼容的 ``tools`` 描述。
2. 执行前先校验参数（必填、类型、枚举），把"模型传错参"挡在业务函数之外。
3. 执行异常统一转成 ``{"ok": False, "error": ...}`` 回灌给模型，
   让模型自己决定重试或向用户解释，而不是让整个对话崩掉。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


class ToolParamError(Exception):
    """参数校验失败（模型调用参数不合法）。"""


class ToolExecError(Exception):
    """工具执行阶段的业务错误（如订单不存在、城市不支持）。"""


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any]

    # ------------------------------------------------------------------ 校验
    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """按 JSON Schema 子集校验并做温和的类型转换。

        模型常把数字写成 ``"42"``，这里做一次容错转换；
        但缺失必填项、类型无法转换、枚举越界都会直接抛 ToolParamError。
        """
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ToolParamError("参数必须是 JSON 对象")

        schema = self.parameters or {}
        props: Dict[str, Any] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])

        missing = [k for k in required if k not in args or args[k] in (None, "")]
        if missing:
            raise ToolParamError(f"缺少必填参数: {', '.join(missing)}")

        cleaned: Dict[str, Any] = {}
        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                # 未声明的参数直接忽略，避免污染业务函数签名
                continue
            cleaned[key] = self._coerce(key, value, spec)

        return cleaned

    @staticmethod
    def _coerce(key: str, value: Any, spec: Dict[str, Any]) -> Any:
        expect = spec.get("type", "string")

        if value is None:
            return value

        target = _TYPE_MAP.get(expect)
        if target is not None and not isinstance(value, target):
            # 仅对"字符串 -> 数字/布尔"做容错，其余视为类型错误
            if expect in ("number", "integer") and isinstance(value, str):
                try:
                    value = float(value) if expect == "number" else int(float(value))
                except ValueError:
                    raise ToolParamError(f"参数 {key} 需要 {expect}，收到 {value!r}")
            elif expect == "boolean" and isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes"):
                    value = True
                elif low in ("false", "0", "no"):
                    value = False
                else:
                    raise ToolParamError(f"参数 {key} 需要 boolean，收到 {value!r}")
            else:
                raise ToolParamError(f"参数 {key} 需要 {expect}，收到 {type(value).__name__}")

        enum = spec.get("enum")
        if enum and value not in enum:
            raise ToolParamError(f"参数 {key} 只能取 {enum}，收到 {value!r}")

        return value

    # ------------------------------------------------------------------ 执行
    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cleaned = self.validate(args)
        except ToolParamError as exc:
            return {"ok": False, "error": f"参数校验失败：{exc}"}

        try:
            return {"ok": True, "result": self.func(**cleaned)}
        except ToolExecError as exc:
            return {"ok": False, "error": str(exc)}
        except TypeError as exc:
            # 模型漏参 / 多参时，把签名问题说清楚，方便它自我修正
            return {"ok": False, "error": f"工具调用签名不匹配：{exc}"}
        except Exception as exc:  # noqa: BLE001 - 兜底，保证对话不中断
            return {"ok": False, "error": f"工具内部异常：{exc.__class__.__name__}: {exc}"}

    # ------------------------------------------------------------ 描述导出
    def to_spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表，同时负责把调用分发到正确的工具上。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ValueError(f"工具重复注册: {name}")
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                func=func,
            )
            return func

        return decorator

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        return self._tools[name]

    def specs(self) -> List[Dict[str, Any]]:
        return [tool.to_spec() for tool in self._tools.values()]

    def execute(self, name: str, raw_args: str | Dict[str, Any]) -> Dict[str, Any]:
        """执行工具。``raw_args`` 可以是 JSON 字符串（模型原始输出）。"""
        if name not in self._tools:
            return {"ok": False, "error": f"未知工具：{name}"}

        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"参数不是合法 JSON：{exc}"}

        return self._tools[name].run(raw_args)
