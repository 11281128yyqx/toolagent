"""LLM 适配层：真实 DeepSeek 客户端 + 离线可用的规则模拟器。

Agent 只依赖 ``chat(messages, tools) -> message`` 这一个接口，
因此"有没有 API Key"不影响主流程验证——这正是简历里"异常降级"的落点。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

Message = Dict[str, Any]


# --------------------------------------------------------------------------
# 真实调用
# --------------------------------------------------------------------------
class DeepSeekClient:
    """DeepSeek Chat Completions（OpenAI 兼容协议）最小客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 30,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"deepseek:{model}"

    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> Message:
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:200]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                # 4xx（鉴权/参数）不重试，5xx 与限流才退避重试
                if 400 <= exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = RuntimeError(f"网络异常: {exc}")
            except (KeyError, json.JSONDecodeError) as exc:
                last_error = RuntimeError(f"响应解析失败: {exc}")
                break

            if attempt < self.max_retries:
                time.sleep(0.6 * (2**attempt))

        raise RuntimeError(f"DeepSeek 调用失败（重试 {self.max_retries} 次）：{last_error}")


# --------------------------------------------------------------------------
# 离线模拟
# --------------------------------------------------------------------------
_SUPPORTED_CITIES = ["长春", "北京", "上海", "深圳"]
_UNIT_PATTERN = r"(公里|千米|米|厘米|英里|英尺|斤|公斤|千克|克|磅)"
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*" + _UNIT_PATTERN + r"\s*(?:换算成|换成|转成|等于|是多少|合)\s*" + _UNIT_PATTERN)
_SKU_RE = re.compile(r"\b([A-Z]\d{3})\b")
_ORDER_RE = re.compile(r"(ORD-\d{4})", re.IGNORECASE)


class MockLLM:
    """规则驱动的假模型：只做两件事——决定调哪个工具、把结果说成人话。

    它的存在是为了让 Function Calling 的**链路**（路由→校验→执行→回灌→总结）
    在没有 API Key 的机器上也能跑通并被单元测试覆盖。
    """

    name = "mock:rule-based"

    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None) -> Message:
        last = messages[-1] if messages else {}

        # 工具结果回灌后，生成最终回答
        if last.get("role") == "tool":
            return {"role": "assistant", "content": self._finalize(last)}

        user_text = self._last_user_text(messages)
        call = self._route(user_text)
        if call is None:
            return {
                "role": "assistant",
                "content": "我可以帮你查天气、换算单位、查订单和库存。你可以说得具体一点，比如「长春天气怎么样」。",
            }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{int(time.time() * 1000) % 100000}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
            ],
        }

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _last_user_text(messages: List[Message]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    def _route(self, text: str) -> Optional[Dict[str, Any]]:
        order = _ORDER_RE.search(text)
        if order:
            return {"name": "query_order", "arguments": {"order_id": order.group(1).upper()}}

        if "天气" in text:
            city = next((c for c in _SUPPORTED_CITIES if c in text), None)
            if city is None:
                return {"name": "get_weather", "arguments": {"city": "长春"}}
            return {"name": "get_weather", "arguments": {"city": city}}

        unit = _UNIT_RE.search(text)
        if unit:
            return {
                "name": "convert_unit",
                "arguments": {
                    "value": float(unit.group(1)),
                    "from_unit": unit.group(2),
                    "to_unit": unit.group(3),
                },
            }

        if "库存" in text:
            sku = _SKU_RE.search(text)
            return {"name": "get_stock", "arguments": {"sku": sku.group(1) if sku else "A001"}}

        return None

    @staticmethod
    def _finalize(tool_message: Message) -> str:
        raw = tool_message.get("content") or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return "工具返回格式异常，我先记下这次失败，稍后再试。"

        if not payload.get("ok"):
            return f"这次查询没有成功：{payload.get('error', '未知错误')}。你可以换个订单号或城市再试一次。"

        result = payload.get("result") or {}
        name = payload.get("tool")

        if name == "get_weather":
            return (
                f"{result['city']}当前{result['condition']}，气温 {result['temp_c']}℃，"
                f"湿度 {result['humidity']}%，{result['wind']}。"
            )
        if name == "convert_unit":
            return (
                f"{result['value']:g}{result['from_unit']} = "
                f"{result['result']:g}{result['to_unit']}。"
            )
        if name == "query_order":
            return (
                f"订单 {result['order_id']} 当前状态：{result['status']}，"
                f"商品 {result['sku']} × {result['qty']}，金额 ¥{result['amount']:.2f}，"
                f"预计送达 {result['eta']}。"
            )
        if name == "get_stock":
            return f"{result['sku']} 当前库存 {result['stock']} 件。"

        return json.dumps(result, ensure_ascii=False)


def build_llm(api_key: Optional[str] = None) -> Any:
    """有 Key 用真模型，没有就退回模拟器（保证 demo 永远能跑）。"""
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    return DeepSeekClient(api_key=key) if key else MockLLM()
