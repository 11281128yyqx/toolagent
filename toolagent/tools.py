"""三个示例工具：天气查询、单位换算、订单查询。

说明
----
天气与订单使用**本地示例数据源**，目的是让 Function Calling 链路可以离线跑通、
单元测试可复现；真实接入时只需替换这两个函数的内部实现，注册方式与调用链路不变。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from .registry import ToolExecError, ToolRegistry

registry = ToolRegistry()


# --------------------------------------------------------------------------
# 1. 天气
# --------------------------------------------------------------------------
_WEATHER_DB: Dict[str, Dict[str, Any]] = {
    "长春": {"condition": "多云", "temp_c": 6, "humidity": 48, "wind": "西南风 3 级"},
    "北京": {"condition": "晴", "temp_c": 14, "humidity": 35, "wind": "北风 2 级"},
    "上海": {"condition": "小雨", "temp_c": 19, "humidity": 78, "wind": "东风 2 级"},
    "深圳": {"condition": "阴", "temp_c": 26, "humidity": 70, "wind": "南风 3 级"},
}

_SUPPORTED_CITIES = list(_WEATHER_DB)


@registry.register(
    name="get_weather",
    description="查询指定城市的当前天气（温度、天气状况、湿度、风力）。",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如：长春、北京、上海、深圳",
                "enum": _SUPPORTED_CITIES,
            }
        },
        "required": ["city"],
    },
)
def get_weather(city: str) -> Dict[str, Any]:
    data = _WEATHER_DB.get(city)
    if data is None:
        raise ToolExecError(f"暂不支持城市「{city}」，当前支持：{'、'.join(_SUPPORTED_CITIES)}")
    return {"city": city, **data}


# --------------------------------------------------------------------------
# 2. 单位换算
# --------------------------------------------------------------------------
_LENGTH = {"米": 1.0, "千米": 1000.0, "公里": 1000.0, "厘米": 0.01, "英里": 1609.344, "英尺": 0.3048}
_WEIGHT = {"克": 1.0, "千克": 1000.0, "公斤": 1000.0, "斤": 500.0, "磅": 453.59237}
_ALL_UNITS = {**_LENGTH, **_WEIGHT}


@registry.register(
    name="convert_unit",
    description="在长度或重量单位之间换算，例如公里/英里、斤/千克。",
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "待换算的数值"},
            "from_unit": {"type": "string", "description": "原单位，如：公里、英里、斤、千克"},
            "to_unit": {"type": "string", "description": "目标单位，如：公里、英里、斤、千克"},
        },
        "required": ["value", "from_unit", "to_unit"],
    },
)
def convert_unit(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    if from_unit not in _ALL_UNITS:
        raise ToolExecError(f"不支持的单位：{from_unit}")
    if to_unit not in _ALL_UNITS:
        raise ToolExecError(f"不支持的单位：{to_unit}")
    if (from_unit in _LENGTH) != (to_unit in _LENGTH):
        raise ToolExecError(f"不能在长度单位「{from_unit}」和重量单位「{to_unit}」之间换算")

    table = _LENGTH if from_unit in _LENGTH else _WEIGHT
    result = value * table[from_unit] / table[to_unit]
    return {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": round(result, 4),
    }


# --------------------------------------------------------------------------
# 3. 订单查询
# --------------------------------------------------------------------------
_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {"status": "已发货", "sku": "A001 机械键盘", "qty": 30, "amount": 8970.00, "eta": "2026-11-07"},
    "ORD-1002": {"status": "待付款", "sku": "A002 无线鼠标", "qty": 5, "amount": 495.00, "eta": "-"},
    "ORD-1003": {"status": "已完成", "sku": "B003 显示器支架", "qty": 2, "amount": 398.00, "eta": "已签收"},
}


@registry.register(
    name="query_order",
    description="按订单号查询订单状态、商品、数量、金额与预计送达时间。",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "订单号，格式如 ORD-1001",
            }
        },
        "required": ["order_id"],
    },
)
def query_order(order_id: str) -> Dict[str, Any]:
    key = order_id.strip().upper()
    order = _ORDERS.get(key)
    if order is None:
        raise ToolExecError(f"订单 {key} 不存在，请核对订单号")
    return {"order_id": key, **order}


# --------------------------------------------------------------------------
# 4. 演示用：一个必然抛异常的工具（用于验证异常降级链路）
# --------------------------------------------------------------------------
@registry.register(
    name="get_stock",
    description="按 SKU 查询库存（示例工具，用于演示异常处理）。",
    parameters={
        "type": "object",
        "properties": {"sku": {"type": "string", "description": "商品 SKU"}},
        "required": ["sku"],
    },
)
def get_stock(sku: str) -> Dict[str, Any]:
    # 用哈希保证同一 SKU 结果稳定，便于测试断言
    digest = int(hashlib.md5(sku.encode("utf-8")).hexdigest(), 16)
    if digest % 7 == 0:
        raise ToolExecError(f"库存服务暂时不可用（SKU={sku}）")
    return {"sku": sku, "stock": digest % 500}
