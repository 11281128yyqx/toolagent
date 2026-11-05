"""工具注册与参数校验的单元测试。"""

import unittest

from toolagent.registry import ToolExecError, ToolParamError, ToolRegistry


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register(
        name="add",
        description="两数相加",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "integer"},
                "mode": {"type": "string", "enum": ["fast", "safe"]},
            },
            "required": ["a", "b"],
        },
    )
    def add(a, b, mode="safe"):
        if mode == "fast" and b < 0:
            raise ToolExecError("fast 模式不支持负数")
        return a + b

    @reg.register(
        name="boom",
        description="必定抛异常",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def boom():
        raise RuntimeError("模拟内部故障")

    return reg


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = build_registry()

    def test_specs_format(self):
        specs = self.reg.specs()
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]["type"], "function")
        self.assertEqual(specs[0]["function"]["name"], "add")

    def test_ok(self):
        out = self.reg.execute("add", '{"a": 1, "b": 2}')
        self.assertTrue(out["ok"])
        self.assertEqual(out["result"], 3)

    def test_missing_required(self):
        out = self.reg.execute("add", {"a": 1})
        self.assertFalse(out["ok"])
        self.assertIn("缺少必填参数", out["error"])

    def test_type_coercion(self):
        # 模型常把数字写成字符串，这里应能容错
        out = self.reg.execute("add", '{"a": "1.5", "b": "2"}')
        self.assertTrue(out["ok"])
        self.assertEqual(out["result"], 3.5)

    def test_bad_type(self):
        out = self.reg.execute("add", {"a": "abc", "b": 2})
        self.assertFalse(out["ok"])
        self.assertIn("参数校验失败", out["error"])

    def test_enum_violation(self):
        out = self.reg.execute("add", {"a": 1, "b": 2, "mode": "turbo"})
        self.assertFalse(out["ok"])
        self.assertIn("只能取", out["error"])

    def test_unknown_tool(self):
        out = self.reg.execute("not_exist", "{}")
        self.assertFalse(out["ok"])
        self.assertIn("未知工具", out["error"])

    def test_invalid_json(self):
        out = self.reg.execute("add", "{a:1}")
        self.assertFalse(out["ok"])
        self.assertIn("不是合法 JSON", out["error"])

    def test_business_error_degraded(self):
        out = self.reg.execute("add", {"a": 1, "b": -2, "mode": "fast"})
        self.assertFalse(out["ok"])
        self.assertIn("fast 模式不支持负数", out["error"])

    def test_unexpected_exception_degraded(self):
        out = self.reg.execute("boom", {})
        self.assertFalse(out["ok"])
        self.assertIn("工具内部异常", out["error"])

    def test_duplicate_register(self):
        with self.assertRaises(ValueError):
            self.reg.register(name="add", description="重复")(lambda: None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
