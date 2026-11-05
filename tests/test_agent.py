"""Agent 主循环的端到端测试（使用离线规则模拟器，不依赖网络与 API Key）。"""

import unittest

from toolagent import Agent, MockLLM, registry


def new_agent() -> Agent:
    return Agent(llm=MockLLM(), registry=registry)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = new_agent()

    def test_weather_route(self):
        answer = self.agent.chat("长春今天天气怎么样？")
        self.assertEqual(self.agent.last_tool_used(), "get_weather")
        self.assertIn("长春", answer)
        self.assertIn("℃", answer)

    def test_unit_conversion(self):
        answer = self.agent.chat("120 公里换算成英里是多少")
        self.assertEqual(self.agent.last_tool_used(), "convert_unit")
        self.assertIn("英里", answer)
        self.assertIn("74.5", answer)  # 120 / 1.609344 ≈ 74.5645

    def test_order_query(self):
        answer = self.agent.chat("帮我查一下订单 ORD-1001 到哪了")
        self.assertEqual(self.agent.last_tool_used(), "query_order")
        self.assertIn("已发货", answer)

    def test_unknown_order_degrades(self):
        answer = self.agent.chat("ORD-9999 这个订单什么状态")
        self.assertEqual(self.agent.last_tool_used(), "query_order")
        self.assertIn("没有成功", answer)
        self.assertIn("不存在", answer)

    def test_no_tool_needed(self):
        answer = self.agent.chat("你好")
        self.assertIsNone(self.agent.last_tool_used())
        self.assertIn("天气", answer)

    def test_context_accumulates(self):
        self.agent.chat("长春天气")
        self.agent.chat("北京天气")
        roles = [m["role"] for m in self.agent.messages]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles.count("user"), 2)
        self.assertIn("tool", roles)

    def test_reset(self):
        self.agent.chat("长春天气")
        self.agent.reset()
        self.assertEqual(len(self.agent.messages), 1)
        self.assertIsNone(self.agent.last_tool_used())

    def test_llm_failure_degrades(self):
        class BrokenLLM:
            name = "broken"

            def chat(self, messages, tools=None):
                raise RuntimeError("连接超时")

        agent = Agent(llm=BrokenLLM(), registry=registry)
        answer = agent.chat("长春天气")
        self.assertIn("模型服务暂时不可用", answer)
        self.assertIn("连接超时", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
