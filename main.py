"""命令行入口：跑一组示例问题，展示工具调用链路。

用法：
    python main.py            # 离线模式（无需 API Key，规则模拟器）
    python main.py --live     # 接入真实 DeepSeek（需 DEEPSEEK_API_KEY）
"""

from __future__ import annotations

import argparse
import json
import sys

from toolagent import Agent, build_llm, registry

DEMO_QUESTIONS = [
    "长春今天天气怎么样？",
    "120 公里换算成英里是多少",
    "帮我查一下订单 ORD-1001 到哪了",
    "A001 现在还有库存吗",
    "ORD-9999 这个订单什么状态",
]


def run(questions, live: bool = False) -> int:
    llm = build_llm() if live else build_llm(api_key="")
    if live:
        print(f"[模式] 在线 · {llm.name}")
    else:
        print("[模式] 离线（规则模拟器，设 DEEPSEEK_API_KEY 可切换真实模型）")

    agent = Agent(llm=llm, registry=registry)
    print(f"[工具] 已注册 {len(registry)} 个：{', '.join(agent.tool_names)}\n")

    total_calls = 0
    for index, question in enumerate(questions, 1):
        print("=" * 64)
        print(f"[{index}] 用户：{question}")
        answer = agent.chat(question)
        print(f"    助手：{answer}")

        if agent.trace:
            total_calls += len(agent.trace)
            for item in agent.trace:
                args = item["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                flag = "OK " if item["ok"] else "ERR"
                print(f"    ↳ [{flag}] {item['tool']}({json.dumps(args, ensure_ascii=False)})")
        else:
            print("    ↳ （本轮未调用工具）")
        print()

    print("=" * 64)
    print(f"共 {len(questions)} 轮对话，工具调用 {total_calls} 次。上下文消息数：{len(agent.messages)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ToolAgent demo")
    parser.add_argument("--live", action="store_true", help="使用真实 DeepSeek API（需 DEEPSEEK_API_KEY）")
    parser.add_argument("--ask", type=str, default=None, help="只问一个问题")
    args = parser.parse_args()

    questions = [args.ask] if args.ask else DEMO_QUESTIONS
    return run(questions, live=args.live)


if __name__ == "__main__":
    sys.exit(main())
