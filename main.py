import os
from dotenv import load_dotenv
from agents import Runner

from agent import agent

from harness import AgentHarness





load_dotenv(override=True)


def main():

    harness = AgentHarness(agent)
    print("Agent Harness 已启动")
    print("输入 exit 退出")


    while True:
            user_input = input("\n你：")

            if user_input.lower() == "exit":
                print("程序退出")
                break

            result = harness.run(user_input)

            if result["status"] == "success":
                print("\nAgent：", result["output"])
                print(f"耗时：{result['latency']:.2f}s")
                print(f"Run ID：{result['run_id']}")

            else:
                print("\n执行失败")
                print("错误：", result["error"])


if __name__ == "__main__":
    main()