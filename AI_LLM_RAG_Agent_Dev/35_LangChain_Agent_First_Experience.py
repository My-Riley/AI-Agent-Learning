"""
LangChain Agent 智能体初体验示例（基于通义 ChatTongyi）

本示例对应课件中关于「Agent 智能体」的图片，重点演示：

1. 如何使用 LangChain 定义一个最简单的工具（查询天气）
2. 如何将聊天模型（ChatTongyi）和工具组合成一个 Agent 智能体
3. 如何向 Agent 发送用户消息，并打印出 Agent 返回的消息列表
4. 如何配合 StrOutputParser，将消息对象统一解析为字符串输出

核心概念：
- 工具（Tool）：Agent 可以调用的函数能力，例如：查天气、查数据库、调用 API 等
- Agent：拥有「规划 + 调用工具 + 记忆」能力的智能体，本质上是对大模型的封装
- system_prompt：系统提示词，用来规定 Agent 的角色和行为规范
- messages：对话消息列表（role + content），与前面消息示例保持一致

为了让你快速“上手有感觉”，本示例刻意保持简单：
- 只定义 1 个工具：`get_weather`，永远返回“晴天”
- 不引入复杂的记忆、规划逻辑，只展示最基本的调用链路
- 重点放在：看懂 Agent 的输入 / 输出结构
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool


def init_chat_model() -> ChatTongyi:
    """
    初始化 ChatTongyi 聊天模型实例。

    说明：
    - 与项目中其他示例保持一致，优先从以下环境变量中读取密钥：
      1. DASHSCOPE_API_KEY（阿里云官方推荐）
      2. API_KEY（与本项目其他示例兼容）
    - 使用 qwen-max 作为聊天模型，适合 Agent 场景
    """
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    # LangChain 的 ChatTongyi 封装会自动从环境变量中读取 key，
    # 这里设置一份到 DASHSCOPE_API_KEY，确保兼容性。
    os.environ["DASHSCOPE_API_KEY"] = api_key

    chat = ChatTongyi(model="qwen-max")
    return chat


@tool(description="查询天气（示例工具，固定返回“晴天”）")
def get_weather() -> str:
    """
    一个极简的天气查询工具。

    说明：
    - 为了和课件中的示例保持一致，这里不接收任何参数
    - 实际项目中可以改写为：根据城市、日期等参数调用真实天气 API
    - 本示例只关心「工具调用流程」，不追求真实天气数据
    """

    return "晴天"


def create_weather_agent() -> any:
    """
    创建一个可以调用 `get_weather` 工具的 Agent 智能体。

    使用 LangChain 提供的 `create_agent` 辅助方法，直接将：
    - 聊天模型（ChatTongyi）
    - 工具列表（[get_weather]）
    - system_prompt（系统角色提示）
    组合成一个可调用的 Agent。
    """
    model = init_chat_model()

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个聊天助手，可以回答用户问题，并在需要时调用查询天气工具。",
    )

    return agent


def invoke_agent_and_print_messages(agent: any, user_question: str) -> None:
    """
    调用 Agent，并打印它返回的消息列表。

    说明：
    - Agent 的输入是一个字典，其中关键字段是 "messages"
    - "messages" 是一个列表，每个元素是一个 dict：
      {"role": "user" / "assistant" / "system", "content": "..."}
    - Agent 的输出同样包含 "messages"，其中记录了整个对话过程：
      例如：系统提示词、用户提问、Agent 思考过程、工具调用、最终回答等
    """
    print("=" * 80)
    print("【示例】LangChain Agent 智能体初体验：查询天气")
    print("=" * 80)
    print(f"用户问题：{user_question}")
    print("-" * 80)

    # 调用 Agent
    res = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": user_question},
            ]
        }
    )

    # Agent 返回的消息列表
    messages: List[BaseMessage] = res.get("messages", [])

    # 使用 StrOutputParser，将不同类型的消息统一解析为字符串，方便打印
    parser = StrOutputParser()

    print("Agent 返回的消息列表：")
    print("-" * 80)
    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__
        text = parser.invoke(msg)
        print(f"{i:02d}. {msg_type}: {text}")

    print("-" * 80)
    print("提示：")
    print("1. 上面的每一条消息都可能来自：系统提示词、用户输入、Agent 思考过程、工具调用或最终回答；")
    print("2. 通过查看消息列表，你可以清楚地看到 Agent 是如何一步步完成任务的。")
    print("=" * 80)
    print()


def main() -> None:
    """
    入口函数：演示 LangChain Agent 智能体的最小可用示例。
    """
    print("=" * 80)
    print("LangChain Agent 智能体初体验（基于通义 ChatTongyi）")
    print("=" * 80)
    print()

    # 1. 创建可以调用天气工具的 Agent
    agent = create_weather_agent()

    # 2. 构造一个和课件中类似的问题
    user_question = "明天深圳的天气如何呀？"

    # 3. 调用 Agent 并打印消息列表
    invoke_agent_and_print_messages(agent, user_question)

    print("=" * 80)
    print("示例执行完毕，可以根据需要修改工具逻辑，扩展更多能力。")
    print("=" * 80)


if __name__ == "__main__":
    main()

