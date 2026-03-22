"""
LangChain Agent 智能体流式输出示例（基于通义 ChatTongyi）

本示例对应课件中关于「Agent.stream 流式输出」的图片代码，重点演示：

1. 如何创建一个可以调用工具（查询股票信息）的 Agent 智能体
2. 如何使用 `agent.stream(..., stream_mode="values")` 持续接收增量消息
3. 如何从每个 `chunk` 中取出最新一条消息，并根据是「普通回复」还是「工具调用」做不同处理

核心概念回顾：
- invoke / 调用：一次性得到完整结果（上一节 `35_LangChain_Agent_First_Experience.py` 已演示）
- stream / 流式：连续收到多个结果块（chunk），可以一边生成、一边展示
- stream_mode="values"：每个 chunk 都是一个「完整状态快照」，其中 `messages` 字段包含当前为止的所有消息
"""

import os
import time
from typing import Any, Iterable, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool


def init_chat_model() -> ChatTongyi:
    """
    初始化 ChatTongyi 聊天模型实例。

    说明：
    - 与项目中其他 Tongyi 示例保持一致，优先从以下环境变量中读取密钥：
      1. DASHSCOPE_API_KEY（阿里云官方推荐）
      2. API_KEY（与本项目其他示例兼容）
    - 使用 qwen-max 作为聊天模型，适合 Agent + Tool 场景
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


@tool(description="获取股票价格，传入股票名称，返回字符串信息（示例工具）")
def get_price(name: str) -> str:
    """
    获取股票价格（示例工具）。

    为了方便演示，这里直接返回一个固定文案，不调用真实行情接口。
    """

    return f"股票{name}的价格是20元"


@tool(description="获取股票信息，传入股票名称，返回字符串信息（示例工具）")
def get_info(name: str) -> str:
    """
    获取股票基本信息（示例工具）。

    同样使用固定文案，方便你把注意力放在「Agent 调用工具 + 流式输出」流程上。
    """

    return f"股票{name}，是一家A股上市公司，专注于IT职业教育。"


def create_stock_agent() -> Any:
    """
    创建一个可以回答股票相关问题的 Agent 智能体。

    该 Agent：
    - 底层模型：ChatTongyi（qwen-max）
    - 工具列表：[get_price, get_info]
    - system_prompt：引导模型在需要时调用工具，并解释调用原因
    """
    model = init_chat_model()

    agent = create_agent(
        model=model,
        tools=[get_price, get_info],
        system_prompt=(
            "你是一个智能助手，可以回答股票相关问题。"
            "在需要查询价格或公司简介时，请合理调用对应工具。"
            "在思考和调用工具时，请用简短中文解释你的思路。"
        ),
    )

    return agent


def pretty_print_latest_message(latest_message: BaseMessage, chunk_index: int) -> None:
    """
    根据消息类型，打印当前 chunk 中最新一条消息的关键信息。

    逻辑与课件截图大体一致：
    - 如果 latest_message.content 存在：认为是 Agent 的自然语言回复
    - 否则如果 latest_message.tool_calls 存在：认为是在发起工具调用
    """
    # 消息内容（普通自然语言回复）
    content = getattr(latest_message, "content", None)

    # 工具调用（如 Function / ToolCall）
    tool_calls = getattr(latest_message, "tool_calls", None)

    prefix = f"[chunk {chunk_index:02d}]"
    msg_type = type(latest_message).__name__

    if content:
        print(f"{prefix} {msg_type}: {content}")
    elif tool_calls:
        # tool_calls 可能是字典列表，也可能是包含 name 属性的对象列表
        names: List[str] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                names.append(str(tc.get("name") or tc.get("tool", "unknown_tool")))
            else:
                names.append(str(getattr(tc, "name", "unknown_tool")))
        print(f"{prefix} {msg_type} 调用工具: {names}")
    else:
        # 某些内部消息可能没有 content / tool_calls，这里仅做类型展示，方便调试
        print(f"{prefix} {msg_type}: (无 content/tool_calls，可能是内部控制消息)")


def stream_agent_messages(agent: Any, user_question: str) -> None:
    """
    使用 `agent.stream(..., stream_mode="values")` 流式打印 Agent 的思考过程。

    重点说明：
    - stream_mode="values"：每个 chunk 都包含当前时刻的完整状态，我们只关心其中的 messages
    - chunk["messages"][-1]：永远是“最新追加的那条消息”
    - 通过判断 latest_message.content / latest_message.tool_calls，可以区分：
      - Agent 自然语言回复
      - Agent 准备调用 / 已调用的工具
    """
    print("=" * 80)
    print("【示例】LangChain Agent 智能体流式输出（基于股票查询工具）")
    print("=" * 80)
    print(f"用户问题：{user_question}")
    print("-" * 80)

    start_time = time.time()

    # 使用 stream 接口，注意显式传入 stream_mode="values"
    stream: Iterable[dict] = agent.stream(
        input={
            "messages": [
                {"role": "user", "content": user_question},
            ]
        },
        stream_mode="values",
    )

    final_answer_parts: List[str] = []
    chunk_count = 0
    # 记录当前已经“看过”的消息数量，避免在 values 模式下重复打印
    seen_len = 0

    for chunk in stream:
        chunk_count += 1
        messages: List[BaseMessage] = chunk.get("messages", [])
        if not messages:
            continue

        # 本次 chunk 新增的消息（相对于上一个 chunk）
        new_messages: List[BaseMessage] = messages[seen_len:]
        seen_len = len(messages)

        if not new_messages:
            continue

        for msg in new_messages:
            # 打印当前新增消息的大致含义
            pretty_print_latest_message(msg, chunk_index=chunk_count)

            # 如果是自然语言内容，则顺便拼接到最终答案中
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                final_answer_parts.append(content)

    elapsed_time = time.time() - start_time

    print("-" * 80)
    print(f"共接收到 {chunk_count} 个 chunk，耗时 {elapsed_time:.2f} 秒")
    print("\nAgent 最终整合后的回答（根据 content 粗略拼接，仅作演示）：")
    print("".join(final_answer_parts))
    print("=" * 80)
    print()


def main() -> None:
    """
    入口函数：演示 LangChain Agent 的流式输出用法。
    """
    print("=" * 80)
    print("LangChain Agent 流式输出示例（基于通义 ChatTongyi）")
    print("=" * 80)
    print()

    agent = create_stock_agent()

    # 与课件截图风格一致的问题：询问某只股票的价格和简介
    user_question = "传智教育股价多少，并介绍一下？"

    stream_agent_messages(agent, user_question)

    print("=" * 80)
    print("示例执行完毕，你可以修改工具逻辑或 system_prompt，探索更多 Agent 行为。")
    print("=" * 80)


if __name__ == "__main__":
    main()

