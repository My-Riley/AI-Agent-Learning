"""
LangChain Agent ReAct 行动框架示例（基于通义 ChatTongyi）

本示例对应课件中关于「ReAct 思考-行动-观察」框架的图片代码，重点演示：

1. 如何定义可供 Agent 调用的工具（获取体重 / 身高）
2. 如何在 `system_prompt` 中显式约束 Agent 必须按照「思考 → 行动 → 观察 → 再思考」的流程解决问题
3. 如何结合 `agent.stream(..., stream_mode="values")` 观察 ReAct 框架下的思考过程与工具调用

ReAct 核心概念回顾：
- Thought（思考）：模型用自然语言分析当前信息、规划下一步
- Action（行动）：模型决定调用哪个工具，以及调用参数
- Observation（观察）：接收工具返回结果，并基于结果进行下一轮思考
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
    - 使用 qwen-max 作为聊天模型，适合 Agent + Tool + ReAct 场景
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


@tool(description="获取体重，返回值是整数，单位千克（示例工具）")
def get_weight() -> int:
    """
    获取用户体重（示例工具）。

    为了专注演示 ReAct 流程，这里直接返回固定体重 90kg。
    你可以根据需要改为从实际输入或系统中获取。
    """

    return 90


@tool(description="获取身高，返回值是整数，单位厘米（示例工具）")
def get_height() -> int:
    """
    获取用户身高（示例工具）。

    与课件截图一致，这里返回固定身高 172cm。
    """

    return 172


def create_react_agent() -> Any:
    """
    创建一个遵循 ReAct 思考-行动-观察流程的 Agent 智能体。

    该 Agent：
    - 底层模型：ChatTongyi（qwen-max）
    - 工具列表：[get_weight, get_height]
    - system_prompt：明确要求模型：
      - 必须按「思考 → 行动 → 观察 → 再思考」的结构解决问题
      - 每轮最多只调用 1 个工具，不能一次调用多个工具
      - 思考时要简要说明为什么需要该工具，以及观察到的结果如何影响下一步
    """
    model = init_chat_model()

    system_prompt = """
你是严格遵循 ReAct 框架的智能体，必须按「思考→行动→观察→再思考」的流程解决问题：

1. 思考（Thought）：用简短中文说明你当前要解决什么子问题、准备如何做
2. 行动（Action）：如果需要外部信息，只能调用一个合适的工具，并给出工具名称与参数
3. 观察（Observation）：接收工具返回结果，描述你从中得到的关键信息
4. 再思考（Thought）：基于新的信息继续推理，直至得到最终答案

约束要求：
- 每轮最多只能调用 1 个工具，禁止单次调用多个工具
- 在作出最终回答前，至少展示一次完整的「思考→行动→观察→再思考」过程
- 请用清晰的中文解释你在每一步的理由，让用户能看懂你的 ReAct 过程
""".strip()

    agent = create_agent(
        model=model,
        tools=[get_weight, get_height],
        system_prompt=system_prompt,
    )

    return agent


def pretty_print_latest_message(latest_message: BaseMessage, chunk_index: int) -> None:
    """
    根据消息类型，打印当前 chunk 中最新一条消息，并标注其在 ReAct 中的角色。

    - 如果 latest_message.content 存在：通常对应 ReAct 中的 Thought / 最终 Answer
    - 如果 latest_message.tool_calls 存在：通常对应 ReAct 中的 Action（调用工具）
    - 工具返回值会以新的消息形式出现在 messages 中，可视为 Observation
    """
    content = getattr(latest_message, "content", None)
    tool_calls = getattr(latest_message, "tool_calls", None)

    prefix = f"[chunk {chunk_index:02d}]"
    msg_type = type(latest_message).__name__

    if tool_calls:
        # 将本条消息视作 ReAct 中的「Action」
        names: List[str] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                names.append(str(tc.get("name") or tc.get("tool", "unknown_tool")))
            else:
                names.append(str(getattr(tc, "name", "unknown_tool")))
        print(f"{prefix} {msg_type}（Action 行动）调用工具: {names}")
    elif content:
        # 简单基于内容判断是 Thought/Observation/Answer，这里只做教学演示
        label = "Thought 思考"
        if "结果" in str(content) or "BMI" in str(content):
            label = "Answer 最终回答"
        elif "返回" in str(content) or "工具" in str(content):
            label = "Observation 观察"
        print(f"{prefix} {msg_type}（{label}）: {content}")
    else:
        print(f"{prefix} {msg_type}: (无 content/tool_calls，可能是内部控制消息)")


def stream_react_process(agent: Any, user_question: str) -> None:
    """
    使用 `agent.stream(..., stream_mode="values")` 观察 ReAct 思考-行动-观察全过程。

    关键点：
    - stream_mode="values"：每个 chunk 都是“当前完整状态”
    - chunk["messages"][-1]：为最新追加的那条消息，通常是本轮 Thought / Action / Observation / Answer
    - 通过 `pretty_print_latest_message` 我们可以直观看到 ReAct 的执行轨迹
    """
    print("=" * 80)
    print("【示例】LangChain Agent ReAct 行动框架（计算 BMI）")
    print("=" * 80)
    print(f"用户问题：{user_question}")
    print("-" * 80)

    start_time = time.time()

    stream: Iterable[dict] = agent.stream(
        input={
            "messages": [
                {"role": "user", "content": user_question},
            ]
        },
        stream_mode="values",
    )

    chunk_count = 0
    seen_len = 0

    for chunk in stream:
        chunk_count += 1
        messages: List[BaseMessage] = chunk.get("messages", [])
        if not messages:
            continue

        new_messages: List[BaseMessage] = messages[seen_len:]
        seen_len = len(messages)
        if not new_messages:
            continue

        for msg in new_messages:
            pretty_print_latest_message(msg, chunk_index=chunk_count)

    elapsed_time = time.time() - start_time
    print("-" * 80)
    print(f"共接收到 {chunk_count} 个 chunk，耗时 {elapsed_time:.2f} 秒")
    print("=" * 80)
    print()


def main() -> None:
    """
    入口函数：演示 LangChain Agent 在 ReAct 框架下计算 BMI 的完整过程。
    """
    print("=" * 80)
    print("LangChain Agent ReAct 行动框架示例（基于通义 ChatTongyi）")
    print("=" * 80)
    print()

    agent = create_react_agent()

    # 与课件截图一致的问题：让 Agent 通过调用工具获取身高体重并计算 BMI
    user_question = "计算我的BMI，并给出是否正常的结论。"

    stream_react_process(agent, user_question)

    print("=" * 80)
    print("示例执行完毕，你可以修改工具实现或 system_prompt，进一步体验 ReAct。")
    print("=" * 80)


if __name__ == "__main__":
    main()

