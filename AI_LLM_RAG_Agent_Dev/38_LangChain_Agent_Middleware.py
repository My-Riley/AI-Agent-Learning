"""
LangChain Agent 中间件（Middleware）示例（基于通义 ChatTongyi）

本示例对应课件中关于「中间件」的图片代码，重点演示：

1. 中间件的作用：对智能体的每一步工作进行控制和自定义的执行
2. 节点式钩子（执行点顺序拦截）：
   - before_agent: agent 执行之前拦截
   - after_agent: agent 执行后拦截
   - before_model: 模型执行前拦截
   - after_model: 模型执行后拦截
3. 包装式钩子（针对工具和模型）：
   - wrap_model_call: 每个模型调用时候拦截
   - wrap_tool_call: 每个工具调用时候拦截
4. 中间件的应用场景：
   - 日志记录、分析、调试
   - 转换提示词、工具选择
   - 重试、备用、提前终止等逻辑控制
   - 安全防护、个人身份检测等

核心概念：
- 中间件通过 Hooks 钩子来实现拦截
- 自定义中间件可以简单的使用装饰器来定义
- LangChain 中内置了一些基础的中间件，参见：
  https://docs.langchain.com/oss/python/langchain/middleware/built-in
"""

import os
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langgraph.runtime import Runtime
from langchain_community.chat_models.tongyi import ChatTongyi
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


@tool(description="查询天气")
def get_weather() -> str:
    """
    一个简单的天气查询工具。

    说明：
    - 为了演示中间件功能，这里返回固定值"晴天"
    - 实际项目中可以改为调用真实天气 API
    """
    return "晴天"


# ============================================================================
# 节点式钩子（执行点顺序拦截）
# ============================================================================


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    """
    Agent 执行之前的拦截钩子。

    应用场景：
    - 记录 Agent 启动日志
    - 验证输入参数
    - 修改初始状态
    """
    print(f"[before_agent] Starting agent with {len(state['messages'])} messages")


@after_agent
def log_completion(state: AgentState, runtime: Runtime) -> None:
    """
    Agent 执行之后的拦截钩子。

    应用场景：
    - 记录 Agent 完成日志
    - 统计执行时间
    - 保存对话历史
    """
    print(f"[after_agent] Agent completed with {len(state['messages'])} messages")


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    """
    模型执行之前的拦截钩子。

    应用场景：
    - 记录模型调用日志
    - 转换提示词
    - 添加系统提示
    """
    print(f"[before_model] About to call model with {len(state['messages'])} messages")


@after_model
def log_latest_message(state: AgentState, runtime: Runtime) -> None:
    """
    模型执行之后的拦截钩子。

    应用场景：
    - 记录模型返回结果
    - 分析模型输出
    - 修改模型响应
    """
    if state["messages"]:
        print(f"[after_model] {state['messages'][-1].content}")


# ============================================================================
# 包装式钩子（针对工具和模型）
# ============================================================================


@wrap_model_call
def retry_on_error(request: Any, handler: Callable) -> Any:
    """
    模型调用的包装钩子，实现重试逻辑。

    应用场景：
    - 自动重试失败的模型调用
    - 实现降级策略
    - 添加超时控制
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print("[wrap_model_call]")
            return handler(request)
        except Exception as e:
            if attempt == max_retries - 1:
                # 最后一次尝试失败，抛出异常
                raise e
            print(f"[wrap_model_call] Retry {attempt + 1}/{max_retries} due to: {e}")


@wrap_tool_call
def monitor_tool(
    request: Any,
    handler: Callable[[Any], Any],
) -> Any:
    """
    工具调用的包装钩子，实现监控和日志记录。

    应用场景：
    - 记录工具调用日志
    - 监控工具执行时间
    - 实现工具调用的权限控制
    - 添加工具调用的错误处理
    """
    print(f"[wrap_tool_call] Executing tool: {request.tool_call['name']}")
    print(f"[wrap_tool_call] Arguments: {request.tool_call['args']}")

    try:
        result = handler(request)
        print(f"[wrap_tool_call] Tool completed successfully")
        return result
    except Exception as e:
        print(f"[wrap_tool_call] Tool failed: {e}")
        raise


def create_agent_with_middleware() -> Any:
    """
    创建一个带有中间件的 Agent 智能体。

    说明：
    - middleware 参数接收一个中间件函数列表
    - 中间件的执行顺序与列表中的顺序相关
    - 不同类型的钩子会在不同的执行点被调用
    """
    model = init_chat_model()

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[
            monitor_tool,  # 工具调用监控
            retry_on_error,  # 模型调用重试
            log_latest_message,  # 模型执行后日志
            log_before_model,  # 模型执行前日志
            log_completion,  # Agent 执行后日志
            log_before_agent,  # Agent 执行前日志
        ],
    )

    return agent


def invoke_agent_with_middleware(agent: Any, user_question: str) -> None:
    """
    调用带有中间件的 Agent，并打印结果。

    说明：
    - 中间件会在 Agent 执行的各个阶段被调用
    - 可以通过打印输出观察中间件的执行顺序
    """
    print("=" * 80)
    print("【示例】LangChain Agent 中间件（Middleware）")
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

    print("-" * 80)
    print("最终结果：")
    print("**********")
    print(res)
    print("=" * 80)
    print()


def main() -> None:
    """
    入口函数：演示 LangChain Agent 中间件的完整功能。
    """
    print("=" * 80)
    print("LangChain Agent 中间件（Middleware）示例（基于通义 ChatTongyi）")
    print("=" * 80)
    print()
    print("中间件的作用：对智能体的每一步工作进行控制和自定义的执行")
    print()
    print("节点式钩子（执行点顺序拦截）：")
    print("  - before_agent: agent 执行之前拦截")
    print("  - after_agent: agent 执行后拦截")
    print("  - before_model: 模型执行前拦截")
    print("  - after_model: 模型执行后拦截")
    print()
    print("包装式钩子（针对工具和模型）：")
    print("  - wrap_model_call: 每个模型调用时候拦截")
    print("  - wrap_tool_call: 每个工具调用时候拦截")
    print()
    print("应用场景：")
    print("  - 日志记录、分析、调试")
    print("  - 转换提示词、工具选择")
    print("  - 重试、备用、提前终止等逻辑控制")
    print("  - 安全防护、个人身份检测等")
    print("=" * 80)
    print()

    # 创建带有中间件的 Agent
    agent = create_agent_with_middleware()

    # 调用 Agent，观察中间件的执行过程
    user_question = "今天天气如何呀，如何穿衣"
    invoke_agent_with_middleware(agent, user_question)

    print("=" * 80)
    print("示例执行完毕，你可以根据需要修改中间件逻辑，实现更多自定义功能。")
    print("=" * 80)


if __name__ == "__main__":
    main()
