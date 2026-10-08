"""B2 · 工具的实现在调另一个模型

学什么
    你已经写过"工具里调 Tavily"、"工具里调本地函数"。
    这一课是第三种：工具里再发起一次 LLM 调用。
    链路变成  模型 -> 工具 -> 模型，这是"多 Agent"最原始的形态。

    主模型负责"决策什么时候用这个工具"，
    工具内部的模型负责"用一套专用 prompt 完成具体子任务"，两者职责分离、prompt 互不干扰。

对应司南
    src/app/agent/sinan_agent.py:77    recognize_homework
      -> src/app/agent/tools/vision.py:71   recognize_photo
        -> glm-4v-flash 真的在识图

怎么跑
    python examples/b02_tool_calls_model.py
    （需要 .env 里的 DEEPSEEK_API_KEY）

观察点
    stream_mode="updates" 会分步吐出中间状态：
      ① 主模型决定调用 summarize_notes（你能看到 tool_calls）
      ② 工具内部静默地又跑了一次模型（外层只看得到最终文本）
      ③ 结果回到主模型，它再组织成人话

TODO（学完自己改）
    1) 再加一个 translate_to_english(text) 工具，内部也调模型
    2) 把 stream_mode 换成 "messages"，观察 token 是怎么逐块流出来的
    3) 想一想：工具内部那次调用失败了，应该抛异常还是返回一句"识别失败"？
       （提示：司南的 vision.py 选择抛异常，由外层决定降级 —— 为什么？）

       工具内部失败时，要让错误信号对主模型"可见且可区分"，而不是伪装成正常结果混进去
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

_MODEL = None


def get_model():
    """懒加载单例 —— 顺带复习 B10。"""
    global _MODEL
    if _MODEL is None:
        _MODEL = init_chat_model(model="deepseek-chat")
    return _MODEL


# ======================================================================
# 关键：工具的实现体里，再发起一次模型调用
# ======================================================================
@tool
def summarize_notes(text: str) -> str:
    """把一段学习笔记压缩成 3 条要点。

    当用户要求"总结 / 提炼 / 压缩"笔记内容时，使用本工具。

    Args:
        text: 原始笔记内容
    """
    response = get_model().invoke(
        [
            HumanMessage(
                content=(
                    "把下面的学习笔记压缩成 3 条要点，每条不超过 20 字，"
                    "只输出要点本身，不要任何开场白：\n\n" + text
                )
            )
        ]
    )
    content = response.content
    return content if isinstance(content, str) else str(content)


@tool
def translate_to_english(text: str) -> str:
    """把一段学习笔记翻译成英文。

    当用户要求"翻译"笔记内容时，使用本工具。

    Args:
        text: 原始笔记内容
    """
    response = get_model().invoke(
        [
            HumanMessage(
                content=(
                    "把下面的学习笔记翻译成英文：\n\n" + text
                )
            )
        ]
    )
    content = response.content
    return content if isinstance(content, str) else str(content)


NOTES = """
今天复习了极限这一章。等价无穷小替换要小心，只有乘除能用、加减不能随便换。
洛必达法则用了三次，发现自己一遇到 0/0 就条件反射洛必达，其实很多题泰勒展开更快。
泰勒公式记不牢，尤其是 sinx、ln(1+x) 的三阶展开容易写错系数。
晚上做了 15 道题，错了 4 道，其中 3 道是泰勒展开系数写错。
"""


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("跳过：未检测到 DEEPSEEK_API_KEY，请先在 .env 里配置。")
        return

    agent = create_agent(
        model=get_model(),
        tools=[summarize_notes, translate_to_english],
        system_prompt=(
            "你是学习助手。用户要求总结、提炼、压缩笔记时，"
            "必须调用 summarize_notes 工具，不要自己直接总结。"
            "用户要求翻译笔记时，必须调用 translate_to_english 工具。"
        ),
    )

    print("=" * 64)
    print("用户输入：帮我把今天的笔记总结一下")
    print("=" * 64)

    # ---- updates 模式解析（已注释，留作对比）----
    step_no = 0
    for step in agent.stream(
        {"messages": [HumanMessage(content="帮我把今天的笔记总结一下。并翻译为英文\n" + NOTES)]},
        stream_mode="updates",
    ):
        step_no += 1
        print(f"\n--- 第 {step_no} 步 ---")
        for node_name, update in step.items():
            msgs = (update or {}).get("messages", []) if isinstance(update, dict) else []
            print(f"  节点：{node_name}   产出 {len(msgs)} 条消息")
            for msg in msgs:
                if isinstance(msg, AIMessage):
                    calls = getattr(msg, "tool_calls", None)
                    if calls:
                        for c in calls:
                            print(f"    [主模型] 决定调用工具：{c['name']}")
                            print(f"             参数：{str(c['args'])[:80]}...")
                    elif msg.content:
                        print(f"    [主模型] 最终回答：{str(msg.content)[:200]}")
                elif isinstance(msg, ToolMessage):
                    print(f"    [工具返回] {str(msg.content)[:160]}...")

    # ---- messages 模式解析：观察 token 逐块流出 ----
    """step_no = 0
    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content="帮我把今天的笔记总结一下。并翻译为英文\n" + NOTES)]},
        stream_mode="messages",
    ):
        step_no += 1
        node_name = metadata.get("langgraph_node", "?") if isinstance(metadata, dict) else "?"
        content = getattr(chunk, "content", "")
        text = content if isinstance(content, str) else ""
        if not text:
            continue
        print(f"[第{step_no:4d}块] [节点:{node_name:10s}] {text!r}")
"""
    print()
    print("-" * 64)
    print("看清楚这条链路了吗：")
    print("    主模型(deepseek) --tool_call--> summarize_notes")
    print("                                       |")
    print("                                       v")
    print("                              又一个 deepseek 调用")
    print("                                       |")
    print("                                       v")
    print("    主模型 <----ToolMessage-------- 返回文本")
    print()
    print("司南里的 recognize_homework 是一模一样的结构，")
    print("只不过工具内部调的不是 deepseek，而是 glm-4v-flash（视觉模型）。")
    print("也就是说：你其实已经会写多 Agent 了，只是还不知道它叫什么。")

    # TODO: 加 translate_to_english 工具，再跑一遍看两个工具会不会被串联调用
    # 完成

if __name__ == "__main__":
    main()