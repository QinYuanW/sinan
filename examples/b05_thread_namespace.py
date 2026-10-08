"""B5 · thread_id 命名空间隔离

学什么
    SqliteSaver 把所有人的会话都存在同一张表里，靠 thread_id 区分。
    你 Test_Saverdb.ipynb 里用的是裸 "1"。
    如果你想区分"用户 + 身份"，就必须自己拼命名空间：
        f"{user_id}:{role}:{thread_id}"
    否则：A 用户的会话会被 B 用户读到，或者同一个人切换身份时串档。

对应司南
    src/app/agent/sinan_agent.py:162、206、235
        thread_id = f"{user_id}:{role}:{thread_id}"

怎么跑
    python examples/b05_thread_namespace.py
    （需要 DEEPSEEK_API_KEY；会真的写一个 _demo_checkpoints.db）

观察点
    PART 1 用裸 thread_id：Bob 读到了 Alice 的会话 —— 事故现场。
    PART 2 加命名空间：各读各的，互不可见。

TODO（学完自己改）
    1) 打开跑完生成的 _demo_checkpoints.db，
       SELECT * FROM checkpoints; 看看 thread_id 长什么样
    2) 想清楚：如果 thread_id 只拼 user_id 不拼 role，
       用户在"考研"和"教师"之间切换会发生什么？

       #无法区分用户身份，会话会串档
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

DB = Path(__file__).resolve().parent / "_demo_checkpoints.db"


def fresh_checkpointer() -> SqliteSaver:
    """每次跑都从干净库开始，避免上次的残留干扰观察。"""
    if DB.exists():
        DB.unlink()
    conn = sqlite3.connect(str(DB), check_same_thread=False)
    cp = SqliteSaver(conn)# 初始化检查点保存器
    cp.setup()# 初始化数据库表
    return cp


def build_agent(checkpointer: SqliteSaver, user_name: str):
    return create_agent(
        model=init_chat_model(model="deepseek-chat"),
        tools=[],
        system_prompt=f"你是学习助手。当前用户叫 {user_name}。回答要简短。",
        checkpointer=checkpointer,
    )


def say(agent, thread_id: str, text: str) -> None:
    """往某个会话里说一句话（这会写入 checkpoint）"""
    agent.invoke(
        {"messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": thread_id}},
    )


def read_history(checkpointer: SqliteSaver, thread_id: str) -> list[str]:
    """读回某个会话的历史（不调模型，直接看库）"""
    try:
        snapshot = checkpointer.get({"configurable": {"thread_id": thread_id}})
    except Exception:  # noqa: BLE001
        return []
    if not snapshot:
        return []
    channel = (snapshot or {}).get("channel_values") or {}
    out = []
    for msg in channel.get("messages") or []:
        if isinstance(msg, HumanMessage):
            out.append(f"user: {str(msg.content)[:40]}")
        elif isinstance(msg, AIMessage):
            out.append(f"ai:   {str(msg.content)[:40]}")
    return out


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("跳过：未检测到 DEEPSEEK_API_KEY，请先在 .env 里配置。")
        return

    cp = fresh_checkpointer()
    agent_alice = build_agent(cp, "爱丽丝")
    agent_bob = build_agent(cp, "鲍勃")

    # ==================================================================
    print("=" * 68)
    print("PART 1 · 反面教材：裸 thread_id")
    print("=" * 68)
    say(agent_alice, "1", "我叫爱丽丝。")
    alice_hist = read_history(cp, "1")
    bob_hist = read_history(cp, "1")
    print(f"  Alice 用 thread_id='1' 说了一句话")
    print(f"  Alice 读 '1' 的历史 -> {alice_hist}")
    print(f"  Bob   读 '1' 的历史 -> {bob_hist}")
    print()
    if bob_hist:
        print("  ⚠️ 事故：Bob 什么都没说，却读到了 Alice 的会话！")
        print("     因为 thread_id 相同，checkpointer 认为这是同一个对话。")
    else:
        print("  （本次未复现，但风险客观存在）")

    # ==================================================================
    print()
    print("=" * 68)
    print("PART 2 · 正解：拼上 user_id 和 role 命名空间")
    print("=" * 68)
    tid_alice = "alice:kaoyan:1"
    tid_bob = "bob:kaoyan:1"
    tid_alice_teacher = "alice:teacher:1"

    say(agent_alice, tid_alice, "我叫爱丽丝，考研。")
    say(agent_bob, tid_bob, "我叫鲍勃，也考研。")
    say(agent_alice, tid_alice_teacher, "我是爱丽丝，这次是教师身份。")

    for label, tid in [
        ("Alice + 考研", tid_alice),
        ("Bob   + 考研", tid_bob),
        ("Alice + 教师", tid_alice_teacher),
    ]:
        hist = read_history(cp, tid)
        print(f"  {label}  thread_id={tid}")
        for line in hist:
            print(f"      {line}")
        print()

    print("-" * 68)
    print("验证结论：")
    print("  · Alice 和 Bob 用同一个裸 '1'  -> 串档")
    print("  · 拼上 user_id:role:id 之后    -> 三个会话完全独立")
    print()
    print("顺带看一眼库里的 reality：")
    try:
        conn = sqlite3.connect(str(DB))
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
        for (tid,) in rows:
            print(f"      thread_id = {tid}")
        conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"      （查库失败：{exc}）")

    print()
    print("为什么 role 也必须拼进去？")
    print("    同一用户在考研身份下的对话，和教师身份下的对话，")
    print("    上下文完全不同 —— system_prompt 都是两套。")
    print("    不拼 role，切身份时会带着上一个身份的对话历史，模型会精神分裂。")

    # TODO: 想想如果再加一层 thread（一个用户多个对话窗口），命名空间该怎么设计


if __name__ == "__main__":
    main()
