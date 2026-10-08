"""B1 · 工具工厂（闭包绑定上下文）

学什么
    你已经会写模块级 @tool 了。但一旦涉及"当前是谁"，模块级就不够用：
    要么把 user_id 暴露成参数（模型可能乱填别人的 ID），
    要么用一个外层函数把它"关"进去（闭包），模型只看得见业务参数。

对应司南
    src/app/agent/tools/mastery.py:24   build_mastery_tools(repo, user_id, role)

怎么跑
    python examples/b01_tool_factory.py
    （不需要 API key —— 只看两个工具的参数 schema 差异）

观察点
    对比两个工具的 args：
      · 模块级版本：{user_id, keyword}   <- user_id 裸露，模型能改
      · 工厂版本：  {keyword}            <- user_id 被闭包藏起来了

TODO（学完自己改）
    1) 给工厂再加一个工具 get_study_streak()，返回"连续打卡 N 天"
    完成
    2) 想清楚：如果一定要让模型传 user_id，会有什么安全问题？
    大模型可能随机填入一个 user_id，导致模型使用错误的 user_id。

    这是典型的"按用户动态生成的特定工具"——它专门为了把 user_id 锁死在工具创建时刻，
    牺牲了"全局复用"，换来的是"身份不可伪造"。 这就是它和普通全局工具在设计上的核心权衡。
"""

from __future__ import annotations

import json

from langchain.tools import tool


# ======================================================================
# 反例：模块级工具，把 user_id 暴露给模型
# ======================================================================
@tool
def search_notes_bad(user_id: str, keyword: str) -> str:
    """按关键词搜索某个用户的笔记。

    Args:
        user_id: 用户 ID
        keyword: 搜索关键词
    """
    return f"[反例] 查询 {user_id} 的笔记：{keyword}"


# ======================================================================
# 正例：工厂函数，user_id 被闭包捕获
#闭包：闭包就是函数把一个不属于自己的外部变量"捕获并锁进自己的记忆"里，
# 即使外层函数执行完，它仍然能用那个值。
# ======================================================================
def make_note_tools(user_id: str) -> list:
    """为某个用户构建工具集。user_id 通过闭包被捕获，不进 schema。"""

    @tool
    def search_notes(keyword: str) -> str:
        """按关键词搜索我的笔记。

        Args:
            keyword: 搜索关键词
        """
        return f"[正例] 查询 {user_id} 的笔记：{keyword}"

    @tool
    def count_notes() -> str:
        """统计我的笔记总数。"""
        return f"[正例] {user_id} 共有 42 条笔记"

    @tool
    def get_study_streak() -> str:
        """返回我的连续打卡天数。"""
        return f"[正例] {user_id} 连续打卡 10 天"

    return [search_notes, count_notes, get_study_streak]


def main() -> None:
    tools = make_note_tools("tangyuan")

    print("=" * 64)
    print("模块级工具的参数 schema —— user_id 暴露给模型")
    print("=" * 64)
    print(json.dumps(search_notes_bad.args, ensure_ascii=False, indent=2))

    print()
    print("=" * 64)
    print("工厂工具的参数 schema —— 只剩 keyword，user_id 被闭包藏起来")
    print("=" * 64)
    for t in tools:
        print(f"  {t.name:16s} args = {sorted(t.args.keys())}")

    print()
    print("实际调用效果：")
    print("  ", search_notes_bad.invoke({"user_id": "someone_else", "keyword": "极限"}))
    print("  ", tools[0].invoke({"keyword": "极限"}))

    print()
    print("-" * 64)
    print("结论：工厂版的模型压根看不到 user_id，也就无法伪造。")
    print("这正是 build_mastery_tools(repo, user_id, role) 存在的理由 ——")
    print("掌握度工具必须知道'查谁的'，但又不能把决定权交给模型。")


if __name__ == "__main__":
    main()
