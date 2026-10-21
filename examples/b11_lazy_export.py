"""B11 · 模块级惰性导出 __getattr__

学什么
    langgraph.json 要求 `sinan_agent.py:agent` 这样一个模块级变量。
    但司南的 Agent 是按 (user_id, role) 动态构建的，根本不存在全局实例。
    解法：模块级 __getattr__ —— 只有真正访问 .agent 时才去构建。

对应司南
    src/app/agent/sinan_agent.py:242-250

怎么跑
    python examples/b11_lazy_export.py
    （不需要 API key）

观察点
    导入之后 _built 是空的 —— 说明导入期什么都没发生。
    访问一次 me.agent 才触发构建。
    ⚠️ 再访问一次，它又构建了一遍 —— 这就是为什么真实项目里
       get_agent 内部还要再套一层缓存（B6）。

TODO（学完自己改）
    1) 把 __getattr__ 改成"第一次访问后把自己的实例塞进模块 globals"，
       让第二次访问直接命中缓存。提示：globals()["agent"] = obj
    2) 打开 langgraph.json，看看它引用的到底是什么路径。
"""

from __future__ import annotations

import sys

# 记录构建过几次，用来观察"什么时候真的发生了构建"
_built: list[str] = []


def _build_agent(user_id: str, role: str) -> str:
    """模拟真实项目里的 create_agent(...) —— 会读库、拼提示词、连模型。"""
    label = f"Agent(user={user_id}, role={role})"
    _built.append(label)
    print(f"    [构建] {label}")
    return label


# ======================================================================
# 模块级 __getattr__：只在属性找不到时才被调用
# ======================================================================
def __getattr__(name: str):
    """PEP 562 —— 模块属性的兜底查找。

    只有 `sinan_agent.xxx` 这种访问、且常规查找失败时才会走到这里。
    """
    if name == "agent":
        # 真实项目：return get_agent("default", "kaoyan")
        obj = _build_agent("default", "kaoyan")
        globals()[name] = obj  # ← 加在这里(缓存版)
        """globals()[name] = obj 就是把"刚构建出来的对象"塞进模块的属性字典，
        让后续的同名属性访问在常规查找阶段就能命中，从而跳过 __getattr__，
        实现一次性构建 + 永久缓存。"""
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    me = sys.modules[__name__]

    print("=" * 64)
    print("PART 1 · 导入之后，什么都没发生")
    print("=" * 64)
    print(f"    _built = {_built}          <- 空列表，导入期零副作用")

    print()
    print("=" * 64)
    print("PART 2 · 访问 me.agent，触发构建")
    print("=" * 64)
    a1 = me.agent
    print(f"    拿到：{a1}")

    print()
    print("=" * 64)
    print("PART 3 · 再访问一次 —— 注意发生了什么")
    print("=" * 64)
    a2 = me.agent
    print(f"    拿到：{a2}")
    print(f"    是不是同一个对象：{a1 is a2}")
    print()
    print(f"    _built 现在有 {len(_built)} 条 —— __getattr__ 每次访问都会执行！")

    print()
    print("=" * 64)
    print("PART 4 · 访问一个不存在的名字")
    print("=" * 64)
    try:
        me.this_does_not_exist
    except AttributeError as exc:
        print(f"    ✗ {exc}")
        print("    ↑ raise AttributeError 是必须的，否则 hasattr() 会误判")

    print()
    print("-" * 64)
    print("为什么需要它？")
    print("    langgraph.json 里写的是：")
    print('        "graphs": {"sinan_agent": "./src/app/agent/sinan_agent.py:agent"}')
    print("    LangGraph CLI 启动时会去取这个 `agent` 属性。")
    print("    惰性导出让它拿到一个能用的实例，同时又不会在 import 阶段")
    print("    就要求数据库和 .env 全部就绪。")

    # TODO: 实现"缓存版"__getattr__，让第二次访问 a1 is a2 为 True


if __name__ == "__main__":
    main()
