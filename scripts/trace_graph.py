"""司南 · 业务图结构导出（L7 验收工具）

打印 `app/agent/core/graph.py` 定义的那张图：节点、边、条件边、状态字段。
同时对照打印 `create_agent` **自动生成**的图结构——
两张图摆在一起，就能讲清"手写图 vs 自动图"的差别在哪。

用法（项目根目录）：
    .venv/Scripts/python.exe scripts/trace_graph.py
    .venv/Scripts/python.exe scripts/trace_graph.py --mermaid   # 输出 Mermaid

不需要 key、不需要建库、不需要联网。图是纯静态结构。
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from app.agent.core.graph import describe_graph  # noqa: E402
from app.agent.report_agent import describe_report_graph  # noqa: E402

SEP = "=" * 68


def main() -> int:
    argv = sys.argv[1:]
    d = describe_graph()

    if "--mermaid" in argv:
        print("```mermaid")
        print("graph TD")
        print("    START([START]) --> route")
        for t in d["conditional"]["targets"]:
            print(f"    route -.->|{t}| {t}")
        # L8：长期规划是两跳，且第二跳是中断点
        print("    long_draft --> long_confirm")
        print("    long_confirm{{interrupt}}:::wait")
        for t in d["conditional"]["targets"]:
            if t == "long_draft":
                continue
            print(f"    {t} --> END([END])")
        print("    long_confirm --> END")
        print("    classDef wait fill:#ffe6cc,stroke:#d79b00,stroke-width:2px")
        print("```")
        return 0

    print(SEP)
    print("司南业务图（手写 · StateGraph）")
    print(SEP)

    print(f"\n状态字段 ({len(d['state_fields'])} 个)")
    for f in d["state_fields"]:
        print(f"    · {f}")

    print(f"\n节点 ({len(d['all_nodes'])} 个，含内置)")
    for n in d["all_nodes"]:
        builtin = "  ← 内置" if n.startswith("__") else ""
        desc = d["branches"].get(n, "")
        print(f"    · {n:<12} {desc}{builtin}")

    print(f"\n边 ({len(d['edges'])} 条)")
    for e in d["edges"]:
        print(f"    {e['from']:>12} ──► {e['to']}")

    c = d["conditional"]
    print("\n条件边 (1 组)")
    print(f"    源节点   : {c['from']}")
    print(f"    判定函数 : {c['decider']}(state) -> 节点名")
    for t in c["targets"]:
        print(f"      判定值 {t:<10} → 节点 {t}")

    # ---- L8：中断点 ----
    for it in d.get("interrupts", []):
        print(f"\n中断点 (L8) · 共 {len(d['interrupts'])} 处")
        print(f"    节点     : {it['node']}   （上游：{it['next_after']}）")
        print(f"    恢复方式 : {it['resume_with']}")
        print(f"    前置条件 : 必须挂 checkpointer")
        print(f"    铁律     : {it['rule']}")

    # ---- L9：子图 ----
    subs = d.get("subgraphs") or []
    if subs:
        print(f"\n子图 (L9) · 共 {len(subs)} 个")
        for name in subs:
            sd = describe_report_graph()
            inner = [n for n in sd["all_nodes"] if not n.startswith("__")]
            print(f"\n    ● {name}  展开为 {len(inner)} 个节点")
            for n in inner:
                print(f"        · {n:<8} {sd['branches'].get(n, '')}")
            print(f"\n      条件边 : {sd['conditional']['from']} "
                  f"--{sd['conditional']['decider']}--> {sd['conditional']['targets']}")
            print(f"      环     : {sd['loop']}")
            print(f"      中断点 : {sd['interrupt']['node']} / {sd['interrupt']['kind']}")
            print(f"      止损   : 最多追问 {sd['max_ask_rounds']} 轮")
            print(f"      共享字段（与父图同名对接）: {sd['shared_with_parent']}")
            print(f"      私有字段（留在子图内部）  : {sd['private_fields']}")

    # ---------------------------------------------------------------
    print()
    print(SEP)
    print("对照：create_agent 自动生成的图（附录 A 实证三）")
    print(SEP)
    print("""
节点 : ['__start__', 'model', 'tools', '__end__']

边：
  __start__ -> model
  model     -> __end__    [条件边]
  model     -> tools      [条件边]
  tools     -> model      [条件边]      ← 这条让图「转」起来了
""")
    print("两张图的差别：")
    print("    自动图  —— 4 节点，1 个环（model ⇄ tools），ReAct 循环，逻辑不可干预")
    print("    手写图  —— 7 节点 + 1 个子图；敢停下来等人（L8），也装得下自己的小环（L9）")
    print("\n    自动图解决「一个 agent 怎么调工具」，")
    print("    手写图解决「多个业务步骤怎么编排」。")
    print("    后者能把前者装进去——这就是「子图」的用法（L9 已完成）。")

    print()
    print(SEP)
    print("Mermaid 视图：加 --mermaid 参数")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
