"""司南 · 引擎自检脚本

验证链路（不需要启动服务、不需要调用大模型）：
    建表 -> 导入数二考点树 -> 组装上下文 -> 优先度计算 -> 生成日报
    -> 打卡 -> 汇报重规划 -> 长期规划 -> 意图路由

用法（在项目根目录执行）：
    .venv/Scripts/python.exe scripts/selftest_engine.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 脚本在 scripts/ 下，项目根是上一级
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

TEST_DB = BASE / "_selftest.db"
if TEST_DB.exists():
    TEST_DB.unlink()

# 图的人工介入（L8）会把执行状态写进 checkpoints 库；
# 默认位置是线上 data 目录，自检要用自己的，避免污染真实会话状态。
TEST_CHECKPOINT_DB = BASE / "_selftest_checkpoints.db"
if TEST_CHECKPOINT_DB.exists():
    TEST_CHECKPOINT_DB.unlink()

from app.agent.core.graph import decide_branch, describe_graph  # noqa: E402
from app.agent.core.planner import (  # noqa: E402
    build_context,
    ensure_kp_tree,
    generate_daily_plan,
    generate_long_plan,
    replan_after_report,
)
from app.agent.core.priority import rank_kps  # noqa: E402
from app.agent.core.router import route  # noqa: E402
from app.agent.roles import get_pack, list_packs  # noqa: E402
from app.agent.tools.mastery import build_mastery_tools  # noqa: E402
from app.db.models import init_db  # noqa: E402
from app.db.repository import SQLiteRepository  # noqa: E402

SEP = "=" * 68


def main() -> int:
    init_db(TEST_DB)
    print(SEP)
    print("1. 身份包注册")
    print(SEP)
    packs = list_packs()
    for p in packs:
        print(f"  {p['icon']} {p['key']:<14} {p['name']:<8} 字段 {len(p['goal_fields'])} 个 "
              f"| 因子 {[f['key'] for f in p['factors']]}")
    assert len(packs) == 3, "身份包注册数量不对"

    # ---------------------------------------------------------------
    print(SEP)
    print("2. 设置目标 + 导入考点树")
    print(SEP)
    repo = SQLiteRepository(TEST_DB)
    uid, role = "selftest", "kaoyan"
    pack = get_pack(role)

    repo.upsert_profile(
        uid, role,
        name="汤园",
        goal_payload={
            "target_school": "某大学", "target_major": "电子信息",
            "exam_date": "2027-12-20", "math_type": "数二",
        },
        daily_hours=4.0, subjects=["数学"],
    )
    imported = ensure_kp_tree(repo, uid, role, pack)
    total_kp = repo.count_kp(uid, role)
    print(f"  导入考点 {imported} 个，库存 {total_kp} 个")
    subjects = sorted({k["subject"] for k in repo.list_kp(uid, role)})
    print(f"  科目: {subjects}")
    assert total_kp >= 60, "考点树数量异常"

    # ---------------------------------------------------------------
    print(SEP)
    print("3. 组装上下文")
    print(SEP)
    ctx = build_context(repo, uid, role, pack)
    print(f"  剩余天数 {ctx.days_left} | 每日 {ctx.daily_hours}h | 打卡 {ctx.streak} 天 "
          f"| 知识点 {len(ctx.kp_list)}")

    # ---------------------------------------------------------------
    print(SEP)
    print("4. 优先度排序（权重与因子明细）")
    print(SEP)
    print(f"  生效权重: {ctx and pack.adjust_weights(ctx)}")
    ranked = rank_kps(ctx, pack)
    print(f"  参与排序 {len(ranked)} 个（已剔除掌握度 ≥0.85 的）")
    for item in ranked[:5]:
        print(f"  [{item['priority']:.3f}] {item['name']}")
        print(f"          {item['reason']}")
    assert ranked[0]["priority"] >= ranked[-1]["priority"], "排序方向不对"

    # ---------------------------------------------------------------
    print(SEP)
    print("5. 生成今日作战清单（4 小时预算装箱）")
    print(SEP)
    plan = generate_daily_plan(repo, uid, role)
    print(f"  日期 {plan['date']} | 任务 {len(plan['tasks'])} 条 | 总耗时 {plan['total_hours']}h "
          f"/ 预算 {plan['budget_hours']}h")
    for t in plan["tasks"]:
        print(f"    - [{t['priority']:.2f}] {t['kp']}（{t['est_hours']}h）")
    print(f"  节奏提醒: {plan['coach_note'][:160]}")
    assert plan["total_hours"] <= plan["budget_hours"] + 1e-6, "装箱超出预算"

    # 幂等性：二次调用应命中缓存
    again = generate_daily_plan(repo, uid, role)
    print(f"  幂等复读: cached={again.get('cached')} 任务数={len(again['tasks'])}")
    assert again.get("cached") is True, "当日计划幂等失效"

    # ---------------------------------------------------------------
    print(SEP)
    print("6. 打卡（不应触发重规划）")
    print(SEP)
    log = repo.check_in(uid, role)
    print(f"  打卡时间 {log['checked_in_at']} | 连续 {repo.check_in_streak(uid, role)} 天")

    # ---------------------------------------------------------------
    print(SEP)
    print("7. 汇报 -> 更新画像 -> 重规划")
    print(SEP)
    first_kp = plan["tasks"][0]["kp"]
    result = replan_after_report(
        repo, uid, role, plan["date"],
        accuracy={first_kp: 0.6},
        completed_kps=[first_kp],
        problems=["极限计算中泰勒展开阶数取错"],
    )
    print(f"  掌握度变化: {result['mastery_delta']}")
    print(f"  计划是否变更: {result['changed']}")
    print(f"  原因: {result['replan_reason']}")
    if result.get("next_plan"):
        print(f"  次日清单 {len(result['next_plan']['tasks'])} 条, 顺延 "
              f"{len(result['next_plan']['carry_over'])} 条")

    # 指数平滑验证：再来一次同样的证据，掌握度应继续上升但幅度收窄
    after_first = repo.get_mastery(uid, role, first_kp)
    repo.update_mastery(uid, role, first_kp, 0.6)
    after_second = repo.get_mastery(uid, role, first_kp)
    print(f"  指数平滑验证: 首次 {after_first:.4f} -> 二次 {after_second:.4f} "
          f"(Δ={after_second - after_first:+.4f})")
    assert after_second > after_first, "二次证据未推动掌握度"

    # ---------------------------------------------------------------
    print(SEP)
    print("8. 长期阶段规划")
    print(SEP)
    long_plan = generate_long_plan(repo, uid, role)
    print(f"  {long_plan['coverage_note']}")
    for m in long_plan["milestones"]:
        print(f"    {m['name']} {m['start_date']}~{m['end_date']} 覆盖 {len(m['target_kps'])} 考点")

    # ---------------------------------------------------------------
    print(SEP)
    print("9. 意图路由")
    print(SEP)
    cases = [
        ("@汇报 今天做了极限8道题错了3道", "report"),
        ("打卡", "checkin"),
        ("帮我生成今天的计划", "plan_daily"),
        ("给我做个整体规划", "plan_long"),
        ("你好，你是谁", "chat"),
    ]
    for msg, expect in cases:
        intent, cleaned = route(msg)
        flag = "OK " if intent.value == expect else "!! "
        print(f"  {flag}{msg[:22]:<24} -> {intent.value:<12} 正文={cleaned[:20]!r}")

    # ---------------------------------------------------------------
    print(SEP)
    print("10. 掌握度概览")
    print(SEP)
    summary = repo.mastery_summary(uid, role)
    print(f"  已记录 {summary['total']} 个点 | 平均 {summary['average']}")

    # ---------------------------------------------------------------
    print(SEP)
    print("11. 掌握度工具集（B1 · 工厂函数 + 闭包绑定上下文）")
    print(SEP)
    tools = build_mastery_tools(repo, uid, role)
    for t in tools:
        print(f"  · {t.name:<24} 参数 {list(t.args.keys())}")
    print(f"  工具总数 {len(tools)}")
    assert len(tools) == 5, "工具工厂应产出 5 个工具"
    # 闭包验证：user_id / role 关在工厂里，不应出现在任何工具的参数表中
    for t in tools:
        assert "user_id" not in t.args, f"{t.name} 泄漏了 user_id 参数"
    print("  OK 5 个工具齐备，且无一个把 user_id 暴露给模型（闭包生效）")

    # ---------------------------------------------------------------
    print(SEP)
    print("12. 业务编排图（L7 · StateGraph 裸写）")
    print(SEP)
    g = describe_graph()
    print(f"  状态字段 {len(g['state_fields'])} 个: {', '.join(g['state_fields'])}")

    biz_nodes = [n for n in g["all_nodes"] if not n.startswith("__")]
    print(f"  业务节点 {len(biz_nodes)} 个: {biz_nodes}")
    assert set(biz_nodes) == {
        "route", "checkin", "report", "daily",
        "long_draft", "long_confirm", "chat",
    }, "图节点与预期不符"

    print(f"  边 {len(g['edges'])} 条")
    c = g["conditional"]
    print(f"  条件边: {c['from']} --{c['decider']}--> {c['targets']}")
    assert c["from"] == "route" and len(c["targets"]) == 5, "条件边拓扑不对"

    # 条件边函数本身：意图 -> 节点名 的映射必须可判定且能兜底
    expect_branch = {
        "checkin": "checkin", "report": "report",
        "plan_daily": "daily", "plan_long": "long_draft",  # ← L8：进"起草"而非"确认"
        "chat": "chat",
        "未登记的意图": "chat",   # 兜底：绝不走进死路
    }
    for intent_name, node in expect_branch.items():
        got = decide_branch({"intent": intent_name})
        flag = "OK " if got == node else "!! "
        print(f"  {flag}{intent_name:<14} -> {got}")
        assert got == node, f"条件边映射错误: {intent_name}"
    print("  OK 四场景全部可路由，且未登记意图降级到 chat")

    # ---------------------------------------------------------------
    print(SEP)
    print("13. 人工介入（L8 · interrupt：确认才落库）")
    print(SEP)

    import asyncio as _asyncio
    from datetime import date as _date_cls

    import app.agent.sinan_agent as _sa
    from app.agent.core import graph as _graph
    from app.agent.core import planner as _planner

    # 让图用自检的仓储与 checkpointer（默认都指向线上 data 目录）
    _sa._repo = repo
    _sa.CHECKPOINT_DB = TEST_CHECKPOINT_DB

    uid2 = "selftest_confirm"
    repo.upsert_profile(
        uid2, role,
        name="汤园(确认测试)",
        goal_payload={
            "target_school": "某大学", "target_major": "电子信息",
            "exam_date": "2027-12-20", "math_type": "数二",
        },
        daily_hours=3.0, subjects=["数学"],
    )
    ensure_kp_tree(repo, uid2, role, pack)

    # 给"起草"打计数：interrupt 恢复时它【不该】重跑
    _orig_gen = _planner.generate_long_plan
    _draft_calls: list[int] = []

    def _counted(*a, **k):
        _draft_calls.append(1)
        return _orig_gen(*a, **k)

    _planner.generate_long_plan = _counted

    today_s = _date_cls.today().isoformat()

    def _saved():
        return repo.get_plan(uid2, role, "long", today_s)

    async def _flow():
        tid = "selftest-confirm"

        # ① 发起 -> 应当挂起，且【不落库】
        r1 = await _graph.run_graph("给我做个整体规划", tid, uid2, role)
        print(f"  ① 发起     挂起={r1.interrupted} 落库={'是 ❌' if _saved() else '否 ✅'} "
              f"起草次数={len(_draft_calls)}")
        assert r1.interrupted, "长期规划应当挂起等人确认"
        assert _saved() is None, "未确认就落库了"
        assert (r1.interrupt or {}).get("kind") == "plan_confirm", "中断类型不对"

        # ② 含糊回复 -> 应当继续追问，仍不落库
        r2 = await _graph.run_graph("嗯……让我想想", tid, uid2, role)
        print(f"  ② 含糊回复 挂起={r2.interrupted} 落库={'是 ❌' if _saved() else '否 ✅'} "
              f"起草次数={len(_draft_calls)}")
        assert r2.interrupted, "含糊回复后应当继续追问"
        assert _saved() is None, "含糊回复竟然落库了"

        # ③ 明确确认 -> 应当落库，且起草不该重跑
        r3 = await _graph.run_graph("确认保存", tid, uid2, role)
        rec = _saved()
        print(f"  ③ 确认     挂起={r3.interrupted} 落库={'是 ✅' if rec else '否 ❌'} "
              f"起草次数={len(_draft_calls)}")
        assert not r3.interrupted, "确认后不应再挂起"
        assert rec is not None, "确认后没落库"
        assert rec.get("status") == "active", "落库状态不对"
        assert len(_draft_calls) == 1, "恢复时起草被重跑了 —— 节点没拆开！"

        pending = await _graph.get_pending_interrupt(tid, uid2, role)
        print(f"     收尾挂起查询: {pending}   (应为 None)")
        assert pending is None, "确认后仍处于挂起态"

        # ④ 取消路径 -> 回执正确，且不该再挂起
        t2 = "selftest-cancel"
        await _graph.run_graph("给我做个整体规划", t2, uid2, role)
        r5 = await _graph.run_graph("取消", t2, uid2, role)
        kinds = [k for k, _ in r5.events]
        print(f"  ④ 取消     事件={kinds} 落库记录仍在={'是 ✅' if _saved() else '否'}")
        assert not r5.interrupted, "取消后不应再挂起"
        assert "confirm_result" in kinds, "取消没有回执"

        # 收尾：释放 aiosqlite 连接，否则解释器退不出去（实测 exit=124）
        await _graph.close_graph()

    _asyncio.run(_flow())

    # 判定函数边界：宁可多问一次，不可误存一次
    print("  判定函数边界：")
    for text, want in [("确认", "approve"), ("可以，存吧", "approve"), ("嗯嗯", "approve"),
                       ("取消", "reject"), ("不要保存", "reject"), ("不用了", "reject"),
                       ("？", "unclear"), ("", "unclear"), ("这版不行", "unclear")]:
        got = _graph.judge_confirmation(text)
        flag = "OK " if got == want else "!! "
        print(f"      {flag}{text!r:<14} -> {got}")
        assert got == want, f"确认判定错误: {text!r} -> {got}"

    print("  OK 挂起→追问→确认落库→取消不落库，且起草全程只算 1 次")

    # ---------------------------------------------------------------
    print(SEP)
    print("14. 汇报子图（L9 · 子图 + 环 + 追问止损）")
    print(SEP)

    from app.agent.report_agent import (  # noqa: E402
        MAX_ASK_ROUNDS,
        describe_report_graph,
        get_report_graph,
    )
    from app.agent.core.graph import SinanState  # noqa: E402
    from app.models.schemas import ReportParse  # noqa: E402
    import app.agent.report_agent as _ra  # noqa: E402

    sg = describe_report_graph()
    sub_nodes = [n for n in sg["all_nodes"] if not n.startswith("__")]
    print(f"  子图节点: {sub_nodes}")
    print(f"  条件边  : {sg['conditional']['from']} --{sg['conditional']['decider']}--> "
          f"{sg['conditional']['targets']}")
    print(f"  环      : {sg['loop']}")
    print(f"  中断点  : {sg['interrupt']['node']} / {sg['interrupt']['kind']}")
    print(f"  止损轮次: {sg['max_ask_rounds']}")

    assert set(sub_nodes) == {"parse", "ask", "apply"}, "汇报子图节点不对"
    assert sg["conditional"]["from"] == "parse", "子图条件边源不对"
    assert MAX_ASK_ROUNDS == 3, "止损轮次变了"

    # 父子状态契约：共享字段必须都能在父图 state 里找到
    missing = set(sg["shared_with_parent"]) - set(SinanState.__annotations__)
    print(f"  共享字段: {sg['shared_with_parent']}")
    print(f"  私有字段: {sg['private_fields']}")
    assert not missing, f"共享字段在父图 state 里不存在: {missing}"

    # 主图确实把 report 当子图挂
    assert describe_graph().get("subgraphs") == ["report"], "主图未挂载汇报子图"

    # ★ 子图能独立编译、独立存在（子图最大的好处之一）
    standalone = get_report_graph()
    assert standalone is get_report_graph(), "子图应缓存单例"
    print("  OK 子图可独立编译（不依赖主图）")

    # ---------- 功能验证：用可控的假解析器 ----------
    uid3 = "selftest_report"
    repo.upsert_profile(
        uid3, role,
        name="汤园(汇报子图)",
        goal_payload={
            "target_school": "某大学", "target_major": "电子信息",
            "exam_date": "2027-12-20", "math_type": "数二",
        },
        daily_hours=3.0, subjects=["数学"],
    )
    ensure_kp_tree(repo, uid3, role, pack)

    _seen_texts: list = []

    def _fake_parse(text=None, photos=None, report_date=None, role="kaoyan"):
        _seen_texts.append(text)
        if "补充" not in (text or ""):
            return ReportParse(
                date=report_date, needs_followup=True,
                followup_question="正确率大概多少？哪几道错了？",
            )
        return ReportParse(
            date=report_date,
            completed_kps=["极限的基本计算（等价无穷小 / 洛必达 / 泰勒）"],
            accuracy={"极限的基本计算（等价无穷小 / 洛必达 / 泰勒）": 0.75},
            problems=["泰勒展开阶数取错"],
        )

    _orig_parse = _ra.parse_report
    _ra.parse_report = _fake_parse

    async def _report_flow():
        tid = "selftest-report"

        # ① 信息不足 → 子图挂起追问
        r1 = await _graph.run_graph("@汇报 今天做了极限8道题", tid, uid3, role)
        ip = r1.interrupt or {}
        print(f"  ① 信息不足 挂起={r1.interrupted} 类型={ip.get('kind')} "
              f"轮次={ip.get('round')}/{ip.get('max_rounds')}")
        assert r1.interrupted, "信息不足应挂起追问"
        assert ip.get("kind") == "report_followup", "中断类型不对"
        assert ip.get("round") == 1, "应为第 1 轮"
        assert ip.get("question"), "追问语为空"
        assert len(_seen_texts) == 1, "首轮应只解析一次"

        # ② 补充 → 环回到 parse 重新解析 → 落库并回执
        r2 = await _graph.run_graph("补充：正确率75%，泰勒展开阶数取错了", tid, uid3, role)
        kinds = [k for k, _ in r2.events]
        print(f"  ② 补充后   挂起={r2.interrupted} 事件={kinds} parse调用={len(_seen_texts)}")
        assert not r2.interrupted, "补充后不应再挂起"
        assert "replan" in kinds, "缺少重规划回执"
        assert len(_seen_texts) == 2, "环没生效（parse 应被调用两次）"
        assert "【补充说明】" in (_seen_texts[1] or ""), "补充材料没有并入"
        print(f"     材料1: {_seen_texts[0]!r}")
        print(f"     材料2: {_seen_texts[1]!r}")

        # ③ 止损：一直说不清，3 轮后必须停
        _ra.parse_report = lambda text=None, photos=None, report_date=None, role="kaoyan": ReportParse(
            date=report_date, needs_followup=True, followup_question="还是说说正确率？",
        )
        tid2 = "selftest-report-stop"
        rs = await _graph.run_graph("@汇报 今天学了一会儿", tid2, uid3, role)
        rounds = [(rs.interrupt or {}).get("round")]
        while rs.interrupted:
            rs = await _graph.run_graph("嗯……就是那样吧", tid2, uid3, role)
            rounds.append((rs.interrupt or {}).get("round"))
        print(f"  ③ 止损    追问轮次轨迹={rounds[:-1]} → 停止（最终事件={[k for k, _ in rs.events]}）")
        assert not rs.interrupted, "应已停止追问"
        assert MAX_ASK_ROUNDS in rounds or rounds[-1] is None, "轮次轨迹异常"
        assert "followup" in [k for k, _ in rs.events], "止损回执缺少说明"

        await _graph.close_graph()

    _asyncio.run(_report_flow())
    _ra.parse_report = _orig_parse  # 还原

    print("  OK 子图：解析→追问(挂起)→补充→环回重解析→落库；3 轮止损生效")

    # ---------------------------------------------------------------
    print(SEP)
    print("15. 长期记忆 Store（L10 · 掌握度跨会话）")
    print(SEP)

    from app.db import memory_store as _ms  # noqa: E402

    # 隔离：自检用独立的 store 文件，不碰线上 memory_store.db
    _ms.STORE_DB = BASE / "_selftest_store.db"
    if _ms.STORE_DB.exists():
        _ms.STORE_DB.unlink()
    _ms.reset_store()

    info = _ms.describe_store()
    print(f"  后端 {info['backend']} · 文件 {Path(info['file']).name}")
    print(f"  命名空间 {info['namespace']} · search limit {info['search_limit']}")
    assert info["namespace"] == "(user_id, role, mastery)", "命名空间结构变了"

    kp_a = "极限的基本计算（等价无穷小 / 洛必达 / 泰勒）"
    kp_b = "矩阵运算与转置"
    _ops = [(kp_a, "高等数学", [0.6, 0.6, 0.8]), (kp_b, "线性代数", [0.2, 0.5])]

    def _snapshot(uid: str) -> dict:
        for kp, subj, accs in _ops:
            for acc in accs:
                repo.update_mastery(uid, role, kp, acc, subject=subj)
        summ = repo.mastery_summary(uid, role)
        return {
            "map": repo.get_mastery_map(uid, role),
            "last": repo.get_last_review_map(uid, role),
            "total": summ["total"],
            "average": summ["average"],
            "weakest": [i["kp_name"] for i in summ["weakest"]],
        }

    # ① 表后端
    os.environ["SINAN_USE_STORE"] = "0"
    snap_table = _snapshot("selftest_cmp_table")
    print(f"  ① 表后端    平均 {snap_table['average']} · {snap_table['total']} 条")

    # ② Store 后端（同样输入）
    os.environ["SINAN_USE_STORE"] = "1"
    snap_store = _snapshot("selftest_cmp_store")
    print(f"  ② Store 后端 平均 {snap_store['average']} · {snap_store['total']} 条")

    assert snap_table == snap_store, "两种后端结果不一致 —— 换存储穿透了业务代码"
    print("  OK 两后端逐字段一致（指数平滑小数 / 排序 / 聚合）")

    # ③ 迁移 + 切换（这是真正的上线路径）
    uid5 = "selftest_migrate"
    os.environ["SINAN_USE_STORE"] = "0"
    repo.update_mastery(uid5, role, kp_a, 0.6, subject="高等数学")
    repo.update_mastery(uid5, role, kp_b, 0.4, subject="线性代数")
    before = repo.get_mastery_map(uid5, role)

    migrated = _ms.migrate_from_table(repo)
    print(f"  ③ 迁移 {migrated} 条进 Store")

    os.environ["SINAN_USE_STORE"] = "1"
    _ms.reset_store()
    after = repo.get_mastery_map(uid5, role)
    assert before == after, "迁移后画像不一致"
    print(f"     切后端后读到一致 ✅ ({len(after)} 条)")

    # ④ 切换后继续累加：指数平滑在 Store 上必须照常工作
    old = repo.get_mastery(uid5, role, kp_a)
    new = repo.update_mastery(uid5, role, kp_a, 0.8, subject="高等数学")
    expect = round(old + 0.3 * (0.8 - old), 4)
    print(f"  ④ 切换后继续学习 {old} -> {new}（期望 {expect}）")
    assert abs(new - expect) < 1e-9, "Store 上的指数平滑不对"

    # ⑤ 层级命名空间：跨身份可检索 + 用户之间隔离
    st = _ms.get_store()
    st.put((uid5, "teacher", "mastery"), "教学法", {"mastery": 0.7, "subject": "教育学"})
    cross = st.search((uid5,))
    ns_list = st.list_namespaces(prefix=(uid5,))
    print(f"  ⑤ 跨身份前缀检索 {len(cross)} 条 · 命名空间 {ns_list}")
    assert len(cross) == 3, "跨身份前缀检索异常"
    assert st.search(("nobody", role, "mastery")) == [], "用户隔离失效"
    print("  OK Store：两后端等价 · 迁移无缝 · 跨身份可检索 · 用户隔离")

    os.environ["SINAN_USE_STORE"] = "0"  # 还原默认（走 mastery 表）
    _ms.reset_store()

    # ---------------------------------------------------------------
    print(SEP)
    print("16. 知识库检索（L11 · RAG：分块 / BM25 / 融合 / 工具）")
    print(SEP)

    from app.agent.tools.exam_kb import search_exam_kb  # noqa: E402
    from app.kb import describe_kb, get_kb, tokenize  # noqa: E402
    from app.kb.chunker import load_chunks  # noqa: E402
    from app.kb.retriever import _rrf  # noqa: E402

    chunks = load_chunks()
    docs = {c.source for c in chunks}
    print(f"  知识源 {len(docs)} 篇 · 分块 {len(chunks)} 块")
    assert len(docs) >= 2 and len(chunks) >= 30, "文档未加载全"

    info = describe_kb()
    print(f"  稀疏 {info['sparse']}")
    print(f"  稠密 {info['dense']}")
    print(f"  默认模式 {info['default_mode']}")
    assert info["default_mode"] in ("sparse", "hybrid")

    # ① 分词：中英混合（零依赖方案的地基）
    toks = tokenize("泰勒展开 Taylor 8道题")
    assert "泰勒" in toks and "taylor" in toks and "8" in toks, f"分词异常: {toks[:12]}"
    print(f"  ① 分词示例 {toks[:10]} …")

    kb = get_kb()

    # ② 检索质量：查询必须命中对应考点卡片
    cases = [
        ("泰勒展开阶数怎么定", "极限的基本计算"),
        ("矩阵乘法能不能交换顺序", "矩阵运算与转置"),
        ("二重积分怎么交换积分次序", "二重积分"),
        ("特征值重根要不要正交化", "二次型"),
        ("一阶线性微分方程通解公式", "一阶线性微分方程"),
    ]
    for q, expect in cases:
        hits = kb.search(q, top_k=3, mode="sparse")
        titles = [h["title"] for h in hits]
        ok = any(expect in t for t in titles)
        print(f"  ② {'OK ' if ok else '!! '}{q[:16]:<18} -> {titles[0][:24]}")
        assert ok, f"检索未命中: {q} -> {titles}"

    # ③ 工具路径（Agent 实际走这条）
    out = search_exam_kb.invoke({"query": "洛必达法则什么时候能用"})
    assert "【1】" in out, "工具输出格式不对"
    print(f"  ③ 工具输出 {len(out)} 字 · 含编号标题路径 ✅")

    # ④ RRF 融合（纯函数，不依赖 embedding）
    fused = _rrf([[0, 1, 2], [2, 0, 3]])
    assert fused[0] == 0, f"RRF 融合异常: {fused}"
    print(f"  ④ RRF 融合 [[0,1,2],[2,0,3]] -> {fused}（两路都靠前的胜出）")

    # ⑤ 降级：dense 不可用时 auto 模式必须照常返回
    auto_hits = kb.search("泰勒展开", top_k=2, mode="auto")
    assert auto_hits, "降级后检索为空"
    print(f"  ⑤ auto 模式（稠密{'可用' if kb.has_dense else '不可用'}）返回 {len(auto_hits)} 条 ✅")

    # ⑥ 稠密路径：有真 embedding 就用真的做对比；没有就用测试替身验证管线
    #    ★ 为什么两条路都要测：外部模型可用 ≠ 自己写的向量检索代码正确；
    #      外部模型不可用也 ≠ 可以不验证自己的代码。
    if kb.has_dense:
        # 同义词场景：这是稀疏检索的经典短板（"求导" 与 "导数" 字面不同）
        #
        # ⚠️ 断言只查"Top-3 里有没有语义相关的卡片"，**不锁具体排名**。
        #    原因：知识库是会长的。这条断言最初写的是 `"导数" in top1`，
        #    后来新加了一张「全微分与多元复合函数求导」——它本身就讲链式法则，
        #    抢走 Top-1 完全合理，断言却因此挂了。
        #    **测试断言绑死具体排名，等于给未来的自己埋雷。**
        q_syn = "求导法则有哪些"
        RELATED = ("导数", "求导")  # 任意含其一即算命中相关语义
        rows = {}
        for mode in ("sparse", "dense", "hybrid"):
            tops = [h["title"] for h in kb.search(q_syn, top_k=3, mode=mode)]
            hit = any(any(k in t for k in RELATED) for t in tops)
            rows[mode] = (hit, tops[0][:22] if tops else "(空)")
            print(f"  ⑥ {mode:<7} {'OK' if hit else '!!'} Top1={tops[0][:22] if tops else '(空)'}")
        assert rows["dense"][0], "稠密检索 Top-3 未命中任何相关卡片"
        assert rows["hybrid"][0], "混合检索 Top-3 未命中任何相关卡片"
        print("     ✅ 语义相关卡片进入 Top-3（不锁具体排名，知识库增长不会误报）")
    else:
        print("  ⑥ 稠密不可用（未配置 embedding）→ 用测试替身验证管线")
        import hashlib  # noqa: E402

        from app.kb.retriever import KnowledgeBase as _KB  # noqa: E402

        def _fake_embed(texts: list[str]) -> list[list[float]]:
            out = []
            for t in texts:
                v = [0.0] * 256
                for tok in tokenize(t):
                    h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                    v[h % 256] += 1.0
                out.append(v)
            return out

        kb2 = _KB(
            kb.chunks,
            _fake_embed([c.retrieval_text for c in kb.chunks]),
            embed_fn=_fake_embed,  # ★ 查询也走替身（与建索引同一个）
        )
        assert kb2.has_dense, "向量索引构建失败"
        d_hits = kb2.search("泰勒展开阶数", top_k=2, mode="dense")
        h_hits = kb2.search("泰勒展开阶数", top_k=2, mode="hybrid")
        assert d_hits and h_hits, "dense/hybrid 检索为空"
        print(f"     dense {len(d_hits)} 条 · hybrid {len(h_hits)} 条 ✅")

    print("  OK 知识库：分块→BM25→向量→融合→工具 全链路可用，缺 embedding 时自动降级")

    # ---------------------------------------------------------------
    print(SEP)
    print("17. 多租户部署（L12 · 第二个身份包 + 隔离 + 上线配置）")
    print(SEP)

    from app.agent.roles.base import PlanningContext  # noqa: E402

    # ① 第二个身份包「落地」（不再是占位）
    zsb, ky = get_pack("zhuanshengben"), get_pack("kaoyan")
    zkps = zsb.build_kp_tree()
    zsubj = sorted({k["subject"] for k in zkps})
    print(f"  ① 专升本考点 {len(zkps)} 个 · 科目 {zsubj}")
    print(f"     因子 {len(zsb.factors)} 个（考研 {len(ky.factors)} 个）· "
          f"考研考点 {len(ky.build_kp_tree())} 个")
    assert len(zkps) >= 60, "专升本考点树仍是占位规模"
    assert len(zsb.factors) == 5, "专升本应有 5 个因子（含难度惩罚）"

    # adjust_weights 钩子（base.py 留的扩展点，本关第一次真正用上）
    w_long = zsb.adjust_weights(
        PlanningContext(user_id="u", role="zhuanshengben", days_left=300))
    w_sprint = zsb.adjust_weights(
        PlanningContext(user_id="u", role="zhuanshengben", days_left=30))
    print(f"     调权 长线 exam_freq={w_long['exam_freq']:.2f}/缺口={w_long['mastery_gap']:.2f}"
          f" → 冲刺 exam_freq={w_sprint['exam_freq']:.2f}/缺口={w_sprint['mastery_gap']:.2f}")
    assert w_sprint["exam_freq"] > w_long["exam_freq"], "冲刺期应上调考频权重"
    assert w_long["mastery_gap"] > w_sprint["mastery_gap"], "长线期应上调缺口权重"
    print("     ✅ adjust_weights 钩子生效：同一套因子，不同阶段不同话语权")

    # ② 多租户隔离：user × role 正交
    uidA, uidB = "selftest_tenantA", "selftest_tenantB"
    for u, r in ((uidA, "zhuanshengben"), (uidB, "kaoyan")):
        repo.upsert_profile(u, r, name=u, goal_payload={}, daily_hours=3.0, subjects=[])
        ensure_kp_tree(repo, u, r, get_pack(r))
    ensure_kp_tree(repo, uidA, "kaoyan", ky)  # 同一人再开一个身份

    a_zsb = repo.count_kp(uidA, "zhuanshengben")
    a_ky = repo.count_kp(uidA, "kaoyan")
    b_zsb = repo.count_kp(uidB, "zhuanshengben")
    print(f"  ② 隔离  A/专升本={a_zsb} A/考研={a_ky} B/专升本={b_zsb}")
    assert a_zsb == 64 and a_ky == 69, "同用户多身份数据串了"
    assert b_zsb == 0, "跨用户数据泄漏"

    kp_x = "核心词汇与词义辨析"
    repo.update_mastery(uidA, "zhuanshengben", kp_x, 0.9)
    vA = repo.get_mastery(uidA, "zhuanshengben", kp_x)
    vB = repo.get_mastery(uidB, "zhuanshengben", kp_x)
    vA_ky = repo.get_mastery(uidA, "kaoyan", kp_x)
    assert vA is not None and vB is None and vA_ky is None, "掌握度跨租户泄漏"
    print(f"     掌握度 A/专升本={vA}  B/专升本={vB}  A/考研={vA_ky} → user×role 正交 ✅")

    # ③ 两个身份在同一份部署里各自跑通完整链路
    for u, r in ((uidA, "zhuanshengben"), (uidB, "kaoyan")):
        plan = generate_daily_plan(repo, u, r)
        lp = generate_long_plan(repo, u, r)
        assert plan["tasks"] and len(lp["milestones"]) == 3, f"{u}/{r} 规划链路异常"
        print(f"     {u:<18} {r:<14} 日计划 {len(plan['tasks'])} 条 · 长期 {len(lp['milestones'])} 阶段")
    print("     ✅ 两份身份包并存，互不干扰")

    # ④ 上线配置项就位
    import app.main as _main  # noqa: E402

    print(f"  ④ 上线配置  env={_main.IS_PROD}  auth={'on' if _main.API_TOKEN else 'off'}  "
          f"cors={_main.ALLOWED_ORIGINS or '通配(仅开发)'}")
    has_health = any(getattr(r, "path", "") == "/health" for r in _main.app.routes)
    assert has_health, "缺少 /health 健康检查端点"
    assert hasattr(_main, "auth_middleware"), "缺少访问令牌中间件"
    print("     ✅ /health 端点 + 令牌中间件 + CORS 可配置 均就位")
    print("  OK L12：第二个身份包落地 · 租户三维隔离 · 上线三件套齐备")

    # ---------------------------------------------------------------
    print(SEP)
    print("18. 每日早报（L13 · 复盘 + 薄弱点 + 资讯）")
    print(SEP)

    from datetime import date as _d, timedelta as _td  # noqa: E402

    from app.agent.briefing import build_briefing, describe_briefing  # noqa: E402

    uidE, roleE = "selftest_briefing", "kaoyan"
    y_day = (_d.today() - _td(days=1)).isoformat()

    repo.upsert_profile(uidE, roleE, name="早报测试",
                        goal_payload={"exam_date": "2027-12-20", "math_type": "数二"},
                        daily_hours=4.0, subjects=["数学"])
    ensure_kp_tree(repo, uidE, roleE, pack)

    # 造"昨天"：有清单、打卡了、完成 1 项、暴露 1 个问题
    y_plan = generate_daily_plan(repo, uidE, roleE, date=y_day, force_refresh=True)
    done_kp = y_plan["tasks"][0]["kp"] if y_plan["tasks"] else "极限的基本计算"
    repo.check_in(uidE, roleE, y_day)
    repo.save_report(
        uidE, roleE, y_day,
        raw="昨天做了几道题",
        parsed={
            "date": y_day, "completed_kps": [done_kp],
            "accuracy": {done_kp: 0.5},
            "problems": ["泰勒展开：阶数选取方法不会"],
            "photos_recognized": [], "needs_followup": False, "followup_question": "",
        },
    )

    # ① 生成（跳过资讯，避免自检依赖外部搜索）
    b = build_briefing(repo, uidE, roleE, with_news=False, force=True)
    print(f"  ① 生成 date={b['date']} yesterday={b['yesterday']}")
    print(f"     复盘: 打卡={b['recap']['checked_in']} 计划={b['recap']['planned_count']} "
          f"完成={len(b['recap']['completed_in_plan'])} 完成率={b['recap']['completion_rate']}")
    print(f"     薄弱点 {len(b['weak_points'])} 个 · 建议 {len(b['suggestions'])} 条")
    assert b["recap"]["checked_in"], "昨日打卡状态没读到"
    assert b["recap"]["has_report"], "昨日汇报没读到"
    assert b["recap"]["completion_rate"] == 1.0, "完成率算错"
    assert b["weak_points"], "薄弱点为空"
    # 昨日暴露的问题必须排第一（设计约定：最新鲜的优先）
    assert "泰勒" in b["weak_points"][0]["kp"], f"薄弱点排序不对: {b['weak_points'][0]}"
    print("     ✅ 事实由算法算（完成率 1.0 / 问题排第一）")

    # ② 幂等：第二次不重算
    b2 = build_briefing(repo, uidE, roleE, with_news=False)
    print(f"  ② 幂等: cached={b2.get('cached')}（应为 True）")
    assert b2.get("cached") is True, "早报幂等失效"

    # ③ 渲染结构
    md = b["markdown"]
    for section in ("## 昨天", "## 今天最该补的", "## 今天怎么走", "## 外面发生了什么"):
        assert section in md, f"Markdown 缺少段落: {section}"
    print(f"  ③ Markdown {len(md)} 字符 · 四段齐全 ✅")

    info = describe_briefing()
    print(f"     事实来源: {info['fact_source']}")
    print("  OK 早报：复盘靠算、薄弱点排序正确、当天幂等、四段渲染完整")

    repo.close()

    print(SEP)
    print("自检通过")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
