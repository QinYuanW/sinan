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
    repo.close()

    print(SEP)
    print("自检通过")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
