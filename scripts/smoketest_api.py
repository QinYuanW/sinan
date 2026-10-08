"""司南 · API 冒烟测试（真实 HTTP 调用，验证服务端而非仅模块）"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
UID = "smoke"
ROLE = "kaoyan"

# 绕过系统代理（本机 curl/urllib 默认会走代理导致 502）
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(method: str, path: str, body: dict | None = None):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:400]


def main() -> int:
    sep = "=" * 66

    print(sep, "\n1. PUT /profile  设置目标（goal_schema 校验）\n", sep)
    status, data = call("PUT", "/profile", {
        "user_id": UID, "role": ROLE,
        "goal_payload": {
            "target_school": "华东某大学", "target_major": "电子信息",
            "exam_date": "2026-12-19", "math_type": "数二",
        },
        "daily_hours": 6.0,
    })
    print(f"  HTTP {status}")
    print(f"  {data.get('message') if isinstance(data, dict) else data}")
    if isinstance(data, dict):
        print(f"  目标摘要: {data.get('goal_summary')}")

    print(sep, "\n2. PUT /profile  脏数据校验（math_type 传错类型应被拦截或转换）\n", sep)
    status, data = call("PUT", "/profile", {
        "user_id": UID, "role": ROLE, "goal_payload": {"target_score": "不是数字"},
    })
    print(f"  HTTP {status} -> {'已拦截（422）' if status == 422 else '接受'}")

    print(sep, "\n3. POST /plan/daily  生成今日清单\n", sep)
    status, data = call("POST", "/plan/daily", {"user_id": UID, "role": ROLE})
    print(f"  HTTP {status}")
    plan = data.get("plan", {})
    print(f"  日期 {plan.get('date')} | 任务 {len(plan.get('tasks', []))} 条 | "
          f"总耗时 {plan.get('total_hours')}h / 预算 {plan.get('budget_hours')}h")
    for t in plan.get("tasks", []):
        partial = " [部分]" if t.get("partial") else ""
        print(f"    [{t['priority']:.2f}] {t['kp']}{partial} 今日{t['planned_hours']}h/共{t['est_hours']}h")
        print(f"           {t['reason']}")
    print(f"  提醒: {plan.get('coach_note', '')[:140]}")

    print(sep, "\n4. GET /plan/long  长期规划\n", sep)
    status, data = call("GET", f"/plan/long?user_id={UID}&role={ROLE}")
    lp = data.get("plan", {})
    print(f"  HTTP {status}")
    print(f"  {lp.get('coverage_note')}")
    for m in lp.get("milestones", []):
        print(f"    {m['name']} {m['start_date']}~{m['end_date']} 覆盖 {len(m['target_kps'])} 考点")

    print(sep, "\n5. POST /log/checkin  打卡\n", sep)
    status, data = call("POST", "/log/checkin", {"user_id": UID, "role": ROLE})
    print(f"  HTTP {status} | {data.get('message')}")

    print(sep, "\n6. GET /log/today  今日状态\n", sep)
    status, data = call("GET", f"/log/today?user_id={UID}&role={ROLE}")
    print(f"  HTTP {status} | 已打卡={data.get('checked_in')} 已汇报={data.get('reported')} "
          f"连续={data.get('streak')}")

    first_kp = plan.get("tasks", [{}])[0].get("kp", "")
    print(sep, f"\n7. POST /log/report  汇报（明确信息，应直接记录）\n  {first_kp}\n", sep)
    status, data = call("POST", "/log/report", {
        "user_id": UID, "role": ROLE,
        "text": f"我今天把「{first_kp}」做了 8 道题，对了 5 道。",
    })
    print(f"  HTTP {status}")
    if isinstance(data, dict):
        print(f"  状态: {data.get('status')}")
        if data.get("status") == "ok":
            print(f"  掌握度变化: {data.get('mastery_delta')}")
            print(f"  计划变更: {data.get('plan_changed')}")
            print(f"  处理说明: {data.get('replan_reason', '')[:260]}")
            nxt = data.get("next_plan") or {}
            if nxt:
                print(f"  次日清单: {len(nxt.get('tasks', []))} 条, "
                      f"顺延 {len(nxt.get('carry_over', []))} 条")
                for t in nxt.get("tasks", [])[:3]:
                    print(f"      [{t['priority']:.2f}] {t['kp']}")
        else:
            print(f"  追问: {data.get('question', '')[:200]}")

    print(sep, "\n7b. POST /log/report  汇报（模糊信息，应追问不猜）\n", sep)
    status, data = call("POST", "/log/report", {
        "user_id": UID, "role": ROLE,
        "text": "今天学了会儿数学，感觉还行。",
    })
    print(f"  HTTP {status} | 状态={data.get('status') if isinstance(data, dict) else data}")
    if isinstance(data, dict) and data.get("status") != "ok":
        print("  -> 正确：信息不足时追问而不是编造正确率")

    print(sep, "\n8. GET /mastery  掌握度画像\n", sep)
    status, data = call("GET", f"/mastery?user_id={UID}&role={ROLE}")
    s = data.get("summary", {})
    print(f"  HTTP {status} | 已记录 {s.get('total')} 个点 | 平均 {s.get('average')}")
    for i in (s.get("weakest") or [])[:4]:
        print(f"    {i['mastery']:.3f}  {i['kp_name']}")

    print(sep, "\n9. GET /roles  身份包（前端动态表单的数据源）\n", sep)
    status, data = call("GET", "/roles")
    for r in data.get("roles", []):
        print(f"  {r['icon']} {r['name']:<6} 目标字段 {len(r['goal_fields'])} 个 | "
              f"因子 {[f['label'] for f in r['factors']]}")

    print(sep, "\n冒烟测试结束\n", sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
