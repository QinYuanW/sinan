"""司南 · 演示数据准备

给指定用户灌入一份可演示的状态：目标 + 打卡 + 今日清单 + 若干汇报。
用途：演示前快速把界面填满，或在新环境里复现展示效果。

用法：
    .venv/Scripts/python.exe scripts/seed_demo.py [user_id] [--reset]

    --reset  先清空该用户的计划 / 打卡 / 掌握度记录，避免重复灌入时数据叠加
"""

from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8010/api/v1"
ROLE = "kaoyan"

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
UID = _args[0] if _args else "default"
RESET = "--reset" in sys.argv

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def reset_user(uid: str) -> None:
    db = Path(__file__).resolve().parent.parent / "src" / "app" / "db" / "sinan.db"
    if not db.exists():
        return
    con = sqlite3.connect(str(db))
    with con:
        for table in ("plan", "daily_log", "mastery"):
            con.execute(f"DELETE FROM {table} WHERE user_id = ? AND role = ?", (uid, ROLE))
    con.close()
    print(f"  已清空 {uid} 的历史计划 / 打卡 / 掌握度")


def call(method: str, path: str, body: dict | None = None):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    with opener.open(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print(f"[司南] 为 {UID} 准备演示数据 …")
    if RESET:
        reset_user(UID)

    r = call("PUT", "/profile", {
        "user_id": UID,
        "role": ROLE,
        "goal_payload": {
            "target_school": "华东师范大学",
            "target_major": "电子信息（专硕）",
            "exam_date": "2026-12-19",
            "math_type": "数二",
            "target_score": 360,
            "subjects": ["数学二", "英语二", "政治", "专业课"],
        },
        "daily_hours": 6.0,
    })
    print(f"  目标: {r.get('goal_summary')}")

    plan = call("POST", "/plan/daily", {"user_id": UID, "role": ROLE})["plan"]
    print(f"  今日清单: {len(plan['tasks'])} 条 / 共 {plan['total_hours']} 小时")
    for t in plan["tasks"]:
        print(f"    [{t['priority']:.2f}] {t['kp']}")

    log = call("POST", "/log/checkin", {"user_id": UID, "role": ROLE})
    print(f"  {log.get('message')}")

    # 汇报若干条，让掌握度画像有层次（依赖大模型，失败不影响其余演示数据）
    reports = [
        "今天把「矩阵的秩」做了 6 道题，对了 3 道，求秩的时候初等变换老是算错。",
        "「极限的基本计算（等价无穷小 / 洛必达 / 泰勒）」做了 10 道对了 8 道，泰勒展开阶数偶尔取错。",
        "「线性相关与线性无关的判定」只做了 5 道，对了 2 道，定义会用但证明题没思路。",
    ]
    for text in reports:
        try:
            rep = call("POST", "/log/report", {"user_id": UID, "role": ROLE, "text": text})
            if rep.get("status") == "ok":
                delta = rep.get("mastery_delta") or {}
                print(f"  汇报: 掌握度 {delta} | 计划变更={rep.get('plan_changed')}")
            else:
                print(f"  汇报: 模型要求追问（{rep.get('status')}），跳过")
        except Exception as exc:  # noqa: BLE001 — 演示数据不应因模型失败而中断
            print(f"  汇报: 跳过（{type(exc).__name__}: {exc}）")

    print("[司南] 完成。打开 http://127.0.0.1:8010/ 查看。")


if __name__ == "__main__":
    main()
