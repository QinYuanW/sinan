"""司南 · 生成每日早报（L13）

用法（项目根目录）：
    .venv/Scripts/python.exe scripts/daily_briefing.py
    .venv/Scripts/python.exe scripts/daily_briefing.py --user wangtangyuan --role kaoyan
    .venv/Scripts/python.exe scripts/daily_briefing.py --no-news          # 只复盘，秒回
    .venv/Scripts/python.exe scripts/daily_briefing.py --refresh          # 强制重算
    .venv/Scripts/python.exe scripts/daily_briefing.py --out 早报.md       # 同时存文件

设计说明：
    这个脚本存在的意义是**把"定时"从应用里剥离出来**。
    项目本身不内置调度器（APScheduler 之类），而是提供一个幂等的命令行入口：

        Linux/macOS:  crontab -e  →  30 7 * * * cd /path && .venv/bin/python scripts/daily_briefing.py
        Windows:      任务计划程序 →  每天 7:30 运行同一条命令
        Docker:       由外部 cron 容器或 K8s CronJob 调用

    ★ 为什么不在 FastAPI 进程里起后台定时：
        uvicorn 多 worker 时每个进程都会起一个调度器，同一分钟生成 N 次早报；
        而且进程重启会让"今天还没生成"的状态丢失。
        **定时任务的正确归属是操作系统，不是 Web 进程。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BASE / ".env")

from app.agent.briefing import build_briefing  # noqa: E402
from app.agent.sinan_agent import get_repo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="生成司南每日早报")
    ap.add_argument("--user", default="default", help="用户 ID")
    ap.add_argument("--role", default="kaoyan", help="身份包 key")
    ap.add_argument("--date", default=None, help="日期 YYYY-MM-DD，默认今天")
    ap.add_argument("--no-news", action="store_true", help="跳过资讯（快，不花钱）")
    ap.add_argument("--refresh", action="store_true", help="强制重算，忽略当天缓存")
    ap.add_argument("--out", default=None, help="同时写入的文件路径（Markdown）")
    args = ap.parse_args()

    repo = get_repo()
    try:
        b = build_briefing(
            repo,
            args.user,
            args.role,
            args.date,
            with_news=not args.no_news,
            force=args.refresh,
        )
    except Exception as exc:  # noqa: BLE001 — 命令行入口要把错误说人话
        print(f"[早报生成失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    md = b["markdown"]
    print(md)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n[已写入] {args.out}", file=sys.stderr)

    if b.get("cached"):
        print("\n[提示] 命中当天缓存（加 --refresh 可强制重算）", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
