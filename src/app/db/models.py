"""司南 · 数据层建表 DDL

设计要点：
1. 所有业务表同时带 user_id（多用户隔离）与 role（身份隔离）两个维度。
   平台定位是「对外可演示的作品」，这两个字段现在加是零成本，后期补是灾难。
2. 用户目标字段（考研是考试日期、教师是教学周期）字段本身身份相关，
   因此统一收敛为 goal_payload JSON，由各 RolePack 的 goal_schema 校验。
3. LangGraph 自带的 checkpoints / writes 表由 SqliteSaver 自行管理
   （会话级短期记忆），不在此处创建，两套记忆职责分离：
       checkpoints/writes -> 会话内上下文
       mastery            -> 跨会话长期画像

用法：
    from app.db.models import init_db
    init_db()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# 数据库文件绝对路径（src/app/db/sinan.db），避免依赖启动时的 cwd
DB_PATH: Path = Path(__file__).resolve().parent / "sinan.db"

DDL: list[str] = [
    # ---------- 用户画像 ----------
    """
    CREATE TABLE IF NOT EXISTS user_profile (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       TEXT    NOT NULL,
        role          TEXT    NOT NULL,
        name          TEXT,
        goal_payload  TEXT    NOT NULL DEFAULT '{}',
        daily_hours   REAL    NOT NULL DEFAULT 4.0,
        subjects      TEXT    NOT NULL DEFAULT '[]',
        created_at    TEXT    NOT NULL,
        updated_at    TEXT    NOT NULL,
        UNIQUE (user_id, role)
    )
    """,

    # ---------- 知识点树 ----------
    """
    CREATE TABLE IF NOT EXISTS kp_node (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     TEXT    NOT NULL,
        role        TEXT    NOT NULL,
        subject     TEXT    NOT NULL,
        chapter     TEXT    NOT NULL,
        name        TEXT    NOT NULL,
        parent_id   INTEGER,
        exam_freq   REAL    NOT NULL DEFAULT 0.5,
        est_hours   REAL    NOT NULL DEFAULT 2.0,
        difficulty  REAL    NOT NULL DEFAULT 0.5,
        created_at  TEXT    NOT NULL,
        UNIQUE (user_id, role, subject, name)
    )
    """,

    # ---------- 计划（长期 / 每日） ----------
    """
    CREATE TABLE IF NOT EXISTS plan (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     TEXT    NOT NULL,
        role        TEXT    NOT NULL,
        ptype       TEXT    NOT NULL,
        date        TEXT    NOT NULL,
        payload     TEXT    NOT NULL DEFAULT '{}',
        status      TEXT    NOT NULL DEFAULT 'active',
        created_at  TEXT    NOT NULL,
        updated_at  TEXT    NOT NULL
    )
    """,

    # ---------- 打卡 + 汇报（一张表，两个入口） ----------
    # checked_in_at    <- 打卡入口写（轻量，不触发重规划）
    # report_*         <- 汇报入口写（触发掌握度更新与重规划）
    """
    CREATE TABLE IF NOT EXISTS daily_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         TEXT    NOT NULL,
        role            TEXT    NOT NULL,
        date            TEXT    NOT NULL,
        checked_in_at   TEXT,
        report_raw      TEXT,
        report_photos   TEXT    NOT NULL DEFAULT '[]',
        report_parsed   TEXT,
        actual_hours    REAL,
        mastery_delta   TEXT    NOT NULL DEFAULT '{}',
        created_at      TEXT    NOT NULL,
        updated_at      TEXT    NOT NULL,
        UNIQUE (user_id, role, date)
    )
    """,

    # ---------- 掌握度画像（跨会话长期记忆） ----------
    """
    CREATE TABLE IF NOT EXISTS mastery (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         TEXT    NOT NULL,
        role            TEXT    NOT NULL,
        kp_name         TEXT    NOT NULL,
        subject         TEXT,
        mastery         REAL    NOT NULL DEFAULT 0.5,
        evidence_count  INTEGER NOT NULL DEFAULT 0,
        last_review_at  TEXT,
        updated_at      TEXT    NOT NULL,
        UNIQUE (user_id, role, kp_name)
    )
    """,

    # ---------- 索引 ----------
    "CREATE INDEX IF NOT EXISTS idx_kp_scope ON kp_node (user_id, role, subject)",
    "CREATE INDEX IF NOT EXISTS idx_plan_scope ON plan (user_id, role, ptype, date)",
    "CREATE INDEX IF NOT EXISTS idx_mastery_scope ON mastery (user_id, role)",
    "CREATE INDEX IF NOT EXISTS idx_log_scope ON daily_log (user_id, role, date)",
]


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """打开数据库连接。

    check_same_thread=False 是为了配合 FastAPI 的多线程请求处理。
    """
    conn = sqlite3.connect(str(db_path or DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = None) -> Path:
    """初始化数据库：建表 + 建索引。幂等，可重复调用。"""
    conn = get_connection(db_path)
    try:
        with conn:
            for stmt in DDL:
                conn.execute(stmt)
    finally:
        conn.close()
    return Path(db_path or DB_PATH)


if __name__ == "__main__":
    path = init_db()
    print(f"[司南] 数据库已初始化: {path}")
