"""司南 · 数据访问层

对外暴露一个 SQLiteRepository，屏蔽所有 SQL 细节。
引擎层与工具层只调用这里的方法，不直接写 SQL。

记忆职责划分（重要）：
    LangGraph checkpoints/writes  -> 会话内短期上下文（由 SqliteSaver 管理）
    mastery 表                   -> 跨会话长期画像（由本仓储管理）
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, date as _date
from pathlib import Path
from typing import Any, Iterable

from app.db.models import DB_PATH, get_connection

# 掌握度的先验值：知识点在没有任何证据时的初始判断（0.3 = 薄弱）。
# 必须与 app.agent.roles.base.PlanningContext.mastery_of 的默认取值保持一致，
# 否则"排序时的缺口"与"更新后的画像"会打架。
PRIOR_MASTERY = 0.3


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _today() -> str:
    return _date.today().isoformat()


def _loads(raw: Any, default: Any) -> Any:
    """容错 JSON 反序列化"""
    if raw is None or raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class SQLiteRepository:
    """司南的数据仓储。一个实例可服务多个 user_id / role 组合。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or DB_PATH)
        self.conn: sqlite3.Connection = get_connection(self.db_path)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # 用户画像
    # ------------------------------------------------------------------
    def upsert_profile(
        self,
        user_id: str,
        role: str,
        name: str | None = None,
        goal_payload: dict[str, Any] | None = None,
        daily_hours: float | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        now = _now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO user_profile
                    (user_id, role, name, goal_payload, daily_hours, subjects,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, role) DO UPDATE SET
                    name         = COALESCE(excluded.name, user_profile.name),
                    goal_payload = CASE WHEN excluded.goal_payload = '{}'
                                        THEN user_profile.goal_payload
                                        ELSE excluded.goal_payload END,
                    daily_hours  = excluded.daily_hours,
                    subjects     = CASE WHEN excluded.subjects = '[]'
                                        THEN user_profile.subjects
                                        ELSE excluded.subjects END,
                    updated_at   = excluded.updated_at
                """,
                (
                    user_id,
                    role,
                    name,
                    json.dumps(goal_payload or {}, ensure_ascii=False),
                    float(daily_hours) if daily_hours is not None else 4.0,
                    json.dumps(subjects or [], ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def get_profile(self, user_id: str, role: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM user_profile WHERE user_id = ? AND role = ?",
            (user_id, role),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["goal_payload"] = _loads(data.get("goal_payload"), {})
        data["subjects"] = _loads(data.get("subjects"), [])
        return data

    # ------------------------------------------------------------------
    # 知识点树
    # ------------------------------------------------------------------
    def bulk_upsert_kp(self, user_id: str, role: str, kps: Iterable[dict[str, Any]]) -> int:
        """批量写入知识点。已存在则只更新考频/耗时/难度，不覆盖用户数据。"""
        now = _now()
        count = 0
        with self.conn:
            for kp in kps:
                self.conn.execute(
                    """
                    INSERT INTO kp_node
                        (user_id, role, subject, chapter, name, parent_id,
                         exam_freq, est_hours, difficulty, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (user_id, role, subject, name) DO UPDATE SET
                        chapter    = excluded.chapter,
                        exam_freq  = excluded.exam_freq,
                        est_hours  = excluded.est_hours,
                        difficulty = excluded.difficulty
                    """,
                    (
                        user_id,
                        role,
                        kp["subject"],
                        kp.get("chapter", ""),
                        kp["name"],
                        kp.get("parent_id"),
                        float(kp.get("exam_freq", 0.5)),
                        float(kp.get("est_hours", 2.0)),
                        float(kp.get("difficulty", 0.5)),
                        now,
                    ),
                )
                count += 1
        return count

    def list_kp(
        self,
        user_id: str,
        role: str,
        subject: str | None = None,
        chapter: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM kp_node WHERE user_id = ? AND role = ?"
        params: list[Any] = [user_id, role]
        if subject:
            sql += " AND subject = ?"
            params.append(subject)
        if chapter:
            sql += " AND chapter = ?"
            params.append(chapter)
        sql += " ORDER BY subject, chapter, id"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def count_kp(self, user_id: str, role: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM kp_node WHERE user_id = ? AND role = ?",
            (user_id, role),
        ).fetchone()
        return int(row["c"]) if row else 0

    # ------------------------------------------------------------------
    # 计划
    # ------------------------------------------------------------------
    def save_plan(
        self,
        user_id: str,
        role: str,
        ptype: str,
        date: str,
        payload: dict[str, Any],
        status: str = "active",
    ) -> int:
        """保存计划。同一 (user, role, ptype, date) 视为覆盖更新。"""
        now = _now()
        with self.conn:
            existing = self.conn.execute(
                """
                SELECT id FROM plan
                WHERE user_id = ? AND role = ? AND ptype = ? AND date = ?
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, role, ptype, date),
            ).fetchone()
            raw = json.dumps(payload, ensure_ascii=False)
            if existing:
                self.conn.execute(
                    "UPDATE plan SET payload = ?, status = ?, updated_at = ? WHERE id = ?",
                    (raw, status, now, existing["id"]),
                )
                return int(existing["id"])
            cur = self.conn.execute(
                """
                INSERT INTO plan (user_id, role, ptype, date, payload, status,
                                  created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, role, ptype, date, raw, status, now, now),
            )
            return int(cur.lastrowid)

    def get_plan(
        self, user_id: str, role: str, ptype: str, date: str
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM plan
            WHERE user_id = ? AND role = ? AND ptype = ? AND date = ?
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, role, ptype, date),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["payload"] = _loads(data.get("payload"), {})
        return data

    def list_plans(
        self, user_id: str, role: str, ptype: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM plan
            WHERE user_id = ? AND role = ? AND ptype = ?
            ORDER BY date DESC LIMIT ?
            """,
            (user_id, role, ptype, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = _loads(d.get("payload"), {})
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # 打卡 / 汇报
    # ------------------------------------------------------------------
    def check_in(self, user_id: str, role: str, date: str | None = None) -> dict[str, Any]:
        """打卡：只写 checked_in_at 一个字段，不触发任何重规划。"""
        date = date or _today()
        now = _now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO daily_log
                    (user_id, role, date, checked_in_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, role, date) DO UPDATE SET
                    checked_in_at = COALESCE(daily_log.checked_in_at, excluded.checked_in_at),
                    updated_at    = excluded.updated_at
                """,
                (user_id, role, date, now, now, now),
            )
        return self.get_daily_log(user_id, role, date) or {}

    def save_report(
        self,
        user_id: str,
        role: str,
        date: str | None = None,
        raw: str | None = None,
        photos: list[str] | None = None,
        parsed: dict[str, Any] | None = None,
        actual_hours: float | None = None,
        mastery_delta: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """汇报：写 report_* 字段，由调用方负责触发掌握度更新与重规划。"""
        date = date or _today()
        now = _now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO daily_log
                    (user_id, role, date, report_raw, report_photos, report_parsed,
                     actual_hours, mastery_delta, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, role, date) DO UPDATE SET
                    report_raw    = COALESCE(excluded.report_raw, daily_log.report_raw),
                    report_photos = CASE WHEN excluded.report_photos = '[]'
                                         THEN daily_log.report_photos
                                         ELSE excluded.report_photos END,
                    report_parsed = COALESCE(excluded.report_parsed, daily_log.report_parsed),
                    actual_hours  = COALESCE(excluded.actual_hours, daily_log.actual_hours),
                    mastery_delta = CASE WHEN excluded.mastery_delta = '{}'
                                         THEN daily_log.mastery_delta
                                         ELSE excluded.mastery_delta END,
                    updated_at    = excluded.updated_at
                """,
                (
                    user_id,
                    role,
                    date,
                    raw,
                    json.dumps(photos or [], ensure_ascii=False),
                    json.dumps(parsed, ensure_ascii=False) if parsed else None,
                    actual_hours,
                    json.dumps(mastery_delta or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return self.get_daily_log(user_id, role, date) or {}

    def get_daily_log(self, user_id: str, role: str, date: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM daily_log WHERE user_id = ? AND role = ? AND date = ?",
            (user_id, role, date),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["report_photos"] = _loads(d.get("report_photos"), [])
        d["report_parsed"] = _loads(d.get("report_parsed"), {})
        d["mastery_delta"] = _loads(d.get("mastery_delta"), {})
        return d

    def list_recent_logs(
        self, user_id: str, role: str, limit: int = 14
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM daily_log
            WHERE user_id = ? AND role = ?
            ORDER BY date DESC LIMIT ?
            """,
            (user_id, role, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def check_in_streak(self, user_id: str, role: str) -> int:
        """连续打卡天数（从今天或昨天往回数）"""
        rows = self.conn.execute(
            """
            SELECT date FROM daily_log
            WHERE user_id = ? AND role = ? AND checked_in_at IS NOT NULL
            ORDER BY date DESC
            """,
            (user_id, role),
        ).fetchall()
        if not rows:
            return 0

        from datetime import timedelta

        days = [r["date"] for r in rows]
        today = _date.today()
        # 允许从今天或昨天开始计（今天还没打卡不打断连续）
        start = today if days[0] == today.isoformat() else today - timedelta(days=1)
        if days[0] not in (today.isoformat(), start.isoformat()):
            return 0

        streak = 0
        cursor = start
        day_set = set(days)
        while cursor.isoformat() in day_set:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    # ------------------------------------------------------------------
    # 掌握度画像
    # ------------------------------------------------------------------
    def get_mastery_map(self, user_id: str, role: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT kp_name, mastery FROM mastery WHERE user_id = ? AND role = ?",
            (user_id, role),
        ).fetchall()
        return {r["kp_name"]: float(r["mastery"]) for r in rows}

    def get_mastery(self, user_id: str, role: str, kp_name: str) -> float | None:
        row = self.conn.execute(
            "SELECT mastery FROM mastery WHERE user_id = ? AND role = ? AND kp_name = ?",
            (user_id, role, kp_name),
        ).fetchone()
        return float(row["mastery"]) if row else None

    def update_mastery(
        self,
        user_id: str,
        role: str,
        kp_name: str,
        accuracy: float,
        subject: str | None = None,
        alpha: float = 0.3,
    ) -> float:
        """指数平滑更新掌握度。

            mastery_new = mastery_old + alpha * (accuracy - mastery_old)

        alpha=0.3 意味着单次打卡不会推翻既有判断，需要多次证据才推动画像，
        这样"越用越准"是可解释的，而不是黑箱跳变。

        首次出现的知识点以先验值 PRIOR_MASTERY 起步，走**完全相同**的公式，
        不设特例。若直接以 accuracy 作为初值，等于"做一次题就认定掌握度 60%"，
        会系统性高估——一次做对可能是蒙的、也可能只是这道题简单。
        """
        now = _now()
        accuracy = max(0.0, min(1.0, float(accuracy)))
        old = self.get_mastery(user_id, role, kp_name)

        base = PRIOR_MASTERY if old is None else old
        new = round(base + alpha * (accuracy - base), 4)

        if old is None:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO mastery
                        (user_id, role, kp_name, subject, mastery, evidence_count,
                         last_review_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (user_id, role, kp_name, subject, new, now, now),
                )
        else:
            with self.conn:
                self.conn.execute(
                    """
                    UPDATE mastery
                    SET mastery = ?, evidence_count = evidence_count + 1,
                        subject = COALESCE(?, subject),
                        last_review_at = ?, updated_at = ?
                    WHERE user_id = ? AND role = ? AND kp_name = ?
                    """,
                    (new, subject, now, now, user_id, role, kp_name),
                )
        return new

    def bulk_update_mastery(
        self,
        user_id: str,
        role: str,
        results: dict[str, float],
        subject: str | None = None,
    ) -> dict[str, float]:
        """按 {知识点: 正确率} 批量更新，返回更新后的掌握度"""
        return {
            kp: self.update_mastery(user_id, role, kp, acc, subject=subject)
            for kp, acc in results.items()
        }

    def mastery_summary(self, user_id: str, role: str) -> dict[str, Any]:
        """掌握度概览：总数 / 平均 / 薄弱项"""
        rows = self.conn.execute(
            """
            SELECT kp_name, subject, mastery, evidence_count
            FROM mastery WHERE user_id = ? AND role = ?
            ORDER BY mastery ASC
            """,
            (user_id, role),
        ).fetchall()
        items = [dict(r) for r in rows]
        if not items:
            return {"total": 0, "average": None, "weakest": [], "strongest": []}
        avg = round(sum(i["mastery"] for i in items) / len(items), 4)

        n = len(items)

        # 记录太少时不足以区分强弱，只列薄弱项。
        # 否则同一条会同时出现在"最薄弱"和"已较稳固"里，看起来像 bug。
        if n < 4:
            return {"total": n, "average": avg, "weakest": items, "strongest": []}

        # 两组各取 n/3（2~5 条），保证互不重叠
        k = max(2, min(5, n // 3))
        return {
            "total": n,
            "average": avg,
            "weakest": items[:k],
            "strongest": items[-k:][::-1],
        }
