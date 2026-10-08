"""司南 · 掌握度工具

这些工具是「动态构建」的：需要绑定具体的 user_id 与 role，
因此用工厂函数返回工具列表，而不是模块级函数。

与记忆体系的关系：
    checkpointer（LangGraph）-> 会话内上下文
    本模块操作 mastery 表   -> 跨会话长期画像
模型可以在对话中主动查询与更新画像，但更新必须基于真实证据。
"""

from __future__ import annotations

import json

from langchain.tools import tool

from app.db.repository import SQLiteRepository

# 掌握度更新的平滑系数：单次证据不推翻既有判断，多次证据才推动画像
ALPHA = 0.3


def build_mastery_tools(
    repo: SQLiteRepository, user_id: str, role: str
) -> list:
    """构建绑定用户上下文的掌握度工具集"""

    @tool
    def get_mastery_overview() -> str:
        """查询当前用户的掌握度画像总览：已记录的知识点数、平均掌握度、
        最薄弱的 5 个知识点与最强的 5 个知识点。

        在制定计划前应先调用本工具，了解用户当前的真实水平。
        """
        summary = repo.mastery_summary(user_id, role)
        return json.dumps(summary, ensure_ascii=False)

    @tool
    def get_kp_mastery(kp_name: str) -> str:
        """查询指定知识点的当前掌握度（0-1）。没有记录表示尚未开始学习。

        Args:
            kp_name: 知识点名称，应与考点树中的名称一致
        """
        value = repo.get_mastery(user_id, role, kp_name)
        return json.dumps(
            {"kp_name": kp_name, "mastery": value, "recorded": value is not None},
            ensure_ascii=False,
        )

    @tool
    def list_weak_kps(limit: int = 8) -> str:
        """列出当前最薄弱的知识点（掌握度由低到高），用于确定补漏优先级。

        Args:
            limit: 返回条数，默认 8
        """
        rows = repo.conn.execute(
            """
            SELECT kp_name, subject, mastery, evidence_count
            FROM mastery WHERE user_id = ? AND role = ?
            ORDER BY mastery ASC LIMIT ?
            """,
            (user_id, role, max(1, int(limit))),
        ).fetchall()
        return json.dumps([dict(r) for r in rows], ensure_ascii=False)

    @tool
    def record_practice_result(kp_name: str, accuracy: float) -> str:
        """记录一次作答证据并更新该知识点的掌握度（指数平滑）。

        只有掌握真实证据时才调用。accuracy 是本次作答的正确率（0-1），
        不是主观感觉的"我学会了"。

        Args:
            kp_name: 知识点名称
            accuracy: 本次作答正确率，0-1 之间
        """
        acc = max(0.0, min(1.0, float(accuracy)))
        before = repo.get_mastery(user_id, role, kp_name)
        after = repo.update_mastery(user_id, role, kp_name, acc, alpha=ALPHA)
        return json.dumps(
            {
                "kp_name": kp_name,
                "accuracy": acc,
                "mastery_before": before,
                "mastery_after": after,
                "alpha": ALPHA,
            },
            ensure_ascii=False,
        )

    return [
        get_mastery_overview,
        get_kp_mastery,
        list_weak_kps,
        record_practice_result,
    ]
