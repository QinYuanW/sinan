"""司南 · 身份包抽象（RolePack）

这是整个平台的扩展点：**加一个身份 = 加一个文件，引擎一行不改。**

一个身份包负责回答四个问题：
    1. 这个身份的目标长什么样？        -> goal_schema
    2. 这个身份的知识体系是什么？      -> build_kp_tree()
    3. 什么该优先做？                  -> factors + weights (+ adjust_weights 钩子)
    4. "汇报"在这个身份下意味着什么？  -> reporter_prompt / parse_hint

引擎层只依赖本文件的接口，不认识任何具体身份。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, Callable

from pydantic import BaseModel


# ======================================================================
# 规划上下文：引擎喂给因子与规划器的一切信息
# ======================================================================
@dataclass
class PlanningContext:
    user_id: str
    role: str
    kp_list: list[dict[str, Any]] = field(default_factory=list)
    mastery_map: dict[str, float] = field(default_factory=dict)
    last_review: dict[str, str] = field(default_factory=dict)
    daily_hours: float = 4.0
    today: str = field(default_factory=lambda: _date.today().isoformat())
    exam_date: str | None = None
    days_left: int | None = None
    recent_logs: list[dict[str, Any]] = field(default_factory=list)
    streak: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------- 供因子使用的便捷派生量 ----------
    @property
    def max_est_hours(self) -> float:
        hours = [float(k.get("est_hours", 2.0)) for k in self.kp_list] or [2.0]
        return max(hours)

    def mastery_of(self, kp_name: str) -> float:
        """没有记录的知识点视为薄弱（0.3）。

        刻意不用 0.5 的中性值：中性值会把真实缺口抹平，
        导致"从未学过"和"学过一半"在排序里得分一样。
        该常量须与 app.db.repository.PRIOR_MASTERY 一致。
        """
        return float(self.mastery_map.get(kp_name, 0.3))

    def days_since_review(self, kp_name: str) -> int | None:
        last = self.last_review.get(kp_name)
        if not last:
            return None
        try:
            return (_date.fromisoformat(self.today) - _date.fromisoformat(last)).days
        except ValueError:
            return None


# ======================================================================
# 优先度因子
# ======================================================================
class PriorityFactor:
    """一个可计算的打分因子。compute() 返回 0-1 的分数，方向由权重决定。

    权重为负表示"越大越不优先"（例如耗时成本）。
    """

    def __init__(self, key: str, label: str, fn: Callable[[PlanningContext, dict[str, Any]], float]):
        self.key = key
        self.label = label
        self._fn = fn

    def compute(self, ctx: PlanningContext, kp: dict[str, Any]) -> float:
        return max(0.0, min(1.0, float(self._fn(ctx, kp))))


# ---------- 通用因子实现（各身份包按需取用） ----------

def factor_exam_freq(ctx: PlanningContext, kp: dict[str, Any]) -> float:
    """考频权重：该知识点在真题中出现的频率，直接来自 KP 树"""
    return float(kp.get("exam_freq", 0.5))


def factor_mastery_gap(ctx: PlanningContext, kp: dict[str, Any]) -> float:
    """掌握度缺口：1 - 当前掌握度。缺口越大越该补"""
    return 1.0 - ctx.mastery_of(kp["name"])


def factor_urgency(ctx: PlanningContext, kp: dict[str, Any]) -> float:
    """时效紧急度（遗忘曲线）。

    - 从未学过/复习过  -> 1.0（全新内容，需要尽早安排）
    - 有复习记录       -> min(1, 距上次复习天数 / 14)，14 天视为完全遗忘
    """
    days = ctx.days_since_review(kp["name"])
    if days is None:
        return 1.0
    return min(1.0, max(0.0, days / 14.0))


def factor_cost(ctx: PlanningContext, kp: dict[str, Any]) -> float:
    """学习成本：预估耗时归一化。配合负权重使用，越耗时越降权"""
    return float(kp.get("est_hours", 2.0)) / ctx.max_est_hours


def factor_difficulty(ctx: PlanningContext, kp: dict[str, Any]) -> float:
    """难度：可作为加分项（难点易失分）或减分项（先捡软柿子）"""
    return float(kp.get("difficulty", 0.5))


# ======================================================================
# 身份包基类
# ======================================================================
class RolePack(ABC):
    # ---- 元信息 ----
    key: str = ""
    name: str = ""
    icon: str = ""
    description: str = ""

    # ---- 目标模型：不同身份的目标字段完全不同，由各自定义 ----
    goal_schema: type[BaseModel] = BaseModel

    # ---- 规划形态 ----
    plan_horizon: str = "long+daily"       # long+daily | weekly
    default_daily_hours: float = 4.0

    # ---- 优先度因子与权重 ----
    factors: list[PriorityFactor] = []
    weights: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 子类必须实现
    # ------------------------------------------------------------------
    @abstractmethod
    def build_kp_tree(self) -> list[dict[str, Any]]:
        """返回该身份的知识体系初始结构。

        每条至少包含: subject / chapter / name / exam_freq / est_hours / difficulty
        """

    @abstractmethod
    def reporter_prompt(self) -> str:
        """汇报解析提示词。不同身份"汇报"的含义不同：
        学生汇报做题情况，教师汇报教学进度与学情。
        """

    @abstractmethod
    def system_prompt(self, ctx: PlanningContext) -> str:
        """主 Agent 的系统提示词（可注入当前目标与画像）"""

    # ------------------------------------------------------------------
    # 可选覆写
    # ------------------------------------------------------------------
    def adjust_weights(self, ctx: PlanningContext) -> dict[str, float]:
        """动态调权钩子。例如临近考试时上调考频权重。

        默认原样返回，子类可按剩余天数等条件覆写。
        """
        return dict(self.weights)

    def output_template(self) -> str:
        return (
            "输出结构：先给结论，再给依据。任务项必须写清「为什么先做它」"
            "与「做完怎么算会了」。"
        )

    def goal_summary(self, profile: dict[str, Any]) -> str:
        """把 goal_payload 渲染成一句人话，供提示词使用"""
        payload = profile.get("goal_payload") or {}
        if not payload:
            return "（尚未设置目标）"
        return "；".join(f"{k}={v}" for k, v in payload.items() if v not in (None, "", []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "icon": self.icon,
            "description": self.description,
            "plan_horizon": self.plan_horizon,
            "goal_fields": self._describe_goal_fields(),
            "factors": [
                {"key": f.key, "label": f.label, "weight": self.weights.get(f.key)}
                for f in self.factors
            ],
        }

    def _describe_goal_fields(self) -> list[dict[str, Any]]:
        """把 goal_schema 的字段描述成前端可直接渲染的表单定义。

        不同身份的目标字段不同，前端据此动态生成表单，
        新增身份时前端无需改动——这是插件化设计的最后一公里。
        """
        fields: list[dict[str, Any]] = []
        for name, info in getattr(self.goal_schema, "model_fields", {}).items():
            annotation = str(info.annotation)
            if "list" in annotation.lower():
                ftype = "list"
            elif "float" in annotation or "int" in annotation:
                ftype = "number"
            else:
                ftype = "text"

            default = info.default
            if default is None or str(default).startswith("PydanticUndefined"):
                default = ""

            fields.append(
                {
                    "name": name,
                    "label": info.description or name,
                    "type": ftype,
                    "default": default if isinstance(default, (str, int, float)) else "",
                }
            )
        return fields


# ======================================================================
# 注册表
# ======================================================================
_REGISTRY: dict[str, RolePack] = {}


def register(pack: RolePack) -> RolePack:
    if not pack.key:
        raise ValueError("RolePack.key 不能为空")
    _REGISTRY[pack.key] = pack
    return pack


def get_pack(key: str) -> RolePack:
    if key not in _REGISTRY:
        available = ", ".join(_REGISTRY) or "(空)"
        raise KeyError(f"未知身份包 '{key}'，可选: {available}")
    return _REGISTRY[key]


def list_packs() -> list[dict[str, Any]]:
    return [p.to_dict() for p in _REGISTRY.values()]


def has_pack(key: str) -> bool:
    return key in _REGISTRY
