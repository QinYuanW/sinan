"""司南 · 业务编排图（L7 · StateGraph 裸写）

═══════════════════════════════════════════════════════════════════════════
这个文件存在的全部意义：掀开 create_agent 那层玻璃
═══════════════════════════════════════════════════════════════════════════

在此之前，司南的"分流"是写在 API 层的一串 if-elif：

    if intent is Intent.CHECKIN:      ...
    elif intent is Intent.REPORT:     ...
    elif intent is Intent.PLAN_DAILY: ...
    else:                             ...

能跑，但有三个硬伤：

  1. **流程藏在函数体里。** 想知道"用户说打卡会走哪些步骤"，只能从上往下读代码。
  2. **没有状态。** 每一步之间只能靠局部变量传递，中间想加"把识别结果暂存"就没地方放。
  3. **加不了环，也加不了人工介入。** 想实现"汇报信息不足 → 回头再问一次 → 再解析"，
     if-elif 里只能写 while + 一堆标志位；想实现"计划生成后等用户点确认"，
     在函数体里根本无处安放。

本文件把同一套业务流改写为 **显式图**：

    START ──► route_node ──(条件边：读 state.intent)──┬──► checkin_node ──► END
                                                       ├──► report_node  ──► END
                                                       ├──► daily_node   ──► END
                                                       ├──► long_draft_node ──► long_confirm_node ──► END
                                                       │    （只算，不存）        （interrupt！确认才落库）
                                                       └──► chat_node    ──► END

图本身是可检视的数据结构：节点名、边、条件映射都能被打印出来
（见 `scripts/trace_graph.py`）。这就是 L7 的验收标准
——「四场景拆成节点图 + 条件边」。

═══════════════════════════════════════════════════════════════════════════
L8 追加：人工介入（interrupt）
═══════════════════════════════════════════════════════════════════════════

L7 回答了「图怎么编排多个步骤」，但没回答一个更硬的问题：
**凭什么所有步骤都要一口气跑完？**

长期规划是几十天、上百个考点的排布。一口气生成完直接写库，
用户连看一眼的机会都没有。真实产品里这一步必须停下来等人点头。

`interrupt()` 就是这个"停下来"：

    answer = interrupt({"question": "这份规划存不存？"})   # ← 图在此挂起
    # ……用户回话后，用 Command(resume="确认") 唤醒……
    # 这次 answer 拿到 "确认"

★ 代价（本关唯一必须记牢的东西）：
    **被中断的那个节点，恢复时从头重跑。**
    官方原话：resumes from the start of the node, re-executing all logic。

    所以 interrupt 之前**不能有副作用**——写库、扣费、发消息、
    调模型（除非幂等），否则用户点一次"确认"，世界就被搅动一次。

    这就是 `long` 为什么拆成两个节点：
        long_draft_node   —— 只算不存（无副作用，结果写进 state）
        long_confirm_node —— 只问不干别的（interrupt 挂起；确认后才落库）
    拆完，恢复时只重跑"问"那一步，"算"的结果安稳躺在 checkpoint 里。

    **节点的粒度 = 重跑的粒度。** 记住这句，就记住了 interrupt。

★ 意外收获（顺带把 L7 那个"没有状态"的坑填了一半）：
    因为 interrupt 强制要求 checkpointer，图被迫上了持久化。
    于是"确认前"的规划不再等于"丢了"——它存在 checkpoints 表里：
        业务库干净（没确认 = plan 表里没有记录）
        图状态完整（关掉页面、明天再来，它还在等你点头）
    「确认才落库」因此**不需要任何额外的草稿表**，plan 表也一行没改。

───────────────────────────────────────────────────────────────────────────
三条设计约定（先看这里，再读代码）
───────────────────────────────────────────────────────────────────────────

① **图是编排层，不是执行层。**
   节点内部仍然调用 `service.handle_*` 和 `sinan_agent.stream_chat`。
   图负责"什么时候走哪条路"，节点负责"这条路怎么走"。
   不在这里重写 ReAct 循环——那会把 B2/B3/B6/B7 的成果全部推倒。

② **节点只写状态，绝不自己 yield。**
   Python 的 generator 不能跨节点传递（`asyncio.to_thread` 里迭代出来的帧
   没法塞进 state 再被下一个节点接着 yield）。
   所以节点的产出统一写进 `state["events"]`，
   由**外层**（api 层）在图跑完后按序下发。分工很干净：
   图管顺序与分支，api 管传输格式。

③ **graph.py 与 chat.py 是并行的两条路。**
   `chat.py` 里有一个开关决定走图还是走原来的 if-elif。
   原链路一行没删——有分歧时可以直接对拍（这是学习型项目的安全做法）。

④ **中断只断"节点"，不断"整张图"。**（L8）
   用户确认后，图只重跑被中断的那个节点，前面跑过的节点结果都在
   checkpoint 里，不会重来。所以拆节点的标准是"重跑代价"：
      副作用重 / 计算贵 的步骤 → 单独一个节点，放在 interrupt 前面或后面
      纯等待人工 的步骤        → 单独一个节点，里面只有 interrupt
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date as _date
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from app.agent.core import service
from app.agent.core.router import Intent, route, strip_at_mentions
from app.agent.report_agent import get_report_graph  # L9：汇报子图
from app.common.logger import logger

# ⚠️ 注意：这里【没有】在模块顶层 import sinan_agent。
#    sinan_agent 顶层会 load_dotenv() 并持有仓储/checkpointer 单例，
#    而 core.service 的 _repo() 又是反向 import sinan_agent —— 直接互导会形成环。
#    因此需要用到它的地方一律延迟导入（与 service._repo 的既有做法保持一致）。


# ======================================================================
# 1. 状态定义 —— 图的"血管"
# ======================================================================
# 为什么用 TypedDict 而不是 Pydantic：
#   LangGraph 的 state 是"通道（channel）"的载体，每次节点返回的局部字典
#   会按 key 合并回全局 state。TypedDict 是最贴近这个语义的声明方式，
#   而且写起来最轻。用 Pydantic 也能跑，但这层不需要校验，只需要形状。


def _last_wins(_old: str | None, new: str | None) -> str | None:
    """reducer：同一次执行中若被写多次，取最后一次（默认行为就是覆盖）。"""
    return new


class SinanState(TypedDict, total=False):
    """一次对话请求在图里流转的全部数据。

    出入参（由 api 层填）：
        message / image_url / thread_id / user_id / role
    图内派生：
        intent        —— route_node 写入，供条件边读取
        cleaned       —— 去掉 @提及后的正文
        events        —— 节点产出的 SSE 事件（有序），api 层据此下发
        meta          —— 图级别的元信息（命中意图、耗时等），供观测/调试
    """

    # ---- 入参 ----
    message: str
    image_url: str | None
    thread_id: str
    user_id: str
    role: str

    # ---- 路由产物 ----
    intent: str

    # ---- 人工确认（L8）----
    # long_draft_node 算出的规划草稿，**不落业务库**。
    # 它的唯一用途：让 long_confirm_node 被重跑时不必重新计算。
    # 换句话说，这个字段就是「节点重跑代价」的解法本身。
    draft_plan: Annotated[dict[str, Any], _last_wins]

    # ---- 节点产出 ----
    # ⚠️ events 这里用 list 并配 `operator.add` 之类的 reducer 会更"函数式"，
    #    但本项目每个分支只会被走一次，用覆盖语义反而更直观、
    #    也避免"上次请求的事件残留到下次"这类幽灵 bug。
    events: Annotated[list[tuple[str, Any]], _last_wins]
    meta: Annotated[dict[str, Any], _last_wins]


# ======================================================================
# 2. 事件构造小工具
# ======================================================================
# 节点产出的 (kind, payload) 二元组与 B6/L6 定义的语义事件保持一致：
#   ("meta",       {"intent": "checkin"})
#   ("text",       "正文片段")
#   ("checkin",    {...}) / ("plan", {...}) / ("replan", {...}) / ...
#   ("tool_call",  {"name": "..."})   —— 只有 chat 分支会产出
#
# 复用同一套 kind，是为了让 api 层的下发循环**不需要为图另写一套**。


def _evt(kind: str, payload: Any) -> tuple[str, Any]:
    return (kind, payload)


# ======================================================================
# 2.5 人工确认的判定（L8）
# ======================================================================
# 用户回话是自由文本："确认" / "好" / "可以，存吧" / "不用了" / "我再想想"。
# 这里用规则法判定，与 router.py 的意图路由同源——零成本、可预测、可测试。
#
# ⚠️ 两条刻意的设计：
#   ① **先判否定，再判肯定。** "不要保存"里既有"不要"又有"保存"，
#      顺序反了就会把拒绝读成同意。宁可多问一次，不可误存一次。
#   ② **判不出来就返回 unclear，绝不默认通过。** 落库是不可逆的对外动作，
#      模型可以说错话，数据库不能写错行。
#
# 【作业】这套规则法很快会遇到边界（"这版不行，改改第三阶段"算通过还是打回？）。
#         正解是让模型判定 + 结构化输出（你 B4 学过的 with_structured_output），
#         但要给它留一个"拿不准就问人"的出口。等你做到那一步再回头改这里。

_APPROVE_WORDS: tuple[str, ...] = (
    "确认", "保存", "同意", "通过", "可以", "好的", "没问题", "就这样",
    "就这么", "存吧", "行吧", "行的", "好吧", "ok", "yes", "嗯嗯", "对",
)

_REJECT_WORDS: tuple[str, ...] = (
    "取消", "不用", "不要", "算了", "放弃", "不存", "重来", "再想想",
    "不满意", "先不", "改改", "打回", "否", "no",
)


def judge_confirmation(text: Any) -> Literal["approve", "reject", "unclear"]:
    """把用户的一句回话判成 同意 / 拒绝 / 说不清。

    返回 unclear 时调用方应当**重新询问**，而不是替用户做主。
    """
    if isinstance(text, bool):  # 前端将来加了按钮，可能直接传布尔
        return "approve" if text else "reject"

    s = str(text or "").strip().lower()
    if not s:
        return "unclear"

    # ① 否定优先（见上方注释为什么）
    if any(w in s for w in _REJECT_WORDS):
        return "reject"
    # ② 再判肯定
    if any(w in s for w in _APPROVE_WORDS):
        return "approve"
    # ③ 拿不准：交给调用方重问
    return "unclear"


# ======================================================================
# 3. 节点定义
# ======================================================================
async def route_node(state: SinanState) -> dict[str, Any]:
    """① 路由节点：判断用户这句话想干什么。

    这是**唯一**产出 `intent` 的地方——条件边只认这个字段。
    把"判断"独立成节点而不是塞在入口，好处是：
      后续要换成 LLM 分类器 / 小模型意图识别，只改这一个节点，
      边和下游节点完全不用动。（这是账本里"识别置信度低 → 回退"的落脚点）
    """
    intent, cleaned = route(state.get("message", ""))
    logger.info(
        f"[司南·graph] 路由 intent={intent.value} "
        f"role={state.get('role')} user={state.get('user_id')}"
    )
    return {
        "intent": intent.value,
        "meta": {"intent": intent.value, "route": "rule_based_v1"},
        # cleaned 不单独占字段：正文清洗是与意图绑定的，直接合进 meta，
        # 避免 state 里出现"半个字段"这种模糊地带。
        "message": cleaned,
    }


async def checkin_node(state: SinanState) -> dict[str, Any]:
    """② 打卡节点：只记出勤，不触发重规划。

    注意节点内部**原样调用 service.handle_checkin**。
    这正是"图是编排层"的含义：图的单位是"一步业务"，不是"一行代码"。
    """
    result = await service.handle_checkin(state["user_id"], state["role"])
    return {
        "events": [
            _evt("meta", {"intent": "checkin"}),
            _evt("text", result["message"] + "\n\n" + result["next_action"]),
            _evt("checkin", {"streak": result["streak"], "date": result["date"]}),
        ]
    }


# ======================================================================
# ③ 汇报节点 —— L9 起升级为「子图」，本文件不再持有它的实现
# ======================================================================
# 这里原本是一个 `report_node` 函数，内部调用 service.handle_report，
# 用 if-else 处理"要不要追问"。L9 把它整体搬进了 `app/agent/report_agent.py`，
# 展开成一张真正的子图：
#
#     START → parse ──(信息够吗?)──┬─否─→ ask（interrupt 追问）
#                ▲                 │        │
#                └─────────────────┘        │   （补充后重新解析）
#                                          └─是─→ apply（落库）
#
# 为什么不把这个结构直接写在主图里：
#   主图是放射状的（route 一分五，各自到 END），塞进一个带环的子流程
#   会让整张图失去"一眼看清主干"的能力。子图给了这段流程一个自己的边界。
#
# 想读它的实现 → `src/app/agent/report_agent.py`
# 想单独跑它   → `get_report_graph().ainvoke({...})`（不需要主图）


# ======================================================================
# 3. 节点定义（续）
# ======================================================================
async def daily_node(state: SinanState) -> dict[str, Any]:
    """④ 今日清单节点。"""
    result = await service.handle_daily_plan(state["user_id"], state["role"])
    return {
        "events": [
            _evt("meta", {"intent": "plan_daily"}),
            _evt("text", render_daily_plan(result["plan"])),
            _evt("plan", {"plan": result["plan"]}),
        ]
    }


async def long_draft_node(state: SinanState) -> dict[str, Any]:
    """⑤ 起草长期规划 —— **只算不存**。

    这是整个 L8 的地基。它必须满足一个苛刻条件：
        「无论被重跑多少次，世界都不变。」

    所以它只做两件事：读数据库、算出一份 plan。
    算完**不写库**，只把结果放进 state 的 draft_plan 字段。

    为什么这么较真：
        它虽然自己不被中断，但它**下游的 long_confirm_node 会被中断**。
        而 LangGraph 恢复时是"重跑被中断的节点"——本例中 long_confirm
        会重跑，但 long_draft **不会**。老话说得好：
        把贵的东西放在不会重跑的地方，把 interrupt 放在没副作用的地方。
    """
    from app.agent.core.planner import generate_long_plan  # 延迟导入，避开环
    from app.agent.sinan_agent import get_repo

    repo = get_repo()
    # ★ persist=False 是这一关的关键参数：算，但不落库
    plan = await asyncio.to_thread(
        generate_long_plan, repo, state["user_id"], state["role"], False
    )

    logger.info(
        f"[司南·graph] 长期规划已起草（未落库）user={state.get('user_id')} "
        f"节点数={len(plan.get('milestones', []))}"
    )

    return {
        "draft_plan": plan,
        "events": [_evt("meta", {"intent": "plan_long"})],
    }


async def long_confirm_node(state: SinanState) -> dict[str, Any]:
    """⑥ 人工确认 —— 图在这里**挂起**，等用户点头才落库。

    ★ 这个节点被重跑时会发生什么？我们把顺序安排好了：

        第一遍：draft_plan 已在 state 里 → interrupt() 抛异常 → 图暂停
                （此时还没碰过数据库）
        第二遍：用户回"确认" → 节点重跑 → 直接走到 interrupt()，
                这次它**返回**确认值而不是抛异常 → 才调用 save_long_plan

    也就是说：**落库那一行代码，只在第二遍才会被执行到。**
    "确认才落库"因此不是靠自觉，是靠代码结构。

    ⚠️ 但请注意第二遍的 interrupt() **之前**仍然执行了 state 读取
       和日志。这就是"interrupt 之前不能有副作用"的实际含义——
       允许读、允许算，禁止写。

    ★ 说不清怎么办：judge_confirmation 返回 unclear 时，这里**再挂起一次**。
        同一个节点里多次 interrupt，LangGraph 按调用顺序匹配 resume 值。
        用户因此会被追问到明确表态为止——而不是被默认保存。
    """
    from app.agent.core.planner import save_long_plan
    from app.agent.sinan_agent import get_repo

    plan = dict(state.get("draft_plan") or {})
    if not plan:
        # 防御：没有草稿就没什么可确认的（正常流程不会走到这）
        return {"events": [_evt("text", "没有待确认的规划，请先生成一份长期规划。")]}

    # ---------- 第一次挂起 ----------
    answer = interrupt(
        {
            "kind": "plan_confirm",
            "intent": "plan_long",
            "question": "确认保存这份长期规划吗？（回复「确认」保存，或「取消」放弃）",
            "display": render_long_plan(plan),  # api 层拿它渲染给用户看
            "plan": plan,                       # 结构化副本，前端将来可画图
        }
    )

    verdict = judge_confirmation(answer)
    logger.info(f"[司南·graph] 长期规划确认判定: {verdict} (原文={str(answer)[:40]!r})")

    # ---------- 说不清 → 再挂起一次 ----------
    # 同一个节点里可以连续 interrupt；LangGraph 按调用序号配对 resume 值。
    # 这实现了"追问到明确为止"。注意：这里的 while 不会真的循环——
    # 每次 run 只会走到下一个 interrupt 就停下，直到集齐所有 resume 值。
    while verdict == "unclear":
        answer = interrupt(
            {
                "kind": "plan_confirm",
                "intent": "plan_long",
                "question": "没太明白。请明确回复「确认保存」或「取消」。",
                "display": render_long_plan(plan),
                "plan": plan,
            }
        )
        verdict = judge_confirmation(answer)
        logger.info(f"[司南·graph] 追问后判定: {verdict} (原文={str(answer)[:40]!r})")

    # ---------- 用户拒绝 ----------
    if verdict == "reject":
        logger.info("[司南·graph] 用户取消了长期规划，未落库")
        return {
            "events": [
                _evt("text", "好的，这份规划**没有保存**。你可以调整目标后重新生成，或者直接告诉我哪里不满意。"),
                _evt("confirm_result", {"approved": False, "plan": plan}),
            ]
        }

    # ---------- 用户同意 → 到这里才碰数据库 ----------
    repo = get_repo()
    await asyncio.to_thread(
        save_long_plan, repo, state["user_id"], state["role"], plan
    )
    logger.info("[司南·graph] 用户确认，长期规划已落库")

    return {
        "events": [
            _evt("text", "✅ 已保存这份阶段规划。\n\n" + render_long_plan(plan)),
            _evt("confirm_result", {"approved": True, "plan": plan}),
        ]
    }


async def chat_node(state: SinanState) -> dict[str, Any]:
    """⑥ 闲聊 / 自主决策节点：把主 Agent 的流式产出收进 events。

    ★ 这是全图**唯一**一个"循环内部再迭代"的节点。

    为什么不能像别的节点那样直接向上 yield：
      节点的返回值必须是普通字典（要合并进 state），generator 无法跨节点传递。
      所以这里先把 Agent 的流消费完、攒成列表，再一次性写回 state。

    ★ 副作用：这会让 chat 分支**失去逐字流式的观感**——模型说完整段才下发。
      这是本次改动已知的、刻意接受的代价（也是留给你的作业）：
          【作业】改成"边跑图边下发"的正解是 LangGraph 的 `astream()` +
                 `stream_mode="custom"` + `get_stream_writer()`，
                 在节点里 writer(...) 直接往下游推事件。
                 学完 L8 的 interrupt 之后再回头处理它，收益最大。
    """
    from app.agent.sinan_agent import stream_chat  # 延迟导入，避开循环依赖

    events: list[tuple[str, Any]] = [_evt("meta", {"intent": "chat"})]

    # ↓ 与原 chat.py 完全一致的参数顺序与语义，保证行为可对拍
    async for kind, payload in stream_chat(
        strip_at_mentions(state.get("message", "")),
        state.get("image_url"),
        state["thread_id"],
        state["user_id"],
        state["role"],
    ):
        events.append(_evt(kind, payload))

    return {"events": events}


# ======================================================================
# 4. 条件边 —— 本关的核心
# ======================================================================
def decide_branch(state: SinanState) -> Literal[
    "checkin", "report", "daily", "long_draft", "chat"
]:
    """条件边函数：只做一件事——把 state.intent 翻译成节点名。

    它的签名（state -> 节点名）就是 LangGraph 对条件边的全部要求。
    返回值必须与 `add_conditional_edges` 的映射表 key 对得上，
    否则运行时会报 "branch returned unknown node"。

    为什么要单独一个映射表而不是直接 `return state["intent"]`：
      意图枚举名和节点名**现在是同名**的巧合。留一层显式映射，
      将来改名（比如 plan_long → long_draft）时只改这张表，不会牵动意图定义。
      —— L8 就正好用上了：`plan_long` 要进的是 `long_draft`（起草），
         不是 `long_confirm`。条件边永远只指向"分支入口"。
    """
    mapping: dict[str, str] = {
        "checkin": "checkin",
        "report": "report",
        "plan_daily": "daily",
        "plan_long": "long_draft",  # ← 长期规划走"起草 → 确认"两跳
        "chat": "chat",
    }
    intent = state.get("intent") or "chat"
    # 兜底：出现未登记的意图时一律降级到 chat，绝不让图走进死路
    return mapping.get(intent, "chat")  # type: ignore[return-value]


# ======================================================================
# 5. 建图 + 编译
# ======================================================================
def build_graph() -> StateGraph:
    """组装图（未编译）。拆出来是为了让 trace 脚本能看"毛坯"结构。"""
    builder = StateGraph(SinanState)

    # ---- 节点 ----
    builder.add_node("route", route_node)
    builder.add_node("checkin", checkin_node)
    # ★ L9：汇报是一个**子图**。add_node 的第二个参数既可以是函数，也可以是
    #   一个编译好的图——LangGraph 会自动把它当子图处理：
    #       · 父图的同名 state 字段自动注入子图
    #       · 子图的同名 state 字段自动冒泡回父图（本例只放行 events）
    #       · 子图里的 interrupt 由父图的 checkpointer 落盘（所以子图自己不挂 saver）
    builder.add_node("report", get_report_graph())
    builder.add_node("daily", daily_node)
    # ★ L8：长期规划拆成两跳。"起草"和"确认"分开，
    #   是因为 interrupt 的重跑单位是**节点**——挤在一个节点里，
    #   恢复时会把"算"也重做一遍。
    builder.add_node("long_draft", long_draft_node)
    builder.add_node("long_confirm", long_confirm_node)
    builder.add_node("chat", chat_node)

    # ---- 边 ----
    builder.add_edge(START, "route")

    # ★ 条件边：从 route 出发，按 decide_branch 的返回值选一条路
    builder.add_conditional_edges(
        "route",
        decide_branch,
        {
            "checkin": "checkin",
            "report": "report",
            "daily": "daily",
            "long_draft": "long_draft",
            "chat": "chat",
        },
    )

    # ★ L8：起草完直接进确认节点（这是图里第一条"业务内部"的边，
    #   前面所有边都是 START→route→分支→END 这种放射状结构）
    builder.add_edge("long_draft", "long_confirm")

    # 各分支的终点收束到 END
    # 注意 long_draft 不在列表里——它的下游是 long_confirm，不是 END
    for node in ("checkin", "report", "daily", "long_confirm", "chat"):
        builder.add_edge(node, END)

    return builder


_graphs: dict[int, CompiledStateGraph] = {}
_conns: list[Any] = []  # aiosqlite 连接，显式关闭用（见 close_graph）


async def get_graph() -> CompiledStateGraph:
    """取编译后的图 —— 按事件循环缓存。

    ★ L8 实测踩到的坑，就记在这里（这是本关第一个真实的报错）：

        用 `ainvoke` 跑图时，checkpointer **必须是异步的**。
        项目原有的 `SqliteSaver`（sinan_agent 用的那个）是同步实现，
        一旦被 async 调用就当场抛：

            NotImplementedError: The SqliteSaver does not support async methods.
            Consider using AsyncSqliteSaver instead.

        所以要换成 `AsyncSqliteSaver`（底层是 aiosqlite，L8 新装的依赖）。
        教训：**同步/异步不是风格问题，是能不能跑的问题。**
        看一个库支不支持异步，最直接的办法是找它的 `a` 前缀方法
        （ainvoke / aget_state / astream）和 aio 版实现类。

    ★ 为什么按事件循环（loop）缓存，而不是全局单例：
        aiosqlite 的连接**绑定创建它的那个 loop**。
        uvicorn 全程只有一个 loop，线上因此只建一次；
        而脚本里每次 `asyncio.run()` 都是新 loop，旧连接会失效。
        按 loop 缓存能同时伺候这两种环境，不用区分线上和调试。

    ★ 与 sinan_agent 那个 checkpointer 的关系：
        两者指向同一个 checkpoints.db、同一张表，但**实例必须不同**：
            会话记忆 -> 同步 SqliteSaver     （create_agent 的 stream 是同步的）
            图状态   -> 异步 AsyncSqliteSaver（图的 ainvoke 是异步的）
        靠 thread_id 前缀区分（见 thread_config）。
    """
    loop_key = id(asyncio.get_running_loop())
    cached = _graphs.get(loop_key)
    if cached is not None:
        return cached

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.agent.sinan_agent import CHECKPOINT_DB  # 复用同一个库文件

    conn = await aiosqlite.connect(str(CHECKPOINT_DB))
    saver = AsyncSqliteSaver(conn)
    await saver.setup()  # 建表；幂等

    # ★ L10：图按规范挂上长期记忆 Store。
    #
    #   为什么要挂：LangGraph 的 Store 是**通过 compile 注入**的，
    #   挂上去之后，节点和工具里才能用 `langgraph.config.get_store()` 拿到它。
    #   不挂的话，节点里调 get_store() 会直接报"no store configured"。
    #
    #   现状与将来：
    #     现在掌握度仍走 repository（同步 API，在 to_thread 里跑），
    #     所以这个 store 暂时没被节点直接使用——它是**接口就位**，
    #     等你要做"Agent 自己记长期记忆"（作业里的语义检索）时立刻能用上。
    #   这也解释了为什么 checkpointer 用异步版、store 用同步版：
    #     前者是图**运行时必须**调用的，后者目前只在节点按需访问，
    #     而 repository 那条路径是同步的（`asyncio.to_thread` 里跑）。
    #     ⚠️ 如果哪天节点里要直接 await store，记得换成 AsyncSqliteStore。
    from app.db.memory_store import get_store as get_memory_store

    graph = build_graph().compile(
        checkpointer=saver,
        store=get_memory_store(),  # L10：长期记忆（跨会话）
    )

    _graphs[loop_key] = graph
    _conns.append(conn)
    logger.info(f"[司南·graph] 业务图已编译（异步 checkpointer + Store · loop={loop_key}）")
    return graph


async def close_graph() -> None:
    """关闭图持有的数据库连接。

    ★ 为什么需要它（自检实测踩到，所以写在这）：
        aiosqlite 的连接背后是一个**非守护线程**。不显式关闭，
        解释器退出时就会一直等这个线程——表现是脚本"打印完最后一行
        却再也回不到命令行"，实测 `exit=124`（被 timeout 杀掉）。
        逻辑全对，但进程不退出，自动化里会被误判成失败。

        uvicorn 场景有进程退出兜底，不会有观感问题；
        脚本 / CLI / 测试必须自己收尾。

    注意：必须在**创建连接的那个事件循环里**调用（连接绑定 loop）。
    """
    for conn in _conns:
        try:
            await conn.close()
        except Exception as exc:  # noqa: BLE001 — 收尾失败不该盖住主流程
            logger.warning(f"[司南·graph] 关闭连接失败: {exc}")
    _conns.clear()
    _graphs.clear()
    logger.info("[司南·graph] 已释放图连接")


def thread_config(thread_id: str, user_id: str, role: str) -> dict[str, Any]:
    """图执行的 config。

    ★ 为什么前缀是 `graph:` 而不是直接复用会话的 thread_id：
        sinan_agent 的会话记忆和图的执行状态共用 checkpoints.db 的同一张表。
        两者虽然都叫 thread_id，但**语义完全不同**：
            会话记忆 = "这个人和我聊过什么"
            图状态   = "这个流程走到哪一步、在等什么"
        混在一起会互相覆盖（比如清空聊天记录顺手把挂起的确认也清了）。
        加前缀 = 给两类状态划了不会串的命名空间。
    """
    return {"configurable": {"thread_id": f"graph:{user_id}:{role}:{thread_id}"}}


async def get_pending_interrupt(
    thread_id: str, user_id: str, role: str
) -> dict[str, Any] | None:
    """查询这个会话是否有"正在等待人工确认"的挂起中断。

    返回 None 表示没有挂起；否则返回中断给用户看的内容。

    ★ 判据是 `snapshot.interrupts`，**不是** `snapshot.next`。
      这里踩过坑，实测数据摊在下面（第一版就是错的）：

        场景                              next               interrupts
        ----------------------------------------------------------------
        从未跑过的 thread                 ()                 ()
        首次挂起（节点内第 1 次 interrupt）  ('long_confirm',)  (Interrupt,)
        含糊回复后再挂起（同节点第 2 次）    ()   ← 空！         (Interrupt,)
        跑完                              ()                 ()

      注意第三行：**同一个节点里第二次 interrupt 时，next 变成空的。**
      因为 LangGraph 对 next 的定义是"下一个待调度的节点"，而这个节点
      已经处于"运行中被挂起"的状态，不算"待调度"，于是 next 空了，
      真正的待办藏在 `tasks` / `interrupts` 里。

      第一版只用 next 判断，结果：用户第一次回答"让我想想"，
      系统正确追问了一句；等用户再说"确认"，系统却**认不出这是在回话**，
      把"确认"当成新消息重新路由，一路走到 chat 节点去闲聊了——
      该存进库的规划，一句没存。

      教训：判断"状态是什么"，要挑**直接表达那个状态**的字段
      （interrupts = "有人正举着手等回答"），而不是挑一个
      在当前场景下**恰好**等价的字段（next）。等价是巧合，不是保证。
    """
    graph = await get_graph()
    snapshot = await graph.aget_state(thread_config(thread_id, user_id, role))

    interrupts = tuple(getattr(snapshot, "interrupts", ()) or ())
    if not interrupts:
        return None

    # 取出待办节点自己抛出的 interrupt 内容（用于回显给用户）
    payload: dict[str, Any] | None = None
    for itr in interrupts:
        value = getattr(itr, "value", None)
        if isinstance(value, dict):
            payload = value
            break

    return {
        "next": list(snapshot.next or ()),  # 仅供展示；判断"是否挂起"别看它
        "interrupt": payload,
    }


# ======================================================================
# 6. 对外执行接口 —— api 层唯一需要碰的东西
# ======================================================================
@dataclass
class GraphRun:
    """一次图执行的结果。

    刻意不直接返回 dict：调用方（api 层）只该关心
    "命中了什么意图"、"有哪些事件要下发"、"是不是停下来等人了"，
    不该看到 state 里那些图内部字段。

    L8 新增三个字段，正好对应 interrupt 的三种状态：
        resumed=False, interrupted=True   -> 这一跑停下来了，正在等人
        resumed=True,  interrupted=False  -> 这一跑是在回答上一次的等人，且跑完了
        resumed=True,  interrupted=True   -> 回答之后又抛了新问题（追问不休）
    """

    intent: str
    events: list[tuple[str, Any]]
    meta: dict[str, Any]
    interrupted: bool = False
    interrupt: dict[str, Any] | None = None
    resumed: bool = False


async def run_graph(
    message: str,
    thread_id: str,
    user_id: str = "default",
    role: str = "kaoyan",
    image_url: str | None = None,
) -> GraphRun:
    """跑一次业务图，返回可下发的语义事件。

    ★ L8 的核心改动：本函数自己判断"这条消息是**新请求**还是**对上一条的回答**"。

        该会话有挂起中断 → 这条消息是回答 → Command(resume=message) 唤醒它
        该会话没有挂起   → 这条消息是新请求 → 正常从头跑

    为什么不让调用方（api 层 / 前端）来判断：
        那样前端就得知道"当前处于什么状态"，还得在 UI 上区分
        "发送"和"确认"两个按钮。而现在前端**一行都没改**——
        用户就是在同一个聊天框里接着打字："确认"。
        这正是 chat.py 开头那句"一个聊天框承载所有交互"的延续。

    ⚠️ 代价：如果用户在等待确认时**不说确认，而是问了个新问题**
        （比如"等一下，考研数学二考不考级数？"），这条消息会被当成
        对规划的答复——judge_confirmation 判成 unclear，于是被再问一遍。
        体验上不算好，但**绝不会误存**。这是刻意的取舍：
        在"确认"这种不可逆动作上，宁可烦一点，不可错一次。

    【作业】想修上面这个体验问题，正解是引入取消/改意图的能力：
        `Command(resume=...)` 之外还有 `Command(goto=...)`，
        可以在恢复的同时把流程**转到另一个节点**去。
        试想：判定为 unclear 且明显是别的问题时，goto 到 chat 节点先回答它，
        再回来继续确认。这是 L9 之后的自然延伸。
    """
    graph = await get_graph()
    config = thread_config(thread_id, user_id, role)

    # ---------- ① 先看这个会话是不是在等人 ----------
    pending = await get_pending_interrupt(thread_id, user_id, role)
    resumed = pending is not None

    if resumed:
        logger.info(
            f"[司南·graph] 该会话正等待 {pending['next']}，本条消息作为答复: "
            f"{str(message)[:40]!r}"
        )
        # Command(resume=...) 唤醒：被中断的节点会重跑，interrupt() 这次返回值而非抛异常
        final_state = await graph.ainvoke(Command(resume=message), config)
    else:
        initial: SinanState = {
            "message": message,
            "image_url": image_url,
            "thread_id": thread_id,
            "user_id": user_id,
            "role": role,
            "events": [],
            "meta": {},
        }
        # ainvoke 而非 invoke：节点里全是 await（service.handle_* 会 to_thread，
        # stream_chat 是异步生成器），必须走异步执行器。
        final_state = await graph.ainvoke(initial, config)

    # ---------- ② 看这一跑是跑完了还是又停下了 ----------
    raw = final_state.get("__interrupt__")
    interrupted = bool(raw)
    interrupt_payload: dict[str, Any] | None = None
    if interrupted:
        first = raw[0]
        value = getattr(first, "value", None)
        interrupt_payload = value if isinstance(value, dict) else {"value": value}

    return GraphRun(
        intent=final_state.get("intent", "chat"),
        events=list(final_state.get("events") or []),
        meta=dict(final_state.get("meta") or {}),
        interrupted=interrupted,
        interrupt=interrupt_payload,
        resumed=resumed,
    )


async def run_graph_async_events(
    message: str,
    thread_id: str,
    user_id: str = "default",
    role: str = "kaoyan",
    image_url: str | None = None,
):
    """★ 可选升级路径：用 astream(stream_mode="updates") 边跑边产出事件。

    与 run_graph 的差别只在于"什么时候能把事件交出去"：
      run_graph              —— 图跑完才有 events（chat 分支因此没有逐字流式）
      run_graph_async_events —— 每个节点一完成就产出它的事件，观感更接近原生

    但它**仍然救不了 chat 的逐字流式**：chat_node 自己是个整体，
    要么全跑完要么不产出。真正的解法见 chat_node 的作业注释
    （`get_stream_writer()`）。

    【作业】把 chat.py 切到这个函数，对比两种下发的时序差异。
    """
    config = thread_config(thread_id, user_id, role)
    initial: SinanState = {
        "message": message,
        "image_url": image_url,
        "thread_id": thread_id,
        "user_id": user_id,
        "role": role,
        "events": [],
        "meta": {},
    }

    graph = await get_graph()
    async for chunk in graph.astream(initial, config, stream_mode="updates"):
        # chunk 形如 {"节点名": {"events": [...], ...}}
        for node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue
            for evt in update.get("events") or []:
                yield node_name, evt


# ======================================================================
# 7. 渲染辅助
# ======================================================================
# ⚠️ 这两个函数是从 api/v1/chat.py **平移**过来的，一行未改。
#
#    为什么不从这里反向 import chat.py：
#      api 层 import core 层是正常方向，core 层 import api 层会形成倒挂——
#      而且 chat.py 一被导入就会建 APIRouter，在脚本里跑图时会平白拉起路由。
#
#    这是本次"图化"暴露出来的一个真实结构问题：
#      "计划 → Markdown" 属于**展示**职责，它既不属于 api 也不属于 core。
#    后续可以抽一个 `app/common/render.py` 收容它。
#    （我故意先不动，留给你判断：抽还是不抽？）


def render_daily_plan(plan: dict[str, Any]) -> str:
    """把今日清单渲染成 Markdown（与 api/v1/chat.py 的 _render_daily_plan 等价）"""
    lines = [f"## 今日作战清单（{plan.get('date', '')}）"]

    if plan.get("carry_over"):
        lines.append("\n### 昨日顺延")
        for t in plan["carry_over"]:
            lines.append(f"- **{t['kp']}**（{t['est_hours']}h）")

    lines.append("\n### 今日任务")
    for i, t in enumerate(plan.get("tasks", []), start=1):
        lines.append(
            f"\n**{i}. {t['kp']}** ｜ {t['subject']} ｜ 预计 {t['est_hours']} 小时 ｜ 优先级 {t['priority']:.2f}"
        )
        lines.append(f"- 为什么先做它：{t['reason']}")
        lines.append(f"- 验收标准：{t['acceptance']}")

    if not plan.get("tasks"):
        lines.append("\n当前没有需要安排的知识点，请先设置目标或补充考点。")

    lines.append(f"\n### 节奏提醒\n{plan.get('coach_note', '')}")
    lines.append(f"\n总预计耗时：{plan.get('total_hours', 0)} 小时")
    return "\n".join(lines)


def render_long_plan(plan: dict[str, Any]) -> str:
    """把长期规划渲染成 Markdown（与 api/v1/chat.py 的 _render_long_plan 等价）"""
    lines = [f"## 长期阶段规划（剩余 {plan.get('days_left', '?')} 天）"]
    lines.append(f"\n{plan.get('coverage_note', '')}")

    for ms in plan.get("milestones", []):
        lines.append(f"\n### {ms['name']}（{ms['start_date']} ~ {ms['end_date']}）")
        lines.append(f"- 阶段说明：{ms['note']}")
        kps = ms.get("target_kps", [])
        preview = "、".join(kps[:12]) + ("…" if len(kps) > 12 else "")
        lines.append(f"- 覆盖考点 {len(kps)} 个：{preview}")
    return "\n".join(lines)


# ======================================================================
# 8. 结构导出 —— 给答辩用
# ======================================================================
def describe_graph() -> dict[str, Any]:
    """把图结构导成可打印的字典。

    这不是调试残余，是**答辩素材**：
      `scripts/trace_graph.py` 用它打印节点与边，
      对照 create_agent 自动生成的那张图（附录 A 实证三），
      就能讲清"手写图 vs 自动图"的差别在哪。
    """
    # 只看结构：不带 checkpointer 编译一份就够了（不需要 interrupt 能力，
    # 也就不必碰异步连接）。describe_graph 因此保持同步，脚本可直接调用。
    drawn = build_graph().compile().get_graph()

    nodes = [n for n in drawn.nodes if not n.startswith("__")]
    edges: list[dict[str, str]] = []
    for e in drawn.edges:
        edges.append({"from": e.source, "to": e.target, "kind": "normal"})

    return {
        "state_fields": list(SinanState.__annotations__.keys()),
        "nodes": nodes,
        "all_nodes": list(drawn.nodes),
        "edges": edges,
        "conditional": {
            "from": "route",
            "decider": decide_branch.__name__,
            "targets": ["checkin", "report", "daily", "long_draft", "chat"],
        },
        "branches": {
            "checkin": "打卡（不触发重规划）",
            "report": "★ 汇报**子图**：parse →（追问环）→ apply；见 report_agent.py",
            "daily": "今日作战清单",
            "long_draft": "起草长期规划（只算不存 · 无副作用）",
            "long_confirm": "★ interrupt 中断点：挂起等确认，确认才落库",
            "chat": "主 Agent 自主决策（图内唯一带循环的节点）",
        },
        "subgraphs": ["report"],
        "interrupts": [
            {
                "node": "long_confirm",
                "next_after": "long_draft",
                "requires_checkpointer": True,
                "resume_with": "Command(resume=<用户回话>)",
                "rule": "被中断节点重跑；故 interrupt 之前禁止副作用",
            }
        ],
    }


__all__ = [
    "SinanState",
    "build_graph",
    "get_graph",
    "close_graph",
    "run_graph",
    "run_graph_async_events",
    "describe_graph",
    "decide_branch",
    "judge_confirmation",
    "thread_config",
    "get_pending_interrupt",
    "GraphRun",
    "render_daily_plan",
    "render_long_plan",
]
