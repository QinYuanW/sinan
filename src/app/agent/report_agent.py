"""司南 · 汇报助手（L9 · 真子图）

═══════════════════════════════════════════════════════════════════════════
为什么要把「汇报」从节点升级成子图
═══════════════════════════════════════════════════════════════════════════

L7 的时候，汇报只是主图里的**一个节点** `report_node`：

    async def report_node(state):
        result = await service.handle_report(...)   # ← 一整个函数，几十行
        ...组装事件...

能跑。但它把两件本该分开的事揉在了一起：

  ① **一段完整的业务流程**（解析 → 可能要追问 → 更新画像 → 重规划）
  ② **主图的一个格子**

结果是：主图上写着一个叫 `report` 的方块，你从图上**看不出它内部有分叉、
有循环、有需要停下来问人的地方**。图的说明能力在这里断掉了。

L9 把它展开成一张**子图**（subgraph）：

    主图：  route ──► report ──► END
                        │
                        ▼  展开
    子图：  START ──► parse ──(信息够吗?)──┬─否──► ask
                       ▲                    │      │
                       └────────────────────┘      │
                              （补充后再解析）        │
                                                    │
                       ┌────────────────────────────┘
                       │
                       └─是──► apply ──► END

现在主图那一格变成了"一整个可复用零件"。

───────────────────────────────────────────────────────────────────────────
子图带来的三件实实在在的好处
───────────────────────────────────────────────────────────────────────────

① **可独立测试。**
   本文件可以直接 `get_report_graph().ainvoke({...})` 单独跑——
   不需要主图、不需要路由、不需要别的分支。
   （对照：L7 的 `service.handle_report` 也能单独调，但它是函数不是图，
     你没法"只看它的流程"。）

② **能装下环。**
   "追问 → 补充 → 重新解析"天然是一个环。主图里塞个环会让整张图难读
   （而且主图是放射状的，混进一个环很突兀）；放在子图里，它就是
   "汇报这件事自己的事"。
   ★ 这也修正了 L7 的一句结论：当时说"本关是单发图，不构成环"——
     环不是不能有，是**该有环的地方要有自己的边界**。子图就是这个边界。

③ **两处复用成为可能。**
   `@汇报助手` 是一条入口。将来如果做"定时提醒用户补汇报"或者
   "另一个身份包的汇报流程"，都能直接挂这张子图，不用抄一遍逻辑。

───────────────────────────────────────────────────────────────────────────
状态怎么传（父子图的接口契约）
───────────────────────────────────────────────────────────────────────────

LangGraph 的父子图状态映射规则很简单：**同名的 key 自动对接**。

    父图 SinanState         子图 ReportState
    ─────────────────       ──────────────────
    user_id          ──────► user_id        ✅ 自动传入
    role             ──────► role           ✅
    message          ──────► message        ✅
    image_url        ──────► image_url      ✅
    events           ◄────── events         ✅ 自动冒泡回父图
    （父图没有）      ──────► material       ← 子图私有，不外泄
    （父图没有）      ──────► parse / ask_round

所以这里的规矩是：**只让 `events` 一个字段过界**，
子图内部的中间状态（材料、解析结果、追问轮次）全部留在子图里。
父图不需要知道汇报内部有几个步骤——这正是封装的意义。

───────────────────────────────────────────────────────────────────────────
和 L8 的分工：interrupt 的"重跑" vs 子图的"环"
───────────────────────────────────────────────────────────────────────────

这两个概念长得很像，必须分清（L8 埋的伏笔在这里揭开）：

    interrupt 的重跑：**机制副作用**。用户答一次话，被中断的节点就从头再跑一遍。
                      所以要防——把"算"和"问"拆开（L8 干的事）。
    子图的环：        **设计意图**。追问一轮就该重新解析一次，这是业务要的。
                      所以要控——用 `ask_round` 计数止损，别让用户被问到底。

一个是"不希望发生的重复"，一个是"希望发生的重复"。
分不清这两个，就会一边乱优化、一边写出死循环。

───────────────────────────────────────────────────────────────────────────
关于 handoff（本次没做，说清楚为什么不）
───────────────────────────────────────────────────────────────────────────

LangGraph 里 "handoff" 通常指 `Command(goto="某节点", update={...})`：
**一个节点跑到一半，把控制权连同数据交给另一个节点/子图**。

典型用法：用户在主 Agent 那边闲聊时说了句"我今天做了极限8道题错了3道"，
主 Agent 识别出这是汇报意图，于是主动 handoff 到汇报子图——
而不是让用户先去敲 `@汇报`。

本次没做，是因为它需要主 Agent 具备"意图识别 + 主动转交"的能力，
而那属于 **L10 之后**的话题（Agent 的自主决策与工具化）。

但地基已经打好了：`get_report_graph()` 是一个**可以被 goto 过去的编译产物**。
等你想做时，改动量是一行 `Command(goto=...)`。

    【作业】在主图的 chat 节点里加一个工具 `handoff_to_report`，
            让它返回 `Command(goto="report", update={"message": ...})`。
            注意：goTo 会绕开 route 节点，所以要自己保证 state 齐备。
"""

from __future__ import annotations

import asyncio
from datetime import date as _date
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.agent.core.reporter import parse_report
from app.agent.core.router import strip_at_mentions
from app.common.logger import logger

# 最多追问几轮。
# 为什么必须有：环 + 用户的含糊回答 = 理论上的死循环。
# 三轮是个经验值——超过三轮还说不清，多半不是"没问对"，而是用户自己也还没想清楚，
# 这时候继续追问只会招人烦。宁可先用现有信息给个回执，把"再补一次"的主动权还给他。
MAX_ASK_ROUNDS = 3

DEFAULT_QUESTION = "请补充：今天做了哪些知识点，正确率大概多少？"

# 材料里追问补充的分隔标记，第二次解析时模型能看出哪些是后补的
FOLLOWUP_TAG = "【补充说明】"


# ======================================================================
# 1. 子图状态
# ======================================================================
class ReportState(TypedDict, total=False):
    """汇报子图的状态。

    ⚠️ 字段名就是父子图的接口契约，改名前先看模块开头那张映射表。
    """

    # ---- 入参：与 SinanState 同名，父图自动注入 ----
    user_id: str
    role: str
    message: str
    image_url: str | None
    date: str

    # ---- 子图内部状态：不外泄 ----
    material: str   # 累积的汇报材料（首轮正文 + 历次追问补充）
    parse: dict     # 解析结果。存 dict 不存 Pydantic 对象——
                    # checkpointer 要把它序列化进 SQLite，对象会被拒。
    ask_round: int  # 已追问轮次（环的刹车片）

    # ---- 产出：同名冒泡回父图 ----
    events: list


# ======================================================================
# 2. 节点
# ======================================================================
async def parse_node(state: ReportState) -> dict[str, Any]:
    """① 解析 —— **纯计算，不碰数据库**。

    这是 L8「算存分离」原则的直接继承：
    interrupt 可能让节点重跑，环可能让节点再跑一遍，
    所以凡是会被反复执行的节点，就该是无副作用的。

    本节点会执行两次以上（首轮 + 每轮追问之后），因此它一个字都不许写库。
    落库全部推迟到 apply_node——那是环外，只执行一次。
    """
    role = state.get("role") or "kaoyan"
    date = state.get("date") or _date.today().isoformat()

    # 材料：优先用累积值（ask 节点拼好的），首轮则取正文
    material = (state.get("material") or "").strip()
    if not material:
        material = strip_at_mentions(state.get("message", ""))

    photos = [state["image_url"]] if state.get("image_url") else []

    logger.info(
        f"[司南·report] 解析中 user={state.get('user_id')} "
        f"轮次={state.get('ask_round') or 0} 材料长度={len(material)} 照片={len(photos)}"
    )

    # parse_report 内部会调视觉模型 + 结构化输出模型，是阻塞调用
    parse = await asyncio.to_thread(
        parse_report, material or None, photos, date, role
    )

    logger.info(
        f"[司南·report] 解析结果 needs_followup={parse.needs_followup} "
        f"识别考点={len(parse.completed_kps)} 正确率项={len(parse.accuracy)}"
    )

    return {
        "material": material,
        "parse": parse.model_dump(),
        # 这两个事件只表达"正在干活"，会被 apply 节点的事件覆盖掉。
        # 之所以还留着：万一后面挂了，用户至少看到过系统在工作。
        "events": [
            ("meta", {"intent": "report"}),
            ("status", {"text": "正在解析汇报内容…"}),
        ],
    }


async def ask_node(state: ReportState) -> dict[str, Any]:
    """② 追问 —— 图在这里**挂起**（复用 L8 的 interrupt）。

    ★ 这个节点恢复时会被重跑，所以：
        interrupt() 之前——只允许"读 state + 拼字符串"；
        interrupt() 之后——把用户的回答并进材料，然后丢回 parse。

    ★ 为什么把 `partial` 一起递给用户：
        只说"信息不够，请补充"是很敷衍的——用户不知道自己刚才哪儿没讲清。
        把"我已经读到了什么"摆出来，他才知道该补哪一块。
        这个字段是本子图里唯一一处**纯体验**设计，但体验就是产品。
    """
    parse = state.get("parse") or {}
    question = (parse.get("followup_question") or "").strip() or DEFAULT_QUESTION
    round_no = int(state.get("ask_round") or 0)

    logger.info(f"[司南·report] 第 {round_no + 1} 轮追问，挂起等人")

    answer = interrupt(
        {
            "kind": "report_followup",
            "intent": "report",
            "question": question,
            "round": round_no + 1,
            "max_rounds": MAX_ASK_ROUNDS,
            "partial": _partial_summary(parse),  # 让用户看到系统已理解的部分
        }
    )

    # ↓ 以下只在恢复后执行
    merged = (state.get("material") or "").strip()
    supplement = str(answer or "").strip()
    merged = f"{merged}\n\n{FOLLOWUP_TAG}{supplement}" if merged else supplement

    return {"material": merged, "ask_round": round_no + 1}


async def apply_node(state: ReportState) -> dict[str, Any]:
    """③ 落库 —— **全子图唯一有副作用的节点**。

    它在环外面，所以只会被执行一次。把"写"集中到唯一一个点上，
    是这张子图最容易讲清楚的设计：想知道汇报动了什么数据，只看这一个函数。

    （对照起来看：`service.handle_report` 把解析、落库、重规划揉在一段流程里，
      读代码时要一路读下去才知道碰了哪些表。这就是"函数式编排"和"图编排"
      最直观的差别——**图把'碰数据'变成了一个可以单独指出来的节点**。）
    """
    # 延迟导入：这些模块在本文件被 langgraph.json 加载时不一定要马上可用
    from app.agent.core.planner import replan_after_report
    from app.agent.core.service import _report_summary
    from app.agent.sinan_agent import get_repo
    from app.models.schemas import ReportParse

    repo = get_repo()
    user_id = state.get("user_id") or "default"
    role = state.get("role") or "kaoyan"
    date = state.get("date") or _date.today().isoformat()
    parse_dict = dict(state.get("parse") or {})
    material = state.get("material") or ""
    photos = [state["image_url"]] if state.get("image_url") else []

    events: list[tuple[str, Any]] = [("meta", {"intent": "report"})]

    # 把 dict 还原成 Pydantic 模型（state 里只能放可序列化的东西）
    try:
        parse = ReportParse(**parse_dict)
    except Exception as exc:  # noqa: BLE001 — 状态损坏时降级为追问
        logger.error(f"[司南·report] 解析结果还原失败: {exc}")
        events.append(("text", "汇报解析结果异常，请重新描述一次。"))
        return {"events": events}

    # ---------- 信息仍不足（追问到上限后的止损分支）----------
    if parse.needs_followup or not parse.accuracy:
        logger.info("[司南·report] 信息不足，不更新画像（仅存档原始材料）")
        # 原始材料照样落库：用户说的话不该因为"没解析明白"就丢了
        await asyncio.to_thread(
            repo.save_report, user_id, role, date,
            raw=material or None, photos=photos, parsed=parse_dict,
        )
        question = parse.followup_question or DEFAULT_QUESTION
        events.append((
            "text",
            f"这次的信息还不足以判断掌握程度（已追问 {state.get('ask_round') or 0} 轮），"
            f"我先把你说的话存档了，**画像这次没动**。\n\n要更新画像的话，可以再发一次：\n\n{question}",
        ))
        events.append(("followup", {"question": question}))
        return {"events": events}

    # ---------- 信息足够 → 更新画像 + 按需重规划 ----------
    result = await asyncio.to_thread(
        replan_after_report,
        repo, user_id, role, date,
        parse.accuracy, parse.completed_kps, parse.problems,
    )

    # 一次性落库：原始材料 + 解析结果 + 掌握度变化
    # （对照 service.handle_report 里的两次 save_report，这里合并成一次）
    await asyncio.to_thread(
        repo.save_report, user_id, role, date,
        raw=material or None,
        photos=photos,
        parsed=parse_dict,
        mastery_delta=result["mastery_delta"],
    )

    logger.info(
        f"[司南·report] 完成 user={user_id} 更新{len(result['mastery_delta'])}个知识点 "
        f"计划变更={result['changed']} 追问{state.get('ask_round') or 0}轮"
    )

    events.append(("text", _report_summary(parse, result)))
    events.append((
        "replan",
        {
            "changed": result["changed"],
            "mastery_delta": result["mastery_delta"],
            "reason": result["replan_reason"],
        },
    ))
    if result.get("next_plan"):
        events.append(("plan", {"plan": result["next_plan"]}))

    return {"events": events}


# ======================================================================
# 3. 条件边
# ======================================================================
def need_more(state: ReportState) -> Literal["ask", "apply"]:
    """条件边：信息够不够？不够就追问，够了（或问到上限）就去落库。

    ★ 刹车片的顺序很讲究：
        先判"够了"—— 够了就直接走，不受轮次限制；
        再判"到上限"—— 只有"不够 + 已经问了 N 轮"才刹车。
      反过来写的话，第三轮即使问清楚了也会被强行扣下，白白浪费一次好数据。
    """
    parse = state.get("parse") or {}
    enough = (not parse.get("needs_followup")) and bool(parse.get("accuracy"))
    if enough:
        return "apply"

    round_no = int(state.get("ask_round") or 0)
    if round_no >= MAX_ASK_ROUNDS:
        logger.info(f"[司南·report] 追问已达上限 {MAX_ASK_ROUNDS} 轮，停止追问")
        return "apply"

    return "ask"


# ======================================================================
# 4. 建图
# ======================================================================
def build_report_graph() -> StateGraph:
    """组装汇报子图（未编译）。"""
    g = StateGraph(ReportState)

    g.add_node("parse", parse_node)
    g.add_node("ask", ask_node)
    g.add_node("apply", apply_node)

    g.add_edge(START, "parse")

    # ★ 子图里就有条件边：和主图那条 decide_branch 是同一种东西，
    #   只是判据从"用户想干什么"变成了"这条汇报说清楚了吗"。
    g.add_conditional_edges("parse", need_more, {"ask": "ask", "apply": "apply"})

    # ★ 环：追问完回到解析。L7 说"本关不构成环"——环在这里出现了，
    #   而且它被关在子图里，主图仍然是一张干净的放射状图。
    g.add_edge("ask", "parse")

    g.add_edge("apply", END)

    return g


_report_graph: CompiledStateGraph | None = None


def get_report_graph() -> CompiledStateGraph:
    """编译后的汇报子图（单例）。

    ★ 注意这里 **没有 checkpointer**，这是刻意的：
        子图的状态由父图的 checkpointer 统一托管。
        如果子图自己也挂一个 saver，就会变成两层持久化——
        挂起时父子各写一份，恢复时到底听谁的会变得很难推理。
        规矩记好：**checkpointer 只在最外层挂一次。**

    ★ 与主图 `get_graph()` 的另一个差别：这里不用按事件循环缓存。
        没有 aiosqlite 连接，编译产物就是纯结构，可以安全地跨 loop 复用。
        （L8 那个按 loop 缓存的写法，是被 aiosqlite 的连接绑定逼出来的。）
    """
    global _report_graph
    if _report_graph is None:
        _report_graph = build_report_graph().compile()
        logger.info("[司南·report] 汇报子图已编译")
    return _report_graph


# ======================================================================
# 5. 渲染辅助
# ======================================================================
def _partial_summary(parse: dict[str, Any]) -> str:
    """把"已经读出来的部分"拼成一句话，追问时一起给用户看。"""
    bits: list[str] = []
    kps = parse.get("completed_kps") or []
    if kps:
        preview = "、".join(kps[:3]) + ("…" if len(kps) > 3 else "")
        bits.append(f"已完成知识点：{preview}")
    acc = parse.get("accuracy") or {}
    if acc:
        bits.append("已读到的作答：" + "、".join(f"{k} {v:.0%}" for k, v in list(acc.items())[:3]))
    problems = parse.get("problems") or []
    if problems:
        bits.append(f"已识别问题 {len(problems)} 条")
    return "；".join(bits) if bits else ""


# ======================================================================
# 6. 结构导出（给 trace 脚本与答辩用）
# ======================================================================
def describe_report_graph() -> dict[str, Any]:
    """导出子图结构。"""
    drawn = build_report_graph().compile().get_graph()
    nodes = [n for n in drawn.nodes if not n.startswith("__")]
    edges = [{"from": e.source, "to": e.target} for e in drawn.edges]

    return {
        "nodes": nodes,
        "all_nodes": list(drawn.nodes),
        "edges": edges,
        "conditional": {
            "from": "parse",
            "decider": need_more.__name__,
            "targets": ["ask", "apply"],
        },
        "loop": "ask → parse  （追问后重新解析）",
        "max_ask_rounds": MAX_ASK_ROUNDS,
        "interrupt": {
            "node": "ask",
            "kind": "report_followup",
            "resume_with": "Command(resume=<用户的补充说明>)",
        },
        "state_fields": list(ReportState.__annotations__.keys()),
        "shared_with_parent": ["user_id", "role", "message", "image_url", "events"],
        "private_fields": ["date", "material", "parse", "ask_round"],
        "branches": {
            "parse": "解析汇报材料（纯计算 · 不落库）",
            "ask": "★ interrupt 中断点：追问到明确为止，最多 3 轮",
            "apply": "★ 唯一有副作用的节点：更新画像 + 重规划 + 一次性落库",
        },
    }


__all__ = [
    "ReportState",
    "build_report_graph",
    "get_report_graph",
    "describe_report_graph",
    "need_more",
    "MAX_ASK_ROUNDS",
]
