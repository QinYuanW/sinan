"""司南 · 长期记忆（L10 · 官方 Store）

═══════════════════════════════════════════════════════════════════════════
checkpointer vs Store —— L10 唯一必须分清的一组对照
═══════════════════════════════════════════════════════════════════════════

    ┌──────────┬────────────────────────┬──────────────────────────────┐
    │          │ checkpointer           │ Store                        │
    ├──────────┼────────────────────────┼──────────────────────────────┤
    │ 记忆单元  │ 一个 thread（会话）      │ 一个 namespace（用户/身份）    │
    │ 隔离键    │ thread_id              │ namespace 元组                │
    │ 生命周期  │ 会话内（可跨进程）        │ 长期（跨会话、跨月份）          │
    │ 回答      │ "这次对话说到哪了"       │ "这个人是谁、他学到哪了"        │
    │ 典型内容  │ 消息历史、挂起的确认      │ 掌握度画像、偏好、笔记          │
    │ 底层表    │ checkpoints.db         │ memory_store.db              │
    └──────────┴────────────────────────┴──────────────────────────────┘

一句话记法：**checkpointer 记"这次聊了什么"，Store 记"这个人是谁"。**

司南的落点：
    checkpoints.db  ← checkpointer（L5 会话记忆 + L8 图状态，已在用）
    memory_store.db ← Store（掌握度画像，L10 从 mastery 表换过来）

───────────────────────────────────────────────────────────────────────────
为什么掌握度"该"放进 Store
───────────────────────────────────────────────────────────────────────────

1. **它天生跨会话。** 今天聊的和明天聊的是两个 thread_id，
   但掌握度必须连续。塞进 checkpointer 的话，会话一换画像就断片。

2. **命名空间层级天然匹配。** `(user_id, role, "mastery")`——
   想跨身份看"这人在所有身份下都学了什么"，一句 `search(("u1",))` 就够。

3. **官方能力白拿。** `filter` 服务端下推、TTL 过期、可选的语义检索（见文末作业）。

4. **Agent 能自己拿到。** 图挂上 store 后，节点和工具用 `get_store()`
   就能读写长期记忆，不必把 repository 一路注入下去。

───────────────────────────────────────────────────────────────────────────
为什么【不该】什么都往 Store 塞（这条比上面更重要）
───────────────────────────────────────────────────────────────────────────

    · 考点树 `kp_node` —— 有明确 schema 与层级关系，是**领域数据**
    · 计划 `plan`      —— 要幂等读、按日期查、带状态机，是**事务数据**
    · 打卡 `daily_log` —— 同上，还要算连续天数

    判断标准就一句话：
        这条数据是「关于用户本人的记忆」，还是「系统运作的业务记录」？
        前者给 Store，后者留给 SQL。

把一个需要"按日期范围聚合 + 幂等 upsert + 状态流转"的表塞进 KV，
是用框架的优雅换工程的痛苦。**知道边界在哪，比会用 API 更值钱。**
（这也是本项目一直保留 SQL 层、只把"画像"这一件事搬过来的原因。）

───────────────────────────────────────────────────────────────────────────
【作业】三个 Store 能力，本次没用上，留给你
───────────────────────────────────────────────────────────────────────────

1. **语义检索**。`put(..., index=["text"])` + `search(query="哪些跟极限有关")`
   可以让"我哪个知识点不熟"支持自然语言查询。需要配 embedding 模型
   （智谱有 embedding-3，`.env` 里已有 key）。这是 Store 相比自建表**最大的增量**。
2. **TTL**。`put(..., ttl=30*24*3600)` 让"临时提示"这类记忆自动过期。
3. **跨身份聚合**。`search(("u1",))` 已经能用，但产品上还没做入口——
   比如"我在考研和教师两个身份下的总学习时长"。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.store.sqlite import SqliteStore

from app.common.logger import logger
from app.db.models import DB_PATH

# 长期记忆库与业务库、会话库并列，互不干扰
STORE_DB = Path(DB_PATH).parent / "memory_store.db"

# 命名空间第三层：记忆类型。
# 现在只有掌握度；将来加"用户偏好""学习笔记"时挂同级，不会串。
MEMORY_KIND = "mastery"

# ⚠️ Store 的 search 默认 limit=10 —— 不显式放大就会**静默丢数据**
#（考研有 69 个考点，只取 10 条会让画像凭空残缺，且不报任何错）
SEARCH_LIMIT = 1000


def use_store() -> bool:
    """掌握度是否走 Store。

    与 L7 的 `SINAN_USE_GRAPH` 同源做法：**可切换、可对拍、可回退**。
    默认关闭（走原来的 mastery 表），保证既有行为不变。
    每次实时读环境变量，方便自检脚本在同一进程里对比两种模式。
    """
    return os.getenv("SINAN_USE_STORE", "0").strip() in {"1", "true", "True", "yes"}


def mastery_namespace(user_id: str, role: str) -> tuple[str, ...]:
    """掌握度的命名空间：`(用户, 身份, 记忆类型)`。

    三层不是装饰：
        第一层 user —— 用户之间的隔离
        第二层 role —— 同一个人的考研 / 教师 / 专升本各自独立
        第三层 kind —— 与将来的其它记忆类型分开

    好处是可以按**前缀**检索：`search(("u1",))` 一次拿到该用户所有身份的记忆。
    （SQL 实现同样的事，得 join 或 UNION；KV 的层级命名空间天然支持。）
    """
    return (user_id, role, MEMORY_KIND)


_store: SqliteStore | None = None


def get_store() -> SqliteStore:
    """Store 单例（SQLite 持久化）。

    ★ 三个必须这么写的细节（实测踩坑换来的）：
      1. `isolation_level=None`（autocommit）—— Store 内部显式 `BEGIN`/`COMMIT` 管事务，
         而 sqlite3 默认走隐式事务，两者撞车会直接抛
         `cannot start a transaction within a transaction`。
      2. `check_same_thread=False` —— 我们多在 `asyncio.to_thread` 里调用它，连接要跨线程。
      3. 建表不用管 —— Store 首次访问时会自己 `setup()`。
    """
    global _store
    if _store is None:
        conn = sqlite3.connect(
            str(STORE_DB), check_same_thread=False, isolation_level=None
        )
        _store = SqliteStore(conn)
        logger.info(f"[司南·store] 长期记忆库已连接: {STORE_DB.name}")
    return _store


def reset_store() -> None:
    """丢掉缓存的 Store 连接（自检里换库文件后必须调用）。"""
    global _store
    _store = None


# ======================================================================
# 读
# ======================================================================
def load_rows(user_id: str, role: str) -> list[dict[str, Any]]:
    """该用户该身份的全部掌握度记录。

    返回的字段与 mastery 表的行保持同名同义（含 kp_name），
    这样 repository 换个后端不必改聚合逻辑。
    """
    items = get_store().search(mastery_namespace(user_id, role), limit=SEARCH_LIMIT)
    rows: list[dict[str, Any]] = []
    for it in items:
        row = dict(it.value)
        row["kp_name"] = it.key  # Store 的 key 就是知识点名
        row.setdefault("mastery", 0.0)
        row.setdefault("subject", "")
        row.setdefault("evidence_count", 0)
        row.setdefault("last_review_at", None)
        rows.append(row)
    return rows


def load_map(user_id: str, role: str) -> dict[str, float]:
    """{知识点: 掌握度}"""
    items = get_store().search(mastery_namespace(user_id, role), limit=SEARCH_LIMIT)
    return {it.key: float(it.value.get("mastery") or 0.0) for it in items}


def load_last_review(user_id: str, role: str) -> dict[str, str]:
    """{知识点: 上次复习日期 YYYY-MM-DD}"""
    out: dict[str, str] = {}
    for it in get_store().search(mastery_namespace(user_id, role), limit=SEARCH_LIMIT):
        value = it.value.get("last_review_at")
        if value:
            out[it.key] = str(value)[:10]
    return out


def get_one(user_id: str, role: str, kp_name: str) -> float | None:
    it = get_store().get(mastery_namespace(user_id, role), kp_name)
    if it is None:
        return None
    return float(it.value.get("mastery") or 0.0)


# ======================================================================
# 写
# ======================================================================
def put_one(
    user_id: str,
    role: str,
    kp_name: str,
    mastery: float,
    subject: str | None = None,
    last_review_at: str | None = None,
) -> None:
    """写入一条掌握度（含 evidence_count 自增）。

    ⚠️ evidence_count 要"读-改-写"，因为 KV 没有 SQL 的 `count = count + 1`。
       这在并发下**不是原子的**——单用户单进程没问题，
       要严格原子就得走事务或干脆用业务表。

       这正好是本文档开头那句"知道边界在哪"的具体样子：
       一个计数器就把 KV 的短板暴露了。Store 适合存**整块快照**，
       不适合做**并发计数的累加器**。
    """
    store = get_store()
    ns = mastery_namespace(user_id, role)
    old = store.get(ns, kp_name)
    prev = dict(old.value) if old else {}
    now = datetime.now().isoformat(timespec="seconds")

    store.put(ns, kp_name, {
        "mastery": round(float(mastery), 4),
        "subject": subject or prev.get("subject") or "",
        "evidence_count": int(prev.get("evidence_count") or 0) + 1,
        "last_review_at": last_review_at or now,
        "updated_at": now,
    })


# ======================================================================
# 迁移：把 mastery 表搬进 Store
# ======================================================================
def migrate_from_table(repo) -> int:
    """把业务表里的掌握度**全量**搬进 Store，返回搬迁条数。

    幂等：重复执行只是重复覆盖，结果一致。
    evidence_count 与 last_review_at 原样带过来——
    迁移要保真，否则搬运完画像就"变浅"了。
    """
    rows = repo.conn.execute(
        """
        SELECT user_id, role, kp_name, subject, mastery, evidence_count, last_review_at
        FROM mastery
        """
    ).fetchall()

    store = get_store()
    n = 0
    for r in rows:
        d = dict(r)
        ns = mastery_namespace(d["user_id"], d["role"])
        old = store.get(ns, d["kp_name"])
        prev = dict(old.value) if old else {}
        store.put(ns, d["kp_name"], {
            "mastery": round(float(d["mastery"] or 0.0), 4),
            "subject": d["subject"] or prev.get("subject") or "",
            "evidence_count": int(d["evidence_count"] or 0),
            "last_review_at": d["last_review_at"] or prev.get("last_review_at"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        n += 1

    logger.info(f"[司南·store] 迁移完成，共 {n} 条掌握度记录")
    return n


# ======================================================================
# 描述（自检 / 答辩）
# ======================================================================
def describe_store() -> dict[str, Any]:
    return {
        "backend": type(get_store()).__name__,
        "file": str(STORE_DB),
        "namespace": "(user_id, role, mastery)",
        "active": use_store(),
        "search_limit": SEARCH_LIMIT,
    }


__all__ = [
    "STORE_DB",
    "MEMORY_KIND",
    "SEARCH_LIMIT",
    "use_store",
    "mastery_namespace",
    "get_store",
    "reset_store",
    "load_rows",
    "load_map",
    "load_last_review",
    "get_one",
    "put_one",
    "migrate_from_table",
    "describe_store",
]
