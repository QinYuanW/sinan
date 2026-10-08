"""B6 · Agent 实例缓存 + 日期做缓存键

学什么
    system_prompt 里塞了"剩余 45 天""连续打卡 7 天"这种每天都变的数据。
    于是产生两难：
      · Agent 永久缓存  -> 提示词过期，明天还显示"剩余 45 天"
      · 每次都重建      -> 白烧 CPU 和 token
    解法：把日期塞进缓存键。同一天复用，跨天自动失效。

对应司南
    src/app/agent/sinan_agent.py:119-125   get_agent()

怎么跑
    python examples/b06_agent_cache.py
    （不需要 API key —— 用假的 builder 模拟，重点看缓存行为）

观察点
    第 3 次调用 get_agent 时，builder 没有被执行 —— 缓存命中了。
    然后手动把"今天"改成"明天"，看它重建。

TODO（学完自己改）
    1) 把缓存换成 LRU（最多 100 个），看 _cache 会不会无限增长
    #会无限增长
    2) 想清楚：为什么缓存键里要有 role，不能只有 user_id？
    #会出现缓存命中问题，导致用户身份切换后，会话会串档
"""
#日期变 → key 变 → 缓存查不到 → 调用 _build_agent → 新 key-value 存入字典。

from __future__ import annotations

from datetime import date as _date

# 假装今天是这个日期，方便观察"跨天"效果
_fake_today = _date(2026, 9, 29)

_cache: dict[tuple, str] = {}
_build_count = 0


def _build_agent(user_id: str, role: str) -> str:
    """真实项目里这里会 create_agent(...)，并构造动态 system_prompt。

    示例里用字符串代替，重点在缓存行为。
    """
    global _build_count
    _build_count += 1
    print(f"    [build] 真的在构建 Agent #{_build_count}  (user={user_id}, role={role})")
    return f"Agent(user={user_id}, role={role}, built_on={_fake_today})"


def get_agent(user_id: str, role: str) -> str:
    """按 (user_id, role, 日期) 缓存。

    日期参与键 -> 每天自动刷新一次，既保证提示词新鲜，又不用每次重建。
    """
    key = (user_id, role, _fake_today.isoformat())
    if key not in _cache:
        _cache[key] = _build_agent(user_id, role)
    else:
        print(f"    [hit]   缓存命中，直接复用  key={key}")
    return _cache[key]


def main() -> None:
    global _fake_today

    print("=" * 64)
    print("场景一：同一天，同一个人，连续请求 3 次")
    print("=" * 64)
    for i in range(1, 4):
        print(f"  第 {i} 次请求：")
        get_agent("tangyuan", "kaoyan")

    print()
    print("=" * 64)
    print("场景二：同一天，切到教师身份")
    print("=" * 64)
    print("  请求：")
    get_agent("tangyuan", "teacher")

    print()
    print("=" * 64)
    print("场景三：换一天（模拟第二天打开 App）")
    print("=" * 64)
    _fake_today = _date(2026, 9, 30)
    print("  请求：")
    get_agent("tangyuan", "kaoyan")

    print()
    print("-" * 64)
    print(f"总计真实构建次数：{_build_count} 次（缓存里 {len(_cache)} 条）")
    print()
    print("关键点：")
    print("  · 同一天重复请求 -> 复用，不重建")
    print("  · 换身份        -> 缓存键不同，必须重建（提示词完全不同）")
    print("  · 跨天          -> 键里的日期变了，自动失效重建")
    print()
    print("如果缓存键少了 role，用户在考研/教师之间切换时")
    print("会拿到对方身份的 system_prompt —— 这是个隐蔽的严重 bug。")

    # TODO: 试试把 _cache 换成 LRU，再跑一遍
    # from functools import lru_cache
    # ... 但注意 lru_cache 不能直接用在"日期会变"的键上，思考为什么


if __name__ == "__main__":
    main()
