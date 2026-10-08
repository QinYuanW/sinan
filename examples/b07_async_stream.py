"""B7 · 异步生成器 + asyncio.to_thread

学什么
    agent.stream() 是【同步】生成器，每产出一块都要等模型。
    FastAPI 是【异步】框架，事件循环只有一个线程。
    如果在 async 函数里直接 for chunk in agent.stream(...)：
        -> 整个事件循环被卡住
        -> 别的用户请求全部排队
    解法：把同步迭代丢进线程池（asyncio.to_thread），事件循环就能在等的时候去干别的。

对应司南
    src/app/agent/sinan_agent.py:136-193   stream_chat()

怎么跑
    python examples/b07_async_stream.py
    （不需要 API key —— 用模拟的慢生成器，重点看耗时对比）

观察点
    PART 1 顺序消费两条流：约 2 秒
    PART 2 并发消费两条流：约 1 秒
    同样的数据量，耗时几乎减半 —— 这就是"不阻塞"的价值。

TODO（学完自己改）
    1) 把 slow_stream_sync 的 delay 调大，看并发优势是否更明显
    2) 写一版【错误】的实现：在 async 里直接 for...in 同步生成器，
       用 time.perf_counter 量一下两个任务的总耗时，见证它退化成一倍

       #同步阻塞代码不能跑在事件循环的线程上，要用 asyncio.to_thread
       把它扔到后台线程池去，这样事件循环才能腾出手来服务别的请求。
"""

from __future__ import annotations

import asyncio
import time

CHUNKS = 4
DELAY = 0.25


# ======================================================================
# 模拟：同步生成器（等价于 agent.stream）
# ======================================================================
def slow_stream_sync(name: str, chunks: int = CHUNKS, delay: float = DELAY):
    """每产出一块就 sleep 一下，模拟等模型的网络耗时。"""
    for i in range(chunks):
        time.sleep(delay)          # ← 阻塞式等待，会卡住调用它的线程
        yield f"{name}#{i}"


# ======================================================================
# 正例：把同步迭代丢进线程池
## ======================================================================
async def stream_chat(name: str):
    """司南 stream_chat 的核心模式。

    注意两点：
      1) next() 本身也是阻塞的，所以每次取下一块都要再 to_thread 一次
      2) 用 iter(lambda: next(it, None), None) 这种写法更简洁
    """
    it = slow_stream_sync(name)

    def _next():
        return next(it, None)      # 不用 StopIteration，直接给 None 更好处理

    while True:
        item = await asyncio.to_thread(_next) # 放进线程池，等模型返回
        if item is None:
            break
        yield item


# ======================================================================
# 反例：在 async 里直接迭代同步生成器
# ======================================================================
async def stream_chat_blocking(name: str):
    """错误示范：事件循环被 sleep 卡死。"""
    for chunk in slow_stream_sync(name):     # ← 这一行会阻塞整个事件循环
        yield chunk


# ======================================================================
async def main() -> None:
    print("=" * 64)
    print("PART 1 · 顺序消费两条流（没有任何并发）")
    print("=" * 64)
    t0 = time.perf_counter()
    out = []
    out += [c async for c in stream_chat("A")]
    out += [c async for c in stream_chat("B")]
    t1 = time.perf_counter()
    print(f"    {out}")
    print(f"    耗时：{t1 - t0:.2f}s   （理论值 {CHUNKS * DELAY * 2:.2f}s）")

    print()
    print("=" * 64)
    print("PART 2 · 用 asyncio.gather 并发消费（to_thread 版）")#asyncio.gather 并发执行多个任务
    print("=" * 64)
    t0 = time.perf_counter()

    async def collect(name: str) -> list[str]:
        return [c async for c in stream_chat(name)]

    results = await asyncio.gather(collect("A"), collect("B"))
    t1 = time.perf_counter()
    print(f"    {results}")
    print(f"    耗时：{t1 - t0:.2f}s   （理论值 {CHUNKS * DELAY:.2f}s）")
    print("    ↑ 两条流的时间重叠了 —— B 没等 A 跑完。")

    print()
    print("=" * 64)
    print("PART 3 · 反例：直接在 async 里迭代同步生成器")
    print("=" * 64)
    t0 = time.perf_counter()
    results = await asyncio.gather(
        _drain_blocking("A"),
        _drain_blocking("B"),
    )
    t1 = time.perf_counter()
    print(f"    耗时：{t1 - t0:.2f}s")
    print("    ↑ 和 PART 1 一样慢 —— 因为 sleep 把唯一的事件循环线程占住了，")
    print("       gather 根本没法切换任务。")

    print()
    print("-" * 64)
    print("所以司南的 stream_chat 长这样：")
    print("    iterator = await asyncio.to_thread(lambda: agent.stream(...))")
    print("    while True:")
    print("        item = await asyncio.to_thread(lambda: next(iterator, None))")
    print("        if item is None: break")
    print("        yield chunk.content")
    print()
    print("代价：每个连接会占一个线程。用户量很大时，要考虑线程池上限。")


async def _drain_blocking(name: str) -> list[str]:
    return [c async for c in stream_chat_blocking(name)]


if __name__ == "__main__":
    asyncio.run(main())
