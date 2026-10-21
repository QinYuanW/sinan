"""B7.1 · 同步生成器在异步框架里到底会造成多大伤害（量化实测）

学什么
    b07 讲了"为什么要把同步生成器丢进线程池"。这一课把伤害【量化】出来：
    直接迭代同步生成器时，整个 asyncio 事件循环会被按住多久。

对应司南
    src/app/agent/sinan_agent.py
        stream_chat       —— 正确写法（每步 await asyncio.to_thread）
        stream_chat_sync  —— 对照写法（直接 for 迭代，只能在同步环境用）

怎么跑
    python examples/b07.1_async_vs_sync.py
    （不需要 API key，全程用假的同步生成器模拟模型输出）

实验设计
    假的同步生成器：10 块 × 0.2 秒 = 2 秒（模拟模型的逐 token 输出）
    同时开一个"心跳"协程：每 50ms 跳一次。事件循环一旦被阻塞，心跳就会卡顿。

观察点
    1) 心跳最大间隔 —— 直接反映"事件循环被按住多久"
    2) 两个用户同时提问的总耗时 —— 直接反映"并发能力"

预期结论
    错误写法：心跳一次都跳不成（事件循环 100% 卡死），两个用户串行。
    正确写法：心跳照跳，两个用户真并发。

实测结果（2026-10-11 本机）
    [实验一] 事件循环阻塞
        反例：流式 2.11s，心跳 0 次（理想 42）-> 整段期间完全卡死
        正例：流式 2.02s，心跳 33 次，最大间隔 64ms
    [实验二] 两个用户同时提问
        反例：4.01s（严格串行，并发度退化为 1）
        正例：2.01s（真并发）
"""

from __future__ import annotations

import asyncio
import time

BLOCKS = 10
DELAY = 0.2
HEARTBEAT = 0.05


def sync_stream(n: int = BLOCKS, delay: float = DELAY):
    """假装这是 agent.stream —— 同步、会阻塞、逐块产出。"""
    for i in range(n):
        time.sleep(delay)  # 模拟模型推理 / 网络往返的耗时
        yield f"块{i} "


# ======================================================================
# 两种消费方式（核心对比）
# ======================================================================
async def consume_wrong(n: int, delay: float) -> int:
    """[反例] 在 async 函数里直接迭代同步生成器。

    `for` 每要一块，都在当前线程里 sleep —— 而这正是事件循环所在的线程。
    所以整段流式期间，事件循环一步都动不了。
    """
    count = 0
    for _ in sync_stream(n, delay):
        count += 1
    return count


async def consume_right(n: int, delay: float) -> int:
    """[正例] 每一步都丢进线程池（stream_chat 的做法）。

    `time.sleep` 发生在工作线程里，事件循环在 await 期间可以自由调度其他任务。
    """
    gen = await asyncio.to_thread(lambda: sync_stream(n, delay))

    def _next():
        try:
            return next(gen)
        except StopIteration:
            return None

    count = 0
    while True:
        item = await asyncio.to_thread(_next)
        if item is None:
            break
        count += 1
    return count


# ======================================================================
# 测量工具
# ======================================================================
async def heartbeat(stop: asyncio.Event) -> list[float]:
    """每 50ms 跳一次，返回每次间隔。间隔暴涨 = 事件循环被阻塞。"""
    gaps: list[float] = []
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(HEARTBEAT)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now
    return gaps


async def case_blocking(consumer, label: str) -> None:
    """实验一：量一量事件循环被按住多久"""
    stop = asyncio.Event()
    hb = asyncio.create_task(heartbeat(stop))

    t0 = time.perf_counter()
    await consumer(BLOCKS, DELAY)
    elapsed = time.perf_counter() - t0

    stop.set()
    gaps = await hb

    print(f"  {label}")
    print(f"    流式总耗时   : {elapsed:.2f}s")
    print(f"    心跳次数     : {len(gaps)}（理想值约 {int(elapsed / HEARTBEAT)}）")
    if gaps:
        print(f"    心跳最大间隔 : {max(gaps) * 1000:.0f} ms   <- 越大说明事件循环被卡得越狠")
    else:
        print("    心跳最大间隔 : 一次都没跳成 —— 事件循环被 100% 按住")
    print()


async def case_concurrency(consumer, label: str) -> None:
    """实验二：两个用户同时提问，量总耗时"""
    t0 = time.perf_counter()
    await asyncio.gather(consumer(BLOCKS, DELAY), consumer(BLOCKS, DELAY))
    elapsed = time.perf_counter() - t0

    verdict = "串行（一个卡死另一个）" if elapsed > DELAY * BLOCKS * 1.8 else "并发（各跑各的）"
    print(f"  {label}")
    print(f"    两个用户总耗时: {elapsed:.2f}s   -> {verdict}")
    print()


# ======================================================================
async def main() -> int:
    print("=" * 68)
    print("假的同步流式生成器：10 块 x 0.2s = 2.00s")
    print("心跳协程：每 50ms 跳一次")
    print("=" * 68)

    print("\n[实验一] 事件循环被阻塞了多久？")
    await case_blocking(consume_wrong, "[反例] async 里直接 for 迭代")
    await case_blocking(consume_right, "[正例] await asyncio.to_thread")

    print("=" * 68)
    print("\n[实验二] 两个用户同时提问，总耗时多少？")
    await case_concurrency(consume_wrong, "[反例] async 里直接 for 迭代")
    await case_concurrency(consume_right, "[正例] await asyncio.to_thread")

    print("=" * 68)
    print("结论")
    print("=" * 68)
    print("  [反例] 单用户时毫无异样，但：")
    print("         - 事件循环被整段按住 -> 健康检查、其他用户请求全部排队")
    print("         - 两个用户同时提问 = 4 秒（并发度退化成 1）")
    print("  [正例] 心跳照跳、用户真并发 —— 代价只是切线程的一点点开销。")
    print()
    print("  所以 sinan_agent.py 的 stream_chat 里，")
    print("  `await asyncio.to_thread(...)` 那两行不是装饰，是并发能力的来源。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
