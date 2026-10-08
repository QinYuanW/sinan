"""B10 · 懒加载单例

学什么
    模型客户端如果写在模块顶层，import 的那一刻就会执行。
    于是"没配 API key"这件小事，会升级成"整个服务起不来"。
    懒加载 = 第一次真正用到时才创建；单例 = 只创建一次。

对应司南
    src/app/agent/tools/vision.py:37    _VISION_MODEL
    src/app/agent/core/reporter.py:32   _PARSER
    src/app/agent/sinan_agent.py:60     _checkpointer

怎么跑
    python examples/b10_lazy_singleton.py
    （不需要 API key —— 缺 key 恰恰是本课要演示的场景）

观察点
    PART 1 在「导入期」就炸 —— 这就是反面教材。
    PART 2 / PART 3 导入期平安无事，只在真正调用时才可能报错。

TODO（学完自己改）
    1) 把司南 .env 里的 ZHIPUAI_API_KEY 临时删掉，启动服务。
       应该能正常起来，只有"上传照片"时才会报错 —— 这就是懒加载的收益。
    2) 想清楚：单例为什么不用每次 new？SqliteSaver 每次 new 会怎样？
"""

from __future__ import annotations

import os


class FakeClient:
    """假装是一个"需要 API key 才能构造"的模型客户端。"""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        print("    [构造] FakeClient 被创建了（真实项目里这里会建连接 / 校验 key）")

    def invoke(self, text: str) -> str:
        return f"<模型回复: {text}>"


# ======================================================================
# 反面教材：模块级立即创建
# ======================================================================
def make_client_immediate() -> FakeClient:
    """模拟"模块顶层直接执行"的写法。

    注意：在真实项目里这行会写在模块顶层，import 的时候就跑。
    这里包成函数只是为了能重复演示。
    """
    key = os.getenv("DEMO_API_KEY")
    if not key:
        raise RuntimeError("缺少环境变量 DEMO_API_KEY —— 但这是在 import 阶段就抛的！")
    return FakeClient(key)


# ======================================================================
# 正例：懒加载 + 单例
# ======================================================================
_CLIENT: FakeClient | None = None


def get_client() -> FakeClient:
    """第一次调用时才创建，之后复用同一个对象。"""
    global _CLIENT                       # <- 单例：模块级变量持有
    if _CLIENT is None:
        key = os.getenv("DEMO_API_KEY")
        if not key:
            raise RuntimeError("缺少环境变量 DEMO_API_KEY")
        _CLIENT = FakeClient(key)
    return _CLIENT


def main() -> None:
    print("=" * 64)
    print("PART 1 · 立即创建 —— 「import 就炸」的反面教材")
    print("=" * 64)
    try:
        make_client_immediate()
    except RuntimeError as exc:
        print(f"    ✗ {exc}")
        print("    ↑ 如果这行写在模块顶层，整个服务连启动都做不到。")

    print()
    print("=" * 64)
    print("PART 2 · 懒加载 —— 导入期什么都不做")
    print("=" * 64)
    print("    模块已加载完毕，get_client 还没有被调用，所以毫无动静。")
    print("    （真实项目里 import vision 就是这个效果）")
    print()
    print("    现在真的要用它了：")
    try:
        get_client()
    except RuntimeError as exc:
        print(f"    ✗ {exc}")
        print("    ↑ 报错时机推迟到了「真正要用」的那一刻，服务本身没受影响。")

    print()
    print("=" * 64)
    print("PART 3 · 单例 —— 设好 key 后，验证只创建一次")
    print("=" * 64)
    os.environ["DEMO_API_KEY"] = "demo-key-123"
    c1 = get_client()
    c2 = get_client()
    c3 = get_client()
    print(f"    三次拿到的对象：{id(c1)} / {id(c2)} / {id(c3)}")
    print(f"    是不是同一个：{c1 is c2 is c3}")
    print()
    print("    ↑ 只打印了一次 [构造]。这就是单例。")

    print()
    print("-" * 64)
    print("为什么 SqliteSaver 必须单例？")
    print("    它持有一个 sqlite3.Connection。每次 new 都会新开一个连接，")
    print("    旧连接没人关 —— 很快就「too many open files」。")
    print("    司南里 get_checkpointer() 用模块级 _checkpointer 保证只建一次。")

    # TODO: 把上面的 get_client 改成「带过期时间的缓存」（比如 1 小时后重建）


if __name__ == "__main__":
    main()
