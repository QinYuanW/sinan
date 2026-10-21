"""B8 · SSE（Server-Sent Events）事件格式

学什么
    text/event-stream 的正文有硬性格式规定：每一行前面必须加 "data: "，
    用空行消息之间分隔。不符合格式，浏览器 EventSource 就收不到东西。
    顺带一个真实 bug 复盘：原私厨项目声明了 text/event-stream，
    却直接 yield 裸文本 —— 声明和实现不一致。

对应司南
    src/app/api/v1/chat.py   sse() 与 SSE_HEADERS

怎么跑
    python examples/b08_sse_events.py            # 用 TestClient 直接看响应体
    python examples/b08_sse_events.py --serve    # 起真服务，浏览器打开 http://127.0.0.1:8099/proper

观察点
    对比 /naive 和 /proper 两个端点的原始响应体 ——
    一个是一坨裸文本，一个每行都有 data: 前缀。

TODO（学完自己改）
    1) 加一个 event: tool_call 事件类型，让前端能显示"正在调用工具"
    2) 加一个 id: 字段 + 前端 Last-Event-ID 重连，试试断线续传
"""

from __future__ import annotations

import json
import sys
import time

from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse, StreamingResponse

# ======================================================================
# 核心：SSE 文本封装
# ======================================================================
def sse_naive(payload) -> str:
    """反面教材 —— 声明了 event-stream 却吐裸文本。

    这是原私厨项目里的真实缺陷：
        return StreamingResponse(gen(), media_type="text/event-stream")
    而 gen() 里 yield 的是普通字符串。
    浏览器 EventSource 解析不到 data: 前缀，等于收不到任何消息。
    """
    return payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)


def sse(payload) -> str:
    """正例 —— 逐行加前缀，末尾补一个空行表示消息结束。

    注意 payload 里如果有换行（比如模型输出的多行 Markdown），
    每一行都要单独加 data: ，否则第二行开始就是非法格式。
    """
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return "".join(f"data: {line}\n" for line in text.split("\n")) + "\n"

#sse
# 规则就三条：
# 数据行以 data: 开头
# 一条消息结束用空行（连续两个 \n）
# 可选 event: xxx 指定事件名，前端用 addEventListener('xxx', ...) 监听

def sse_event(event: str, payload, id: int | None = None) -> str:
    """带事件类型，可选 id 用于断线续传。

    输出格式：
        id: 3
        event: tool_call
        data: {...}

    浏览器收到带 id 的事件后，断线重连会自动在请求头带
    Last-Event-ID: 3，后端据此从 3 之后继续发。
    """
    lines = []
    if id is not None:
        lines.append(f"id: {id}")
    lines.append(f"event: {event}")
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    for line in text.split("\n"):
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache",       # 禁止中间层缓存
    "Connection": "keep-alive",        # 保持长连接
    "X-Accel-Buffering": "no",         # 关键：关掉 Nginx 缓冲，否则流式会被攒成一坨
}


# ======================================================================
# 一个最小服务，两种端点
# ======================================================================
app = FastAPI(title="SSE 示例")

CHUNKS = [
    "## 今日作战清单\n",
    "\n",
    "1. 极限的基本计算（优先级 0.92）\n",
    "   理由：近 5 年考频 0.95，当前掌握度仅 0.33\n",
    "\n",
    "2. 矩阵的秩（优先级 0.88）\n",
]
        #yield 是一个生成器函数
# 调用函数本身不执行函数体，只是创建生成器
# 每次 next 才让它跑一步，跑到下一个 yield 就停下
# 暂停时会记住当前状态（局部变量、执行到哪一行了）
# 再调用 next，从暂停的地方继续

@app.get("/naive")
def naive():
    """反面：声明了 event-stream，但吐的是裸文本。"""

    def gen():
        for c in CHUNKS:
            yield c          # <- 少了 data: 前缀

    return StreamingResponse(gen(), media_type="text/event-stream")
#StreamingResponse是FastAPI提供的一个响应类，用于返回流式响应
# 它可以将生成器或异步生成器的输出直接返回给客户端，而不需要等待所有数据加载完成后再返回
#gen() 是一个生成器函数（用 yield 而不是 return）
#media_type="text/event-stream" 告诉浏览器："这是 SSE 流，请用 EventSource 解析"

@app.get("/proper")
def proper():
    """正例：规范的 SSE。"""

    def gen():
        for c in CHUNKS:
            yield sse(c)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


# 事件序列（id 从 1 开始），真实场景应由业务动态生成
TYPED_EVENTS = [
    ("start",     {"thread_id": "default:kaoyan:1"}),
    ("token",     "极限的基本计算"),
    ("token",     " —— 先做这一章"),
    ("tool_call", {"name": "get_kp_mastery", "args": {"kp_name": "极限的基本计算"}}),
    ("token",     "（掌握度 0.33，需加强）"),
    ("done",      {"status": "ok"}),
]

@app.get("/typed")
def typed(last_event_id: str | None = Header(default=None)):
    """带事件类型 + id 续传 —— 断线重连后从上次位置继续。

    浏览器 EventSource 断线重连时会自动带 Last-Event-ID 头，
    这里据此跳过已发送的事件。
    """
    start_idx = 0
    if last_event_id:
        try:
            start_idx = int(last_event_id)  # 从 id 对应的下一条开始
        except ValueError:
            start_idx = 0

    def gen():
        for i, (evt, payload) in enumerate(TYPED_EVENTS[start_idx:], start=start_idx + 1):
            yield sse_event(evt, payload, id=i)
            # 每个事件之间留间隔，便于观察续传效果
            time.sleep(0.6)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/")
def frontend():
    """前端演示页：用 EventSource 接收事件，tool_call 来时显示"正在调用工具"。"""
    return HTMLResponse("""
<!doctype html>
<html><head><meta charset="utf-8"><title>SSE tool_call 演示</title>
<style>
  body{font-family:system-ui;max-width:640px;margin:40px auto;padding:0 16px}
  #msg{white-space:pre-wrap;border:1px solid #ddd;padding:12px;border-radius:8px;min-height:120px}
  #status{color:#888;margin:8px 0;font-size:14px;min-height:20px}
  .tool{color:#e67e22}
  button{padding:6px 14px;cursor:pointer}
</style></head>
<body>
  <h3>司南 SSE 事件演示</h3>
  <button onclick="connect()">开始对话</button>
  <button onclick="breakConn()">模拟断线重连</button>
  <div id="status"></div>
  <div id="msg"></div>
<script>
    let es = null;
  function connect(){
    document.getElementById('msg').textContent='';
    document.getElementById('status').textContent='';
    if(es) es.close();
    es = new EventSource('/typed');
    es.addEventListener('start', e => {
      document.getElementById('status').textContent='对话开始…（已收到 id=' + e.lastEventId + '）';
    });
    es.addEventListener('token', e => {
      document.getElementById('msg').textContent += e.data;
    });
    es.addEventListener('tool_call', e => {  
      const d = JSON.parse(e.data);
      document.getElementById('status').innerHTML =
        '<span class="tool">🔧 正在调用工具：' + d.name + '（id=' + e.lastEventId + '）</span>';
    });
    es.addEventListener('done', e => {
      document.getElementById('status').textContent='✅ 完成（最后 id=' + e.lastEventId + '）';
      es.close();
    });
    es.onerror = () => {
      document.getElementById('status').textContent = '⚠️ 连接断开，浏览器自动重连中…';
    };
  }
  // 模拟断线：close 后浏览器会自动重连，并带上 Last-Event-ID
  function breakConn(){
    if(es) es.close();
    connect();
  }
</script>
</body></html>
    """)
# ======================================================================
def main() -> None:
    if "--serve" in sys.argv:
        import uvicorn

        print("起服务中 -> http://127.0.0.1:8099/  （前端演示页，点开始对话看 tool_call 效果）")
        print("对比着看：/naive  /proper  /typed")
        uvicorn.run(app, host="127.0.0.1", port=8099)
        return

    print("=" * 72)
    print("先看两段纯文本，理解格式差别")
    print("=" * 72)
    print("裸文本（反面）：")
    print(repr(CHUNKS[0]))
    print()
    print("加了 data: 前缀（正例）：")
    print(repr(sse(CHUNKS[0])))
    print()
    print("多行内容会被逐行加前缀：")
    multi = "第一行\n第二行"
    print("  原文：", repr(multi))
    print("  封装：", repr(sse(multi)))
    print("  ↑ 注意 'data: 第二行' —— 少了这个，第二行就是非法格式")

    print()
    print("=" * 72)
    print("再用 TestClient 看两个端点的真实响应体")
    print("=" * 72)
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        print("  跳过（缺 httpx，装一下：uv add httpx）")
    else:
        client = TestClient(app)

        for path in ("/naive", "/proper"):
            r = client.get(path)
            print(f"\n  GET {path}   status={r.status_code}")
            print(f"    Content-Type: {r.headers.get('content-type')}")
            print(f"    响应体前 160 字：")
            print("   ", repr(r.text[:160]))

        print()
        r = client.get("/typed")
        print(f"  GET /typed   status={r.status_code}")
        print("    响应体：")
        for line in r.text.split("\n")[:12]:
            print("      " + repr(line))

    print()
    print("-" * 72)
    print("结论：")
    print("  · /naive 声称是 text/event-stream，但响应体里一个 data: 都没有")
    print("  · 浏览器 EventSource 收到 /naive 的内容 -> 一条消息都解析不出")
    print("  · /proper 每行都有前缀，前端才能一条条收到")
    print()
    print("X-Accel-Buffering: no 也要记住 ——")
    print("Nginx 默认会把响应攒够了再发，你的流式就变成了'等 10 秒一起冒出来'。")

    # TODO: 把 /typed 加上前端代码，实现"工具调用中…"的实时提示


if __name__ == "__main__":
    main()
