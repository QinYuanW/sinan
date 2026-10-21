"""司南 · 应用入口

一个多身份学习规划平台：
    目标设定 -> 长线规划 -> 每日作战清单 -> 打卡/汇报 -> 动态重规划

启动方式：
    开发：  python -m app.main          （127.0.0.1 + 热重载）
    生产：  python -m app.main --prod   （0.0.0.0，读环境变量）
    或：    uvicorn app.main:app --host 0.0.0.0 --port 8001

═══════════════════════════════════════════════════════════════════════════
L12 多租户部署：这一层要补的三件事
═══════════════════════════════════════════════════════════════════════════

① 健康检查 `/health`
   部署平台靠它判断实例是否活着。没有这个端点，平台只能靠"端口通不通"猜——
   而一个能接受 TCP 连接、但数据库连不上的进程，会被误判为健康。

② 访问令牌（可选，`SINAN_API_TOKEN`）
   这是**多租户的最后一公里**。在此之前 `user_id` 只是请求体里的一个字段——
   意味着任何能访问服务的人都可以冒充任何用户。
   数据层的隔离是成立的（user × role 正交，自检已验证），
   但**入口层没有身份**，隔离就成了摆设。

   ⚠️ 必须说清的边界：本机制是一把**门锁**，不是完整的多租户鉴权。
      它解决"谁能进门"，没解决"哪个 token 对应哪个用户"——
      拿到 token 的人仍可传任意 user_id。真正的做法见 auth_middleware 的作业注释。

③ CORS 收敛
   开发期 `allow_origins=["*"]`，但和 `allow_credentials=True` 组合
   会被浏览器规范直接拒绝（等于配了没生效）。上线必须换成具体域名。

★ 三者的共同点：**本地开发全都不需要，上线全都必须有。**
  所以它们都由"是否配置了环境变量"控制——默认关闭，部署时打开。
  "默认安全"很难做到，"默认能跑 + 上线时有人提醒你打开"更现实。
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.api.v1 import chat, log, oss, plan, profile
from app.common.logger import setup_logging
from app.db.models import init_db

# 初始化日志与数据库（建表幂等）
setup_logging()
init_db()

# 是否生产模式
IS_PROD = os.getenv("SINAN_ENV", "").strip().lower() in {"prod", "production"}

# 访问令牌：留空 = 不校验（本地开发零摩擦）；公网部署前**必须**设置
API_TOKEN = os.getenv("SINAN_API_TOKEN", "").strip()

# 允许的前端来源：逗号分隔。留空 = 开发期通配
ALLOWED_ORIGINS = [
    o.strip() for o in (os.getenv("SINAN_ALLOWED_ORIGINS") or "").split(",") if o.strip()
]

app = FastAPI(
    title="司南 · 学习规划平台 API",
    description="多身份学习规划 Agent：规划、打卡、汇报、动态重规划",
    version="2.0.0",
)

# 1. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=bool(ALLOWED_ORIGINS),  # 有具体域名时才允许带凭证
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# 1.0 访问令牌校验（L12 · 多租户的入口层）
# ======================================================================
# 放行清单：健康检查与接口文档必须匿名可访问，
# 否则部署平台探活会被 401 挡掉、自己也看不到 /docs。
# （静态资源不在 /api/ 前缀下，天然放行——否则用户连首页都打不开。）
_PUBLIC_PATHS = ("/health", "/docs", "/openapi.json", "/redoc")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """校验 `Authorization: Bearer <SINAN_API_TOKEN>`。

    未配置 API_TOKEN 时整体跳过（本地开发零摩擦）。

    【作业】升级成真正的多租户鉴权，思路三步：
        1. 环境变量改成映射式：`SINAN_TENANTS="tokenA:userA,tokenB:userB"`；
        2. 校验通过后把 user_id 写进 `request.state.user_id`；
        3. 业务层**优先取 request.state.user_id**，
           只有在未启用鉴权（本地开发）时才回落到请求体里的字段。
        这样"用户是谁"由 token 决定，客户端说了不算。
        再往后才是完整用户体系（注册 / 登录 / JWT）。
    """
    if not API_TOKEN or request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if not path.startswith("/api/") or path in _PUBLIC_PATHS:
        return await call_next(request)

    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if token != API_TOKEN:
        return JSONResponse(
            status_code=401,
            content={"error": "未授权", "detail": "缺少或错误的访问令牌"},
        )
    return await call_next(request)


# ======================================================================
# 1.05 健康检查（部署平台探活用）
# ======================================================================
@app.get("/health", include_in_schema=False)
async def health():
    """健康检查。

    不只回 "ok"，顺手把关键依赖状态带出来：实例起不来时看一眼就知道
    是库缺了还是没开鉴权，不用翻日志。

    ⚠️ 刻意**不检查外部模型服务**（DeepSeek / SiliconFlow）：
       每次探活都去调一次大模型既慢又费钱，而且"模型暂时不可用"
       不该让整个实例被判定为不健康。
       **依赖检查要回答"这个进程能不能正常服务"，不是"所有下游是否完美"。**
    """
    from app.db.models import DB_PATH

    db_ok = os.path.exists(DB_PATH)
    return {
        "status": "ok" if db_ok else "degraded",
        "env": "prod" if IS_PROD else "dev",
        "auth": "on" if API_TOKEN else "off",
        "db": "ok" if db_ok else "missing",
    }


# 1.1 业务层异常 -> 合理的 HTTP 状态码
# 身份包的 goal_schema 校验失败若直接冒泡会变成 500，
# 那是服务端错误码，会把"用户填错"误报成"服务挂了"。
@app.exception_handler(ValidationError)
async def _validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "目标字段校验失败",
            "detail": [
                {"field": ".".join(str(p) for p in err.get("loc", [])), "msg": err.get("msg")}
                for err in exc.errors()
            ],
        },
    )


@app.exception_handler(KeyError)
async def _key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(
        status_code=400,
        content={"error": "未知身份包", "detail": str(exc).strip("'\"")},
    )

# 2. 业务路由（放在静态挂载之前，保证 API 优先匹配）
app.include_router(profile.router, prefix="/api/v1", tags=["身份与画像"])
app.include_router(plan.router, prefix="/api/v1", tags=["计划"])
app.include_router(log.router, prefix="/api/v1", tags=["打卡与汇报"])
app.include_router(chat.router, prefix="/api/v1", tags=["对话"])
app.include_router(oss.router, prefix="/api/v1", tags=["文件上传"])

# 3. 前端静态资源
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# 4. SPA fallback —— 只处理非 API 请求
@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(path: str):
    if path.startswith("api/"):
        return JSONResponse({"error": "Not Found"}, status_code=404)

    file_path = os.path.join(static_dir, path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)

    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {"message": "司南 API 已启动，但前端静态资源未就绪", "docs": "/docs"}


if __name__ == "__main__":
    import sys

    import uvicorn

    prod = "--prod" in sys.argv or IS_PROD
    if prod:
        # 生产：监听所有网卡，不重载（重载会起额外进程，容器里没必要）
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8001")),
            reload=False,
        )
    else:
        # 开发：只监听本机 + 热重载
        uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
