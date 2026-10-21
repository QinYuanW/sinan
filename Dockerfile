# 司南 · 生产镜像（L12）
#
# 构建：  docker build -t sinan:latest .
# 运行：  docker run -p 8001:8001 --env-file .env sinan:latest
#
# ★ 三条刻意为之的设计，都在注释里说清楚为什么：
#     ① 用 uv + 锁文件复现依赖（不是 pip install -r，因为没有 requirements.txt）
#     ② 先拷依赖清单再拷源码（吃 Docker 层缓存，改代码不用重装依赖）
#     ③ 密钥一律**不**进镜像，全部运行时注入（见文末）

FROM python:3.14-slim

# ① 用 uv 复现锁定依赖
#    为什么不用 pip：本项目依赖由 uv 管理（pyproject.toml + uv.lock），
#    uv.lock 锁的是**完整解析结果**（含传递依赖），pip 装 requirements.txt
#    在版本漂移面前没有抵抗力。
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ② 分层：依赖清单单独一层
#    源码天天改、依赖很少动 —— 分开放，改代码时这一层命中缓存，
#    不用重新解析和下载依赖（几十秒 vs 几毫秒的差别）。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 再拷源码
COPY src/ ./src/
COPY .env.example ./.env.example

# 运行时目录（SQLite 库、知识库索引都写在这几个位置）
# 挂卷时把 /app/src/app/db 和 /app/src/app/kb/index 映射出去，
# 否则容器重建 = 用户数据全丢。
RUN mkdir -p /app/src/app/db /app/src/app/kb/index

ENV SINAN_ENV=prod \
    PORT=8001

EXPOSE 8001

# ③ 健康检查用 /health（L12 新增），而不是"端口通不通"
#    --start-period 给足冷启动时间：首次启动要建表 + 加载知识库
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4).status==200 else 1)"

# 启动：cwd 用 --app-dir 指到 src，等价于开发时的 "cd src && python -m uvicorn ..."
CMD ["uv", "run", "--no-dev", "python", "-m", "uvicorn", "app.main:app", \
     "--app-dir", "src", "--host", "0.0.0.0", "--port", "8001"]

# ═══════════════════════════════════════════════════════════════════════
# 密钥为什么不进镜像
# ═══════════════════════════════════════════════════════════════════════
# 镜像层是可以被翻出来的（`docker history` + `docker save`），
# 用 `COPY .env .env` 或 `ENV DEEPSEEK_API_KEY=xxx` 都等于**把密钥写进发行物**。
# 一旦镜像被推到公共仓库或分享给别人，密钥就跟着走了。
#
# 正确做法（优先级从高到低）：
#     1. 平台密钥管理（K8s Secret / 云厂商 Secret Manager）
#     2. `docker run --env-file .env`（.env 只在宿主机上）
#     3. 启动时注入的环境变量
#
# 本项目对应的是第 2 种：`.env` 留在宿主机，容器只接收环境变量。
