"""司南 · 知识库 · 文档加载与分块（L11 · RAG 第一步）

═══════════════════════════════════════════════════════════════════════════
RAG 的六个环节，本模块负责前两个
═══════════════════════════════════════════════════════════════════════════

    ① 文档加载  ← 本模块
    ② 分块      ← 本模块
    ③ 向量化        embeddings.py
    ④ 索引存储       retriever.py
    ⑤ 检索           retriever.py
    ⑥ 接入生成       tools/exam_kb.py

**为什么分块是 RAG 里最容易被低估的一步**

把整篇文档当一个检索单元会怎样？
    · 向量被"平均"成一个模糊的点，谁都不像
    · 命中后塞给模型的上下文里 90% 是无关内容，浪费 token 还干扰判断

把文档切太碎又会怎样？
    · "泰勒展开"这个短语被切成两半，检索时谁都匹配不上
    · 失去上下文，模型拿到"取错了"三个字不知道怎么用

所以分块的本质是**在"语义完整"和"检索精度"之间找平衡点**。

★ 本项目的优势：知识库是**结构化 markdown**（一卡一考点，二级标题分小节），
  所以可以直接**按标题切**——每一块天然语义完整，不需要靠字数猜。
  这是"数据形态决定分块策略"的典型例子：
  面对格式混乱的 PDF，你得写复杂的切分逻辑；面对自己写的卡片，一行代码就够。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.common.logger import logger

# 知识源目录
KB_ROOT = Path(__file__).resolve().parent
DOCS_DIR = KB_ROOT / "documents"

# 单块字符上限。超过就在段落边界再切一刀。
# 为什么还要这个：万一某个小节日后写了 3000 字，整块塞给模型同样浪费。
MAX_CHUNK_CHARS = 900

# 过短的块直接并入上一块（"## 说明\n\n略" 这种碎片没有检索价值）
MIN_CHUNK_CHARS = 40


@dataclass
class Chunk:
    """一个可检索单元。

    ★ `retrieval_text` 是本模块最值得一提的设计：
      检索用的文本**不等于**原文，而是 `标题路径 + 原文`。
        "极限的基本计算 > 易错点 > 泰勒展开阶数取错"
      这样即使用户只问"泰勒展开"，这一块也能被匹配上——
      因为标题路径把上下文补进了检索文本里。
      而**送给模型的**仍是原文（见 `text`），不会被这些拼接痕迹污染。

      「检索文本 ≠ 生成文本」是 RAG 里一个很实用的工程习惯。
    """

    id: str
    text: str            # 原文（送给模型的部分）
    retrieval_text: str  # 检索用文本 = 标题路径 + 原文
    title: str           # 所属考点卡片标题
    section: str         # 小节名（如"易错点"）
    source: str          # 来源文件
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def heading_path(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title


# ======================================================================
# 文档加载
# ======================================================================
def load_documents(docs_dir: Path | None = None) -> list[dict[str, Any]]:
    """读取 documents/ 下的全部 markdown。返回 [{path, name, raw, meta}]"""
    root = docs_dir or DOCS_DIR
    if not root.exists():
        logger.warning(f"[司南·kb] 知识源目录不存在: {root}")
        return []

    docs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        docs.append({
            "path": str(path),
            "name": path.stem,
            "raw": raw,
            "meta": _parse_front_matter(raw),
        })
    logger.info(f"[司南·kb] 载入 {len(docs)} 篇文档")
    return docs


def _parse_front_matter(raw: str) -> dict[str, Any]:
    """解析文首的 `key: value` 元信息块（可选的轻量 front matter）。

    格式（放在文件最前面，用 `---` 包起来）：
        ---
        subject: 高等数学
        chapter: 极限与连续
        exam_freq: 0.90
        ---

    为什么自己写而不用 PyYAML：这里只有扁平的 key: value，
    引一个 YAML 依赖不划算。够用就行——**能不引依赖就不引**。
    """
    meta: dict[str, Any] = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.S)
    if not m:
        return meta
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        # 数字能转就转，方便后面按 exam_freq 排序
        try:
            meta[k] = float(v) if "." in v else int(v)
        except ValueError:
            meta[k] = v
    return meta


def strip_front_matter(raw: str) -> str:
    """去掉文首元信息块，只留正文"""
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw, count=1, flags=re.S).lstrip()


# ======================================================================
# 分块
# ======================================================================
def chunk_document(doc: dict[str, Any]) -> list[Chunk]:
    """把一篇 markdown 切成若干块。

    切分规则（按优先级）：
        一级标题 `# X`   -> 卡片标题（本轮一个文档一张卡）
        二级标题 `## Y`  -> 一个块
        超长块           -> 在空行处再切，避免整块过大
        超短块           -> 并入上一块，避免碎片
    """
    raw = strip_front_matter(doc["raw"])
    lines = raw.splitlines()

    title = doc["name"]
    section = ""
    buffer: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        buffer = []
        if not body:
            return
        # 短碎片并入上一块（如果有）
        if len(body) < MIN_CHUNK_CHARS and chunks:
            chunks[-1].text += "\n\n" + body
            chunks[-1].retrieval_text += "\n" + body
            return
        chunks.append(_make_chunk(doc, title, section, body))

    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            flush()
            title = line[2:].strip() or title
            continue
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            continue
        buffer.append(line)

        # 超长保护：到了上限就找最近的空行断开
        if sum(len(x) for x in buffer) > MAX_CHUNK_CHARS:
            cut = len(buffer)
            for i in range(len(buffer) - 1, max(len(buffer) - 12, 0), -1):
                if not buffer[i].strip():
                    cut = i
                    break
            head, tail = buffer[:cut], buffer[cut:]
            buffer = head
            flush()
            buffer = tail

    flush()
    return chunks


def _make_chunk(doc: dict[str, Any], title: str, section: str, body: str) -> Chunk:
    path = f"{title} > {section}" if section else title
    return Chunk(
        id=f"{doc['name']}#{section or 'main'}",
        text=body,
        # ★ 检索文本 = 标题路径 + 正文（见类 docstring 的说明）
        retrieval_text=f"{path}\n{body}",
        title=title,
        section=section,
        source=Path(doc["path"]).name,
        meta=dict(doc["meta"]),
    )


def load_chunks(docs_dir: Path | None = None) -> list[Chunk]:
    """一步到位：加载并切分全部文档。"""
    chunks: list[Chunk] = []
    for doc in load_documents(docs_dir):
        chunks.extend(chunk_document(doc))
    logger.info(f"[司南·kb] 分块完成，共 {len(chunks)} 块")
    return chunks


__all__ = [
    "Chunk",
    "DOCS_DIR",
    "KB_ROOT",
    "MAX_CHUNK_CHARS",
    "MIN_CHUNK_CHARS",
    "load_documents",
    "chunk_document",
    "load_chunks",
    "strip_front_matter",
]
