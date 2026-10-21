"""司南 · 知识库 · 检索（L11 · RAG 的 ④⑤ 步）

═══════════════════════════════════════════════════════════════════════════
稀疏 vs 稠密 —— RAG 检索的两条路
═══════════════════════════════════════════════════════════════════════════

    稀疏检索（BM25）                 稠密检索（向量）
    ─────────────────────           ─────────────────────
    靠"词是否出现"打分               靠"语义是否接近"打分
    可解释：能说清哪几个词命中了       不可解释：只有一个相似度数字
    零外部依赖、零成本                要模型、要算力、要钱
    怕"同义词"（"求导"≠"导数"）        不怕同义词（"求导"≈"导数"）
    不怕生僻词、专有名词              生僻词/新词容易失手

**本项目为什么两条都留着**：
    稀疏是保底（现在就能跑，embedding 不可用也不影响）；
    稠密是增强（key 到位后启用）。
    两条都有时用 RRF 融合——这也就是工业界说的"混合检索"。

★ RRF（Reciprocal Rank Fusion）为什么好用：
    它不比较两路的**分数**（BM25 分数 12.5 和余弦 0.83 没有可比性，
    直接加权是外行做法），只比较**排名**：

        score(d) = Σ  1 / (k + rank_i(d))        k 通常取 60

    某文档在稀疏里排第 1、稠密里排第 5 → 1/61 + 1/65
    另一文档稀疏第 3、稠密第 4      → 1/63 + 1/64（更大，胜出）

    好处：**免调参、免归一化、对抗单路噪声**。
    你说"我要给稀疏 0.7 稠密 0.3 的权重"——那 0.7 是怎么来的？
    大概率是拍的。RRF 把这个不靠谱的问题绕开了。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from app.common.logger import logger
from app.kb.chunker import KB_ROOT, Chunk, load_chunks
from app.kb.embeddings import embed_documents, get_embedder

INDEX_DIR = KB_ROOT / "index"
VEC_FILE = INDEX_DIR / "vectors.npz"
META_FILE = INDEX_DIR / "chunks.json"

# 检索返回条数上限（别一次塞太多给模型，上下文也是成本）
DEFAULT_TOP_K = 3

SearchMode = Literal["sparse", "dense", "hybrid", "auto"]


# ======================================================================
# 分词（稀疏检索的地基）
# ======================================================================
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """零依赖中英混合分词。

    中文用「单字 + 相邻双字（bigram）」：
        单字保证"极限"能匹配到含"极"的句子；
        bigram 保证"泰勒"和"勒展"这种搭配有区分度。
    为什么不上 jieba：为一个 12 篇的知识库引一个分词库不划算，
    而且**字符 bigram 对短文本检索的召回率其实相当能打**。
    （代价是词表膨胀、精度略低——这正是"能跑就行"的典型取舍。）

    英文/数字：整词小写。
    """
    text = (text or "").lower()
    tokens = _ASCII_WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)                                       # 单字
        tokens.extend(run[i: i + 2] for i in range(len(run) - 1))  # 双字
    return tokens


# ======================================================================
# 稀疏检索：BM25
# ======================================================================
class BM25Index:
    """BM25 —— 信息检索的经典打分函数（比 TF-IDF 更靠谱的那版）。

    公式（每个查询词累加）：

                    f(q,D) · (k1 + 1)
        score = Σ ────────────────────────────────── · IDF(q)
                 f(q,D) + k1 · (1 - b + b·|D|/avgdl)

    三个部分各司其职：
        IDF(q)          —— 越罕见的词越值钱（"泰勒"比"的"重要）
        f(q,D) 饱和项   —— 一个词出现 10 次不该比 1 次值钱 10 倍
                           （这就是 (k1+1) 分母里那个 k1 做的事）
        长度归一化      —— 长文档天然容易命中更多词，要除以自身长度
                           （b 控制归一化强度：b=1 全归一，b=0 不归一）

    参数用业界默认值 k1=1.5 / b=0.75，不折腾——**默认值是有统计依据的**，
    拍脑袋调参通常不如不调。
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.doc_tokens: list[list[str]] = [tokenize(c.retrieval_text) for c in chunks]
        self.tf: list[Counter] = [Counter(toks) for toks in self.doc_tokens]

        df: Counter = Counter()
        for tks in self.tf:
            df.update(tks.keys())
        self.n_docs = len(chunks)
        self.avgdl = (
            sum(len(t) for t in self.doc_tokens) / self.n_docs if self.n_docs else 0.0
        )
        # 加 0.5 平滑，避免除零；用 ln(1+x) 形式保证 IDF 恒为正
        self.idf: dict[str, float] = {
            t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 [(chunk 下标, 分数)]，按分数降序。"""
        if not self.n_docs:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scored: list[tuple[int, float]] = []
        for i, counts in enumerate(self.tf):
            dl = len(self.doc_tokens[i]) or 1
            norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score = 0.0
            for t in q_tokens:
                f = counts.get(t)
                if not f:
                    continue
                score += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (f + norm)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ======================================================================
# 稠密检索：向量 + 余弦相似
# ======================================================================
class VectorIndex:
    """把向量按行堆成矩阵，检索就是一次矩阵乘法。

    ★ 为什么要先归一化：
        余弦相似度 = a·b / (|a||b|)。把每行预先除以自身模长后，
        余弦就**退化成内积**——一次 `matrix @ q` 搞定，不用每对都除一遍。
        这是向量检索最常用的一个小技巧。

    ⚠️ 本类用**暴力搜索**（和所有向量逐个算）。12 篇文档完全够用；
       上万条就该上 FAISS / HNSW 这类近似最近邻（ANN）索引了。
       量级决定算法，别一上来就上重型武器。
    """

    def __init__(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if np is None:
            raise RuntimeError("需要 numpy 才能做向量检索")
        self.chunks = chunks
        m = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        self.matrix = m / np.maximum(norms, 1e-9)

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = self.matrix @ q
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]


# ======================================================================
# 统一检索入口
# ======================================================================
class KnowledgeBase:
    """知识库：稀疏 + 稠密 + 融合。

    ★ `embed_fn` 为什么是**构造参数**而不是在 search 里现取：

      ① **正确性**：文档向量和查询向量必须出自**同一个模型**。
         如果索引是用 A 模型建的、查询却用了 B 模型，
         两个向量根本不在同一个空间里——余弦相似度会算出一个
         看似合理、实则毫无意义的数字，**而且不报任何错**。
         这是向量检索最隐蔽的故障之一，把 embedder 绑进 KB 对象
         就从结构上杜绝了它。

      ② **可测性**：写自检时可以注入一个测试替身，
         不必依赖外部 API 是否可用。
         （这条是被测试逼出来的——第一版就是在 search 里现取，
          结果自检根本没法验证向量路径。）
    """

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]] | None = None,
        embed_fn: Any = None,
    ):
        self.chunks = chunks
        self.embed_fn = embed_fn if embed_fn is not None else get_embedder()
        self.bm25 = BM25Index(chunks)
        self.vector: VectorIndex | None = None
        if vectors:
            try:
                self.vector = VectorIndex(chunks, vectors)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[司南·kb] 向量索引构建失败，退回稀疏检索: {exc}")

    # ------------------------------------------------------------------
    @property
    def has_dense(self) -> bool:
        return self.vector is not None and self.embed_fn is not None

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: SearchMode = "auto",
    ) -> list[dict[str, Any]]:
        """检索。mode：
            sparse  只用 BM25
            dense   只用向量（不可用时报错，不静默降级）
            hybrid  两路 RRF 融合
            auto    有向量就 hybrid，没有就 sparse  ← 默认
        """
        if mode == "auto":
            mode = "hybrid" if self.has_dense else "sparse"

        cand = max(top_k * 3, top_k)  # 每路多取一些，融合后才有得挑
        rank_lists: list[list[int]] = []
        scores: dict[int, float] = {}

        if mode in ("sparse", "hybrid"):
            sparse_hits = self.bm25.search(query, cand)
            rank_lists.append([i for i, _ in sparse_hits])
            scores.update({i: s for i, s in sparse_hits})

        if mode in ("dense", "hybrid"):
            if not self.has_dense:
                if mode == "dense":
                    raise RuntimeError("已指定 dense 检索，但没有可用的 embedding provider")
                logger.warning("[司南·kb] dense 不可用，本次退回 sparse")
                if not rank_lists:
                    rank_lists.append([i for i, _ in self.bm25.search(query, cand)])
            else:
                # ★ 用 self.embed_fn（与建索引时同一个），不要在这里现取全局 embedder
                qv = self.embed_fn([query])[0]
                dense_hits = self.vector.search(qv, cand)  # type: ignore[union-attr]
                rank_lists.append([i for i, _ in dense_hits])
                for i, s in dense_hits:
                    scores.setdefault(i, s)

        if mode == "hybrid" and len(rank_lists) == 2:
            order = _rrf(rank_lists)
        else:
            order = rank_lists[0] if rank_lists else []

        return [self._to_hit(i, scores.get(i, 0.0), mode) for i in order[:top_k]]

    # ------------------------------------------------------------------
    def _to_hit(self, idx: int, score: float, mode: str) -> dict[str, Any]:
        c = self.chunks[idx]
        return {
            "title": c.title,
            "section": c.section,
            "heading_path": c.heading_path,
            "text": c.text,           # ★ 送模型的是原文，不是检索文本
            "score": round(score, 4),
            "source": c.source,
            "meta": c.meta,
            "mode": mode,
        }


def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion：只融合**排名**，不融合分数（见模块 docstring）"""
    fused: defaultdict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            fused[idx] += 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]


# ======================================================================
# 索引构建与持久化
# ======================================================================
_kb: KnowledgeBase | None = None


def _vectors_path() -> Path:
    return VEC_FILE


def build(force: bool = False, with_dense: bool | None = None) -> KnowledgeBase:
    """构建（或读取缓存的）知识库索引。

    ★ 向量为什么要缓存：
        调一次 embedding API 就是一次网络往返 + 一次计费。
        12 篇文个档几十块，每次启动都重算纯属浪费。
        缓存键是**分块 id 列表的哈希**——文档改了就失效重建，
        没改就直接读盘。这是"索引类系统"的通用做法。
    """
    global _kb
    chunks = load_chunks()
    if not chunks:
        _kb = KnowledgeBase([])
        return _kb

    ids = [c.id for c in chunks]
    vectors: list[list[float]] | None = None

    # ★ 只在构建时解析一次 embedder，然后**同时**用于"建索引"和"查询"
    #   ——两处必须是同一个模型，否则向量不在同一空间（见 KnowledgeBase docstring）
    fn = get_embedder()
    want_dense = with_dense if with_dense is not None else (fn is not None)
    if want_dense and np is not None and fn is not None:
        cached = _load_vector_cache(ids)
        if cached is not None and not force:
            vectors = cached
            logger.info(f"[司南·kb] 命中向量缓存（{len(vectors)} 条）")
        else:
            try:
                vectors = embed_documents(
                    [c.retrieval_text for c in chunks], fn=fn
                )
                _save_vector_cache(ids, vectors)
                logger.info(f"[司南·kb] 向量化完成并落盘（{len(vectors)} 条）")
            except Exception as exc:  # noqa: BLE001 — 向量失败要能退化，不能炸
                logger.warning(f"[司南·kb] 向量化失败，本次只用稀疏检索: {exc}")
                vectors = None

    _kb = KnowledgeBase(chunks, vectors, embed_fn=fn)
    return _kb


def get_kb() -> KnowledgeBase:
    """单例：第一次调用时构建。"""
    global _kb
    if _kb is None:
        build()
    return _kb


def reset_kb() -> None:
    global _kb
    _kb = None


def _load_vector_cache(ids: list[str]) -> list[list[float]] | None:
    if not (VEC_FILE.exists() and META_FILE.exists()):
        return None
    try:
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if meta.get("ids") != ids:
            logger.info("[司南·kb] 文档已变更，向量缓存失效")
            return None
        data = np.load(VEC_FILE)
        return data["vectors"].tolist()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[司南·kb] 向量缓存读取失败，将重建: {exc}")
        return None


def _save_vector_cache(ids: list[str], vectors: list[list[float]]) -> None:
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(VEC_FILE, vectors=np.asarray(vectors, dtype=np.float32))
        META_FILE.write_text(json.dumps({"ids": ids}, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — 缓存失败不影响本次检索
        logger.warning(f"[司南·kb] 向量缓存写入失败: {exc}")


def describe_kb() -> dict[str, Any]:
    kb = get_kb()
    return {
        "chunks": len(kb.chunks),
        "docs": len({c.source for c in kb.chunks}),
        "sparse": "BM25(k1=1.5, b=0.75)",
        "dense": "可用" if kb.has_dense else "不可用（未配置 embedding）",
        "default_mode": "hybrid" if kb.has_dense else "sparse",
        "vector_cache": str(VEC_FILE.name),
    }


__all__ = [
    "tokenize",
    "BM25Index",
    "VectorIndex",
    "KnowledgeBase",
    "build",
    "get_kb",
    "reset_kb",
    "describe_kb",
    "DEFAULT_TOP_K",
]
