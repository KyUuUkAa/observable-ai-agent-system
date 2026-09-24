from pathlib import Path
import time

import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# 配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RESUME_PATH = (
    BASE_DIR
    / "data"
    / "resume.txt"
)

MODEL_NAME = "BAAI/bge-small-zh-v1.5"


# =========================================================
# 加载本地 Embedding 模型
# =========================================================

print(
    "[RAG] 正在加载本地 Embedding 模型..."
)

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print(
    "[RAG] Embedding 模型加载完成"
)


# =========================================================
# 读取简历
# =========================================================

def load_resume() -> str:

    if not RESUME_PATH.exists():

        raise FileNotFoundError(
            f"找不到简历文件："
            f"{RESUME_PATH}"
        )

    return RESUME_PATH.read_text(
        encoding="utf-8"
    )


# =========================================================
# 文本切块
# =========================================================

def chunk_text(
    text: str
) -> list[str]:

    chunks = [
        block.strip()
        for block in text.split("\n\n")
        if block.strip()
    ]

    return chunks


# =========================================================
# Query Embedding
# =========================================================

def get_query_embedding(
    query: str
) -> np.ndarray:

    """
    为用户查询生成向量。
    """

    instruction = (
        "为这个句子生成表示"
        "以用于检索相关文章："
    )

    query_text = (
        instruction
        + query
    )

    embedding = (
        embedding_model.encode(
            query_text,
            normalize_embeddings=True
        )
    )

    return np.asarray(
        embedding,
        dtype=np.float32
    )


# =========================================================
# Document Embedding
# =========================================================

def get_document_embedding(
    text: str
) -> np.ndarray:

    """
    为文档 Chunk 生成向量。
    """

    embedding = (
        embedding_model.encode(
            text,
            normalize_embeddings=True
        )
    )

    return np.asarray(
        embedding,
        dtype=np.float32
    )


# =========================================================
# 构建 Document Cache
# =========================================================

print(
    "[RAG] 正在构建简历向量缓存..."
)

_resume_text = load_resume()

_resume_chunks = chunk_text(
    _resume_text
)


document_embeddings = []

for chunk in _resume_chunks:

    embedding = (
        get_document_embedding(
            chunk
        )
    )

    document_embeddings.append(
        embedding
    )


# list -> numpy matrix
_resume_embeddings = np.vstack(
    document_embeddings
)


print(
    f"[RAG] 缓存完成，共 "
    f"{len(_resume_chunks)} 个 chunks"
)

print(
    f"[RAG] Embedding Matrix Shape: "
    f"{_resume_embeddings.shape}"
)


# =========================================================
# Semantic Search
# =========================================================

def search_resume_rag(
    query: str,
    top_k: int = 2
) -> list[dict]:

    total_start = (
        time.perf_counter()
    )


    # -----------------------------------------------------
    # 1. Query Embedding
    # -----------------------------------------------------

    embedding_start = (
        time.perf_counter()
    )

    query_embedding = (
        get_query_embedding(
            query
        )
    )

    embedding_latency = (
        time.perf_counter()
        - embedding_start
    )


    # -----------------------------------------------------
    # 2. Similarity Search
    # -----------------------------------------------------

    search_start = (
        time.perf_counter()
    )

    # 因为 document 和 query 都已经 normalize
    # 所以 dot product 就等价于 cosine similarity
    scores = (
        _resume_embeddings
        @ query_embedding
    )


    # 防止 top_k 超过 chunk 数量
    actual_top_k = min(
        top_k,
        len(_resume_chunks)
    )


    top_indices = (
        np.argsort(scores)[::-1]
        [:actual_top_k]
    )


    results = []

    for index in top_indices:

        results.append({
            "chunk_id":
                int(index),

            "content":
                _resume_chunks[index],

            "score":
                float(
                    scores[index]
                ),
        })


    search_latency = (
        time.perf_counter()
        - search_start
    )


    # -----------------------------------------------------
    # 3. Total Latency
    # -----------------------------------------------------

    total_latency = (
        time.perf_counter()
        - total_start
    )


    print(
        "[PERF] RAG | "
        f"embedding="
        f"{embedding_latency:.4f}s | "
        f"search="
        f"{search_latency:.4f}s | "
        f"total="
        f"{total_latency:.4f}s"
    )


    return results

def lexical_score(
    query: str,
    document: str
) -> float:
    """
    简单关键词重合分数。

    主要用于补充 Dense Embedding
    对技术专有名词的匹配能力。
    """

    query_lower = query.lower()
    document_lower = document.lower()

    keywords = [
        "agent",
        "rag",
        "fastapi",
        "vue",
        "postgresql",
        "sqlalchemy",
        "yolo",
        "transformer",
        "diffusion",
        "pytorch",
        "tool",
        "calculator",
        "多视图",
        "聚类",
        "图像分类",
        "计算机视觉",
        "数据库",
        "前端",
        "后端",
    ]

    matched = 0
    total = 0

    for keyword in keywords:

        if keyword in query_lower:

            total += 1

            if keyword in document_lower:
                matched += 1

    if total == 0:
        return 0.0

    return matched / total

def search_resume_hybrid(
    query: str,
    top_k: int = 2,
    dense_weight: float = 0.90,
    lexical_weight: float = 0.10,
) -> list[dict]:

    total_start = time.perf_counter()


    # =====================================================
    # Query Embedding
    # =====================================================

    embedding_start = (
        time.perf_counter()
    )

    query_embedding = (
        get_query_embedding(
            query
        )
    )

    embedding_latency = (
        time.perf_counter()
        - embedding_start
    )


    # =====================================================
    # Dense Similarity
    # =====================================================

    search_start = (
        time.perf_counter()
    )

    dense_scores = (
        _resume_embeddings
        @ query_embedding
    )


    # =====================================================
    # Dense + Lexical
    # =====================================================

    results = []

    for index, chunk in enumerate(
        _resume_chunks
    ):

        dense_score = float(
            dense_scores[index]
        )

        lexical = lexical_score(
            query,
            chunk
        )

        final_score = (
            dense_weight
            * dense_score
            +
            lexical_weight
            * lexical
        )

        results.append({

            "chunk_id":
                index,

            "content":
                chunk,

            "score":
                final_score,

            "dense_score":
                dense_score,

            "lexical_score":
                lexical,
        })


    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    search_latency = (
        time.perf_counter()
        - search_start
    )

    total_latency = (
        time.perf_counter()
        - total_start
    )


    print(
        "[PERF] Hybrid RAG | "
        f"embedding="
        f"{embedding_latency:.4f}s | "
        f"search="
        f"{search_latency:.4f}s | "
        f"total="
        f"{total_latency:.4f}s"
    )


    return results[:top_k]