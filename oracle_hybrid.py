from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "oracle_retrieval" / "index.npz"
DEFAULT_HYBRID_THRESHOLD = 0.85


class OracleRetrievalUnavailableError(RuntimeError):
    """Raised when the optional long-tail retrieval index is unavailable."""


class OracleCandidateNotFoundError(RuntimeError):
    """Raised when a candidate or its source image cannot be resolved safely."""


@dataclass(frozen=True)
class OracleRetrievalIndex:
    embeddings: np.ndarray
    candidate_ids: np.ndarray
    pair_ids: np.ndarray
    class_codes: np.ndarray
    glyph_relpaths: np.ndarray
    rubbing_relpaths: np.ndarray
    candidate_lookup: dict[str, int]
    class_lookup: dict[str, np.ndarray]
    fingerprint: str
    dataset_root_hint: str


def get_retrieval_index_path() -> Path:
    configured = os.getenv("ORACLE_RETRIEVAL_INDEX")
    path = Path(configured).expanduser() if configured else DEFAULT_INDEX_PATH
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def get_dataset_root(index: OracleRetrievalIndex | None = None) -> Path:
    configured = os.getenv("ORACLE_DATASET_ROOT", "").strip()
    if not configured and index is not None:
        configured = index.dataset_root_hint
    if not configured:
        raise OracleRetrievalUnavailableError(
            "未设置 ORACLE_DATASET_ROOT，无法读取候选字模和拓片图片。"
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise OracleRetrievalUnavailableError(
            "ORACLE_DATASET_ROOT 不存在或不是目录。"
        )
    return root


def get_hybrid_threshold() -> float:
    raw = os.getenv("ORACLE_HYBRID_THRESHOLD", str(DEFAULT_HYBRID_THRESHOLD))
    try:
        value = float(raw)
    except ValueError as exc:
        raise OracleRetrievalUnavailableError(
            "ORACLE_HYBRID_THRESHOLD 必须是 0 到 1 之间的数字。"
        ) from exc
    if not 0 <= value <= 1:
        raise OracleRetrievalUnavailableError(
            "ORACLE_HYBRID_THRESHOLD 必须在 0 到 1 之间。"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_retrieval_index() -> OracleRetrievalIndex:
    path = get_retrieval_index_path()
    if not path.is_file():
        raise OracleRetrievalUnavailableError(
            "甲骨文字模向量索引不存在，请先运行 build_oracle_retrieval_index.py。"
        )
    try:
        with np.load(path, allow_pickle=False) as stored:
            embeddings = np.asarray(stored["embeddings"], dtype=np.float32)
            candidate_ids = np.asarray(stored["candidate_ids"], dtype=str)
            pair_ids = np.asarray(stored["pair_ids"], dtype=str)
            class_codes = np.asarray(stored["class_codes"], dtype=str)
            glyph_relpaths = np.asarray(stored["glyph_relpaths"], dtype=str)
            rubbing_relpaths = np.asarray(stored["rubbing_relpaths"], dtype=str)
            dataset_root_hint = (
                str(np.asarray(stored["dataset_root"], dtype=str).item())
                if "dataset_root" in stored.files
                else ""
            )
    except (KeyError, OSError, ValueError) as exc:
        raise OracleRetrievalUnavailableError("甲骨文字模向量索引格式无效。") from exc

    lengths = {
        embeddings.shape[0],
        len(candidate_ids),
        len(pair_ids),
        len(class_codes),
        len(glyph_relpaths),
        len(rubbing_relpaths),
    }
    if embeddings.ndim != 2 or len(lengths) != 1 or embeddings.shape[0] == 0:
        raise OracleRetrievalUnavailableError("甲骨文字模向量索引维度不一致。")
    norms = np.sqrt(np.square(embeddings).sum(axis=1, keepdims=True))
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    candidate_lookup = {
        candidate_id: index
        for index, candidate_id in enumerate(candidate_ids.tolist())
    }
    grouped: dict[str, list[int]] = {}
    for index, class_code in enumerate(class_codes.tolist()):
        grouped.setdefault(class_code, []).append(index)
    class_lookup = {
        class_code: np.asarray(indices, dtype=np.int64)
        for class_code, indices in grouped.items()
    }
    return OracleRetrievalIndex(
        embeddings=embeddings,
        candidate_ids=candidate_ids,
        pair_ids=pair_ids,
        class_codes=class_codes,
        glyph_relpaths=glyph_relpaths,
        rubbing_relpaths=rubbing_relpaths,
        candidate_lookup=candidate_lookup,
        class_lookup=class_lookup,
        fingerprint=_sha256(path),
        dataset_root_hint=dataset_root_hint,
    )


def get_retrieval_status() -> dict:
    try:
        index = load_retrieval_index()
        root = get_dataset_root(index)
        return {
            "status": "ready",
            "candidates": int(index.embeddings.shape[0]),
            "classes": len(index.class_lookup),
            "embedding_dimension": int(index.embeddings.shape[1]),
            "index_sha256": index.fingerprint,
            "dataset_available": root.is_dir(),
        }
    except OracleRetrievalUnavailableError as exc:
        return {"status": "unavailable", "detail": str(exc)}


def extract_yolo_embedding(yolo, source: np.ndarray) -> np.ndarray:
    """Extract the normalized penultimate YOLO classification feature."""

    import torch
    from torch.nn import functional as functional

    core = yolo.model
    modules = list(core.model.children())
    if len(modules) < 2:
        raise OracleRetrievalUnavailableError(
            "当前 YOLO 分类模型没有可用的特征骨干。"
        )
    device = next(core.parameters()).device
    tensor = torch.from_numpy(np.ascontiguousarray(source)).to(device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    expected_size = int(os.getenv("ORACLE_IMGSZ", "224"))
    if tensor.shape[-2:] != (expected_size, expected_size):
        tensor = functional.interpolate(
            tensor,
            size=(expected_size, expected_size),
            mode="bilinear",
            align_corners=False,
        )
    with torch.no_grad():
        features = tensor
        for module in modules[:-1]:
            features = module(features)
        features = functional.adaptive_avg_pool2d(features, 1).flatten(1)
        features = functional.normalize(features, dim=-1)
    return features[0].detach().cpu().numpy().astype(np.float32, copy=False)


def _candidate(index: OracleRetrievalIndex, item_index: int, score: float) -> dict:
    candidate_id = str(index.candidate_ids[item_index])
    return {
        "candidate_id": candidate_id,
        "class_code": str(index.class_codes[item_index]),
        "pair_id": str(index.pair_ids[item_index]),
        "similarity": float(score),
        "glyph_url": f"/oracle/candidates/{candidate_id}/glyph",
        "rubbing_url": f"/oracle/candidates/{candidate_id}/rubbing",
    }


def select_visual_candidates(
    embedding: np.ndarray,
    classification_top5: list[dict],
    *,
    confidence: float,
    threshold: float | None = None,
    limit: int = 5,
    force_retrieval: bool = False,
) -> tuple[dict, list[dict]]:
    """Route a query to high-confidence classification or long-tail retrieval."""

    index = load_retrieval_index()
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape[0] != index.embeddings.shape[1]:
        raise OracleRetrievalUnavailableError(
            "查询向量与字模索引维度不一致，请重新构建索引。"
        )
    vector /= max(float(np.sqrt(np.square(vector).sum())), 1e-12)
    import torch

    scores = (
        torch.from_numpy(index.embeddings)
        @ torch.from_numpy(vector)
    ).numpy()
    cutoff = get_hybrid_threshold() if threshold is None else float(threshold)

    candidates = []
    if confidence >= cutoff and not force_retrieval:
        mode = "classification"
        reason = "YOLO Top-1 confidence reached the hybrid threshold."
        for prediction in classification_top5[:limit]:
            class_code = str(prediction["class_code"])
            indices = index.class_lookup.get(class_code)
            if indices is None or len(indices) == 0:
                continue
            local = int(indices[np.argmax(scores[indices])])
            item = _candidate(index, local, float(scores[local]))
            item["source"] = "classification_support"
            item["classification_confidence"] = float(prediction["confidence"])
            candidates.append(item)
    else:
        mode = "retrieval"
        reason = (
            "Retrieval was explicitly requested by the user."
            if force_retrieval and confidence >= cutoff
            else "YOLO Top-1 confidence was below the hybrid threshold."
        )
        seen_classes: set[str] = set()
        for item_index in np.argsort(scores)[::-1]:
            class_code = str(index.class_codes[item_index])
            if class_code in seen_classes:
                continue
            seen_classes.add(class_code)
            item = _candidate(index, int(item_index), float(scores[item_index]))
            item["source"] = "long_tail_retrieval"
            candidates.append(item)
            if len(candidates) >= limit:
                break

    routing = {
        "mode": mode,
        "threshold": cutoff,
        "classification_confidence": float(confidence),
        "reason": reason,
        "retrieval_index_sha256": index.fingerprint,
        "candidate_count": int(index.embeddings.shape[0]),
        "class_count": len(index.class_lookup),
    }
    return routing, candidates


def _safe_source_path(relative_path: str, index: OracleRetrievalIndex) -> Path:
    root = get_dataset_root(index)
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OracleCandidateNotFoundError("候选图片路径越过数据集根目录。") from exc
    if not path.is_file():
        raise OracleCandidateNotFoundError("候选图片不存在。")
    return path


def get_candidate_image(candidate_id: str, kind: str) -> tuple[bytes, str]:
    if kind not in {"glyph", "rubbing"}:
        raise OracleCandidateNotFoundError("不支持的候选图片类型。")
    index = load_retrieval_index()
    item_index = index.candidate_lookup.get(candidate_id)
    if item_index is None:
        raise OracleCandidateNotFoundError("候选记录不存在。")
    relative_path = (
        index.glyph_relpaths[item_index]
        if kind == "glyph"
        else index.rubbing_relpaths[item_index]
    )
    path = _safe_source_path(str(relative_path), index)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.read_bytes(), media_type
