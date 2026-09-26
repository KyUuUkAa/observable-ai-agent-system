from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Iterable


def source_group_key(record: dict) -> str:
    """Return a stricter source identity than the variant-level pair_id."""

    return f"{record['class_code']}:{record['source_id'].lower()}"


def split_conflicts(records: Iterable[dict], key_name: str) -> list[dict]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = (
            source_group_key(record)
            if key_name == "source_group"
            else str(record[key_name])
        )
        grouped[key].add(str(record["split"]))
    return [
        {"key": key, "splits": sorted(splits)}
        for key, splits in sorted(grouped.items())
        if len(splits) > 1
    ]


def allocate_stratified_counts(
    class_counts: dict[str, int],
    sample_size: int,
) -> dict[str, int]:
    """Allocate an exact, proportional sample with one item per class when possible."""

    counts = {str(key): int(value) for key, value in class_counts.items() if value > 0}
    total = sum(counts.values())
    if not 0 < sample_size <= total:
        raise ValueError("sample_size must be between one and the population size")

    classes = sorted(counts)
    allocation = {class_code: 0 for class_code in classes}
    if sample_size >= len(classes):
        allocation = {class_code: 1 for class_code in classes}

    remaining = sample_size - sum(allocation.values())
    capacity = {
        class_code: counts[class_code] - allocation[class_code]
        for class_code in classes
    }
    while remaining:
        active = [class_code for class_code in classes if capacity[class_code] > 0]
        if not active:
            raise ValueError("not enough records to complete stratified allocation")
        active_total = sum(capacity[class_code] for class_code in active)
        quotas = {
            class_code: remaining * capacity[class_code] / active_total
            for class_code in active
        }
        floors = {
            class_code: min(capacity[class_code], math.floor(quotas[class_code]))
            for class_code in active
        }
        assigned = sum(floors.values())
        if assigned:
            for class_code, amount in floors.items():
                allocation[class_code] += amount
                capacity[class_code] -= amount
            remaining -= assigned
            continue

        ranked = sorted(
            active,
            key=lambda class_code: (
                -(quotas[class_code] - math.floor(quotas[class_code])),
                class_code,
            ),
        )
        for class_code in ranked[:remaining]:
            allocation[class_code] += 1
            capacity[class_code] -= 1
        remaining = 0

    return allocation


def select_stratified_records(
    records: Iterable[dict],
    *,
    sample_size: int,
    seed: int,
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["class_code"])].append(record)
    allocation = allocate_stratified_counts(
        {class_code: len(items) for class_code, items in grouped.items()},
        sample_size,
    )

    selected = []
    for class_code in sorted(grouped):
        items = list(grouped[class_code])
        random.Random(f"{seed}:{class_code}").shuffle(items)
        selected.extend(items[: allocation[class_code]])
    selected.sort(key=lambda item: hashlib.sha256(
        f"{seed}:{item['record_id']}".encode("utf-8")
    ).hexdigest())
    return selected


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def classification_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    top1_correct = sum(row["classification_top1"] == row["true_class"] for row in rows)
    top5_correct = sum(row["true_class"] in row["classification_top5"] for row in rows)
    classes = sorted({row["true_class"] for row in rows})

    true_counts = Counter(row["true_class"] for row in rows)
    predicted_counts = Counter(row["classification_top1"] for row in rows)
    true_positive = Counter(
        row["true_class"]
        for row in rows
        if row["classification_top1"] == row["true_class"]
    )
    per_class = []
    for class_code in classes:
        precision = _safe_rate(true_positive[class_code], predicted_counts[class_code]) or 0.0
        recall = _safe_rate(true_positive[class_code], true_counts[class_code]) or 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "class_code": class_code,
                "support": true_counts[class_code],
                "correct": true_positive[class_code],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return {
        "top1_correct": top1_correct,
        "top1_accuracy": _safe_rate(top1_correct, total) or 0.0,
        "top5_correct": top5_correct,
        "top5_accuracy": _safe_rate(top5_correct, total) or 0.0,
        "macro_precision": sum(item["precision"] for item in per_class) / len(per_class),
        "macro_recall": sum(item["recall"] for item in per_class) / len(per_class),
        "macro_f1": sum(item["f1"] for item in per_class) / len(per_class),
        "per_class": per_class,
    }


def retrieval_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    class_at_1 = sum(row["retrieval_top5"][0]["class_code"] == row["true_class"] for row in rows)
    class_at_5 = sum(
        row["true_class"] in [item["class_code"] for item in row["retrieval_top5"]]
        for row in rows
    )
    exact_at_1 = sum(row["retrieval_exact_pair_rank"] == 1 for row in rows)
    exact_at_5 = sum(row["retrieval_exact_pair_rank"] <= 5 for row in rows)
    return {
        "queries": total,
        "class_recall_at_1": _safe_rate(class_at_1, total) or 0.0,
        "class_recall_at_5": _safe_rate(class_at_5, total) or 0.0,
        "exact_pair_recall_at_1": _safe_rate(exact_at_1, total) or 0.0,
        "exact_pair_recall_at_5": _safe_rate(exact_at_5, total) or 0.0,
    }


def threshold_analysis(
    rows: list[dict],
    thresholds: Iterable[float],
    *,
    assumed_review_seconds: float,
) -> list[dict]:
    total = len(rows)
    result = []
    for threshold in thresholds:
        auto_rows = [row for row in rows if row["classification_confidence"] >= threshold]
        review_rows = [row for row in rows if row["classification_confidence"] < threshold]
        auto_correct = sum(
            row["classification_top1"] == row["true_class"] for row in auto_rows
        )
        hybrid_correct = sum(
            (
                row["classification_top1"]
                if row["classification_confidence"] >= threshold
                else row["retrieval_top5"][0]["class_code"]
            )
            == row["true_class"]
            for row in rows
        )
        conflicts = sum(
            row["classification_top1"] != row["retrieval_top5"][0]["class_code"]
            for row in review_rows
        )
        review_per_100 = len(review_rows) * 100 / total if total else 0.0
        result.append(
            {
                "threshold": float(threshold),
                "auto_pass_count": len(auto_rows),
                "auto_pass_coverage": _safe_rate(len(auto_rows), total) or 0.0,
                "auto_pass_correct": auto_correct,
                "auto_pass_accuracy": _safe_rate(auto_correct, len(auto_rows)),
                "false_accept_count": len(auto_rows) - auto_correct,
                "false_accept_rate_among_auto_passed": _safe_rate(
                    len(auto_rows) - auto_correct,
                    len(auto_rows),
                ),
                "review_count": len(review_rows),
                "review_rate": _safe_rate(len(review_rows), total) or 0.0,
                "review_conflict_count": conflicts,
                "review_conflict_rate": _safe_rate(conflicts, len(review_rows)),
                "experimental_replacement_accuracy": _safe_rate(hybrid_correct, total) or 0.0,
                "estimated_reviews_per_100": review_per_100,
                "estimated_manual_minutes_per_100": (
                    review_per_100 * assumed_review_seconds / 60
                ),
            }
        )
    return result


def confusion_rows(rows: list[dict], limit: int = 10) -> list[dict]:
    counts = Counter(
        (row["true_class"], row["classification_top1"])
        for row in rows
        if row["true_class"] != row["classification_top1"]
    )
    return [
        {"true_class": truth, "predicted_class": predicted, "count": count}
        for (truth, predicted), count in counts.most_common(limit)
    ]


def render_markdown(report: dict) -> str:
    classification = report["classification"]
    retrieval = report["retrieval"]
    workflow = report["workflow"]
    performance = report["performance"]
    leakage = report["leakage_audit"]
    lines = [
        "# 300张甲骨文拓片业务模拟实验",
        "",
        "> 本报告是独立测试集上的离线业务模拟，不是博物馆真实用户研究。人工复核时间为透明的规划假设。",
        "",
        "## 核心结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 样本量 / 覆盖类别 | {report['sample']['images']} / {report['sample']['classes']} |",
        f"| 分类 Top-1 | {classification['top1_accuracy']:.2%} |",
        f"| 分类 Top-5 | {classification['top5_accuracy']:.2%} |",
        f"| Macro-F1 | {classification['macro_f1']:.2%} |",
        f"| 检索类别 Recall@1 | {retrieval['class_recall_at_1']:.2%} |",
        f"| 检索类别 Recall@5 | {retrieval['class_recall_at_5']:.2%} |",
        f"| 分类/检索候选集合 Recall@5 | {report['fusion']['assisted_candidate_recall_at_5']:.2%} |",
        f"| 阈值 {workflow['threshold']:.2f} 自动通过覆盖率 | {workflow['auto_pass_coverage']:.2%} |",
        f"| 自动通过准确率 | {workflow['auto_pass_accuracy']:.2%} |",
        f"| 待复核率 | {workflow['review_rate']:.2%} |",
        f"| 低置信度样本中的分类/检索冲突率 | {workflow['review_conflict_rate']:.2%} |",
        f"| 单张平均 / P95 | {performance['mean_ms_per_image']:.2f} / {performance['p95_ms_per_image']:.2f} ms |",
        f"| 纯系统处理100张 | {performance['compute_seconds_per_100']:.2f} 秒 |",
        "",
        "## 数据隔离审计",
        "",
        f"本次样本审计状态：**{'通过' if leakage['passed'] else '未通过'}**。",
        "",
        f"- 全量清单 pair_id 跨拆分冲突：{leakage['dataset_pair_split_conflicts']}。",
        f"- 全量清单更严格的来源组跨拆分冲突：{leakage['dataset_source_group_split_conflicts']}；这些记录已从抽样候选中排除。",
        f"- 300张样本与 train/val 的 pair_id 重叠：{leakage['selected_pair_overlap']}。",
        f"- 300张样本与 train/val 的来源组重叠：{leakage['selected_source_group_overlap']}。",
        f"- 300张样本与 train/val 的图片内容哈希重叠：{leakage['selected_content_hash_overlap']}。",
        "",
        "## 置信度分流",
        "",
        "| 阈值 | 自动通过率 | 自动通过准确率 | 错误接受率 | 待复核率 | 替换式融合准确率* |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["threshold_analysis"]:
        auto_accuracy = item["auto_pass_accuracy"]
        false_rate = item["false_accept_rate_among_auto_passed"]
        lines.append(
            f"| {item['threshold']:.2f} | {item['auto_pass_coverage']:.2%} | "
            f"{auto_accuracy:.2%} | {false_rate:.2%} | {item['review_rate']:.2%} | "
            f"{item['experimental_replacement_accuracy']:.2%} |"
            if auto_accuracy is not None and false_rate is not None
            else f"| {item['threshold']:.2f} | {item['auto_pass_coverage']:.2%} | N/A | N/A | "
            f"{item['review_rate']:.2%} | {item['experimental_replacement_accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "\* 替换式融合表示低置信度时直接采用检索Top-1，仅作为离线对照；线上工作流不会静默覆盖分类结果。",
            "",
            "## 耗时口径",
            "",
            f"- 模型和索引初始化：{performance['initialization_seconds']:.3f} 秒。",
            f"- 300张顺序端到端视觉处理：{performance['processing_seconds']:.3f} 秒。",
            f"- 每100张预计产生 {performance['estimated_reviews_per_100']:.2f} 条待复核记录。",
            f"- 若人工每条复核按 {performance['assumed_review_seconds']:.1f} 秒规划，人工复核约 "
            f"{performance['estimated_manual_minutes_per_100']:.2f} 分钟/100张；这不是实测人员效率。",
            "",
            "## 可复现信息",
            "",
            f"- 抽样种子：`{report['sample']['seed']}`",
            f"- 样本清单 SHA-256：`{report['provenance']['sample_manifest_sha256']}`",
            f"- 模型 SHA-256：`{report['provenance']['model_sha256']}`",
            f"- 检索索引 SHA-256：`{report['provenance']['index_sha256']}`",
            f"- 设备：`{report['provenance']['device']}`",
            "",
            "详细逐样本结果、CSV/JSONL导出和错误样例图库位于本地忽略目录 `reports/oracle_business_simulation/latest/`。",
            "",
        ]
    )
    return "\n".join(lines)
