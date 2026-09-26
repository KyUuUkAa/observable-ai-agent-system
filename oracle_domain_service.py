from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database import (
    get_oracle_model_version_summary,
    get_oracle_review_summary,
    list_oracle_recognition_records,
)
from oracle_workflow import oracle_records_to_csv, oracle_records_to_json


PROJECT_ROOT = Path(__file__).resolve().parent
EXPORT_DIR = PROJECT_ROOT / "reports" / "oracle_exports"
EXPORT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _validate_days(days: int) -> int:
    value = int(days)
    if value == 0:
        # Some local models express "today" as a zero-day offset.
        return 1
    if not 1 <= value <= 3650:
        raise ValueError("查询天数必须在 1 到 3650 之间。")
    return value


def _normalize_status(status: str) -> str | None:
    value = status.strip().lower()
    if value in {"", "all", "全部"}:
        return None
    if value not in {"pending", "auto_accepted", "accepted", "rejected"}:
        raise ValueError("复核状态必须是 all、pending、auto_accepted、accepted 或 rejected。")
    return value


def _candidate_code(record: dict) -> str | None:
    candidates = record.get("model_info", {}).get("visual_candidates", [])
    if not candidates:
        return None
    return candidates[0].get("class_code")


def summarize_oracle_record(record: dict) -> dict:
    retrieval_class = _candidate_code(record)
    routing = record.get("model_info", {}).get("hybrid_routing", {})
    return {
        "record_id": record["id"],
        "filename": record["original_filename"],
        "class_code": record["top1_class_code"],
        "confidence": record["top1_confidence"],
        "review_status": record["review_status"],
        "review_notes": record.get("review_notes"),
        "routing_mode": routing.get("mode", "classification"),
        "retrieval_top1_class": retrieval_class,
        "classification_retrieval_conflict": (
            retrieval_class is not None
            and retrieval_class != record["top1_class_code"]
        ),
        "model_version": record["model_version"],
        "created_at": record["created_at"],
        "reviewed_at": record.get("reviewed_at"),
    }


def query_oracle_records(
    *,
    review_status: str = "all",
    class_code: str = "",
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    days: int = 30,
    conflicts_only: bool = False,
    limit: int = 20,
) -> list[dict]:
    if class_code and (len(class_code) != 6 or not class_code.isdigit()):
        raise ValueError("类别编码必须是六位数字。")
    if not 1 <= int(limit) <= 500:
        raise ValueError("查询数量必须在 1 到 500 之间。")
    records = list_oracle_recognition_records(
        review_status=_normalize_status(review_status),
        class_code=class_code or None,
        min_confidence=float(min_confidence) if min_confidence > 0 else None,
        max_confidence=float(max_confidence) if max_confidence < 1 else None,
        created_after=datetime.now(timezone.utc) - timedelta(days=_validate_days(days)),
        conflicts_only=bool(conflicts_only),
        limit=int(limit),
        offset=0,
    )
    return [summarize_oracle_record(record) for record in records]


def create_oracle_export(
    *,
    output_format: str = "csv",
    review_status: str = "all",
    class_code: str = "",
    min_confidence: float = 0.0,
    max_confidence: float = 1.0,
    days: int = 30,
    conflicts_only: bool = False,
) -> dict:
    file_format = output_format.strip().lower()
    if file_format not in {"csv", "json"}:
        raise ValueError("导出格式只支持 csv 或 json。")
    records = query_oracle_records(
        review_status=review_status,
        class_code=class_code,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        days=days,
        conflicts_only=conflicts_only,
        limit=500,
    )
    export_id = uuid.uuid4().hex
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"{export_id}.{file_format}"
    content = (
        oracle_records_to_csv(records)
        if file_format == "csv"
        else oracle_records_to_json(records)
    )
    path.write_text(
        ("\ufeff" if file_format == "csv" else "") + content,
        encoding="utf-8",
    )
    return {
        "export_id": export_id,
        "format": file_format,
        "records": len(records),
        "download_url": f"/oracle/exports/{export_id}",
        "filters": {
            "review_status": review_status,
            "class_code": class_code or None,
            "min_confidence": min_confidence,
            "max_confidence": max_confidence,
            "days": days,
            "conflicts_only": conflicts_only,
        },
    }


def get_oracle_export(export_id: str) -> tuple[bytes, str, str]:
    if not EXPORT_ID_PATTERN.fullmatch(export_id):
        raise FileNotFoundError("导出文件不存在。")
    for suffix, media_type in (
        ("csv", "text/csv; charset=utf-8"),
        ("json", "application/json; charset=utf-8"),
    ):
        path = EXPORT_DIR / f"{export_id}.{suffix}"
        if path.is_file():
            return path.read_bytes(), media_type, path.name
    raise FileNotFoundError("导出文件不存在或已被清理。")


def _read_report(relative_path: str) -> dict | None:
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_oracle_quality_metrics() -> dict:
    classification = _read_report(
        "reports/oracle_dataset/classification_ge50_test.json"
    )
    calibration = _read_report(
        "reports/oracle_dataset/hybrid_calibration.json"
    )
    comparison = _read_report(
        "reports/oracle_dataset/model_comparison.json"
    )
    regression = _read_report(
        "reports/agent_regression/latest.json"
    )
    business_simulation = _read_report(
        "docs/evidence/oracle_business_300.json"
    )

    return {
        "review_summary": get_oracle_review_summary(),
        "model_versions": get_oracle_model_version_summary(),
        "classification": (
            {
                key: classification.get(key)
                for key in (
                    "images",
                    "classes",
                    "top1_accuracy",
                    "top5_accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                )
            }
            if classification
            else None
        ),
        "hybrid_calibration": (
            {
                "images": calibration.get("images"),
                "hybrid_threshold": calibration.get("hybrid_threshold"),
                "classification_retrieval_agreement": calibration.get(
                    "classification_retrieval_agreement"
                ),
                "accuracy_when_agree": calibration.get("accuracy_when_agree"),
                "milliseconds_per_image": calibration.get(
                    "milliseconds_per_image"
                ),
                "thresholds": calibration.get("thresholds", []),
            }
            if calibration
            else None
        ),
        "model_comparison": (
            {
                "split": comparison.get("split"),
                "shared_classes": comparison.get("shared_classes"),
                "images": comparison.get("images"),
                "models": [
                    {
                        key: model.get(key)
                        for key in (
                            "name",
                            "native_classes",
                            "top1_accuracy",
                            "top5_accuracy",
                            "macro_f1",
                            "error_count",
                            "error_rate",
                            "top_confusions",
                            "milliseconds_per_image",
                        )
                    }
                    for model in comparison.get("models", [])
                ],
            }
            if comparison
            else None
        ),
        "agent_regression": (
            {
                "generated_at": regression.get("generated_at"),
                "suite_name": regression.get("suite_name"),
                "summary": regression.get("summary", {}),
                "baseline_comparison": regression.get(
                    "baseline_comparison", {"status": "not_recorded"}
                ),
            }
            if regression
            else None
        ),
        "business_simulation": (
            {
                "generated_at": business_simulation.get("generated_at"),
                "sample": business_simulation.get("sample", {}),
                "classification": business_simulation.get("classification", {}),
                "retrieval": business_simulation.get("retrieval", {}),
                "fusion": business_simulation.get("fusion", {}),
                "workflow": business_simulation.get("workflow", {}),
                "performance": business_simulation.get("performance", {}),
                "leakage_audit": business_simulation.get("leakage_audit", {}),
            }
            if business_simulation
            else None
        ),
    }
