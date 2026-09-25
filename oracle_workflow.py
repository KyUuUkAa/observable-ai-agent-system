from __future__ import annotations

import csv
import io
import json
import os
import time
from typing import Iterable


DEFAULT_REVIEW_THRESHOLD = 0.85
MAX_BATCH_FILES = 50


def get_default_review_threshold() -> float:
    raw_value = os.getenv(
        "ORACLE_REVIEW_THRESHOLD",
        str(DEFAULT_REVIEW_THRESHOLD),
    )
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "ORACLE_REVIEW_THRESHOLD 必须是 0 到 1 之间的数字。"
        ) from exc
    return validate_review_threshold(threshold)


def validate_review_threshold(threshold: float) -> float:
    value = float(threshold)
    if not 0 <= value <= 1:
        raise ValueError("复核阈值必须在 0 到 1 之间。")
    return value


def build_recognition_trace(
    *,
    started_at: float,
    source: str,
    filename: str,
    routing: dict | None = None,
) -> dict:
    return {
        "recognition": {
            "status": "success",
            "source": source,
            "filename": filename,
            "routing": routing or {"mode": "classification"},
            "latency_seconds": round(
                time.perf_counter() - started_at,
                4,
            ),
            "steps": [
                "validate_upload",
                "decode_image",
                "run_classifier",
                "route_hybrid_candidates",
                "persist_record",
            ],
        }
    }


def oracle_records_to_json(records: Iterable[dict]) -> str:
    return json.dumps(
        list(records),
        ensure_ascii=False,
        indent=2,
    )


def oracle_records_to_csv(records: Iterable[dict]) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        "id",
        "original_filename",
        "image_sha256",
        "top1_class_code",
        "top1_confidence",
        "top5",
        "routing_mode",
        "retrieval_top1_class",
        "classification_retrieval_conflict",
        "model_version",
        "review_threshold",
        "review_status",
        "review_notes",
        "source",
        "conversation_id",
        "created_at",
        "reviewed_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for record in records:
        normalized = {
            **record,
            "id": record.get("id", record.get("record_id")),
            "original_filename": record.get(
                "original_filename", record.get("filename")
            ),
            "top1_class_code": record.get(
                "top1_class_code", record.get("class_code")
            ),
            "top1_confidence": record.get(
                "top1_confidence", record.get("confidence")
            ),
        }
        writer.writerow(
            {
                key: (
                    json.dumps(normalized.get(key), ensure_ascii=False)
                    if isinstance(normalized.get(key), (list, dict))
                    else normalized.get(key)
                )
                for key in fieldnames
            }
        )

    return output.getvalue()
