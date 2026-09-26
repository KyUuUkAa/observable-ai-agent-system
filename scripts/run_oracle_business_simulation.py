from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.oracle_business_core import (
    classification_metrics,
    confusion_rows,
    percentile,
    render_markdown,
    retrieval_metrics,
    select_stratified_records,
    source_group_key,
    split_conflicts,
    threshold_analysis,
)
from oracle_dataset.retrieval import YoloImageEncoder, build_image_transform
from oracle_hybrid import load_retrieval_index


DEFAULT_THRESHOLDS = "0.50,0.60,0.70,0.80,0.85,0.90,0.95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible, leakage-audited Oracle curation simulation on a "
            "stratified held-out sample."
        )
    )
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "oracle" / "best_ge50.pt"),
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "oracle_classification_ge50"),
    )
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "reports" / "oracle_dataset" / "manifest.csv"),
    )
    parser.add_argument(
        "--index",
        default=str(PROJECT_ROOT / "data" / "oracle_retrieval" / "index.npz"),
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--review-threshold", type=float, default=0.85)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--assumed-review-seconds",
        type=float,
        default=20.0,
        help="Planning assumption only; this is not measured human productivity.",
    )
    parser.add_argument("--gallery-size", type=int, default=30)
    parser.add_argument(
        "--sample-manifest",
        default=str(
            PROJECT_ROOT / "evaluation" / "cases" / "oracle_business_300.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT / "reports" / "oracle_business_simulation" / "latest"
        ),
    )
    parser.add_argument(
        "--evidence-prefix",
        default=str(PROJECT_ROOT / "docs" / "evidence" / "oracle_business_300"),
    )
    parser.add_argument(
        "--refresh-sample",
        action="store_true",
        help="Regenerate the fixed sample manifest instead of reusing it.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def materialized_path(data_root: Path, record: dict) -> Path:
    extension = Path(record["rubbing_relpath"]).suffix.lower() or ".bmp"
    path = (
        data_root
        / record["split"]
        / record["class_code"]
        / f"{record['record_id']}{extension}"
    ).resolve()
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Materialized path escapes data root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def candidate_id(pair_id: str) -> str:
    return hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:20]


def _eligible_classes(data_root: Path) -> set[str]:
    test_root = data_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)
    return {path.name for path in test_root.iterdir() if path.is_dir()}


def _strict_candidates(records: list[dict], eligible: set[str]) -> tuple[list[dict], dict]:
    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_splits[source_group_key(record)].add(record["split"])

    candidates = [
        record
        for record in records
        if record["split"] == "test"
        and record["class_code"] in eligible
        and source_splits[source_group_key(record)] == {"test"}
    ]
    return candidates, source_splits


def _training_hashes(records: list[dict], data_root: Path, eligible: set[str]) -> set[str]:
    hashes = set()
    for record in records:
        if record["class_code"] not in eligible or record["split"] not in {"train", "val"}:
            continue
        hashes.add(sha256(materialized_path(data_root, record)))
    return hashes


def _sample_rows(
    records: list[dict],
    *,
    data_root: Path,
    eligible: set[str],
    sample_size: int,
    seed: int,
    sample_manifest: Path,
    refresh: bool,
) -> tuple[list[dict], set[str], dict]:
    by_id = {record["record_id"]: record for record in records}
    strict_candidates, source_splits = _strict_candidates(records, eligible)
    training_hashes = _training_hashes(records, data_root, eligible)

    candidate_rows = []
    excluded_hash_overlap = 0
    for record in strict_candidates:
        content_hash = sha256(materialized_path(data_root, record))
        if content_hash in training_hashes:
            excluded_hash_overlap += 1
            continue
        candidate_rows.append({**record, "content_sha256": content_hash})

    if sample_manifest.is_file() and not refresh:
        stored = load_csv(sample_manifest)
        if len(stored) != sample_size:
            raise ValueError(
                f"Existing sample manifest contains {len(stored)} rows, expected {sample_size}. "
                "Use --refresh-sample to replace it."
            )
        selected = []
        candidate_lookup = {item["record_id"]: item for item in candidate_rows}
        for stored_row in stored:
            record_id = stored_row["record_id"]
            item = candidate_lookup.get(record_id)
            if item is None:
                raise ValueError(
                    f"Stored sample {record_id} no longer passes strict leakage filters."
                )
            if stored_row.get("content_sha256") != item["content_sha256"]:
                raise ValueError(f"Stored sample hash changed: {record_id}")
            selected.append(item)
    else:
        selected = select_stratified_records(
            candidate_rows,
            sample_size=sample_size,
            seed=seed,
        )
        sample_manifest.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "order",
            "record_id",
            "class_code",
            "pair_id",
            "source_id",
            "split",
            "materialized_relpath",
            "rubbing_relpath",
            "glyph_relpath",
            "content_sha256",
        ]
        with sample_manifest.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for order, item in enumerate(selected, start=1):
                path = materialized_path(data_root, item)
                writer.writerow(
                    {
                        "order": order,
                        "record_id": item["record_id"],
                        "class_code": item["class_code"],
                        "pair_id": item["pair_id"],
                        "source_id": item["source_id"],
                        "split": item["split"],
                        "materialized_relpath": path.relative_to(data_root).as_posix(),
                        "rubbing_relpath": item["rubbing_relpath"],
                        "glyph_relpath": item["glyph_relpath"],
                        "content_sha256": item["content_sha256"],
                    }
                )

    selected_ids = {item["record_id"] for item in selected}
    selected_pairs = {item["pair_id"] for item in selected}
    selected_sources = {source_group_key(item) for item in selected}
    train_val = [
        record
        for record in records
        if record["class_code"] in eligible and record["split"] in {"train", "val"}
    ]
    train_val_ids = {item["record_id"] for item in train_val}
    train_val_pairs = {item["pair_id"] for item in train_val}
    train_val_sources = {source_group_key(item) for item in train_val}
    selected_hashes = {item["content_sha256"] for item in selected}
    audit = {
        "dataset_pair_split_conflicts": len(split_conflicts(records, "pair_id")),
        "dataset_source_group_split_conflicts": len(
            split_conflicts(records, "source_group")
        ),
        "strict_candidate_images": len(candidate_rows),
        "source_conflict_images_excluded": (
            sum(
                record["split"] == "test"
                and record["class_code"] in eligible
                and source_splits[source_group_key(record)] != {"test"}
                for record in records
            )
        ),
        "content_overlap_images_excluded": excluded_hash_overlap,
        "selected_record_id_overlap": len(selected_ids & train_val_ids),
        "selected_pair_overlap": len(selected_pairs & train_val_pairs),
        "selected_source_group_overlap": len(selected_sources & train_val_sources),
        "selected_content_hash_overlap": len(selected_hashes & training_hashes),
        "selected_duplicate_content_hashes": len(selected) - len(selected_hashes),
        "all_selected_from_test": all(item["split"] == "test" for item in selected),
    }
    audit["passed"] = (
        audit["selected_record_id_overlap"] == 0
        and audit["selected_pair_overlap"] == 0
        and audit["selected_source_group_overlap"] == 0
        and audit["selected_content_hash_overlap"] == 0
        and audit["selected_duplicate_content_hashes"] == 0
        and audit["all_selected_from_test"]
    )
    return selected, training_hashes, audit


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _model_probabilities(model, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    return outputs[0] if isinstance(outputs, tuple) else outputs.softmax(dim=1)


def _distinct_retrieval_top5(scores: torch.Tensor, index) -> list[dict]:
    order = torch.argsort(scores, descending=True).detach().cpu().tolist()
    result = []
    seen = set()
    for item_index in order:
        class_code = str(index.class_codes[item_index])
        if class_code in seen:
            continue
        seen.add(class_code)
        result.append(
            {
                "candidate_id": str(index.candidate_ids[item_index]),
                "pair_id": str(index.pair_ids[item_index]),
                "class_code": class_code,
                "similarity": float(scores[item_index].item()),
                "glyph_relpath": str(index.glyph_relpaths[item_index]),
                "rubbing_relpath": str(index.rubbing_relpaths[item_index]),
            }
        )
        if len(result) == 5:
            break
    return result


def _run_inference(
    selected: list[dict],
    *,
    data_root: Path,
    model_path: Path,
    index,
    device: torch.device,
    image_size: int,
) -> tuple[list[dict], dict]:
    from ultralytics import YOLO

    init_started = time.perf_counter()
    yolo = YOLO(str(model_path), task="classify")
    classifier = yolo.model.eval().to(device)
    encoder = YoloImageEncoder(
        model_path,
        embedding_dim=index.embeddings.shape[1],
    ).eval().to(device)
    gallery = torch.from_numpy(index.embeddings).to(device)
    transform = build_image_transform(
        image_size,
        training=False,
        normalization="yolo",
    )
    index_to_class = {int(key): str(value) for key, value in yolo.names.items()}
    model_classes = set(index_to_class.values())
    missing = sorted({item["class_code"] for item in selected} - model_classes)
    if missing:
        raise ValueError(f"Sample contains classes missing from model: {missing[:10]}")
    initialization_seconds = time.perf_counter() - init_started

    first_path = materialized_path(data_root, selected[0])
    with Image.open(first_path) as image:
        warmup = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        _model_probabilities(classifier, warmup)
        encoder(warmup) @ gallery.T
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    rows = []
    processing_started = time.perf_counter()
    for record in selected:
        item_started = time.perf_counter()
        path = materialized_path(data_root, record)
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = _model_probabilities(classifier, tensor)
            top5 = probabilities.topk(min(5, probabilities.shape[1]), dim=1)
            embedding = encoder(tensor)
            scores = (embedding @ gallery.T)[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        classification_top5 = [
            index_to_class[int(item)] for item in top5.indices[0].tolist()
        ]
        classification_confidences = [
            float(item) for item in top5.values[0].detach().cpu().tolist()
        ]
        retrieval_top5 = _distinct_retrieval_top5(scores, index)
        exact_index = index.candidate_lookup.get(candidate_id(record["pair_id"]))
        if exact_index is None:
            raise ValueError(f"Exact pair missing from retrieval index: {record['pair_id']}")
        exact_score = scores[exact_index]
        exact_rank = int((scores > exact_score).sum().item()) + 1
        true_indices = index.class_lookup.get(record["class_code"])
        if true_indices is None:
            raise ValueError(f"Class missing from retrieval index: {record['class_code']}")
        true_index_tensor = torch.as_tensor(true_indices, device=device, dtype=torch.long)
        best_true_score = scores[true_index_tensor].max()
        class_rank = int((scores > best_true_score).sum().item()) + 1
        latency_ms = (time.perf_counter() - item_started) * 1000
        rows.append(
            {
                "record_id": record["record_id"],
                "pair_id": record["pair_id"],
                "source_id": record["source_id"],
                "true_class": record["class_code"],
                "materialized_relpath": path.relative_to(data_root).as_posix(),
                "content_sha256": record["content_sha256"],
                "classification_top1": classification_top5[0],
                "classification_confidence": classification_confidences[0],
                "classification_top5": classification_top5,
                "classification_top5_confidences": classification_confidences,
                "retrieval_top5": retrieval_top5,
                "retrieval_exact_pair_rank": exact_rank,
                "retrieval_class_rank": class_rank,
                "latency_ms": latency_ms,
            }
        )
    processing_seconds = time.perf_counter() - processing_started
    return rows, {
        "initialization_seconds": initialization_seconds,
        "processing_seconds": processing_seconds,
    }


def _write_records(rows: list[dict], output_dir: Path, threshold: float) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "records.jsonl"
    csv_path = output_dir / "records.csv"
    started = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as target:
        for row in rows:
            exported = {
                **row,
                "routing_mode": (
                    "classification"
                    if row["classification_confidence"] >= threshold
                    else "retrieval"
                ),
                "review_status": (
                    "auto_accepted"
                    if row["classification_confidence"] >= threshold
                    else "pending"
                ),
                "classification_retrieval_conflict": (
                    row["classification_top1"]
                    != row["retrieval_top5"][0]["class_code"]
                ),
            }
            target.write(json.dumps(exported, ensure_ascii=False) + "\n")

    fields = [
        "record_id",
        "true_class",
        "classification_top1",
        "classification_confidence",
        "classification_top5",
        "retrieval_top1",
        "retrieval_top5",
        "routing_mode",
        "review_status",
        "classification_retrieval_conflict",
        "latency_ms",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            high_confidence = row["classification_confidence"] >= threshold
            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "true_class": row["true_class"],
                    "classification_top1": row["classification_top1"],
                    "classification_confidence": row["classification_confidence"],
                    "classification_top5": json.dumps(row["classification_top5"]),
                    "retrieval_top1": row["retrieval_top5"][0]["class_code"],
                    "retrieval_top5": json.dumps(
                        [item["class_code"] for item in row["retrieval_top5"]]
                    ),
                    "routing_mode": "classification" if high_confidence else "retrieval",
                    "review_status": "auto_accepted" if high_confidence else "pending",
                    "classification_retrieval_conflict": (
                        row["classification_top1"]
                        != row["retrieval_top5"][0]["class_code"]
                    ),
                    "latency_ms": round(row["latency_ms"], 4),
                }
            )
    elapsed = time.perf_counter() - started
    csv_count = len(load_csv(csv_path))
    jsonl_count = sum(1 for line in jsonl_path.open(encoding="utf-8") if line.strip())
    return {
        "csv": str(csv_path.relative_to(PROJECT_ROOT).as_posix()),
        "jsonl": str(jsonl_path.relative_to(PROJECT_ROOT).as_posix()),
        "csv_records": csv_count,
        "jsonl_records": jsonl_count,
        "counts_match": csv_count == jsonl_count == len(rows),
        "export_seconds": elapsed,
    }


def _safe_gallery_source(root: Path, relative_path: str) -> Path | None:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path if path.is_file() else None


def _write_error_gallery(
    rows: list[dict],
    *,
    data_root: Path,
    dataset_root: Path | None,
    output_dir: Path,
    threshold: float,
    limit: int,
) -> dict:
    gallery = output_dir / "error_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    candidates = [
        row
        for row in rows
        if row["classification_top1"] != row["true_class"]
        or row["classification_confidence"] < threshold
        or row["classification_top1"] != row["retrieval_top5"][0]["class_code"]
    ]
    candidates.sort(
        key=lambda row: (
            not (
                row["classification_confidence"] >= threshold
                and row["classification_top1"] != row["true_class"]
            ),
            row["classification_top1"] == row["true_class"],
            row["classification_confidence"],
            row["record_id"],
        )
    )
    selected = candidates[: max(0, limit)]
    cards = []
    for row in selected:
        query_source = data_root / row["materialized_relpath"]
        query_name = f"{row['record_id']}-query{query_source.suffix.lower()}"
        shutil.copy2(query_source, gallery / query_name)
        candidate_name = None
        if dataset_root is not None:
            source = _safe_gallery_source(
                dataset_root,
                row["retrieval_top5"][0]["glyph_relpath"],
            )
            if source is not None:
                candidate_name = f"{row['record_id']}-candidate{source.suffix.lower()}"
                shutil.copy2(source, gallery / candidate_name)
        cards.append((row, query_name, candidate_name))

    body = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>甲骨文错误样例图库</title>",
        "<style>body{font-family:system-ui;margin:24px;background:#f5f1e8;color:#222}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #d8cfbd;border-radius:12px;padding:14px}",
        ".imgs{display:flex;gap:12px}.imgs div{flex:1}.imgs img{width:100%;height:180px;object-fit:contain;background:#eee}",
        ".code{font-family:ui-monospace,monospace}small{color:#665}</style></head><body>",
        f"<h1>错误与待复核样例（{len(cards)}例）</h1>",
        "<p>左侧为测试拓片，右侧为检索Top-1字模。这里只显示六位类别码，不推断现代汉字。</p><div class='grid'>",
    ]
    for row, query_name, candidate_name in cards:
        candidate_html = (
            f"<img src='{html.escape(candidate_name)}' alt='检索候选'>"
            if candidate_name
            else "<p>候选图片不可用</p>"
        )
        body.extend(
            [
                "<section class='card'>",
                f"<h3 class='code'>{html.escape(row['record_id'])}</h3>",
                "<div class='imgs'>",
                f"<div><small>测试拓片</small><img src='{html.escape(query_name)}' alt='测试拓片'></div>",
                f"<div><small>检索Top-1</small>{candidate_html}</div></div>",
                f"<p class='code'>真值 {row['true_class']} · 分类 {row['classification_top1']} "
                f"({row['classification_confidence']:.2%}) · 检索 {row['retrieval_top5'][0]['class_code']}</p>",
                "</section>",
            ]
        )
    body.append("</div></body></html>")
    index_path = gallery / "index.html"
    index_path.write_text("".join(body), encoding="utf-8")
    return {
        "eligible_cases": len(candidates),
        "displayed_cases": len(cards),
        "index": str(index_path.relative_to(PROJECT_ROOT).as_posix()),
    }


def main() -> int:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be greater than zero")
    if not 0 <= args.review_threshold <= 1:
        raise ValueError("review-threshold must be between zero and one")
    if args.assumed_review_seconds < 0:
        raise ValueError("assumed-review-seconds cannot be negative")

    model_path = Path(args.model).expanduser().resolve()
    data_root = Path(args.data).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve()
    sample_manifest = Path(args.sample_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    evidence_prefix = Path(args.evidence_prefix).expanduser().resolve()
    for required in (model_path, manifest_path, index_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    records = load_csv(manifest_path)
    eligible = _eligible_classes(data_root)
    selected, _, leakage_audit = _sample_rows(
        records,
        data_root=data_root,
        eligible=eligible,
        sample_size=args.sample_size,
        seed=args.seed,
        sample_manifest=sample_manifest,
        refresh=args.refresh_sample,
    )
    if not leakage_audit["passed"]:
        raise RuntimeError(f"Selected sample failed leakage audit: {leakage_audit}")

    os.environ["ORACLE_RETRIEVAL_INDEX"] = str(index_path)
    load_retrieval_index.cache_clear()
    index = load_retrieval_index()
    device = _device(args.device)
    rows, timing = _run_inference(
        selected,
        data_root=data_root,
        model_path=model_path,
        index=index,
        device=device,
        image_size=args.image_size,
    )
    thresholds = sorted({float(value.strip()) for value in args.thresholds.split(",")})
    if args.review_threshold not in thresholds:
        thresholds.append(args.review_threshold)
        thresholds.sort()
    threshold_rows = threshold_analysis(
        rows,
        thresholds,
        assumed_review_seconds=args.assumed_review_seconds,
    )
    workflow = next(
        item for item in threshold_rows if item["threshold"] == args.review_threshold
    )
    classification = classification_metrics(rows)
    retrieval = retrieval_metrics(rows)
    low_confidence_rows = [
        row for row in rows if row["classification_confidence"] < args.review_threshold
    ]
    low_confidence_retrieval = retrieval_metrics(low_confidence_rows)
    assisted_hits = sum(
        row["true_class"]
        in set(row["classification_top5"])
        | {item["class_code"] for item in row["retrieval_top5"]}
        for row in rows
    )
    latencies = [row["latency_ms"] for row in rows]
    exports = _write_records(rows, output_dir, args.review_threshold)

    dataset_root = None
    if index.dataset_root_hint:
        hinted = Path(index.dataset_root_hint).expanduser().resolve()
        if hinted.is_dir():
            dataset_root = hinted
    gallery = _write_error_gallery(
        rows,
        data_root=data_root,
        dataset_root=dataset_root,
        output_dir=output_dir,
        threshold=args.review_threshold,
        limit=args.gallery_size,
    )
    sample_class_counts = Counter(item["class_code"] for item in selected)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": "oracle-business-simulation-300",
            "scope": "offline_held_out_business_simulation",
            "human_study": False,
            "notes": [
                "No modern-character dictionary is used or inferred.",
                "Retrieval is candidate evidence; production does not silently replace classification.",
                "Human review time is a configurable planning assumption, not a measured result.",
            ],
        },
        "sample": {
            "images": len(rows),
            "classes": len(sample_class_counts),
            "seed": args.seed,
            "split": "test",
            "stratification": "proportional_with_minimum_one_per_class",
            "minimum_images_per_class": min(sample_class_counts.values()),
            "maximum_images_per_class": max(sample_class_counts.values()),
            "population_images_after_strict_filter": leakage_audit[
                "strict_candidate_images"
            ],
        },
        "provenance": {
            "model_sha256": sha256(model_path),
            "index_sha256": sha256(index_path),
            "source_manifest_sha256": sha256(manifest_path),
            "sample_manifest_sha256": sha256(sample_manifest),
            "model_classes": len(eligible),
            "index_candidates": int(index.embeddings.shape[0]),
            "index_classes": len(index.class_lookup),
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "leakage_audit": leakage_audit,
        "classification": classification,
        "retrieval": {
            **retrieval,
            "low_confidence_subset": low_confidence_retrieval,
        },
        "fusion": {
            "definition": "union of classification Top-5 and distinct-class retrieval Top-5",
            "assisted_candidate_hits_at_5": assisted_hits,
            "assisted_candidate_recall_at_5": assisted_hits / len(rows),
            "experimental_replacement_accuracy": workflow[
                "experimental_replacement_accuracy"
            ],
        },
        "workflow": workflow,
        "threshold_analysis": threshold_rows,
        "performance": {
            **timing,
            "mean_ms_per_image": sum(latencies) / len(latencies),
            "median_ms_per_image": percentile(latencies, 0.5),
            "p95_ms_per_image": percentile(latencies, 0.95),
            "minimum_ms_per_image": min(latencies),
            "maximum_ms_per_image": max(latencies),
            "compute_seconds_per_100": timing["processing_seconds"] * 100 / len(rows),
            "assumed_review_seconds": args.assumed_review_seconds,
            "estimated_reviews_per_100": workflow["estimated_reviews_per_100"],
            "estimated_manual_minutes_per_100": workflow[
                "estimated_manual_minutes_per_100"
            ],
            "estimated_serial_minutes_per_100": (
                timing["processing_seconds"] * 100 / len(rows) / 60
                + workflow["estimated_manual_minutes_per_100"]
            ),
            "export_seconds": exports["export_seconds"],
        },
        "top_confusions": confusion_rows(rows),
        "exports": exports,
        "error_gallery": gallery,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "report.json"
    report_md = output_dir / "report.md"
    markdown = render_markdown(report)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_md.write_text(markdown, encoding="utf-8")

    evidence_prefix.parent.mkdir(parents=True, exist_ok=True)
    evidence_json = evidence_prefix.with_suffix(".json")
    evidence_md = evidence_prefix.with_suffix(".md")
    evidence_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evidence_md.write_text(markdown, encoding="utf-8")

    summary = {
        "sample": len(rows),
        "classes": len(sample_class_counts),
        "leakage_audit": "passed" if leakage_audit["passed"] else "failed",
        "top1": classification["top1_accuracy"],
        "top5": classification["top5_accuracy"],
        "retrieval_recall_at_5": retrieval["class_recall_at_5"],
        "assisted_candidate_recall_at_5": report["fusion"][
            "assisted_candidate_recall_at_5"
        ],
        "auto_pass_coverage": workflow["auto_pass_coverage"],
        "auto_pass_accuracy": workflow["auto_pass_accuracy"],
        "mean_ms_per_image": report["performance"]["mean_ms_per_image"],
        "p95_ms_per_image": report["performance"]["p95_ms_per_image"],
        "report": str(report_md),
        "evidence": str(evidence_md),
        "gallery": str(output_dir / "error_gallery" / "index.html"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
