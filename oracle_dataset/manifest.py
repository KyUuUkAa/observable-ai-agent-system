from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


CLASS_CODE_PATTERN = re.compile(r"^\d{6}$")
SOURCE_ID_PATTERN = re.compile(r"^(?P<source>[A-Za-z]\d+)(?P<variant>.*)$")
RUBBING_INDEX_PATTERN = re.compile(r"_(?P<index>\d+)$")
DEFAULT_THRESHOLDS = (2, 5, 10, 20, 50, 100)
SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class OracleDatasetError(RuntimeError):
    """Raised when the paired Oracle dataset is malformed."""


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def parse_pair_identity(path: Path, class_code: str, *, rubbing: bool) -> dict:
    """Normalize the paired source identity encoded in one filename."""

    stem = path.stem
    if not stem.startswith(class_code):
        raise OracleDatasetError(
            f"文件名必须以所属六位类别码开头：{path.name}（目录 {class_code}）"
        )

    remainder = stem[len(class_code) :]
    rubbing_index = None
    if rubbing:
        match = RUBBING_INDEX_PATTERN.search(remainder)
        if match:
            rubbing_index = int(match.group("index"))
            remainder = remainder[: match.start()]

    normalized = re.sub(r"[\s_]+", "", remainder)
    if not normalized:
        raise OracleDatasetError(f"无法从文件名解析来源编号：{path.name}")

    source_match = SOURCE_ID_PATTERN.match(normalized)
    source_id = source_match.group("source") if source_match else normalized
    variant = source_match.group("variant") if source_match else ""
    pair_id = f"{class_code}:{normalized.lower()}"
    return {
        "pair_id": pair_id,
        "source_id": source_id.lower(),
        "variant": variant,
        "rubbing_index": rubbing_index,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path, *, include_hash: bool) -> dict:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "mode": None,
            "sha256": _file_sha256(path) if include_hash else None,
            "error": str(exc),
        }

    return {
        "valid": True,
        "width": width,
        "height": height,
        "mode": image.mode,
        "sha256": _file_sha256(path) if include_hash else None,
        "error": None,
    }


def _stable_pair_splits(
    pairs_by_class: dict[str, set[str]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    min_eval_pairs: int,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for class_code, pair_ids in sorted(pairs_by_class.items()):
        ordered = sorted(
            pair_ids,
            key=lambda pair_id: hashlib.sha256(
                f"{seed}:{pair_id}".encode("utf-8")
            ).hexdigest(),
        )
        total = len(ordered)
        if total < min_eval_pairs:
            for pair_id in ordered:
                assignments[pair_id] = "train"
            continue

        val_count = max(1, math.floor(total * val_ratio))
        test_count = max(1, math.floor(total * test_ratio))
        while val_count + test_count >= total:
            if test_count >= val_count and test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break

        for index, pair_id in enumerate(ordered):
            if index < test_count:
                assignments[pair_id] = "test"
            elif index < test_count + val_count:
                assignments[pair_id] = "val"
            else:
                assignments[pair_id] = "train"
    return assignments


def build_dataset_manifest(
    dataset_root: str | Path,
    *,
    rubbing_dir_name: str = "拓片数据",
    glyph_dir_name: str = "字模数据",
    validate_images: bool = False,
    include_hash: bool = False,
    seed: int = 42,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    min_eval_pairs: int = 5,
    strict: bool = True,
) -> tuple[list[dict], dict]:
    """Build one manifest row per rubbing image and its paired glyph image."""

    root = Path(dataset_root).expanduser().resolve()
    rubbing_root = root / rubbing_dir_name
    glyph_root = root / glyph_dir_name
    if not rubbing_root.is_dir() or not glyph_root.is_dir():
        raise OracleDatasetError(
            f"数据集根目录必须同时包含 {rubbing_dir_name}/ 和 {glyph_dir_name}/：{root}"
        )

    glyph_files = _image_files(glyph_root)
    rubbing_files = _image_files(rubbing_root)
    glyph_by_pair: dict[str, Path] = {}
    duplicate_glyph_pairs: list[str] = []
    malformed_files: list[dict] = []

    for path in glyph_files:
        class_code = path.parent.name
        if not CLASS_CODE_PATTERN.fullmatch(class_code):
            malformed_files.append({"path": str(path), "error": "invalid_class_code"})
            continue
        try:
            identity = parse_pair_identity(path, class_code, rubbing=False)
        except OracleDatasetError as exc:
            malformed_files.append({"path": str(path), "error": str(exc)})
            continue
        pair_id = identity["pair_id"]
        if pair_id in glyph_by_pair:
            duplicate_glyph_pairs.append(pair_id)
        else:
            glyph_by_pair[pair_id] = path

    parsed_rubbings: list[tuple[Path, str, dict, Path]] = []
    unmatched_rubbings: list[str] = []
    pairs_by_class: dict[str, set[str]] = defaultdict(set)
    for path in rubbing_files:
        class_code = path.parent.name
        if not CLASS_CODE_PATTERN.fullmatch(class_code):
            malformed_files.append({"path": str(path), "error": "invalid_class_code"})
            continue
        try:
            identity = parse_pair_identity(path, class_code, rubbing=True)
        except OracleDatasetError as exc:
            malformed_files.append({"path": str(path), "error": str(exc)})
            continue
        glyph_path = glyph_by_pair.get(identity["pair_id"])
        if glyph_path is None:
            unmatched_rubbings.append(str(path))
            continue
        parsed_rubbings.append((path, class_code, identity, glyph_path))
        pairs_by_class[class_code].add(identity["pair_id"])

    rubbing_pairs = {item[2]["pair_id"] for item in parsed_rubbings}
    unmatched_glyphs = [
        str(path)
        for pair_id, path in glyph_by_pair.items()
        if pair_id not in rubbing_pairs
    ]

    if strict and (
        malformed_files
        or duplicate_glyph_pairs
        or unmatched_rubbings
        or unmatched_glyphs
    ):
        raise OracleDatasetError(
            "配对审计失败："
            f"malformed={len(malformed_files)}, "
            f"duplicate_glyph_pairs={len(duplicate_glyph_pairs)}, "
            f"unmatched_rubbings={len(unmatched_rubbings)}, "
            f"unmatched_glyphs={len(unmatched_glyphs)}"
        )

    split_by_pair = _stable_pair_splits(
        pairs_by_class,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_eval_pairs=min_eval_pairs,
    )
    records: list[dict] = []
    invalid_images: list[dict] = []
    glyph_metadata_cache: dict[Path, dict] = {}

    for rubbing_path, class_code, identity, glyph_path in parsed_rubbings:
        rubbing_meta = (
            _inspect_image(rubbing_path, include_hash=include_hash)
            if validate_images
            else {}
        )
        if validate_images and glyph_path not in glyph_metadata_cache:
            glyph_metadata_cache[glyph_path] = _inspect_image(
                glyph_path,
                include_hash=include_hash,
            )
        glyph_meta = glyph_metadata_cache.get(glyph_path, {})
        if validate_images and not rubbing_meta.get("valid", False):
            invalid_images.append(
                {"kind": "rubbing", "path": str(rubbing_path), **rubbing_meta}
            )
        if validate_images and not glyph_meta.get("valid", False):
            invalid_images.append(
                {"kind": "glyph", "path": str(glyph_path), **glyph_meta}
            )

        record_id = hashlib.sha256(
            f"{identity['pair_id']}|{rubbing_path.name}".encode("utf-8")
        ).hexdigest()[:20]
        records.append(
            {
                "record_id": record_id,
                "class_code": class_code,
                "pair_id": identity["pair_id"],
                "source_id": identity["source_id"],
                "variant": identity["variant"],
                "rubbing_index": identity["rubbing_index"],
                "split": split_by_pair[identity["pair_id"]],
                "rubbing_relpath": rubbing_path.relative_to(root).as_posix(),
                "glyph_relpath": glyph_path.relative_to(root).as_posix(),
                "rubbing_width": rubbing_meta.get("width"),
                "rubbing_height": rubbing_meta.get("height"),
                "rubbing_mode": rubbing_meta.get("mode"),
                "rubbing_sha256": rubbing_meta.get("sha256"),
                "glyph_width": glyph_meta.get("width"),
                "glyph_height": glyph_meta.get("height"),
                "glyph_mode": glyph_meta.get("mode"),
                "glyph_sha256": glyph_meta.get("sha256"),
            }
        )

    class_counts = Counter(record["class_code"] for record in records)
    pair_counts = Counter(record["pair_id"] for record in records)
    split_images = Counter(record["split"] for record in records)
    split_pairs = Counter(split_by_pair.values())
    counts = sorted(class_counts.values())
    total_records = len(records)
    median = (
        counts[len(counts) // 2]
        if len(counts) % 2
        else (counts[len(counts) // 2 - 1] + counts[len(counts) // 2]) / 2
    ) if counts else 0

    thresholds = []
    for threshold in DEFAULT_THRESHOLDS:
        selected = {
            class_code: count
            for class_code, count in class_counts.items()
            if count >= threshold
        }
        images = sum(selected.values())
        thresholds.append(
            {
                "minimum_images": threshold,
                "classes": len(selected),
                "rubbing_images": images,
                "coverage": round(images / total_records, 6) if total_records else 0,
            }
        )

    audit = {
        "dataset_root": str(root),
        "counts": {
            "rubbing_files_discovered": len(rubbing_files),
            "glyph_files_discovered": len(glyph_files),
            "manifest_records": total_records,
            "classes": len(class_counts),
            "unique_pairs": len(pair_counts),
            "matched_rubbing_images": len(parsed_rubbings),
            "unmatched_rubbing_images": len(unmatched_rubbings),
            "unmatched_glyph_images": len(unmatched_glyphs),
            "duplicate_glyph_pairs": len(duplicate_glyph_pairs),
            "malformed_files": len(malformed_files),
            "invalid_images": len(invalid_images),
        },
        "class_distribution": {
            "minimum": counts[0] if counts else 0,
            "median": median,
            "mean": round(total_records / len(counts), 4) if counts else 0,
            "maximum": counts[-1] if counts else 0,
            "single_image_classes": sum(count == 1 for count in counts),
            "classes_below_3": sum(count < 3 for count in counts),
            "classes_below_5": sum(count < 5 for count in counts),
        },
        "pair_multiplicity": {
            "minimum": min(pair_counts.values()) if pair_counts else 0,
            "maximum": max(pair_counts.values()) if pair_counts else 0,
            "mean": round(total_records / len(pair_counts), 4) if pair_counts else 0,
        },
        "splits": {
            split: {
                "rubbing_images": split_images.get(split, 0),
                "unique_pairs": split_pairs.get(split, 0),
            }
            for split in ("train", "val", "test")
        },
        "threshold_tiers": thresholds,
        "top_classes": [
            {"class_code": class_code, "rubbing_images": count}
            for class_code, count in class_counts.most_common(20)
        ],
        "issues": {
            "malformed_files": malformed_files[:100],
            "duplicate_glyph_pairs": duplicate_glyph_pairs[:100],
            "unmatched_rubbings": unmatched_rubbings[:100],
            "unmatched_glyphs": unmatched_glyphs[:100],
            "invalid_images": invalid_images[:100],
        },
        "settings": {
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "min_eval_pairs": min_eval_pairs,
            "validated_images": validate_images,
            "included_sha256": include_hash,
        },
    }
    return records, audit


def write_manifest_csv(records: Iterable[dict], path: str | Path) -> Path:
    rows = list(records)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "record_id",
        "class_code",
        "pair_id",
        "source_id",
        "variant",
        "rubbing_index",
        "split",
        "rubbing_relpath",
        "glyph_relpath",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_manifest_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def render_audit_markdown(audit: dict) -> str:
    counts = audit["counts"]
    distribution = audit["class_distribution"]
    lines = [
        "# Oracle Glyph/Rubbing Dataset Audit",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Rubbing images | {counts['rubbing_files_discovered']:,} |",
        f"| Glyph images | {counts['glyph_files_discovered']:,} |",
        f"| Matched manifest records | {counts['manifest_records']:,} |",
        f"| Six-digit classes | {counts['classes']:,} |",
        f"| Unique glyph/rubbing pairs | {counts['unique_pairs']:,} |",
        f"| Unmatched rubbings | {counts['unmatched_rubbing_images']:,} |",
        f"| Unmatched glyphs | {counts['unmatched_glyph_images']:,} |",
        f"| Invalid images | {counts['invalid_images']:,} |",
        "",
        "## Long-tail Distribution",
        "",
        f"- Minimum / median / mean / maximum images per class: "
        f"{distribution['minimum']} / {distribution['median']} / "
        f"{distribution['mean']} / {distribution['maximum']}",
        f"- Classes with one rubbing image: {distribution['single_image_classes']:,}",
        f"- Classes with fewer than three images: {distribution['classes_below_3']:,}",
        f"- Classes with fewer than five images: {distribution['classes_below_5']:,}",
        "",
        "## Frequency Tiers",
        "",
        "| Minimum images/class | Classes | Rubbing images | Coverage |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for tier in audit["threshold_tiers"]:
        lines.append(
            f"| {tier['minimum_images']} | {tier['classes']:,} | "
            f"{tier['rubbing_images']:,} | {tier['coverage']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Leakage-safe Split",
            "",
            "All rubbing variants that share one `pair_id` are assigned to the same split.",
            "",
            "| Split | Unique pairs | Rubbing images |",
            "| --- | ---: | ---: |",
        ]
    )
    for split in ("train", "val", "test"):
        stats = audit["splits"][split]
        lines.append(
            f"| {split} | {stats['unique_pairs']:,} | {stats['rubbing_images']:,} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Training Strategy",
            "",
            "1. Keep the 39-class (>=100 images) model as the high-frequency baseline.",
            "2. Expand classification to the >=50 tier before attempting the >=20 tier.",
            "3. Use paired metric learning for the full long tail instead of a 4,038-way softmax.",
            "4. Report Recall@K/MRR for glyph retrieval and Macro-F1 by frequency bucket.",
            "",
        ]
    )
    return "\n".join(lines)
