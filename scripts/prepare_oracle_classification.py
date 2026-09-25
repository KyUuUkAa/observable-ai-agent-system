from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_recognition import pad_to_square


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a frequency-tier Oracle classification dataset."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root that contains 字模数据/ and 拓片数据/.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-images", type=int, default=50)
    parser.add_argument(
        "--no-pad-square",
        action="store_true",
        help="Copy original files instead of applying shape-preserving square padding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional development-only limit after filtering.",
    )
    return parser.parse_args()


def load_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def select_frequency_tier(records: list[dict], minimum_images: int) -> tuple[list[dict], set[str]]:
    if minimum_images <= 0:
        raise ValueError("minimum_images must be greater than zero")
    class_counts = Counter(record["class_code"] for record in records)
    eligible = {
        class_code
        for class_code, count in class_counts.items()
        if count >= minimum_images
    }
    return [record for record in records if record["class_code"] in eligible], eligible


def materialize_record(
    record: dict,
    *,
    dataset_root: Path,
    output_dir: Path,
    pad_square: bool,
) -> Path:
    source = (dataset_root / record["rubbing_relpath"]).resolve()
    try:
        source.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes dataset root: {source}") from exc
    if not source.is_file():
        raise FileNotFoundError(source)

    extension = source.suffix.lower() or ".bmp"
    destination = (
        output_dir
        / record["split"]
        / record["class_code"]
        / f"{record['record_id']}{extension}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    if not pad_square:
        shutil.copy2(source, destination)
        return destination

    with Image.open(source) as image:
        padded = pad_to_square(image)
        padded.save(destination)
    return destination


def prepare_classification_dataset(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_dir: str | Path,
    minimum_images: int,
    pad_square: bool = True,
    limit: int | None = None,
) -> dict:
    records = load_manifest(manifest_path)
    selected, eligible = select_frequency_tier(records, minimum_images)
    if limit is not None:
        selected = selected[:limit]

    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for record in selected:
        materialize_record(
            record,
            dataset_root=root,
            output_dir=output,
            pad_square=pad_square,
        )
        split_counts[record["split"]] += 1
        class_counts[record["class_code"]] += 1

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "dataset_root": str(root),
        "output_dir": str(output),
        "minimum_images": minimum_images,
        "pad_square": pad_square,
        "eligible_classes": len(eligible),
        "materialized_images": len(selected),
        "splits": {
            split: split_counts.get(split, 0)
            for split in ("train", "val", "test")
        },
        "class_counts": dict(sorted(class_counts.items())),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = prepare_classification_dataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        minimum_images=args.minimum_images,
        pad_square=not args.no_pad_square,
        limit=args.limit,
    )
    print(
        "[OK] "
        f"{summary['materialized_images']} images across "
        f"{summary['eligible_classes']} classes"
    )
    print(json.dumps(summary["splits"], ensure_ascii=False))
    print(f"Dataset: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
