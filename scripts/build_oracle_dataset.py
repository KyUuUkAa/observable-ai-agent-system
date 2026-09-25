from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_dataset import (
    OracleDatasetError,
    build_dataset_manifest,
    render_audit_markdown,
    write_manifest_csv,
    write_manifest_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paired Oracle glyph/rubbing data and generate a leakage-safe manifest."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Directory containing 字模数据/ and 拓片数据/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "oracle_dataset"),
    )
    parser.add_argument("--validate-images", action="store_true")
    parser.add_argument("--include-hash", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--min-eval-pairs", type=int, default=5)
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Write matched records even if unmatched or malformed files exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        records, audit = build_dataset_manifest(
            args.dataset_root,
            validate_images=args.validate_images,
            include_hash=args.include_hash,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_eval_pairs=args.min_eval_pairs,
            strict=not args.allow_unmatched,
        )
    except OracleDatasetError as exc:
        print(f"[FAIL] {exc}")
        return 1

    csv_path = write_manifest_csv(records, output_dir / "manifest.csv")
    jsonl_path = write_manifest_jsonl(records, output_dir / "manifest.jsonl")
    audit_json = output_dir / "audit.json"
    audit_markdown = output_dir / "audit.md"
    audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_markdown.write_text(
        render_audit_markdown(audit),
        encoding="utf-8",
    )

    counts = audit["counts"]
    print(
        "[OK] "
        f"{counts['manifest_records']} rubbings, "
        f"{counts['unique_pairs']} pairs, "
        f"{counts['classes']} classes"
    )
    print(f"Manifest CSV: {csv_path}")
    print(f"Manifest JSONL: {jsonl_path}")
    print(f"Audit JSON: {audit_json}")
    print(f"Audit Markdown: {audit_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
