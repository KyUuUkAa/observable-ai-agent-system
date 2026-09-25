from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Ultralytics Oracle glyph classifier from a prepared tier."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "oracle" / "best_portable.pt"),
        help="Existing 39-class Oracle YOLO checkpoint used for transfer learning.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--project",
        default=str(PROJECT_ROOT / "runs" / "oracle_classification"),
    )
    parser.add_argument("--name", default="tier-ge50")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable AMP only when Torch and TorchVision CUDA builds are compatible.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def inspect_dataset(data_dir: str | Path) -> dict:
    root = Path(data_dir).expanduser().resolve()
    split_stats = {}
    train_classes: set[str] = set()
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing classification split: {split_dir}")
        classes = {path.name for path in split_dir.iterdir() if path.is_dir()}
        images = sum(1 for path in split_dir.rglob("*") if path.is_file())
        split_stats[split] = {"classes": len(classes), "images": images}
        if split == "train":
            train_classes = classes
        elif classes != train_classes:
            missing = sorted(train_classes - classes)[:10]
            raise ValueError(
                f"{split} class folders differ from train; missing examples: {missing}"
            )
    return {"data": str(root), "splits": split_stats}


def main() -> int:
    args = parse_args()
    inspection = inspect_dataset(args.data)
    config = {
        **inspection,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "amp": args.amp,
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("[OK] Classification dataset and training configuration are valid.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required; install the project requirements first"
        ) from exc

    model = YOLO(args.model, task="classify")
    try:
        model.train(
            data=inspection["data"],
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            amp=args.amp,
            plots=False,
            fliplr=0.0,
            flipud=0.0,
            auto_augment=None,
            erasing=0.0,
            degrees=5.0,
            translate=0.05,
            scale=0.10,
        )
    except NotImplementedError as exc:
        # Some Windows environments combine CUDA PyTorch with CPU TorchVision.
        # Ultralytics may then fail only in its post-training NMS warmup even
        # though classification training and checkpoint saving completed.
        best = getattr(getattr(model, "trainer", None), "best", None)
        if "torchvision::nms" not in str(exc) or best is None or not Path(best).is_file():
            raise
        print(
            "[WARN] Training completed and best.pt was saved, but the optional "
            "Ultralytics final warmup hit a Torch/TorchVision NMS mismatch."
        )
        print(f"Checkpoint: {best}")
        print("Run scripts/evaluate_oracle_classifier.py for task-specific evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
