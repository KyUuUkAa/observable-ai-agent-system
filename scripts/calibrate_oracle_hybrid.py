from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_dataset.retrieval import YoloImageEncoder, build_image_transform
from oracle_hybrid import load_retrieval_index


class CalibrationDataset(Dataset):
    def __init__(self, root: Path, class_to_index: dict[str, int], image_size: int):
        self.root = root
        self.items = []
        for class_code, class_index in sorted(class_to_index.items()):
            class_dir = root / class_code
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file():
                    self.items.append((path, class_code, class_index))
        self.transform = build_image_transform(
            image_size,
            training=False,
            normalization="yolo",
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, class_code, class_index = self.items[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, class_index, class_code, path.relative_to(self.root).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate confidence routing and generate Oracle error cases."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--index",
        default=str(PROJECT_ROOT / "data" / "oracle_retrieval" / "index.npz"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hybrid-threshold", type=float, default=0.85)
    parser.add_argument(
        "--thresholds",
        default="0.50,0.60,0.70,0.80,0.85,0.90,0.95",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "oracle_dataset" / "hybrid_calibration.json"),
    )
    parser.add_argument("--errors", default=None)
    return parser.parse_args()


def threshold_metrics(
    confidences: np.ndarray,
    correct: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    rows = []
    total = len(confidences)
    for threshold in thresholds:
        accepted = confidences >= threshold
        accepted_count = int(accepted.sum())
        correct_count = int((correct & accepted).sum())
        incorrect_count = accepted_count - correct_count
        rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "coverage": accepted_count / total if total else 0.0,
                "selective_accuracy": (
                    correct_count / accepted_count if accepted_count else None
                ),
                "false_accept_rate_among_accepted": (
                    incorrect_count / accepted_count if accepted_count else None
                ),
                "incorrect_auto_accept_rate": (
                    incorrect_count / total if total else 0.0
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if not 0 <= args.hybrid_threshold <= 1:
        raise ValueError("hybrid-threshold must be between zero and one")
    import os
    from ultralytics import YOLO

    os.environ["ORACLE_RETRIEVAL_INDEX"] = str(Path(args.index).resolve())
    load_retrieval_index.cache_clear()
    index = load_retrieval_index()
    model_path = Path(args.model).expanduser().resolve()
    data_root = Path(args.data).expanduser().resolve()
    split_root = data_root / args.split
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    yolo = YOLO(str(model_path), task="classify")
    class_to_index = {str(name): int(i) for i, name in yolo.names.items()}
    index_to_class = {value: key for key, value in class_to_index.items()}
    dataset = CalibrationDataset(split_root, class_to_index, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    classifier = yolo.model.eval().to(device)
    encoder = YoloImageEncoder(
        model_path,
        embedding_dim=index.embeddings.shape[1],
    ).eval().to(device)
    gallery_embeddings = torch.from_numpy(index.embeddings).to(device)

    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for images, targets, true_codes, relative_paths in loader:
            images = images.to(device, non_blocking=True)
            outputs = classifier(images)
            probabilities = (
                outputs[0]
                if isinstance(outputs, tuple)
                else outputs.softmax(dim=1)
            )
            top5 = probabilities.topk(min(5, probabilities.shape[1]), dim=1)
            embeddings = encoder(images)
            similarities = embeddings @ gallery_embeddings.T
            retrieval_indices = similarities.argmax(dim=1)
            for offset in range(images.shape[0]):
                predicted_index = int(top5.indices[offset, 0].item())
                predicted_code = index_to_class[predicted_index]
                confidence = float(top5.values[offset, 0].item())
                retrieval_index = int(retrieval_indices[offset].item())
                retrieval_code = str(index.class_codes[retrieval_index])
                true_code = str(true_codes[offset])
                hybrid_code = (
                    predicted_code
                    if confidence >= args.hybrid_threshold
                    else retrieval_code
                )
                rows.append(
                    {
                        "path": str(relative_paths[offset]),
                        "true_class": true_code,
                        "classification_class": predicted_code,
                        "classification_confidence": confidence,
                        "classification_top5": [
                            index_to_class[int(item)]
                            for item in top5.indices[offset].tolist()
                        ],
                        "retrieval_class": retrieval_code,
                        "retrieval_similarity": float(
                            similarities[offset, retrieval_index].item()
                        ),
                        "retrieval_candidate_id": str(
                            index.candidate_ids[retrieval_index]
                        ),
                        "classification_retrieval_agree": (
                            predicted_code == retrieval_code
                        ),
                        "hybrid_class": hybrid_code,
                    }
                )

    elapsed = time.perf_counter() - started
    confidences = np.asarray(
        [row["classification_confidence"] for row in rows], dtype=np.float32
    )
    classification_correct = np.asarray(
        [row["classification_class"] == row["true_class"] for row in rows]
    )
    retrieval_correct = np.asarray(
        [row["retrieval_class"] == row["true_class"] for row in rows]
    )
    hybrid_correct = np.asarray(
        [row["hybrid_class"] == row["true_class"] for row in rows]
    )
    agreement = np.asarray(
        [row["classification_retrieval_agree"] for row in rows]
    )
    thresholds = [float(value) for value in args.thresholds.split(",")]
    report = {
        "model": str(model_path),
        "index": str(Path(args.index).resolve()),
        "split": args.split,
        "images": len(rows),
        "classes": len(class_to_index),
        "hybrid_threshold": args.hybrid_threshold,
        "classification_accuracy": float(classification_correct.mean()),
        "retrieval_class_accuracy": float(retrieval_correct.mean()),
        "classification_retrieval_agreement": float(agreement.mean()),
        "accuracy_when_agree": (
            float(classification_correct[agreement].mean())
            if agreement.any()
            else None
        ),
        "hybrid_accuracy": float(hybrid_correct.mean()),
        "thresholds": threshold_metrics(
            confidences,
            classification_correct,
            thresholds,
        ),
        "elapsed_seconds": round(elapsed, 3),
        "milliseconds_per_image": (
            elapsed * 1000 / len(rows) if rows else 0.0
        ),
    }

    error_rows = []
    for row in rows:
        reasons = []
        if row["classification_class"] != row["true_class"]:
            reasons.append("classification_error")
        if row["classification_confidence"] < args.hybrid_threshold:
            reasons.append("low_confidence")
        if not row["classification_retrieval_agree"]:
            reasons.append("classification_retrieval_conflict")
        if row["hybrid_class"] != row["true_class"]:
            reasons.append("hybrid_error")
        if reasons:
            error_rows.append({**row, "reasons": reasons})

    output = Path(args.output).expanduser().resolve()
    errors = (
        Path(args.errors).expanduser().resolve()
        if args.errors
        else output.with_name("hybrid_error_cases.jsonl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with errors.open("w", encoding="utf-8") as destination:
        for row in error_rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Error cases: {errors} ({len(error_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
