from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_dataset.retrieval import build_image_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Oracle classifiers fairly on their shared class set."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeat NAME=CHECKPOINT for every model.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "oracle_dataset" / "model_comparison.json"),
    )
    return parser.parse_args()


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Model must use NAME=CHECKPOINT syntax: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name.strip() or not path.is_file():
            raise ValueError(f"Invalid model entry: {value}")
        parsed.append((name.strip(), path))
    if len(parsed) < 2:
        raise ValueError("At least two models are required for comparison.")
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ComparisonDataset(Dataset):
    def __init__(self, root: Path, class_codes: set[str], image_size: int):
        self.items = []
        for class_code in sorted(class_codes):
            for path in sorted((root / class_code).iterdir()):
                if path.is_file():
                    self.items.append((path, class_code))
        self.transform = build_image_transform(
            image_size,
            training=False,
            normalization="yolo",
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, class_code = self.items[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, class_code


def macro_metrics(true_codes: list[str], predicted_codes: list[str], classes: list[str]):
    precisions = []
    recalls = []
    f1s = []
    for class_code in classes:
        tp = sum(
            true == class_code and predicted == class_code
            for true, predicted in zip(true_codes, predicted_codes)
        )
        fp = sum(
            true != class_code and predicted == class_code
            for true, predicted in zip(true_codes, predicted_codes)
        )
        fn = sum(
            true == class_code and predicted != class_code
            for true, predicted in zip(true_codes, predicted_codes)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Oracle Classifier Model Comparison",
        "",
        f"Shared classes: **{report['shared_classes']}**  ",
        f"Test images: **{report['images']}**",
        "",
        "| Model | Native classes | Top-1 | Top-5 | Macro-F1 | ms/image |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["models"]:
        lines.append(
            "| {name} | {native_classes} | {top1_accuracy:.2%} | "
            "{top5_accuracy:.2%} | {macro_f1:.2%} | {milliseconds_per_image:.3f} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "All models are evaluated on the same images from the intersection of their label sets.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model_entries = parse_models(args.model)
    wrappers = [(name, path, YOLO(str(path), task="classify")) for name, path in model_entries]
    shared_classes = set.intersection(
        *[{str(value) for value in wrapper.names.values()} for _, _, wrapper in wrappers]
    )
    split_root = Path(args.data).expanduser().resolve() / args.split
    available_classes = {path.name for path in split_root.iterdir() if path.is_dir()}
    shared_classes &= available_classes
    if not shared_classes:
        raise ValueError("No shared classes are present in the requested split.")
    dataset = ComparisonDataset(split_root, shared_classes, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    results = []
    for name, path, wrapper in wrappers:
        model = wrapper.model.eval().to(device)
        index_to_class = {int(index): str(code) for index, code in wrapper.names.items()}
        true_codes: list[str] = []
        predicted_codes: list[str] = []
        top5_correct = 0
        started = time.perf_counter()
        with torch.no_grad():
            for images, truths in loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probabilities = (
                    outputs[0]
                    if isinstance(outputs, tuple)
                    else outputs.softmax(dim=1)
                )
                top5 = probabilities.topk(min(5, probabilities.shape[1]), dim=1).indices
                for row_index, truth in enumerate(truths):
                    codes = [index_to_class[int(i)] for i in top5[row_index].tolist()]
                    true_codes.append(str(truth))
                    predicted_codes.append(codes[0])
                    top5_correct += int(str(truth) in codes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        top1 = sum(
            predicted == truth
            for predicted, truth in zip(predicted_codes, true_codes)
        )
        results.append(
            {
                "name": name,
                "checkpoint": str(path),
                "checkpoint_sha256": sha256(path),
                "native_classes": len(wrapper.names),
                "top1_accuracy": top1 / len(true_codes),
                "top5_accuracy": top5_correct / len(true_codes),
                **macro_metrics(true_codes, predicted_codes, sorted(shared_classes)),
                "elapsed_seconds": round(elapsed, 3),
                "milliseconds_per_image": elapsed * 1000 / len(true_codes),
            }
        )
        model.to("cpu")

    report = {
        "split": args.split,
        "shared_classes": len(shared_classes),
        "images": len(dataset),
        "models": results,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Markdown: {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
