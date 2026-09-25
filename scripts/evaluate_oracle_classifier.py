from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ClassificationFolderDataset(Dataset):
    def __init__(self, root: Path, class_to_index: dict[str, int], image_size: int):
        self.items = []
        for class_code, class_index in sorted(class_to_index.items()):
            class_dir = root / class_code
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file():
                    self.items.append((path, class_index))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, target = self.items[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, target


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_confusion(confusion: torch.Tensor) -> dict:
    confusion = confusion.double()
    true_positive = confusion.diag()
    predicted = confusion.sum(dim=0)
    actual = confusion.sum(dim=1)
    precision = torch.where(predicted > 0, true_positive / predicted, 0)
    recall = torch.where(actual > 0, true_positive / actual, 0)
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    return {
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Oracle YOLO classifier directly.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model_path = Path(args.model).expanduser().resolve()
    data_root = Path(args.data).expanduser().resolve()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    yolo = YOLO(str(model_path), task="classify")
    class_to_index = {str(name): int(index) for index, name in yolo.names.items()}
    dataset = ClassificationFolderDataset(
        data_root / args.split,
        class_to_index,
        args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = yolo.model.eval().to(device)
    class_count = len(class_to_index)
    confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            probabilities = outputs[0] if isinstance(outputs, tuple) else outputs.softmax(dim=1)
            top5 = probabilities.topk(min(5, class_count), dim=1).indices
            predictions = top5[:, 0]
            top1_correct += int((predictions == targets).sum().item())
            top5_correct += int((top5 == targets[:, None]).any(dim=1).sum().item())
            flat = (targets * class_count + predictions).detach().cpu()
            confusion += torch.bincount(
                flat,
                minlength=class_count * class_count,
            ).reshape(class_count, class_count)
            total += targets.shape[0]

    report = {
        "model": str(model_path),
        "checkpoint_sha256": checkpoint_sha256(model_path),
        "data": str(data_root),
        "split": args.split,
        "images": total,
        "classes": class_count,
        "top1_accuracy": top1_correct / total if total else 0,
        "top5_accuracy": top5_correct / total if total else 0,
        **metrics_from_confusion(confusion),
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path.with_name(f"{model_path.stem}-{args.split}-metrics.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
