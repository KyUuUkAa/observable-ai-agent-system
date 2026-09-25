from __future__ import annotations

import argparse
import csv
import hashlib
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable YOLO feature index for all paired Oracle glyphs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "oracle" / "best_ge50.pt"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "oracle_retrieval" / "index.npz"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_id(pair_id: str) -> str:
    return hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:20]


def safe_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes dataset root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_unique_pairs(manifest: Path, dataset_root: Path) -> list[dict]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as source:
        records = list(csv.DictReader(source))
    pairs: dict[str, dict] = {}
    for record in records:
        item = pairs.setdefault(
            record["pair_id"],
            {
                "candidate_id": candidate_id(record["pair_id"]),
                "pair_id": record["pair_id"],
                "class_code": record["class_code"],
                "glyph_relpath": record["glyph_relpath"],
                "rubbing_relpath": record["rubbing_relpath"],
            },
        )
        if record["rubbing_relpath"] < item["rubbing_relpath"]:
            item["rubbing_relpath"] = record["rubbing_relpath"]
    items = [pairs[key] for key in sorted(pairs)]
    for item in items:
        item["glyph_path"] = safe_path(dataset_root, item["glyph_relpath"])
        safe_path(dataset_root, item["rubbing_relpath"])
    return items


class GlyphIndexDataset(Dataset):
    def __init__(self, items: list[dict], image_size: int):
        self.items = items
        self.transform = build_image_transform(
            image_size,
            training=False,
            normalization="yolo",
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        with Image.open(item["glyph_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    manifest = Path(args.manifest).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not manifest.is_file() or not model_path.is_file():
        raise FileNotFoundError("Manifest and model checkpoint must exist.")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    items = load_unique_pairs(manifest, dataset_root)
    if args.limit is not None:
        items = items[: args.limit]
    dataset = GlyphIndexDataset(items, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    encoder = YoloImageEncoder(
        model_path,
        embedding_dim=args.embedding_dim,
    ).eval().to(device)
    embeddings = np.empty((len(items), args.embedding_dim), dtype=np.float32)
    with torch.no_grad():
        for images, indices in loader:
            encoded = encoder(images.to(device, non_blocking=True)).cpu().numpy()
            embeddings[indices.numpy()] = encoded

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=embeddings.astype(np.float16),
        candidate_ids=np.asarray([item["candidate_id"] for item in items]),
        pair_ids=np.asarray([item["pair_id"] for item in items]),
        class_codes=np.asarray([item["class_code"] for item in items]),
        glyph_relpaths=np.asarray([item["glyph_relpath"] for item in items]),
        rubbing_relpaths=np.asarray([item["rubbing_relpath"] for item in items]),
        dataset_root=np.asarray(str(dataset_root)),
    )
    elapsed = time.perf_counter() - started
    report = {
        "index": str(output),
        "index_sha256": sha256(output),
        "manifest_sha256": sha256(manifest),
        "model_sha256": sha256(model_path),
        "candidates": len(items),
        "classes": len({item["class_code"] for item in items}),
        "embedding_dimension": args.embedding_dim,
        "storage_dtype": "float16",
        "image_size": args.image_size,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
