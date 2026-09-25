from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms

from oracle_recognition import pad_to_square


def _safe_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    path = (dataset_root / relative_path).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes dataset root: {path}") from exc
    return path


def load_manifest_records(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    split: str | None = None,
) -> list[dict]:
    root = Path(dataset_root).expanduser().resolve()
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as source:
        records = list(csv.DictReader(source))

    selected = []
    for record in records:
        if split is not None and record["split"] != split:
            continue
        selected.append(
            {
                **record,
                "rubbing_path": _safe_dataset_path(root, record["rubbing_relpath"]),
                "glyph_path": _safe_dataset_path(root, record["glyph_relpath"]),
            }
        )
    return selected


def build_image_transform(
    image_size: int,
    *,
    training: bool,
    normalization: str = "resnet",
):
    operations = [transforms.Lambda(pad_to_square)]
    if training:
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=4,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    fill=128,
                ),
            ]
        )
    operations.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    if normalization == "resnet":
        operations.append(
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
        )
    elif normalization != "yolo":
        raise ValueError(f"Unsupported normalization: {normalization}")
    return transforms.Compose(operations)


class OraclePairDataset(Dataset):
    """Return one randomly chosen rubbing and its exact paired glyph per pair."""

    def __init__(
        self,
        records: list[dict],
        *,
        image_size: int = 224,
        training: bool = True,
        seed: int = 42,
        normalization: str = "resnet",
    ):
        grouped: dict[str, dict] = {}
        for record in records:
            pair_id = record["pair_id"]
            if pair_id not in grouped:
                grouped[pair_id] = {
                    "pair_id": pair_id,
                    "class_code": record["class_code"],
                    "glyph_path": record["glyph_path"],
                    "rubbing_paths": [],
                }
            grouped[pair_id]["rubbing_paths"].append(record["rubbing_path"])
        self.pairs = [grouped[key] for key in sorted(grouped)]
        self.transform = build_image_transform(
            image_size,
            training=training,
            normalization=normalization,
        )
        self.training = training
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _open_image(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __getitem__(self, index: int):
        item = self.pairs[index]
        rubbing_paths = item["rubbing_paths"]
        rubbing_path = (
            self.random.choice(rubbing_paths)
            if self.training
            else rubbing_paths[0]
        )
        rubbing = self.transform(self._open_image(rubbing_path))
        glyph = self.transform(self._open_image(item["glyph_path"]))
        return {
            "rubbing": rubbing,
            "glyph": glyph,
            "pair_id": item["pair_id"],
            "class_code": item["class_code"],
        }


class OracleQueryDataset(Dataset):
    """Return every rubbing image as an evaluation query."""

    def __init__(
        self,
        records: list[dict],
        *,
        image_size: int = 224,
        normalization: str = "resnet",
    ):
        self.records = records
        self.transform = build_image_transform(
            image_size,
            training=False,
            normalization=normalization,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["rubbing_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "pair_id": record["pair_id"],
            "class_code": record["class_code"],
        }


class OracleGlyphDataset(Dataset):
    """Return one glyph gallery item per unique pair."""

    def __init__(
        self,
        records: list[dict],
        *,
        image_size: int = 224,
        normalization: str = "resnet",
    ):
        unique = {}
        for record in records:
            unique.setdefault(
                record["pair_id"],
                {
                    "pair_id": record["pair_id"],
                    "class_code": record["class_code"],
                    "glyph_path": record["glyph_path"],
                },
            )
        self.records = [unique[key] for key in sorted(unique)]
        self.transform = build_image_transform(
            image_size,
            training=False,
            normalization=normalization,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["glyph_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "pair_id": record["pair_id"],
            "class_code": record["class_code"],
        }


class ResNetImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int, *, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return F.normalize(self.projection(features), dim=-1)


class YoloImageEncoder(nn.Module):
    """Reuse the domain-trained YOLO11 classification backbone as an embedder."""

    def __init__(self, checkpoint: str | Path, embedding_dim: int = 256):
        super().__init__()
        from ultralytics import YOLO

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        yolo = YOLO(str(checkpoint_path), task="classify")
        modules = list(yolo.model.model.children())
        if len(modules) < 2:
            raise ValueError("YOLO classification checkpoint has no extractable backbone")
        self.backbone = nn.Sequential(*modules[:-1])
        feature_dim = int(modules[-1].conv.conv.in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = (
            nn.Identity()
            if embedding_dim == feature_dim
            else nn.Linear(feature_dim, embedding_dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = self.pool(features).flatten(1)
        return F.normalize(self.projection(features), dim=-1)


class OracleDualEncoder(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        shared_encoder: bool = True,
        pretrained: bool = False,
        backbone: str = "resnet18",
        yolo_checkpoint: str | Path | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.shared_encoder = shared_encoder
        self.backbone_name = backbone
        if backbone == "yolo":
            if yolo_checkpoint is None:
                raise ValueError("yolo_checkpoint is required for the YOLO backbone")
            self.rubbing_encoder = YoloImageEncoder(
                yolo_checkpoint,
                embedding_dim=embedding_dim,
            )
            encoder_factory = lambda: YoloImageEncoder(
                yolo_checkpoint,
                embedding_dim=embedding_dim,
            )
        elif backbone == "resnet18":
            self.rubbing_encoder = ResNetImageEncoder(
                embedding_dim,
                pretrained=pretrained,
            )
            encoder_factory = lambda: ResNetImageEncoder(
                embedding_dim,
                pretrained=pretrained,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.glyph_encoder = (
            self.rubbing_encoder
            if shared_encoder
            else encoder_factory()
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_rubbing(self, images: torch.Tensor) -> torch.Tensor:
        return self.rubbing_encoder(images)

    def encode_glyph(self, images: torch.Tensor) -> torch.Tensor:
        return self.glyph_encoder(images)

    def forward(
        self,
        rubbing_images: torch.Tensor,
        glyph_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rubbing = self.encode_rubbing(rubbing_images)
        glyph = self.encode_glyph(glyph_images)
        scale = self.logit_scale.exp().clamp(max=100)
        return rubbing, glyph, scale


def symmetric_contrastive_loss(
    rubbing_embeddings: torch.Tensor,
    glyph_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits = logit_scale * rubbing_embeddings @ glyph_embeddings.T
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.T, targets)
    ) / 2


def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 512,
) -> dict:
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2 tensors")
    if query_embeddings.shape[0] != target_indices.shape[0]:
        raise ValueError("One target index is required for each query")
    max_k = min(max(ks), gallery_embeddings.shape[0])
    hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0

    gallery = gallery_embeddings.T
    for start in range(0, query_embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, query_embeddings.shape[0])
        queries = query_embeddings[start:stop]
        targets = target_indices[start:stop]
        similarities = queries @ gallery
        top_indices = similarities.topk(max_k, dim=1).indices
        for k in ks:
            effective_k = min(k, top_indices.shape[1])
            hits[k] += int(
                (top_indices[:, :effective_k] == targets[:, None])
                .any(dim=1)
                .sum()
                .item()
            )
        true_scores = similarities.gather(1, targets[:, None])
        ranks = 1 + (similarities > true_scores).sum(dim=1)
        reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())

    total = query_embeddings.shape[0]
    metrics = {
        f"recall_at_{k}": hits[k] / total if total else 0.0
        for k in ks
    }
    metrics["mrr"] = reciprocal_rank_sum / total if total else 0.0
    metrics["queries"] = total
    metrics["gallery_items"] = gallery_embeddings.shape[0]
    return metrics


def compute_label_retrieval_metrics(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: list[str],
    gallery_labels: list[str],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 512,
) -> dict:
    """Measure whether retrieval returns any gallery glyph with the same class code."""

    if len(query_labels) != query_embeddings.shape[0]:
        raise ValueError("One query label is required for each query")
    if len(gallery_labels) != gallery_embeddings.shape[0]:
        raise ValueError("One gallery label is required for each gallery item")
    label_to_index: dict[str, int] = {}
    gallery_label_ids = []
    for label in gallery_labels:
        label_to_index.setdefault(label, len(label_to_index))
        gallery_label_ids.append(label_to_index[label])
    missing = sorted(set(query_labels) - set(label_to_index))
    if missing:
        raise ValueError(f"Query labels missing from gallery: {missing[:10]}")

    gallery_ids = torch.tensor(gallery_label_ids, dtype=torch.long)
    query_ids = torch.tensor(
        [label_to_index[label] for label in query_labels],
        dtype=torch.long,
    )
    max_k = min(max(ks), gallery_embeddings.shape[0])
    hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    gallery = gallery_embeddings.T
    for start in range(0, query_embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, query_embeddings.shape[0])
        similarities = query_embeddings[start:stop] @ gallery
        labels = query_ids[start:stop]
        top_indices = similarities.topk(max_k, dim=1).indices
        top_labels = gallery_ids[top_indices]
        for k in ks:
            effective_k = min(k, top_indices.shape[1])
            hits[k] += int(
                (top_labels[:, :effective_k] == labels[:, None])
                .any(dim=1)
                .sum()
                .item()
            )

        positive_mask = gallery_ids[None, :] == labels[:, None]
        best_positive = similarities.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
        ranks = 1 + (similarities > best_positive[:, None]).sum(dim=1)
        reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())

    total = query_embeddings.shape[0]
    metrics = {
        f"recall_at_{k}": hits[k] / total if total else 0.0
        for k in ks
    }
    metrics["mrr"] = reciprocal_rank_sum / total if total else 0.0
    metrics["queries"] = total
    metrics["gallery_items"] = gallery_embeddings.shape[0]
    metrics["gallery_classes"] = len(label_to_index)
    return metrics


def group_records_by_class(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["class_code"]].append(record)
    return grouped
