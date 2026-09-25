from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle_dataset.retrieval import (
    OracleDualEncoder,
    OracleGlyphDataset,
    OraclePairDataset,
    OracleQueryDataset,
    compute_label_retrieval_metrics,
    compute_retrieval_metrics,
    load_manifest_records,
    symmetric_contrastive_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a paired Oracle rubbing/glyph retriever."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "models" / "oracle" / "retrieval_resnet18.pt"),
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--separate-encoders", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=("yolo", "resnet18"),
        default="yolo",
    )
    parser.add_argument(
        "--yolo-checkpoint",
        default=str(PROJECT_ROOT / "models" / "oracle" / "best_ge50.pt"),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the unmodified backbone without contrastive fine-tuning.",
    )
    parser.add_argument("--limit-train-pairs", type=int, default=None)
    parser.add_argument("--limit-eval-records", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def encode_loader(model, loader, *, mode: str, device: torch.device):
    embeddings = []
    pair_ids = []
    class_codes = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            encoded = (
                model.encode_rubbing(images)
                if mode == "rubbing"
                else model.encode_glyph(images)
            )
            embeddings.append(encoded.cpu())
            pair_ids.extend(batch["pair_id"])
            class_codes.extend(batch["class_code"])
    return torch.cat(embeddings, dim=0), pair_ids, class_codes


def evaluate(
    model,
    records,
    *,
    image_size,
    batch_size,
    workers,
    device,
    normalization,
):
    queries = OracleQueryDataset(
        records,
        image_size=image_size,
        normalization=normalization,
    )
    gallery = OracleGlyphDataset(
        records,
        image_size=image_size,
        normalization=normalization,
    )
    query_loader = DataLoader(
        queries,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    gallery_loader = DataLoader(
        gallery,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    query_embeddings, query_pair_ids, query_class_codes = encode_loader(
        model,
        query_loader,
        mode="rubbing",
        device=device,
    )
    gallery_embeddings, gallery_pair_ids, gallery_class_codes = encode_loader(
        model,
        gallery_loader,
        mode="glyph",
        device=device,
    )
    gallery_index = {pair_id: index for index, pair_id in enumerate(gallery_pair_ids)}
    target_indices = torch.tensor(
        [gallery_index[pair_id] for pair_id in query_pair_ids],
        dtype=torch.long,
    )
    return {
        "exact_pair": compute_retrieval_metrics(
            query_embeddings,
            gallery_embeddings,
            target_indices,
        ),
        "class_code": compute_label_retrieval_metrics(
            query_embeddings,
            gallery_embeddings,
            query_class_codes,
            gallery_class_codes,
        ),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    train_records = load_manifest_records(
        args.manifest,
        dataset_root=args.dataset_root,
        split="train",
    )
    val_records = load_manifest_records(
        args.manifest,
        dataset_root=args.dataset_root,
        split="val",
    )
    train_dataset = OraclePairDataset(
        train_records,
        image_size=args.image_size,
        training=True,
        normalization="yolo" if args.backbone == "yolo" else "resnet",
    )
    if args.limit_train_pairs is not None:
        train_dataset.pairs = train_dataset.pairs[: args.limit_train_pairs]
    if args.limit_eval_records is not None:
        val_records = val_records[: args.limit_eval_records]

    config = {
        "train_pairs": len(train_dataset),
        "validation_queries": len(val_records),
        "device": str(device),
        "epochs": args.epochs,
        "batch": args.batch,
        "image_size": args.image_size,
        "embedding_dim": args.embedding_dim,
        "shared_encoder": not args.separate_encoders,
        "pretrained": args.pretrained,
        "backbone": args.backbone,
        "yolo_checkpoint": (
            str(Path(args.yolo_checkpoint).expanduser().resolve())
            if args.backbone == "yolo"
            else None
        ),
    }
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if not train_dataset or not val_records:
        raise ValueError("Both train and validation splits must be non-empty")
    if args.dry_run:
        print("[OK] Retrieval data and training configuration are valid.")
        return 0

    model = OracleDualEncoder(
        embedding_dim=args.embedding_dim,
        shared_encoder=not args.separate_encoders,
        pretrained=args.pretrained,
        backbone=args.backbone,
        yolo_checkpoint=args.yolo_checkpoint,
    ).to(device)
    normalization = "yolo" if args.backbone == "yolo" else "resnet"

    if args.evaluate_only:
        metrics = evaluate(
            model,
            val_records,
            image_size=args.image_size,
            batch_size=args.batch,
            workers=args.workers,
            device=device,
            normalization=normalization,
        )
        report = (
            Path(args.report).resolve()
            if args.report
            else Path(args.output).resolve().with_suffix(".baseline.json")
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {"config": config, "mode": "evaluate_only", "metrics": metrics},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(metrics, ensure_ascii=False))
        print(f"Report: {report}")
        return 0
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= args.batch,
    )

    history = []
    best_recall = -1.0
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in loader:
            rubbing = batch["rubbing"].to(device, non_blocking=True)
            glyph = batch["glyph"].to(device, non_blocking=True)
            rubbing_embeddings, glyph_embeddings, scale = model(rubbing, glyph)
            loss = symmetric_contrastive_loss(
                rubbing_embeddings,
                glyph_embeddings,
                scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        metrics = evaluate(
            model,
            val_records,
            image_size=args.image_size,
            batch_size=args.batch,
            workers=args.workers,
            device=device,
            normalization=normalization,
        )
        epoch_result = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            **metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))
        if metrics["exact_pair"]["recall_at_1"] > best_recall:
            best_recall = metrics["exact_pair"]["recall_at_1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                output,
            )

    report = Path(args.report).resolve() if args.report else output.with_suffix(".json")
    report.write_text(
        json.dumps(
            {"config": config, "best_recall_at_1": best_recall, "history": history},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Checkpoint: {output}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
