import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from oracle_dataset.manifest import (
    build_dataset_manifest,
    parse_pair_identity,
    render_audit_markdown,
)
from scripts.prepare_oracle_classification import prepare_classification_dataset
from oracle_dataset.retrieval import (
    compute_label_retrieval_metrics,
    compute_retrieval_metrics,
)
from oracle_hybrid import (
    get_candidate_image,
    load_retrieval_index,
    select_visual_candidates,
)


def write_image(path: Path, size=(12, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 255).save(path)


class OracleDatasetTests(unittest.TestCase):
    def test_filename_normalization_matches_glyph_and_rubbing(self):
        rubbing = parse_pair_identity(
            Path("038000_b01244正甲_2.bmp"),
            "038000",
            rubbing=True,
        )
        glyph = parse_pair_identity(
            Path("038000b01244 正甲.bmp"),
            "038000",
            rubbing=False,
        )
        self.assertEqual(rubbing["pair_id"], glyph["pair_id"])
        self.assertEqual(rubbing["rubbing_index"], 2)
        self.assertEqual(rubbing["source_id"], "b01244")
        self.assertEqual(rubbing["variant"], "正甲")

    def test_manifest_pairs_images_and_keeps_pair_in_one_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(1, 7):
                source = f"h{index:05d}"
                write_image(root / "字模数据" / "001000" / f"001000{source}.bmp")
                write_image(
                    root / "拓片数据" / "001000" / f"001000_{source}_1.bmp"
                )
                if index == 1:
                    write_image(
                        root / "拓片数据" / "001000" / f"001000_{source}_2.bmp"
                    )

            records, audit = build_dataset_manifest(
                root,
                validate_images=True,
                strict=True,
                min_eval_pairs=5,
            )

        self.assertEqual(len(records), 7)
        self.assertEqual(audit["counts"]["unique_pairs"], 6)
        self.assertEqual(audit["counts"]["classes"], 1)
        self.assertEqual(audit["counts"]["invalid_images"], 0)
        pair_splits = {}
        for record in records:
            pair_splits.setdefault(record["pair_id"], set()).add(record["split"])
        self.assertTrue(all(len(splits) == 1 for splits in pair_splits.values()))
        self.assertIn("Leakage-safe Split", render_audit_markdown(audit))

    def test_prepare_frequency_tier_materializes_padded_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            for index in range(1, 7):
                source = f"h{index:05d}"
                write_image(root / "字模数据" / "001000" / f"001000{source}.bmp")
                write_image(
                    root / "拓片数据" / "001000" / f"001000_{source}_1.bmp",
                    size=(10, 20),
                )
            records, _ = build_dataset_manifest(root, strict=True)
            manifest = Path(temp_dir) / "manifest.csv"
            from oracle_dataset.manifest import write_manifest_csv

            write_manifest_csv(records, manifest)
            output = Path(temp_dir) / "prepared"
            summary = prepare_classification_dataset(
                manifest_path=manifest,
                dataset_root=root,
                output_dir=output,
                minimum_images=5,
            )
            images = [
                path
                for path in output.rglob("*.bmp")
                if path.is_file()
            ]
            with Image.open(images[0]) as prepared:
                prepared_size = prepared.size

        self.assertEqual(summary["eligible_classes"], 1)
        self.assertEqual(summary["materialized_images"], 6)
        self.assertEqual(prepared_size, (20, 20))

    def test_retrieval_metrics_report_recall_and_mrr(self):
        gallery = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        queries = torch.tensor(
            [[0.9, 0.1], [0.1, 0.9]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1], dtype=torch.long)
        metrics = compute_retrieval_metrics(
            queries,
            gallery,
            targets,
            ks=(1, 2),
        )
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["recall_at_2"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

        label_metrics = compute_label_retrieval_metrics(
            queries,
            gallery,
            ["A", "B"],
            ["A", "B", "A"],
            ks=(1, 2),
        )
        self.assertEqual(label_metrics["recall_at_1"], 1.0)
        self.assertEqual(label_metrics["gallery_classes"], 2)

    def test_hybrid_index_routes_and_serves_safe_candidate_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            glyph = root / "字模数据" / "001000" / "glyph.bmp"
            rubbing = root / "拓片数据" / "001000" / "rubbing.bmp"
            write_image(glyph)
            write_image(rubbing)
            index_path = root / "index.npz"
            np.savez_compressed(
                index_path,
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
                candidate_ids=np.asarray(["candidate-a", "candidate-b"]),
                pair_ids=np.asarray(["001000:h1", "002000:h2"]),
                class_codes=np.asarray(["001000", "002000"]),
                glyph_relpaths=np.asarray(
                    ["字模数据/001000/glyph.bmp", "字模数据/001000/glyph.bmp"]
                ),
                rubbing_relpaths=np.asarray(
                    ["拓片数据/001000/rubbing.bmp", "拓片数据/001000/rubbing.bmp"]
                ),
                dataset_root=np.asarray(str(root)),
            )
            with patch.dict(
                "os.environ",
                {"ORACLE_RETRIEVAL_INDEX": str(index_path)},
            ):
                load_retrieval_index.cache_clear()
                routing, candidates = select_visual_candidates(
                    np.asarray([0.99, 0.01], dtype=np.float32),
                    [
                        {"class_code": "001000", "confidence": 0.9},
                        {"class_code": "002000", "confidence": 0.1},
                    ],
                    confidence=0.9,
                    threshold=0.85,
                )
                content, media_type = get_candidate_image("candidate-a", "glyph")
                self.assertEqual(routing["mode"], "classification")
                self.assertEqual(candidates[0]["class_code"], "001000")
                self.assertTrue(content)
                self.assertEqual(media_type, "image/bmp")
                low_routing, low_candidates = select_visual_candidates(
                    np.asarray([0.01, 0.99], dtype=np.float32),
                    [{"class_code": "001000", "confidence": 0.4}],
                    confidence=0.4,
                    threshold=0.85,
                )
                self.assertEqual(low_routing["mode"], "retrieval")
                self.assertEqual(low_candidates[0]["class_code"], "002000")
            load_retrieval_index.cache_clear()


if __name__ == "__main__":
    unittest.main()
