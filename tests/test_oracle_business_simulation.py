from __future__ import annotations

import unittest
from collections import Counter
from unittest.mock import patch

from evaluation.oracle_business_core import (
    allocate_stratified_counts,
    classification_metrics,
    retrieval_metrics,
    select_stratified_records,
    split_conflicts,
    threshold_analysis,
)


def make_row(
    truth: str,
    predicted: str,
    confidence: float,
    retrieval: str,
    *,
    top5: list[str] | None = None,
) -> dict:
    return {
        "true_class": truth,
        "classification_top1": predicted,
        "classification_confidence": confidence,
        "classification_top5": top5 or [predicted],
        "retrieval_top5": [
            {"class_code": retrieval},
            {"class_code": "999999"},
        ],
        "retrieval_exact_pair_rank": 1 if retrieval == truth else 9,
    }


class OracleBusinessSimulationTests(unittest.TestCase):
    def test_stratified_allocation_is_exact_and_covers_classes(self):
        allocation = allocate_stratified_counts(
            {"001000": 10, "002000": 20, "003000": 30},
            12,
        )
        self.assertEqual(sum(allocation.values()), 12)
        self.assertTrue(all(value >= 1 for value in allocation.values()))
        self.assertTrue(all(allocation[key] <= limit for key, limit in {
            "001000": 10,
            "002000": 20,
            "003000": 30,
        }.items()))

    def test_stratified_selection_is_deterministic(self):
        records = [
            {"record_id": f"{class_code}-{index}", "class_code": class_code}
            for class_code in ("001000", "002000", "003000")
            for index in range(10)
        ]
        first = select_stratified_records(records, sample_size=12, seed=42)
        second = select_stratified_records(records, sample_size=12, seed=42)
        self.assertEqual(
            [item["record_id"] for item in first],
            [item["record_id"] for item in second],
        )
        self.assertEqual(len(first), 12)
        self.assertEqual(set(Counter(item["class_code"] for item in first)), {
            "001000",
            "002000",
            "003000",
        })

    def test_pair_and_source_conflicts_are_detected_separately(self):
        records = [
            {
                "class_code": "001000",
                "source_id": "h001",
                "pair_id": "001000:h001a",
                "split": "train",
            },
            {
                "class_code": "001000",
                "source_id": "h001",
                "pair_id": "001000:h001b",
                "split": "test",
            },
        ]
        self.assertEqual(split_conflicts(records, "pair_id"), [])
        self.assertEqual(len(split_conflicts(records, "source_group")), 1)

    def test_metrics_keep_retrieval_and_replacement_semantics_explicit(self):
        rows = [
            make_row("001000", "001000", 0.95, "002000"),
            make_row("002000", "001000", 0.40, "002000", top5=["001000", "002000"]),
        ]
        classification = classification_metrics(rows)
        retrieval = retrieval_metrics(rows)
        thresholds = threshold_analysis(
            rows,
            [0.85],
            assumed_review_seconds=20,
        )
        self.assertEqual(classification["top1_accuracy"], 0.5)
        self.assertEqual(classification["top5_accuracy"], 1.0)
        self.assertEqual(retrieval["class_recall_at_1"], 0.5)
        self.assertEqual(thresholds[0]["auto_pass_coverage"], 0.5)
        self.assertEqual(thresholds[0]["auto_pass_accuracy"], 1.0)
        self.assertEqual(thresholds[0]["experimental_replacement_accuracy"], 1.0)
        self.assertAlmostEqual(thresholds[0]["estimated_manual_minutes_per_100"], 100 / 2 * 20 / 60)

    def test_quality_endpoint_payload_includes_versioned_business_evidence(self):
        from oracle_domain_service import get_oracle_quality_metrics

        with (
            patch("oracle_domain_service.get_oracle_review_summary", return_value={}),
            patch("oracle_domain_service.get_oracle_model_version_summary", return_value=[]),
        ):
            metrics = get_oracle_quality_metrics()
        evidence = metrics["business_simulation"]
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["sample"]["images"], 300)
        self.assertTrue(evidence["leakage_audit"]["passed"])


if __name__ == "__main__":
    unittest.main()
