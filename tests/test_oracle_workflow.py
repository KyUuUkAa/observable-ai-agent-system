import json
import os
import unittest
from unittest.mock import patch

from oracle_workflow import (
    get_default_review_threshold,
    oracle_records_to_csv,
    oracle_records_to_json,
    validate_review_threshold,
)


class OracleWorkflowTests(unittest.TestCase):
    def test_review_threshold_validation(self):
        self.assertEqual(validate_review_threshold(0.85), 0.85)
        with self.assertRaises(ValueError):
            validate_review_threshold(1.1)

    def test_threshold_can_be_configured_from_environment(self):
        with patch.dict(os.environ, {"ORACLE_REVIEW_THRESHOLD": "0.72"}):
            self.assertEqual(get_default_review_threshold(), 0.72)

    def test_export_formats_include_review_fields(self):
        record = {
            "id": "record-1",
            "original_filename": "glyph.png",
            "image_sha256": "abc",
            "top1_class_code": "001000",
            "top1_confidence": 0.9,
            "top5": [{"class_code": "001000", "confidence": 0.9}],
            "model_version": "sha",
            "review_threshold": 0.85,
            "review_status": "accepted",
            "review_notes": "人工确认",
            "source": "single",
            "conversation_id": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "reviewed_at": "2026-01-01T00:01:00+00:00",
        }

        csv_text = oracle_records_to_csv([record])
        json_text = oracle_records_to_json([record])

        self.assertIn("review_status", csv_text)
        self.assertIn("accepted", csv_text)
        self.assertEqual(json.loads(json_text)[0]["top1_class_code"], "001000")


if __name__ == "__main__":
    unittest.main()
