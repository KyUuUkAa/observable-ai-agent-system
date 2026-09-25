import unittest

from deterministic_routing import argument_hints_for_input, required_tool_for_input


class DeterministicRoutingTests(unittest.TestCase):
    def test_explicit_business_intents_are_constrained(self):
        cases = {
            "我的YOLO项目做了什么？": "search_resume",
            "找出最近30天置信度低于0.85的记录": "query_review_queue",
            "查看类别001000的全部识别结果": "query_review_queue",
            "导出本周待复核记录为CSV": "export_oracle_records",
            "确认记录11111111-1111-4111-8111-111111111111": "update_review_result",
            "当前109类模型Top-1是多少？": "get_oracle_quality_metrics",
            "计算 12 * 7": "calculator",
            "[ORACLE_IMAGE_ATTACHMENT]\nimage_id=abc": "recognize_oracle_image",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(required_tool_for_input(query), expected)

    def test_general_questions_are_not_forced(self):
        for query in (
            "什么是置信度校准？",
            "Top-1 和 Top-5 有什么区别？",
            "为什么人工复核很重要？",
        ):
            with self.subTest(query=query):
                self.assertIsNone(required_tool_for_input(query))

    def test_filter_argument_hints_preserve_direction_and_status(self):
        self.assertEqual(
            argument_hints_for_input("把置信度低于0.85的待复核资料导出CSV"),
            [
                "max_confidence=0.85",
                "do not set min_confidence from this upper bound",
                'review_status="pending"',
                'output_format="csv"',
            ],
        )
        self.assertIn(
            "min_confidence=0.8",
            argument_hints_for_input("导出类别038000置信度0.8以上的结果"),
        )


if __name__ == "__main__":
    unittest.main()
