import unittest

from oracle_agent_workflow import (
    OracleWorkflowStage,
    OracleWorkflowTransitionError,
    clear_workflow,
    complete_retrieval,
    register_classification,
    require_retrieval,
)


def classification(confidence: float) -> dict:
    return {
        "prediction": {
            "class_id": 0,
            "class_code": "001000",
            "confidence": confidence,
        }
    }


class OracleAgentWorkflowTests(unittest.TestCase):
    def tearDown(self):
        clear_workflow("low")
        clear_workflow("high")

    def test_low_confidence_requires_one_retrieval(self):
        workflow = register_classification("low", classification(0.60), threshold=0.85)
        self.assertEqual(workflow.stage, OracleWorkflowStage.CLASSIFIED_LOW)

        require_retrieval("low")
        completed = complete_retrieval("low", [{"class_code": "002000"}], forced=False)
        self.assertEqual(completed.stage, OracleWorkflowStage.CANDIDATES_RETRIEVED)
        with self.assertRaises(OracleWorkflowTransitionError):
            require_retrieval("low")

    def test_high_confidence_requires_explicit_force(self):
        workflow = register_classification("high", classification(0.95), threshold=0.85)
        self.assertEqual(workflow.stage, OracleWorkflowStage.CLASSIFIED_HIGH)

        with self.assertRaises(OracleWorkflowTransitionError):
            require_retrieval("high")
        self.assertEqual(require_retrieval("high", force=True).image_id, "high")

    def test_retrieval_cannot_run_before_classification(self):
        with self.assertRaises(OracleWorkflowTransitionError):
            require_retrieval("low")


if __name__ == "__main__":
    unittest.main()
