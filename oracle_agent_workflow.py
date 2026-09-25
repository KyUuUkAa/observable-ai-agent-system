from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock


WORKFLOW_TTL_SECONDS = 15 * 60
WORKFLOW_MAX_ITEMS = 32


class OracleWorkflowStage(str, Enum):
    UPLOADED = "uploaded"
    CLASSIFIED_HIGH = "classified_high_confidence"
    CLASSIFIED_LOW = "classified_low_confidence"
    CANDIDATES_RETRIEVED = "candidates_retrieved"


class OracleWorkflowTransitionError(RuntimeError):
    """Raised when an Agent tool attempts an invalid workflow transition."""


@dataclass
class OracleAgentWorkflow:
    image_id: str
    stage: OracleWorkflowStage
    classification: dict
    expires_at: float
    candidates: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)

    def safe_summary(self) -> dict:
        prediction = self.classification["prediction"]
        return {
            "image_id": self.image_id,
            "stage": self.stage.value,
            "prediction": prediction,
            "candidate_count": len(self.candidates),
            "trace": list(self.trace),
        }


_lock = Lock()
_workflows: OrderedDict[str, OracleAgentWorkflow] = OrderedDict()


def _event(stage: OracleWorkflowStage, reason: str) -> dict:
    return {
        "stage": stage.value,
        "reason": reason,
        "timestamp": time.time(),
    }


def _purge(now: float) -> None:
    expired = [
        image_id
        for image_id, workflow in _workflows.items()
        if workflow.expires_at <= now
    ]
    for image_id in expired:
        _workflows.pop(image_id, None)


def register_classification(
    image_id: str,
    classification: dict,
    *,
    threshold: float,
) -> OracleAgentWorkflow:
    confidence = float(classification["prediction"]["confidence"])
    stage = (
        OracleWorkflowStage.CLASSIFIED_HIGH
        if confidence >= threshold
        else OracleWorkflowStage.CLASSIFIED_LOW
    )
    reason = (
        "classification confidence reached threshold"
        if stage == OracleWorkflowStage.CLASSIFIED_HIGH
        else "classification confidence below threshold"
    )
    now = time.monotonic()
    workflow = OracleAgentWorkflow(
        image_id=image_id,
        stage=stage,
        classification=classification,
        expires_at=now + WORKFLOW_TTL_SECONDS,
        trace=[
            _event(OracleWorkflowStage.UPLOADED, "validated image reference"),
            _event(stage, reason),
        ],
    )
    with _lock:
        _purge(now)
        _workflows[image_id] = workflow
        _workflows.move_to_end(image_id)
        while len(_workflows) > WORKFLOW_MAX_ITEMS:
            _workflows.popitem(last=False)
    return workflow


def require_retrieval(image_id: str, *, force: bool = False) -> OracleAgentWorkflow:
    now = time.monotonic()
    with _lock:
        _purge(now)
        workflow = _workflows.get(image_id)
        if workflow is None:
            raise OracleWorkflowTransitionError(
                "必须先调用 recognize_oracle_image 完成分类，才能检索候选。"
            )
        if workflow.stage == OracleWorkflowStage.CANDIDATES_RETRIEVED:
            raise OracleWorkflowTransitionError(
                "这张图片已经检索过候选，请不要重复调用检索工具。"
            )
        if workflow.stage == OracleWorkflowStage.CLASSIFIED_HIGH and not force:
            raise OracleWorkflowTransitionError(
                "当前结果已达到高置信度阈值；只有用户明确要求相似字模时才能强制检索。"
            )
        workflow.expires_at = now + WORKFLOW_TTL_SECONDS
        _workflows.move_to_end(image_id)
        return workflow


def complete_retrieval(
    image_id: str,
    candidates: list[dict],
    *,
    forced: bool,
) -> OracleAgentWorkflow:
    now = time.monotonic()
    with _lock:
        _purge(now)
        workflow = _workflows.get(image_id)
        if workflow is None:
            raise OracleWorkflowTransitionError("识别工作流已过期，请重新上传图片。")
        if workflow.stage == OracleWorkflowStage.CANDIDATES_RETRIEVED:
            raise OracleWorkflowTransitionError("候选检索已经完成。")
        workflow.stage = OracleWorkflowStage.CANDIDATES_RETRIEVED
        workflow.candidates = list(candidates)
        workflow.expires_at = now + WORKFLOW_TTL_SECONDS
        workflow.trace.append(
            _event(
                OracleWorkflowStage.CANDIDATES_RETRIEVED,
                "explicit user request" if forced else "low-confidence routing",
            )
        )
        _workflows.move_to_end(image_id)
        return workflow


def clear_workflow(image_id: str) -> None:
    with _lock:
        _workflows.pop(image_id, None)

