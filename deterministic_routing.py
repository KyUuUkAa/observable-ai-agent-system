from __future__ import annotations

import re


UUID_PATTERN = re.compile(
    r"(?<![0-9a-fA-F])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}(?![0-9a-fA-F])"
)


def required_tool_for_input(user_input: str) -> str | None:
    """Return a high-precision tool constraint for explicit business intents.

    The LLM still extracts arguments and handles ambiguous/general language. This
    guard only constrains requests whose intent is explicit enough that skipping
    the tool would fabricate an answer without querying the source of truth.
    """

    text = user_input.strip()
    lowered = text.casefold()

    if "[oracle_image_attachment]" in lowered and "image_id=" in lowered:
        return "recognize_oracle_image"

    if any(token in text for token in ("导出", "生成CSV", "生成JSON", "文件")) and any(
        token in text for token in ("记录", "资料", "识别", "复核", "CSV", "JSON")
    ):
        return "export_oracle_records"

    if UUID_PATTERN.search(text) and any(
        token in text for token in ("确认", "接受", "驳回", "拒绝", "错误")
    ):
        return "update_review_result"

    query_verbs = ("找出", "查询", "查看", "列出", "查一下", "哪些", "给我")
    oracle_record_terms = (
        "甲骨",
        "识别结果",
        "识别记录",
        "待复核",
        "自动通过",
        "人工确认",
        "复核状态",
        "被驳回",
        "分类与检索冲突",
        "分类检索冲突",
        "类别",
        "置信度",
    )
    if any(verb in text for verb in query_verbs) and any(
        term in text for term in oracle_record_terms
    ):
        return "query_review_queue"

    metric_terms = (
        "准确率",
        "macro-f1",
        "覆盖率",
        "错误接受率",
        "模型版本",
        "回归测试",
        "质量指标",
        "top-1",
        "top-5",
    )
    project_terms = (
        "当前",
        "现在",
        "本项目",
        "我们的",
        "甲骨文模型",
        "甲骨文识别",
        "agent回归",
        "阈值",
        "自动通过",
    )
    if any(term in lowered for term in metric_terms) and any(
        term in lowered for term in project_terms
    ):
        return "get_oracle_quality_metrics"

    personal_terms = ("我的", "本人", "根据我的简历")
    resume_terms = ("项目", "经历", "技术", "简历", "成果")
    if any(term in text for term in personal_terms) and any(
        term in text for term in resume_terms
    ):
        return "search_resume"

    if any(term in text for term in ("计算", "帮我算", "等于多少")) and re.search(
        r"\d", text
    ):
        return "calculator"

    return None


def argument_hints_for_input(user_input: str) -> list[str]:
    """Extract unambiguous filter semantics for the tool-calling model."""

    text = user_input.strip()
    hints: list[str] = []

    class_match = re.search(r"类别\s*([0-9]{6})", text)
    if class_match:
        hints.append(f'class_code="{class_match.group(1)}"')

    range_match = re.search(
        r"置信度(?:在)?\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*"
        r"(?:到|至|[-~～])\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        text,
    )
    if range_match:
        hints.extend(
            [
                f"min_confidence={range_match.group(1)}",
                f"max_confidence={range_match.group(2)}",
            ]
        )
    else:
        below_match = re.search(
            r"(?:置信度\s*)?(?:低于|小于|不超过|至多)\s*"
            r"(0(?:\.\d+)?|1(?:\.0+)?)",
            text,
        )
        if below_match:
            hints.append(f"max_confidence={below_match.group(1)}")
            hints.append("do not set min_confidence from this upper bound")
        above_match = re.search(
            r"(?:置信度\s*)?(0(?:\.\d+)?|1(?:\.0+)?)\s*"
            r"(?:以上|及以上)",
            text,
        ) or re.search(
            r"(?:置信度\s*)?(?:高于|大于|至少)\s*"
            r"(0(?:\.\d+)?|1(?:\.0+)?)",
            text,
        )
        if above_match:
            hints.append(f"min_confidence={above_match.group(1)}")
            hints.append("do not set max_confidence from this lower bound")

    recent_days = re.search(r"最近\s*(\d+)\s*天", text)
    if "今天" in text:
        hints.append("days=1")
    elif "本周" in text:
        hints.append("days=7")
    elif "最近一年" in text:
        hints.append("days=365")
    elif recent_days:
        hints.append(f"days={recent_days.group(1)}")

    if "待复核" in text:
        hints.append('review_status="pending"')
    elif any(token in text for token in ("自动通过", "自动接受")):
        hints.append('review_status="auto_accepted"')
    elif any(token in text for token in ("人工确认", "已经确认", "已确认")):
        hints.append('review_status="accepted"')
    elif any(token in text for token in ("被驳回", "已驳回")):
        hints.append('review_status="rejected"')
    elif any(token in text for token in ("全部识别", "全部记录", "所有复核状态")):
        hints.append('review_status="all"')

    if "冲突" in text:
        hints.append("conflicts_only=true")
    if "CSV" in text.upper():
        hints.append('output_format="csv"')
    elif "JSON" in text.upper():
        hints.append('output_format="json"')

    return hints
