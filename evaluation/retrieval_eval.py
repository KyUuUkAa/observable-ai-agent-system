import csv
import json
from pathlib import Path
from statistics import mean

from rag import (
    _resume_chunks,
    search_resume_hybrid,
)

# =========================================================
# 配置
# =========================================================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = REPORT_DIR / "session_growth_eval.json"
REPORT_CSV = REPORT_DIR / "session_growth_eval.csv"
BASE_DIR = Path(__file__).resolve().parent

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(
    exist_ok=True
)

JSON_PATH = (
    REPORT_DIR
    / "retrieval_eval.json"
)

CSV_PATH = (
    REPORT_DIR
    / "retrieval_eval.csv"
)


# Relevant Margin 小于这个值，
# 即使 Hit@1 正确，也认为检索结果不够稳定
WEAK_MARGIN_THRESHOLD = 0.05


# =========================================================
# 当前知识库 Chunk 设计
#
# Chunk 0:
# 个人信息 / 求职方向
#
# Chunk 1:
# Agent / Tool Calling / Evaluation
#
# Chunk 2:
# RAG / FastAPI / Vue / PostgreSQL / 工程化
#
# Chunk 3:
# 不完整多视图甲骨文聚类
#
# Chunk 4:
# YOLO / 甲骨文图像分类
#
# Chunk 5:
# 技术栈 / 综合能力
# =========================================================


# =========================================================
# Retrieval Test Cases
#
# relevant_chunk_ids:
# 人工定义 Ground Truth
#
# 不再依赖关键词自动推断正确 Chunk
# =========================================================

TEST_CASES = [

    # =====================================================
    # Agent
    # =====================================================

    {
        "id": "agent_01",
        "query": "我的Agent项目主要做了什么？",
        "relevant_chunk_ids": [1],
    },

    {
    "id": "agent_02",
    "query": "我的智能体项目用了哪些技术？",
    "relevant_chunk_ids": [1, 2, 5],
},

    {
        "id": "agent_03",
        "query": "我的项目里有没有做工具调用？",
        "relevant_chunk_ids": [1],
    },

    {
        "id": "agent_04",
        "query": "我有没有做过Agent评估？",
        "relevant_chunk_ids": [1],
    },


    # =====================================================
    # RAG / 工程化
    # =====================================================

    {
        "id": "rag_01",
        "query": "我的项目里有没有做RAG？",
        "relevant_chunk_ids": [2],
    },

    {
        "id": "rag_02",
        "query": "我有没有做过FastAPI相关开发？",
        "relevant_chunk_ids": [2],
    },

    {
        "id": "rag_03",
        "query": "我有没有使用PostgreSQL做会话持久化？",
        "relevant_chunk_ids": [2],
    },

    {
        "id": "rag_04",
        "query": "我的Agent系统前后端是怎么实现的？",
        "relevant_chunk_ids": [2],
    },


    # =====================================================
    # YOLO / 图像分类
    # =====================================================

    {
        "id": "yolo_01",
        "query": "我的YOLO项目主要做了什么？",
        "relevant_chunk_ids": [4],
    },

    {
        "id": "yolo_02",
        "query": "我的甲骨文字图像识别项目效果怎么样？",
        "relevant_chunk_ids": [4],
    },

    {
        "id": "yolo_03",
        "query": "我做过图像分类项目吗？",
        "relevant_chunk_ids": [4],
    },

    {
        "id": "yolo_04",
        "query": "我的视觉识别项目做过哪些数据优化？",
        "relevant_chunk_ids": [4],
    },


    # =====================================================
    # 多视图聚类
    # =====================================================

    {
        "id": "mvc_01",
        "query": "我的多视图聚类项目主要研究什么？",
        "relevant_chunk_ids": [3],
    },

    {
        "id": "mvc_02",
        "query": "我有没有做过Transformer相关研究？",
        "relevant_chunk_ids": [3, 5],
    },

    {
        "id": "mvc_03",
        "query": "我的甲骨文聚类项目用了什么模型？",
        "relevant_chunk_ids": [3],
    },

    {
        "id": "mvc_04",
        "query": "我做过Diffusion相关研究吗？",
        "relevant_chunk_ids": [3, 5],
    },


    # =====================================================
    # 技术栈 / 综合能力
    # =====================================================

    {
        "id": "skill_01",
        "query": "我主要掌握哪些AI技术？",
        "relevant_chunk_ids": [5],
    },

    {
        "id": "skill_02",
        "query": "我会哪些后端和数据库技术？",
        "relevant_chunk_ids": [2, 5],
    },

    {
        "id": "skill_03",
        "query": "我有哪些计算机视觉项目经历？",
        "relevant_chunk_ids": [3, 4],
    },

    {
        "id": "skill_04",
        "query": "我有哪些大模型应用开发经验？",
        "relevant_chunk_ids": [1, 2, 5],
    },
]


# =========================================================
# 打印知识库 Chunk
# =========================================================

def print_chunks():

    print("\n")
    print("=" * 80)
    print("Resume Chunks")
    print("=" * 80)

    for index, chunk in enumerate(
        _resume_chunks
    ):

        preview = (
            chunk
            .replace("\n", " ")
            [:300]
        )

        print(
            f"\nChunk {index}"
        )

        print(
            preview
        )


# =========================================================
# 验证 Ground Truth
# =========================================================

def validate_relevant_chunk_ids(
    relevant_ids
):

    if not relevant_ids:

        return (
            False,
            "relevant_chunk_ids is empty"
        )

    invalid_ids = [
        chunk_id
        for chunk_id
        in relevant_ids
        if (
            chunk_id < 0
            or
            chunk_id
            >= len(_resume_chunks)
        )
    ]

    if invalid_ids:

        return (
            False,
            (
                "Invalid relevant chunk ids: "
                f"{invalid_ids}"
            )
        )

    return True, None


# =========================================================
# 单条 Query Evaluation
# =========================================================

def evaluate_case(
    case
):

    query = (
        case["query"]
    )

    relevant_ids = (
        case[
            "relevant_chunk_ids"
        ]
    )


    # =====================================================
    # Ground Truth 检查
    # =====================================================

    valid, reason = (
        validate_relevant_chunk_ids(
            relevant_ids
        )
    )

    if not valid:

        return {

            "id":
                case["id"],

            "query":
                query,

            "valid":
                False,

            "relevant_chunk_ids":
                relevant_ids,

            "reason":
                reason,
        }


    # =====================================================
    # 检索所有 Chunk
    #
    # 为了计算 MRR，
    # top_k 使用全部 Chunk 数量
    # =====================================================

    results = search_resume_hybrid(
    query=query,
    top_k=len(_resume_chunks),
)


    if not results:

        return {

            "id":
                case["id"],

            "query":
                query,

            "valid":
                False,

            "relevant_chunk_ids":
                relevant_ids,

            "reason":
                "RAG returned no results",
        }


    ranked_ids = [
        item["chunk_id"]
        for item
        in results
    ]


    # =====================================================
    # Hit@1
    # =====================================================

    hit_at_1 = (
        ranked_ids[0]
        in relevant_ids
    )


    # =====================================================
    # Hit@2
    # =====================================================

    hit_at_2 = any(
        chunk_id
        in relevant_ids
        for chunk_id
        in ranked_ids[:2]
    )


    # =====================================================
    # Reciprocal Rank
    # =====================================================

    first_relevant_rank = None

    for rank, chunk_id in enumerate(
        ranked_ids,
        start=1
    ):

        if chunk_id in relevant_ids:

            first_relevant_rank = (
                rank
            )

            break


    if first_relevant_rank is None:

        reciprocal_rank = 0.0

    else:

        reciprocal_rank = (
            1.0
            / first_relevant_rank
        )


    # =====================================================
    # Top1 - Top2 Margin
    #
    # 单纯观察第一名与第二名之间的分数差
    # =====================================================

    if len(results) >= 2:

        top1_top2_margin = (
            results[0]["score"]
            - results[1]["score"]
        )

    else:

        top1_top2_margin = 0.0


    # =====================================================
    # Relevant Margin
    #
    # 最佳 Relevant Chunk 分数
    # -
    # 最佳 Non-Relevant Chunk 分数
    #
    # > 0:
    # 正确内容排名领先
    #
    # < 0:
    # 错误内容排名领先
    # =====================================================

    relevant_scores = [
        item["score"]
        for item
        in results
        if item["chunk_id"]
        in relevant_ids
    ]


    non_relevant_scores = [
        item["score"]
        for item
        in results
        if item["chunk_id"]
        not in relevant_ids
    ]


    best_relevant_score = (
        max(relevant_scores)
        if relevant_scores
        else 0.0
    )


    if non_relevant_scores:

        best_non_relevant_score = (
            max(
                non_relevant_scores
            )
        )

        relevant_margin = (
            best_relevant_score
            - best_non_relevant_score
        )

    else:

        best_non_relevant_score = None

        relevant_margin = (
            best_relevant_score
        )


    # =====================================================
    # Ranking Details
    # =====================================================

    ranking_details = []

    for rank, item in enumerate(
        results,
        start=1
    ):

        chunk_id = (
            item["chunk_id"]
        )

        ranking_details.append({

            "rank":
                rank,

            "chunk_id":
                chunk_id,

            "score":
                item["score"],

            "is_relevant":
                chunk_id
                in relevant_ids,
        })


    return {

        "id":
            case["id"],

        "query":
            query,

        "valid":
            True,

        "relevant_chunk_ids":
            relevant_ids,

        "ranked_chunk_ids":
            ranked_ids,

        "ranking_details":
            ranking_details,

        "top1_chunk_id":
            results[0][
                "chunk_id"
            ],

        "top1_score":
            results[0][
                "score"
            ],

        "hit_at_1":
            int(
                hit_at_1
            ),

        "hit_at_2":
            int(
                hit_at_2
            ),

        "first_relevant_rank":
            first_relevant_rank,

        "reciprocal_rank":
            reciprocal_rank,

        "top1_top2_margin":
            top1_top2_margin,

        "best_relevant_score":
            best_relevant_score,

        "best_non_relevant_score":
            best_non_relevant_score,

        "relevant_margin":
            relevant_margin,
    }


# =========================================================
# 打印单条结果
# =========================================================

def print_case_result(
    case,
    result
):

    print("\n")
    print("-" * 80)

    print(
        f"ID: "
        f"{case['id']}"
    )

    print(
        f"Query: "
        f"{case['query']}"
    )


    if not result["valid"]:

        print(
            "[INVALID]"
        )

        print(
            f"Relevant Chunks: "
            f"{result.get('relevant_chunk_ids')}"
        )

        print(
            f"Reason: "
            f"{result['reason']}"
        )

        return


    print(
        "Relevant Chunks:",
        result[
            "relevant_chunk_ids"
        ]
    )

    print(
        "Ranking:",
        result[
            "ranked_chunk_ids"
        ]
    )

    print(
        f"Top1: "
        f"Chunk "
        f"{result['top1_chunk_id']} "
        f"| score="
        f"{result['top1_score']:.4f}"
    )

    print(
        f"Hit@1: "
        f"{result['hit_at_1']}"
    )

    print(
        f"Hit@2: "
        f"{result['hit_at_2']}"
    )

    print(
        f"RR: "
        f"{result['reciprocal_rank']:.4f}"
    )

    print(
        f"Top1-Top2 Margin: "
        f"{result['top1_top2_margin']:.4f}"
    )

    print(
        f"Relevant Margin: "
        f"{result['relevant_margin']:.4f}"
    )


# =========================================================
# Top1 Failure Analysis
# =========================================================

def print_failed_cases(
    valid_records
):

    failed_cases = [
        item
        for item
        in valid_records
        if item[
            "hit_at_1"
        ] == 0
    ]


    print("\n")
    print("=" * 80)
    print(
        "Failed Top1 Retrieval Cases"
    )
    print("=" * 80)

    print(
        f"Failed Cases: "
        f"{len(failed_cases)}"
    )


    if not failed_cases:

        print(
            "没有 Hit@1 失败样本。"
        )

        return


    for item in failed_cases:

        print("\n")
        print("-" * 80)

        print(
            f"ID: "
            f"{item['id']}"
        )

        print(
            f"Query: "
            f"{item['query']}"
        )

        print(
            f"Relevant Chunks: "
            f"{item['relevant_chunk_ids']}"
        )

        print(
            f"Ranking: "
            f"{item['ranked_chunk_ids']}"
        )

        print(
            f"Top1 Chunk: "
            f"{item['top1_chunk_id']}"
        )

        print(
            f"Top1 Score: "
            f"{item['top1_score']:.4f}"
        )

        print(
            f"First Relevant Rank: "
            f"{item['first_relevant_rank']}"
        )

        print(
            f"Relevant Margin: "
            f"{item['relevant_margin']:.4f}"
        )


        print(
            "\nRanking Details:"
        )

        for ranking in (
            item[
                "ranking_details"
            ]
        ):

            if ranking[
                "is_relevant"
            ]:

                marker = (
                    " <-- RELEVANT"
                )

            else:

                marker = ""


            print(
                f"Rank "
                f"{ranking['rank']}: "
                f"Chunk "
                f"{ranking['chunk_id']} "
                f"| score="
                f"{ranking['score']:.4f}"
                f"{marker}"
            )


# =========================================================
# Weak Retrieval Analysis
# =========================================================

def print_weak_cases(
    valid_records
):

    weak_cases = [
        item
        for item
        in valid_records
        if (
            item[
                "hit_at_1"
            ] == 1
            and
            item[
                "relevant_margin"
            ]
            < WEAK_MARGIN_THRESHOLD
        )
    ]


    print("\n")
    print("=" * 80)
    print(
        "Weak Retrieval Cases"
    )
    print("=" * 80)

    print(
        f"Threshold: "
        f"{WEAK_MARGIN_THRESHOLD:.4f}"
    )

    print(
        f"Weak Cases: "
        f"{len(weak_cases)}"
    )


    if not weak_cases:

        print(
            "没有发现弱检索样本。"
        )

        return


    for item in weak_cases:

        print("\n")
        print("-" * 80)

        print(
            f"ID: "
            f"{item['id']}"
        )

        print(
            f"Query: "
            f"{item['query']}"
        )

        print(
            f"Relevant Chunks: "
            f"{item['relevant_chunk_ids']}"
        )

        print(
            f"Top1 Chunk: "
            f"{item['top1_chunk_id']}"
        )

        print(
            f"Top1 Score: "
            f"{item['top1_score']:.4f}"
        )

        print(
            f"Relevant Margin: "
            f"{item['relevant_margin']:.4f}"
        )

        print(
            f"Ranking: "
            f"{item['ranked_chunk_ids']}"
        )


# =========================================================
# Invalid Ground Truth Analysis
# =========================================================

def print_invalid_cases(
    invalid_records
):

    print("\n")
    print("=" * 80)
    print(
        "Invalid Ground Truth Cases"
    )
    print("=" * 80)

    print(
        f"Invalid Cases: "
        f"{len(invalid_records)}"
    )


    if not invalid_records:

        print(
            "没有 Invalid Case。"
        )

        return


    for item in invalid_records:

        print("\n")
        print("-" * 80)

        print(
            f"ID: "
            f"{item['id']}"
        )

        print(
            f"Query: "
            f"{item['query']}"
        )

        print(
            f"Relevant Chunks: "
            f"{item.get('relevant_chunk_ids')}"
        )

        print(
            f"Reason: "
            f"{item['reason']}"
        )


# =========================================================
# 保存 JSON
# =========================================================

def save_json_report(
    records,
    summary
):

    output = {

        "summary":
            summary,

        "results":
            records,
    }


    JSON_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# =========================================================
# 保存 CSV
# =========================================================

def save_csv_report(
    records
):

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow([
            "id",
            "query",
            "valid",
            "relevant_chunk_ids",
            "ranked_chunk_ids",
            "top1_chunk_id",
            "top1_score",
            "hit_at_1",
            "hit_at_2",
            "first_relevant_rank",
            "reciprocal_rank",
            "top1_top2_margin",
            "best_relevant_score",
            "best_non_relevant_score",
            "relevant_margin",
            "reason",
        ])


        for item in records:

            writer.writerow([

                item.get(
                    "id"
                ),

                item.get(
                    "query"
                ),

                item.get(
                    "valid"
                ),

                item.get(
                    "relevant_chunk_ids"
                ),

                item.get(
                    "ranked_chunk_ids"
                ),

                item.get(
                    "top1_chunk_id"
                ),

                item.get(
                    "top1_score"
                ),

                item.get(
                    "hit_at_1"
                ),

                item.get(
                    "hit_at_2"
                ),

                item.get(
                    "first_relevant_rank"
                ),

                item.get(
                    "reciprocal_rank"
                ),

                item.get(
                    "top1_top2_margin"
                ),

                item.get(
                    "best_relevant_score"
                ),

                item.get(
                    "best_non_relevant_score"
                ),

                item.get(
                    "relevant_margin"
                ),

                item.get(
                    "reason"
                ),
            ])


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # 打印当前知识库
    # =====================================================

    print_chunks()


    print("\n")
    print("=" * 80)
    print(
        "Retrieval Evaluation"
    )
    print("=" * 80)


    records = []


    # =====================================================
    # 执行 Retrieval Evaluation
    # =====================================================

    for case in TEST_CASES:

        result = (
            evaluate_case(
                case
            )
        )

        records.append(
            result
        )

        print_case_result(
            case,
            result
        )


    # =====================================================
    # Valid / Invalid
    # =====================================================

    valid_records = [
        item
        for item
        in records
        if item.get(
            "valid"
        )
    ]


    invalid_records = [
        item
        for item
        in records
        if not item.get(
            "valid"
        )
    ]


    # =====================================================
    # 没有有效样本
    # =====================================================

    if not valid_records:

        print(
            "\n没有有效测试样本。"
        )

        print_invalid_cases(
            invalid_records
        )

        return


    # =====================================================
    # Aggregate Metrics
    # =====================================================

    hit_at_1 = mean(
        item[
            "hit_at_1"
        ]
        for item
        in valid_records
    )


    hit_at_2 = mean(
        item[
            "hit_at_2"
        ]
        for item
        in valid_records
    )


    mrr = mean(
        item[
            "reciprocal_rank"
        ]
        for item
        in valid_records
    )


    mean_top_margin = mean(
        item[
            "top1_top2_margin"
        ]
        for item
        in valid_records
    )


    mean_relevant_margin = mean(
        item[
            "relevant_margin"
        ]
        for item
        in valid_records
    )


    failed_top1_count = sum(
        1
        for item
        in valid_records
        if item[
            "hit_at_1"
        ] == 0
    )


    weak_case_count = sum(
        1
        for item
        in valid_records
        if (
            item[
                "hit_at_1"
            ] == 1
            and
            item[
                "relevant_margin"
            ]
            < WEAK_MARGIN_THRESHOLD
        )
    )


    # =====================================================
    # Summary
    # =====================================================

    summary = {

        "total_cases":
            len(records),

        "valid_cases":
            len(
                valid_records
            ),

        "invalid_cases":
            len(
                invalid_records
            ),

        "hit_at_1":
            hit_at_1,

        "hit_at_2":
            hit_at_2,

        "mrr":
            mrr,

        "mean_top1_top2_margin":
            mean_top_margin,

        "mean_relevant_margin":
            mean_relevant_margin,

        "failed_top1_cases":
            failed_top1_count,

        "weak_cases":
            weak_case_count,

        "weak_margin_threshold":
            WEAK_MARGIN_THRESHOLD,
    }


    print("\n")
    print("=" * 80)
    print(
        "Retrieval Evaluation Summary"
    )
    print("=" * 80)


    print(
        f"Total Cases: "
        f"{len(records)}"
    )

    print(
        f"Valid Cases: "
        f"{len(valid_records)}"
    )

    print(
        f"Invalid Cases: "
        f"{len(invalid_records)}"
    )

    print(
        f"Hit@1: "
        f"{hit_at_1:.2%}"
    )

    print(
        f"Hit@2: "
        f"{hit_at_2:.2%}"
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    print(
        f"Mean Top1-Top2 Margin: "
        f"{mean_top_margin:.4f}"
    )

    print(
        f"Mean Relevant Margin: "
        f"{mean_relevant_margin:.4f}"
    )

    print(
        f"Failed Top1 Cases: "
        f"{failed_top1_count}"
    )

    print(
        f"Weak Retrieval Cases: "
        f"{weak_case_count}"
    )


    # =====================================================
    # Error Analysis
    # =====================================================

    print_failed_cases(
        valid_records
    )

    print_weak_cases(
        valid_records
    )

    print_invalid_cases(
        invalid_records
    )


    # =====================================================
    # 保存结果
    # =====================================================

    save_json_report(
        records,
        summary
    )

    save_csv_report(
        records
    )


    print("\n")
    print("=" * 80)
    print(
        "Reports Saved"
    )
    print("=" * 80)

    print(
        f"JSON Report: "
        f"{JSON_PATH}"
    )

    print(
        f"CSV Report: "
        f"{CSV_PATH}"
    )


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    main()