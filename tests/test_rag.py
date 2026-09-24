from rag import search_resume_rag


queries = [
    "我有什么大模型应用开发方面的经历？",
    "我的Agent项目主要做了什么？",
    "我做过哪些智能体相关项目？",
    "我的YOLO项目主要做了什么？",
    "我的多视图聚类项目用了什么技术？"
]


for query in queries:

    print("\n")
    print("=" * 60)

    print(
        f"Query: {query}"
    )

    print("=" * 60)

    results = search_resume_rag(
        query=query,
        top_k=2
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nTop-{rank}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print("\nContent:")

        print(
            result["content"]
        )

        print(
            "\n--------------------"
        )