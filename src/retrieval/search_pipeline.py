class SearchPipeline:

    def __init__(
        self,
        hybrid_retriever,
        reranker,
    ):
        self.hybrid_retriever = (
            hybrid_retriever
        )
        self.reranker = reranker

    def search(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
    ):
        candidates = (
            self.hybrid_retriever.retrieve(
                query,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )
        )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )