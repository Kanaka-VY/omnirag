import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class LexicalResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class BM25Retriever:
    def __init__(
        self,
        documents: list[dict],
    ):
        self.documents = documents

        tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w[\w-]*\b",
            text.lower(),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[LexicalResult]:

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            LexicalResult(
                chunk_id=self.documents[index][
                    "chunk_id"
                ],
                text=self.documents[index]["text"],
                score=float(scores[index]),
                metadata=self.documents[index].get(
                    "metadata",
                    {},
                ),
            )
            for index in ranked_indices
        ]