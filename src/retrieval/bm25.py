from dataclasses import dataclass
import math
import re


@dataclass
class BM25Result:
    chunk_id: str
    score: float
    text: str


class BM25Retriever:
    """
    Simple BM25 keyword retriever.

    Used as the lexical retrieval component
    of the OmniRAG hybrid retrieval pipeline.
    """

    def __init__(
        self,
        documents: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.avg_document_length = (
            sum(self.document_lengths)
            / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

        self.document_frequency = self._build_document_frequency()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into normalized word tokens.
        """
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def _build_document_frequency(self) -> dict[str, int]:
        """
        Count how many documents contain each term.
        """
        frequency = {}

        for tokens in self.tokenized_documents:
            unique_terms = set(tokens)

            for term in unique_terms:
                frequency[term] = (
                    frequency.get(term, 0) + 1
                )

        return frequency

    def _score(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:
        """
        Calculate the BM25 score for one document.
        """

        if not self.documents:
            return 0.0

        document_tokens = self.tokenized_documents[
            document_index
        ]

        document_length = self.document_lengths[
            document_index
        ]

        if document_length == 0:
            return 0.0

        total_documents = len(self.documents)

        score = 0.0

        term_frequencies = {}

        for token in document_tokens:
            term_frequencies[token] = (
                term_frequencies.get(token, 0) + 1
            )

        for term in query_tokens:

            if term not in term_frequencies:
                continue

            df = self.document_frequency.get(
                term,
                0,
            )

            if df == 0:
                continue

            # BM25 inverse document frequency
            idf = math.log(
                1
                + (
                    total_documents
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )

            tf = term_frequencies[term]

            denominator = (
                tf
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self.avg_document_length
                    )
                )
            )

            score += (
                idf
                * (
                    tf
                    * (self.k1 + 1)
                    / denominator
                )
            )

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25Result]:
        """
        Retrieve documents using BM25 keyword matching.
        """

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        if not self.documents:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        results = []

        for index, document in enumerate(
            self.documents
        ):
            score = self._score(
                query_tokens,
                index,
            )

            results.append(
                BM25Result(
                    chunk_id=str(
                        document["chunk_id"]
                    ),
                    score=score,
                    text=document["text"],
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]