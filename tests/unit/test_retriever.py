from src.retrieval.config import RetrievalConfig
from src.retrieval.retriever import Retriever


class FakeEmbeddingModel:
    """
    Fake embedding model for unit tests.
    """

    def encode(self, texts):
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


class FakeResult:
    """
    Fake Qdrant search result.
    """

    def __init__(
        self,
        point_id,
        score,
        payload,
    ):
        self.id = point_id
        self.score = score
        self.payload = payload


class FakeVectorStore:
    """
    Fake Qdrant repository for unit tests.
    """

    def __init__(self):
        self.last_document_id = None

    def search(
        self,
        query_vector,
        limit,
        document_id=None,
    ):
        self.last_document_id = document_id

        return [
            FakeResult(
                point_id="chunk-1",
                score=0.9,
                payload={
                    "text": "Employees receive annual leave.",
                    "document_id": "doc1",
                    "document_name": "sample.pdf",
                    "section": "Leave Policy",
                    "page_numbers": [2],
                    "element_types": [
                        "NarrativeText"
                    ],
                },
            )
        ]


def test_retriever_returns_results():
    """
    Retriever should return RetrievedChunk objects.
    """

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        repository=FakeVectorStore(),
    )

    results = retriever.retrieve(
        "What is the leave policy?"
    )

    assert len(results) == 1

    assert results[0].score == 0.9

    assert (
        results[0].text
        == "Employees receive annual leave."
    )

    assert (
        results[0].document_id
        == "doc1"
    )


def test_retriever_empty_query():
    """
    Empty queries should return no results.
    """

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        repository=FakeVectorStore(),
    )

    results = retriever.retrieve("")

    assert results == []


def test_score_threshold_filters_results():
    """
    Results below the similarity threshold
    should be removed.
    """

    class LowScoreVectorStore:
        def search(
            self,
            query_vector,
            limit,
            document_id=None,
        ):
            return [
                FakeResult(
                    point_id="chunk-1",
                    score=0.4,
                    payload={
                        "text": "Irrelevant content.",
                        "document_id": "doc1",
                        "document_name": "sample.pdf",
                        "section": None,
                        "page_numbers": [1],
                        "element_types": [
                            "NarrativeText"
                        ],
                    },
                )
            ]

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        repository=LowScoreVectorStore(),
    )

    results = retriever.retrieve(
        "What is the leave policy?",
        score_threshold=0.7,
    )

    assert results == []


def test_document_filter_is_passed_to_vector_store():
    """
    document_id should be forwarded to the repository.
    """

    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        repository=vector_store,
    )

    retriever.retrieve(
        "What is the leave policy?",
        document_id="doc1",
    )

    assert vector_store.last_document_id == "doc1"


def test_config_controls_top_k():
    """
    Retrieval configuration should provide the default top_k.
    """

    class RecordingVectorStore:
        def __init__(self):
            self.received_limit = None

        def search(
            self,
            query_vector,
            limit,
            document_id=None,
        ):
            self.received_limit = limit
            return []

    vector_store = RecordingVectorStore()

    config = RetrievalConfig(
        top_k=10,
        score_threshold=None,
    )

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        repository=vector_store,
        config=config,
    )

    retriever.retrieve(
        "What is the leave policy?"
    )

    assert vector_store.received_limit == 10

def test_score_threshold_filters_results():
    """
    Results below the similarity threshold
    should be removed.
    """

    class LowScoreVectorStore:
        def search(
            self,
            query_vector,
            limit,
            document_id=None,
        ):
            return [
                FakeResult(
                    point_id="chunk-1",
                    score=0.4,
                    payload={
                        "text": "Irrelevant content.",
                        "document_id": "doc1",
                        "document_name": "sample.pdf",
                        "section": None,
                        "page_numbers": [1],
                        "element_types": [
                            "NarrativeText"
                        ],
                    },
                )
            ]

    retriever = Retriever(
        embedding_model=FakeEmbeddingModel(),
        repository=LowScoreVectorStore(),
    )

    results = retriever.retrieve(
        "What is the leave policy?",
        score_threshold=0.7,
    )

    assert results == []