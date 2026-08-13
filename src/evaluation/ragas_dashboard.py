import json
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# Configuration
# =========================================================

RESULTS_PATH = Path(
    "data/evaluation/results/ragas_results.json"
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="OmniRAG - RAGAS Evaluation",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# Load results
# =========================================================

@st.cache_data
def load_results() -> dict:

    if not RESULTS_PATH.exists():
        return {
            "error": (
                f"Evaluation results not found: "
                f"{RESULTS_PATH}"
            )
        }

    try:

        with RESULTS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except json.JSONDecodeError as exc:

        return {
            "error": (
                "The evaluation JSON file is invalid: "
                f"{exc}"
            )
        }


data = load_results()


# =========================================================
# Error handling
# =========================================================

if "error" in data:

    st.error(data["error"])

    st.info(
        "Run the RAGAS evaluation first."
    )

    st.stop()


# =========================================================
# Header
# =========================================================

st.title("📊 OmniRAG — RAGAS Evaluation Dashboard")

st.caption(
    "Evaluation results for the OmniRAG "
    "multimodal enterprise RAG pipeline."
)


# =========================================================
# Evaluation status
# =========================================================

num_records = data.get(
    "num_records",
    len(data.get("results", [])),
)

st.success(
    f"Evaluation completed successfully — "
    f"{num_records} questions evaluated."
)


# =========================================================
# Overall metrics
# =========================================================

faithfulness = float(
    data.get(
        "average_faithfulness",
        0.0,
    )
)

context_precision = float(
    data.get(
        "average_context_precision",
        0.0,
    )
)

context_recall = float(
    data.get(
        "average_context_recall",
        0.0,
    )
)

answer_relevancy = float(
    data.get(
        "average_answer_relevancy",
        0.0,
    )
)


# =========================================================
# Metric cards
# =========================================================

st.subheader("Overall RAGAS Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Faithfulness",
        f"{faithfulness:.4f}",
    )

with col2:

    st.metric(
        "Context Precision",
        f"{context_precision:.4f}",
    )

with col3:

    st.metric(
        "Context Recall",
        f"{context_recall:.4f}",
    )

with col4:

    st.metric(
        "Answer Relevancy",
        f"{answer_relevancy:.4f}",
    )


# =========================================================
# Metric interpretation
# =========================================================

st.subheader("Metric Interpretation")

interpretation_col1, interpretation_col2 = (
    st.columns(2)
)

with interpretation_col1:

    st.markdown(
        """
### Faithfulness

Measures whether the generated answer
is supported by the retrieved context.

### Context Precision

Measures whether the retrieved context
contains relevant information for the question.
"""
    )

with interpretation_col2:

    st.markdown(
        """
### Context Recall

Measures whether the required information
was successfully retrieved.

### Answer Relevancy

Measures how relevant the generated answer
is to the user's question.
"""
    )


# =========================================================
# Metric comparison
# =========================================================

st.subheader("Metric Comparison")

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Faithfulness",
            "Context Precision",
            "Context Recall",
            "Answer Relevancy",
        ],
        "Score": [
            faithfulness,
            context_precision,
            context_recall,
            answer_relevancy,
        ],
    }
)

chart_df = metrics_df.set_index(
    "Metric"
)

st.bar_chart(
    chart_df,
    y="Score",
)


# =========================================================
# Per-question results
# =========================================================

results = data.get(
    "results",
    [],
)

st.subheader("Per-Question Evaluation")

if not results:

    st.warning(
        "No question-level results found."
    )

else:

    question_rows = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        question_rows.append(
            {
                "Question": (
                    f"Q{index}: "
                    f"{result.get('question', '')}"
                ),
                "Faithfulness": round(
                    float(
                        result.get(
                            "faithfulness",
                            0.0,
                        )
                    ),
                    4,
                ),
                "Context Precision": round(
                    float(
                        result.get(
                            "context_precision",
                            0.0,
                        )
                    ),
                    4,
                ),
                "Context Recall": round(
                    float(
                        result.get(
                            "context_recall",
                            0.0,
                        )
                    ),
                    4,
                ),
                "Answer Relevancy": round(
                    float(
                        result.get(
                            "answer_relevancy",
                            0.0,
                        )
                    ),
                    4,
                ),
            }
        )

    question_df = pd.DataFrame(
        question_rows
    )

    st.dataframe(
        question_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# Detailed question inspector
# =========================================================

st.subheader("🔎 Question Inspector")

if results:

    question_labels = []

    for index, result in enumerate(
        results,
        start=1,
    ):

        question_labels.append(
            f"Q{index}: "
            f"{result.get('question', '')}"
        )

    selected_question = st.selectbox(
        "Select a question",
        question_labels,
    )

    selected_index = question_labels.index(
        selected_question
    )

    selected = results[selected_index]


    # -----------------------------------------------------
    # Question
    # -----------------------------------------------------

    st.markdown("### Question")

    st.write(
        selected.get(
            "question",
            "",
        )
    )


    # -----------------------------------------------------
    # Reference answer
    # -----------------------------------------------------

    st.markdown("### Reference Answer")

    st.info(
        selected.get(
            "reference",
            "",
        )
    )


    # -----------------------------------------------------
    # Generated answer
    # -----------------------------------------------------

    st.markdown("### Generated Answer")

    st.success(
        selected.get(
            "response",
            "",
        )
    )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    st.markdown("### RAGAS Scores")

    (
        detail_col1,
        detail_col2,
        detail_col3,
        detail_col4,
    ) = st.columns(4)

    with detail_col1:

        st.metric(
            "Faithfulness",
            f"{float(selected.get('faithfulness', 0.0)):.4f}",
        )

    with detail_col2:

        st.metric(
            "Context Precision",
            f"{float(selected.get('context_precision', 0.0)):.4f}",
        )

    with detail_col3:

        st.metric(
            "Context Recall",
            f"{float(selected.get('context_recall', 0.0)):.4f}",
        )

    with detail_col4:

        st.metric(
            "Answer Relevancy",
            f"{float(selected.get('answer_relevancy', 0.0)):.4f}",
        )


    # -----------------------------------------------------
    # Retrieved contexts
    # -----------------------------------------------------

    st.markdown("### Retrieved Contexts")

    contexts = selected.get(
        "retrieved_contexts",
        [],
    )

    if contexts:

        for index, context in enumerate(
            contexts,
            start=1,
        ):

            with st.expander(
                f"Context {index}"
            ):

                st.text(
                    context
                )

    else:

        st.warning(
            "No retrieved contexts available."
        )


    # -----------------------------------------------------
    # Retrieved chunk IDs
    # -----------------------------------------------------

    st.markdown("### Retrieved Chunk IDs")

    chunk_ids = selected.get(
        "retrieved_context_ids",
        [],
    )

    if chunk_ids:

        for index, chunk_id in enumerate(
            chunk_ids,
            start=1,
        ):

            st.code(
                f"{index}. {chunk_id}"
            )

    else:

        st.info(
            "No chunk IDs available."
        )


    # -----------------------------------------------------
    # Citations
    # -----------------------------------------------------

    st.markdown("### Citations")

    citations = selected.get(
        "citations",
        [],
    )

    if citations:

        citation_rows = []

        for citation in citations:

            page_numbers = citation.get(
                "page_numbers",
                [],
            )

            if isinstance(
                page_numbers,
                list,
            ):

                page_text = ", ".join(
                    str(page)
                    for page in page_numbers
                )

            else:

                page_text = str(
                    page_numbers
                )

            citation_rows.append(
                {
                    "Chunk ID": citation.get(
                        "chunk_id",
                        "",
                    ),
                    "Document ID": citation.get(
                        "document_id",
                        "",
                    ),
                    "Document": citation.get(
                        "document_name",
                        "",
                    ),
                    "Page": page_text,
                    "Content Type": citation.get(
                        "content_type",
                        "",
                    ),
                }
            )

        citation_df = pd.DataFrame(
            citation_rows
        )

        st.dataframe(
            citation_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No citations available."
        )


# =========================================================
# Raw JSON
# =========================================================

with st.expander(
    "View Raw Evaluation JSON"
):

    st.json(data)


# =========================================================
# Refresh
# =========================================================

st.divider()

if st.button(
    "🔄 Refresh Evaluation Results"
):

    st.cache_data.clear()

    st.rerun()


# =========================================================
# Footer
# =========================================================

st.caption(
    f"Source: {RESULTS_PATH}"
)