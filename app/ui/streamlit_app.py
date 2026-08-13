import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="OmniRAG",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "uploaded_document" not in st.session_state:
    st.session_state.uploaded_document = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📚 OmniRAG")
st.caption(
    "Multimodal Enterprise RAG Platform"
)

st.divider()


# ---------------------------------------------------------
# Sidebar - Document Upload
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    multimodal = st.checkbox(
        "Enable multimodal processing",
        value=False,
        help=(
            "Enable extraction of images and other "
            "multimodal document elements."
        ),
    )

    if uploaded_file is not None:

        if st.button(
            "🚀 Ingest Document",
            use_container_width=True,
        ):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            }

            params = {
                "multimodal": multimodal,
            }

            try:

                with st.spinner(
                    "Processing document..."
                ):

                    response = requests.post(
                        f"{API_BASE_URL}/api/documents/upload",
                        files=files,
                        params=params,
                        timeout=300,
                    )

                if response.ok:

                    result = response.json()

                    st.session_state.uploaded_document = result

                    st.success(
                        "Document ingested successfully!"
                    )

                else:

                    try:
                        error = response.json()
                    except Exception:
                        error = response.text

                    st.error(
                        f"Upload failed: {error}"
                    )

            except requests.RequestException as exc:

                st.error(
                    "Could not connect to the FastAPI server."
                )

                st.caption(str(exc))


    # -----------------------------------------------------
    # Uploaded document information
    # -----------------------------------------------------

    if st.session_state.uploaded_document:

        document = (
            st.session_state.uploaded_document
        )

        st.divider()

        st.subheader("Indexed Document")

        st.write(
            f"**File:** "
            f"{document.get('document_name', '-')}"
        )

        st.write(
            f"**Elements:** "
            f"{document.get('elements', 0)}"
        )

        st.write(
            f"**Chunks:** "
            f"{document.get('chunks', 0)}"
        )

        st.write(
            f"**Embeddings:** "
            f"{document.get('embeddings', 0)}"
        )

        st.write(
            f"**Vector dimension:** "
            f"{document.get('vector_dimension', '-')}"
        )


# ---------------------------------------------------------
# Main chat area
# ---------------------------------------------------------

st.header("💬 Ask your documents")

if not st.session_state.uploaded_document:

    st.info(
        "Upload and ingest a PDF from the sidebar "
        "before asking questions."
    )


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("result")
        ):

            result = message["result"]

            # ---------------------------------------------
            # Citations
            # ---------------------------------------------

            citations = result.get(
                "citations",
                [],
            )

            if citations:

                with st.expander(
                    f"📚 Sources ({len(citations)})"
                ):

                    for index, citation in enumerate(
                        citations,
                        start=1,
                    ):

                        st.markdown(
                            f"**Source {index}**"
                        )

                        st.json(citation)

            # ---------------------------------------------
            # Retrieved context
            # ---------------------------------------------

            contexts = result.get(
                "retrieved_contexts",
                [],
            )

            context_ids = result.get(
                "retrieved_context_ids",
                [],
            )

            if contexts:

                with st.expander(
                    "🔎 Retrieved Context"
                ):

                    for index, context in enumerate(
                        contexts,
                        start=1,
                    ):

                        chunk_id = (
                            context_ids[index - 1]
                            if index - 1
                            < len(context_ids)
                            else "unknown"
                        )

                        st.markdown(
                            f"**Chunk {index} — "
                            f"`{chunk_id}`**"
                        )

                        st.write(context)

                        if index < len(contexts):
                            st.divider()


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    if not st.session_state.uploaded_document:

        st.warning(
            "Please upload and ingest a PDF first."
        )

        st.stop()

    # -----------------------------------------------------
    # Display user question
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------------------------------
    # Call FastAPI
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{API_BASE_URL}/api/chat",
                    json={
                        "question": question,
                    },
                    timeout=300,
                )

                if response.ok:

                    result = response.json()

                    answer = result.get(
                        "answer",
                        "No answer returned.",
                    )

                    st.markdown(answer)

                    # -------------------------------------
                    # Sources
                    # -------------------------------------

                    citations = result.get(
                        "citations",
                        [],
                    )

                    if citations:

                        with st.expander(
                            f"📚 Sources ({len(citations)})"
                        ):

                            for index, citation in enumerate(
                                citations,
                                start=1,
                            ):

                                st.markdown(
                                    f"**Source {index}**"
                                )

                                st.json(citation)

                    # -------------------------------------
                    # Retrieved context
                    # -------------------------------------

                    contexts = result.get(
                        "retrieved_contexts",
                        [],
                    )

                    context_ids = result.get(
                        "retrieved_context_ids",
                        [],
                    )

                    if contexts:

                        with st.expander(
                            "🔎 Retrieved Context"
                        ):

                            for index, context in enumerate(
                                contexts,
                                start=1,
                            ):

                                chunk_id = (
                                    context_ids[index - 1]
                                    if index - 1
                                    < len(context_ids)
                                    else "unknown"
                                )

                                st.markdown(
                                    f"**Chunk {index} — "
                                    f"`{chunk_id}`**"
                                )

                                st.write(context)

                                if (
                                    index
                                    < len(contexts)
                                ):
                                    st.divider()

                    # -------------------------------------
                    # Save assistant response
                    # -------------------------------------

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "result": result,
                        }
                    )

                else:

                    try:
                        error = response.json()
                    except Exception:
                        error = response.text

                    error_message = (
                        f"RAG query failed: {error}"
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )

            except requests.RequestException as exc:

                error_message = (
                    "Could not connect to the FastAPI server."
                )

                st.error(error_message)

                st.caption(str(exc))

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )