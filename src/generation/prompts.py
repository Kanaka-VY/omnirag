SYSTEM_PROMPT = """
You are an enterprise document question-answering assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not use information that is not supported by the context.
2. If the context does not contain enough information, say that
   the information was not found in the provided documents.
3. Do not invent facts, numbers, names, dates, or policies.
4. Give a concise and clear answer.
5. When possible, refer to the source information provided with the context.
"""


def build_user_prompt(
    query: str,
    context: str,
) -> str:
    return f"""
User question:
{query}

Retrieved context:
{context}

Answer the question using ONLY the retrieved context.
"""

def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a grounded prompt using the question and
    retrieved context.

    Kept as a simple public helper for prompt testing.
    """

    return build_user_prompt(
        query=question,
        context=context,
    )

def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a grounded prompt using the question and context.
    """

    return build_user_prompt(
        query=question,
        context=context,
    )