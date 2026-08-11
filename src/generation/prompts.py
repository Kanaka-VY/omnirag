SYSTEM_PROMPT = """
You are an enterprise document question-answering assistant.

Answer the user's question using only the provided context.

Rules:

1. Do not use information that is not supported by the context.
2. If the context does not contain enough information, say that
   the information was not found in the provided documents.
3. Do not invent facts, numbers, names, dates, or policies.
4. Give a concise and clear answer.
5. Always answer in a complete sentence.
6. Include the relevant person's name, entity, or subject when needed
   so the answer is understandable without seeing the question.
7. Do not answer with only a number, name, date, or short fragment.
8. Do not add unnecessary explanations or information.
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
Provide a concise, complete sentence as the answer.
"""


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