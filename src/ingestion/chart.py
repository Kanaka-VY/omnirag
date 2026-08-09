def build_chart_representation(
    title: str | None,
    description: str | None,
    data_text: str | None,
) -> str:

    parts = [
        "Chart/Figure:"
    ]

    if title:
        parts.append(
            f"Title: {title}"
        )

    if description:
        parts.append(
            f"Description: {description}"
        )

    if data_text:
        parts.append(
            f"Data: {data_text}"
        )

    return "\n".join(parts)