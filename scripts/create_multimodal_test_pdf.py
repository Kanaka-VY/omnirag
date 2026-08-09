from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "multimodal_test.pdf"
CHART_PATH = OUTPUT_DIR / "revenue_chart.png"
IMAGE_PATH = OUTPUT_DIR / "organization_chart.png"


def create_chart() -> None:
    years = ["2023", "2024", "2025"]
    revenue = [10, 15, 22]

    plt.figure(figsize=(6, 4))

    bars = plt.bar(
        years,
        revenue,
        color=["#4C78A8", "#59A14F", "#F28E2B"],
    )

    plt.title("Annual Revenue Growth")
    plt.xlabel("Year")
    plt.ylabel("Revenue (Million INR)")

    for bar, value in zip(bars, revenue):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value}M",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()


def create_image() -> None:
    width = 1000
    height = 500

    image = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (430, 30),
        "Company Organization",
        fill="black",
    )

    boxes = [
        ("CEO", 400, 100),
        ("Engineering", 150, 300),
        ("HR", 400, 300),
        ("Finance", 650, 300),
    ]

    box_width = 180
    box_height = 70

    for label, x, y in boxes:
        draw.rectangle(
            [
                x,
                y,
                x + box_width,
                y + box_height,
            ],
            outline="black",
            width=3,
        )

        draw.text(
            (x + 45, y + 25),
            label,
            fill="black",
        )

    # Connections from CEO
    ceo_center_x = 400 + box_width / 2
    ceo_bottom_y = 100 + box_height

    for _, x, y in boxes[1:]:
        target_x = x + box_width / 2
        target_y = y

        draw.line(
            [
                ceo_center_x,
                ceo_bottom_y,
                target_x,
                target_y,
            ],
            fill="black",
            width=3,
        )

    image.save(IMAGE_PATH)


def create_pdf() -> None:
    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    story = []

    # ---------------------------------------------------------
    # PAGE 1 - TEXT + TABLE
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "OmniRAG Multimodal Test Document",
            styles["Title"],
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Employee Performance Report",
            styles["Heading1"],
        )
    )

    story.append(
        Paragraph(
            """
            This document is created specifically to test the
            OmniRAG multimodal document ingestion pipeline.
            It contains normal text, structured tables,
            charts, and an organizational image.
            """,
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Employee Performance Table",
            styles["Heading2"],
        )
    )

    table_data = [
        [
            "Employee",
            "Department",
            "Rating",
            "Salary",
        ],
        [
            "Ravi",
            "AI",
            "4.5",
            "60000",
        ],
        [
            "Priya",
            "HR",
            "4.2",
            "55000",
        ],
        [
            "Arun",
            "Finance",
            "4.7",
            "65000",
        ],
    ]

    table = Table(
        table_data,
        colWidths=[
            1.5 * inch,
            1.5 * inch,
            1.0 * inch,
            1.2 * inch,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#D9EAF7"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.black,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.black,
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
            ]
        )
    )

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            """
            The table contains employee names, departments,
            performance ratings, and salary information.
            """
            ,
            styles["BodyText"],
        )
    )

    # ---------------------------------------------------------
    # PAGE 2 - CHART + IMAGE
    # ---------------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Revenue Analysis",
            styles["Heading1"],
        )
    )

    story.append(
        Paragraph(
            """
            The company experienced steady revenue growth
            between 2023 and 2025.
            """,
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Annual Revenue Growth Chart",
            styles["Heading2"],
        )
    )

    story.append(
        RLImage(
            str(CHART_PATH),
            width=5.5 * inch,
            height=3.6 * inch,
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            """
            Revenue increased from 10 million INR in 2023
            to 15 million INR in 2024 and 22 million INR
            in 2025.
            """,
            styles["BodyText"],
        )
    )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Company Organization",
            styles["Heading1"],
        )
    )

    story.append(
        Paragraph(
            """
            The following figure represents the company's
            organizational structure.
            """,
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        RLImage(
            str(IMAGE_PATH),
            width=6.5 * inch,
            height=3.25 * inch,
        )
    )

    document.build(story)

    print(f"Created PDF: {PDF_PATH}")
    print(f"Created chart: {CHART_PATH}")
    print(f"Created image: {IMAGE_PATH}")


def main() -> None:
    create_chart()
    create_image()
    create_pdf()


if __name__ == "__main__":
    main()