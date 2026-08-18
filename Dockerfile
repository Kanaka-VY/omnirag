FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

# OCR + PDF/document processing dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Unstructured OCR configuration
ENV OCR_AGENT=unstructured.partition.utils.ocr_models.tesseract_ocr.OCRAgentTesseract

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install --no-cache-dir -r requirements.txt

# Ensure only headless OpenCV remains installed
RUN python -m pip uninstall -y \
    opencv-python \
    opencv-contrib-python \
    opencv-contrib-python-headless \
    || true && \
    python -m pip install --no-cache-dir --force-reinstall \
    opencv-python-headless==4.12.0.88

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]