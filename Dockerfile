FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OCR, PDF, and table extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy French model
RUN python -m spacy download fr_core_news_sm

# Copy application code
COPY . .

# Set environment variables (but don't hardcode credentials)
ENV PYTHONUNBUFFERED=1
# The credential will be mounted at runtime; do not set ENV for it here

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]