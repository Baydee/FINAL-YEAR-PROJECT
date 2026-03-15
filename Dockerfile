FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies with retries and longer timeout
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

# Copy application files
COPY app.py .
COPY fraud_detection_model.py .
COPY credit_card_fraud_10k.csv .
COPY *.pkl .
COPY templates/ ./templates/

# Expose port
EXPOSE 5000

# Run the Flask app directly to keep container startup deterministic for this project.
CMD ["python", "app.py"]