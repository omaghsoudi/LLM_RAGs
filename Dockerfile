# ================================
# Base image (Python 3.11, multi-arch)
# ================================
FROM python:3.11-slim

# ================================
# System dependencies
# ================================
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Environment variables
# ================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

# ================================
# Working directory
# ================================
WORKDIR /app

# ================================
# Install Poetry
# ================================
RUN pip install --upgrade pip && \
    pip install poetry==1.8.3

# ================================
# Copy dependency files first (cache-friendly)
# ================================
COPY pyproject.toml poetry.lock* ./

# ================================
# Install dependencies
# ================================
RUN poetry install --no-root

# ================================
# Copy the rest of the repo
# ================================
COPY . .

# ================================
# Expose port (optional, if you add API later)
# ================================
EXPOSE 8000

# ================================
# Default command
# ================================
CMD ["python", "src/Agentic_AI/run_multimodal_rag.py"]
