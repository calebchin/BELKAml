FROM python:3.12-slim

# Install system dependencies for RDKit and other scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Suppress pip root user warning in containers (expected behavior in Docker)
ENV PIP_ROOT_USER_ACTION=ignore

# Copy project files needed for installation
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Install Python dependencies in editable mode
RUN pip install --no-cache-dir -e .

# Set PYTHONPATH to include src directory
ENV PYTHONPATH=/app/src:/app

# The entry point will be specified by the KFP component
# This allows KFP to execute Python code directly in the container
