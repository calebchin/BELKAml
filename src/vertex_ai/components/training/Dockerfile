FROM python:3.12-slim

# Install system dependencies for RDKit and other scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml for dependency installation
COPY pyproject.toml /app/

# Install Python dependencies
# We install the package in editable mode to include all dependencies
RUN pip install --no-cache-dir -e .

# Copy entire src directory after installing dependencies (for better caching)
COPY src/ /app/src/

# Set PYTHONPATH to include src directory
ENV PYTHONPATH=/app/src:/app

# The entry point will be specified by the KFP component
# This allows KFP to execute Python code directly in the container
