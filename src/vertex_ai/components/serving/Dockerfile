FROM python:3.12-slim

# 1. Install system dependencies (KEPT SAME)
# RDKit and scientific packages need these
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Suppress warnings (KEPT SAME)
ENV PIP_ROOT_USER_ACTION=ignore

# 3. Copy project files (KEPT SAME)
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# 4. Install Python dependencies (MODIFIED)
# We add 'fastapi', 'uvicorn', and 'google-cloud-storage' which are not in your pyproject.toml
RUN pip install --no-cache-dir fastapi uvicorn[standard] google-cloud-storage
# Install your actual package (belkaml) in editable mode
RUN pip install --no-cache-dir -e .

# 5. Copy the serving script (NEW)
# We copy the specific file we created to the app root for easy execution
COPY src/vertex_ai/components/serving.py /app/serving.py

# 6. Configuration (NEW)
# Set PYTHONPATH so the code can find 'belkaml' modules
ENV PYTHONPATH=/app/src:/app
# Expose the port Vertex AI expects
EXPOSE 8080

# 7. Entry Point (NEW)
# Start the web server pointing to the 'app' object inside 'serving.py'
CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8080"]
