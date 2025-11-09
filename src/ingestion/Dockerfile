FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG FUNCTION_TARGET=new_data_ingest_entrypoint

ENV FUNCTION_TARGET=${FUNCTION_TARGET}

CMD ["sh", "-c", "python -m functions_framework --target=${FUNCTION_TARGET} --port=${PORT:-8080}"]