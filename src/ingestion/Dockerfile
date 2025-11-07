# Use the official Python 3.12 image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source
COPY . .

# Set the function target as a build argument (can be overridden)
ARG FUNCTION_TARGET=new_data_ingest_entrypoint

# Set the command to run the function
CMD ["functions-framework", "--target=${FUNCTION_TARGET}"]