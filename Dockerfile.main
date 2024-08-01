# Dockerfile.main

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application code and aggiocorp.json
COPY main.py .
COPY aggiocorp.json .
COPY index.html .

# Set environment variables (you can also set these in docker-compose.yml)
ENV WEAVIATE_URL=weaviate
ENV OPENAI_API_KEY=your_openai_api_key

# Run the main application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]