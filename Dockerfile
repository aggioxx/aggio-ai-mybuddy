FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY data/aggiocorp.json .
COPY index.html .

ENV WEAVIATE_URL=weaviate
ENV OPENAI_API_KEY=your_openai_api_key

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]