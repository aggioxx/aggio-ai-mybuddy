from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI()

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

class TextRequest(BaseModel):
    text: str


@app.get("/vectorize/.well-known/ready")
def read_ready():
    return {"status": "ready"}


@app.post("/vectorize")
def vectorize_text(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return {"vector": embeddings}

# Run the app using: uvicorn transformers_api:app --host 0.0.0.0 --port 8000
