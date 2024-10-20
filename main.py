from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import os
import json
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, HfApiEngine
import weaviate.classes.config as wc
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AggiocorpData:
    def __init__(self, problema, solucao):
        self.problema = problema
        self.solucao = solucao

    def __repr__(self):
        return f"AggiocorpData(problema='{self.problema[:50]}...', solucao='{self.solucao[:50]}...')"


def initialize_weaviate(weaviate_url):
    client = weaviate.Client(weaviate_url)
    return client


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def insert_data(client, data):
    aggiocorp = client.data_object.get("AggiocorpData")
    client.batch(data)
    print("Data inserted into Weaviate.")


def conversational_chat(query, chain):
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    file_path = "data/aggiocorp.json"
    weaviate_url = os.getenv("WEAVIATE_URL")

    api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    HfApiEngine().client.token = api_token

    # Initialize LLaMA model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Initialize Weaviate client
    weaviate_client = initialize_weaviate(weaviate_url)
    print("Weaviate and LLM clients initialized.")

    # Read data and initialize schema in Weaviate
    data = read_json(file_path)

    try:
        weaviate_client.schema.delete_class("AggiocorpData")
    except Exception as e:
        print(f"Error deleting collection: {e}")

    # Create schema
    schema = {
        "class": "AggiocorpData",
        "description": "Schema for Aggiocorp data",
        "properties": [
            {"name": "problema", "dataType": ["text"]},
            {"name": "solucao", "dataType": ["text"]}
        ]
    }
    weaviate_client.schema.create_class(schema)
    print("Collection AggiocorpData was created.")

    # Insert data
    insert_data(weaviate_client, data)

    # Embedding model
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_auth_token=api_token)
    embedding_model = AutoModel.from_pretrained(embedding_model_name, )

    def generate_embeddings(texts):
        inputs = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    # Generate embeddings for the problems
    data_texts = [item['problema'] for item in data]
    embeddings = generate_embeddings(data_texts)

    # Initialize WeaviateVectorStore
    vectordb = WeaviateVectorStore(weaviate_client, "AggiocorpData", "problema", embeddings)

    # Define prompt template
    my_prompt_template = """
    Você é o chatbot da aggiocorp uma empresa de operações de TI e deve sempre responder como OpsBuddy um especialista em tecnologia.
    Considere o contexto definido entre [].
    Responda a pergunta entre <>.

    Contexto: [{context}]

    Pergunta: <{question}>

    Se a pergunta não estiver relacionada à tecnologia e você não encontrar contexto, diga que não sabe.
    """

    my_prompt = PromptTemplate.from_template(my_prompt_template)
    print(my_prompt_template)

    # Initialize ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=local_pipeline,
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        verbose=True,
        combine_docs_chain_kwargs={'prompt': my_prompt}
    )
    yield
    print("Cleaning up clients.")


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get():
    with open("index.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/query")
async def query(request: Request):
    data = await request.json()
    query = data.get("query")
    response = conversational_chat(query, chain)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
