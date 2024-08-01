from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import os
import json
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import weaviate.classes.config as wc
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
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
    weaviate_client = weaviate.connect_to_local(host=weaviate_url, port=8080, headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY"),
    })
    return weaviate_client


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def insert_data(client, data):
    aggiocorp = client.collections.get("AggiocorpData")
    aggiocorp.data.insert_many(data)
    print("Data inserted on weaviate.")


def conversational_chat(query, chain):
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    file_path = "./aggiocorp.json"
    weaviate_url = os.getenv("WEAVIATE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    MyOpenAI = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)
    weaviate_client = initialize_weaviate(weaviate_url)
    print("Clients initialized.")

    data = read_json(file_path)

    try:
        weaviate_client.collections.delete("AggiocorpData")
    except Exception as e:
        print(f"Error deleting collection: {e}")
    finally:
        weaviate_client.collections.create(
            name="AggiocorpData",
            description="Schema for Aggiocorp data",
            vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
                model="ada",
            ),
            generative_config=wc.Configure.Generative.openai(),
            properties=[
                wc.Property(name="problema", data_type=wc.DataType.TEXT),
                wc.Property(name="solucao", data_type=wc.DataType.TEXT)]
        )
        print("Collection AggiocorpData was created.")

    insert_data(weaviate_client, data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model='text-embedding-ada-002')
    vectordb = WeaviateVectorStore(weaviate_client, "AggiocorpData", "problema", embeddings)

    my_prompt_template = """
    Você é o chatbot da aggiocorp uma empresa de operações de TI e deve sempre responder como OpsBuddy um especialista em tecnologia.
    Considere o contexto definido entre [].
    Responda a pergunta entre <>

    Contexto: [{context}]

    Pergunta: <{question}>

    Se a pergunta não estiver relacionada à tecnologia e você não encontrar contexto, diga que não sabe.
    """

    my_prompt = PromptTemplate.from_template(my_prompt_template)
    print(my_prompt_template)

    chain = ConversationalRetrievalChain.from_llm(
        llm=MyOpenAI,
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