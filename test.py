import json
import time
import warnings
import weaviate
import requests
import pandas as pd
from typing import List
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.weaviate import Weaviate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
warnings.filterwarnings("ignore", category=UserWarning)

client = weaviate.Client("url to the weaviate vector database")
print("Client is available: ", client.is_ready())

print("Creating a new schema")
class_obj = {
    "class": "MenuOption",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        }
    }
}

try:
    client.schema.delete_class("MenuOption")
    print("class MenuOption was deleted.")
except:
    pass
finally:
    print("class MenuOption was created.")
    client.schema.create_class(class_obj)

MyOpenAI = ChatOpenAI(temperature=0.5,model_name='gpt-3.5-turbo',openai_api_key="here-you-set-the-openai-api-key")
# Suppose file_content contains a list of json objects
data = json.loads(file_content)
batch_size = 100
# Configure a batch process
with client.batch(batch_size=batch_size) as batch:
    # Batch import all menu options
    for i, doc in enumerate(data["response"]["docs"]):
        print(f"importing menu options: {i+1}")

        properties = {
            "title": doc["displayName"],
            "description": doc["description"],
        }

        client.batch.add_data_object(properties, "MenuOption",)

print("We are ready to run.")
embeddings = OpenAIEmbeddings(openai_api_key="your-openai-api-key-goes-here", model='gpt-3.5-turbo')
vectorstore = Weaviate(client, "MenuOption", "description", embeddings, by_text=False)
from langchain import PromptTemplate

my_template = r"""
Você é o chatbot do Itaú SuperApp e deve sempre responder como um Pirata do Caribe.
Considere o texto definido entre [].
Responda a questão entre <>

Contexto: [{context}]

Pergunta: <{question}>

Se você não encontrou contexto para esta pergunta ou se não sabe respondê-la,
responda "Sem resposta."
"""

my_prompt = PromptTemplate.from_template(my_template)

chain = ConversationalRetrievalChain.from_llm(
    llm=MyOpenAI,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    verbose=True,
    combine_docs_chain_kwargs={'prompt': my_prompt}
)

chat_history = []
def conversational_chat(query):
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]

while True:
    print("Faça uma pergunta sobre nossos menus")
    query = input("")
    start = time.time()
    response = conversational_chat(query)
    end = time.time()
    print("Elapsed time: ", end-start, "\tResposta:", response)


