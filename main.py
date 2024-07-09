import os
import json
import time
import weaviate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores import Weaviate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from weaviate.collections.classes.config import Property, DataType


class AggiocorpData:
    def __init__(self, problema, solucao):
        self.problema = problema
        self.solucao = solucao

    def __repr__(self):
        return f"AggiocorpData(problema='{self.problema[:50]}...', solucao='{self.solucao[:50]}...')"


def initialize_weaviate(weaviate_url):
    weaviate_client = weaviate.connect_to_local(host=weaviate_url, port=8080)
    return weaviate_client


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def batch_insert(client, data, batch_size=100):
    try:
        with client.batch.fixed_size(batch_size=batch_size, concurrent_requests=4) as batch:
            for item in data:
                problem = AggiocorpData(
                    problema=item['problema'],
                    solucao=item['solução']
                )

                problem_data = {
                    "problema": problem.problema,
                    "solucao": problem.solucao
                }

                # Add to batch request
                batch.add_object(properties=problem_data, collection="AggiocorpData")
    finally:
        print("Batch request completed.")
        client.close()


def conversational_chat(query, chain):
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]


def main():
    file_path = "./aggiocorp.json"
    weaviate_url = os.getenv("WEAVIATE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    MyOpenAI = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo', openai_api_key="here-you-set-the-openai-api-key")
    weaviate_client = initialize_weaviate(weaviate_url)
    print("Clients initialized.")

    data = read_json(file_path)
    # V3 weaviate client (deprecated)
    # class_schema = {
    #     "class": "HistoricalEvent",
    #     "vectorizer": "text2vec-transformers",
    #     "properties": [
    #         {
    #             "name": "Event",
    #             "dataType": ["text"]
    #         },
    #         {
    #             "name": "Year",
    #             "dataType": ["int"]
    #         },
    #         {
    #             "name": "Location",
    #             "dataType": ["text"]
    #         },
    #         {
    #             "name": "Significance",
    #             "dataType": ["text"]
    #         }
    #     ]
    # }

    try:
        weaviate_client.collections.delete("aggiocorp")
    except:
        pass
    finally:
        class_schema = {
            "name": "AggiocorpData",
            "description": "Schema for Aggiocorp data",
            "vectorizer_config": "text2vec-transformers",
            "properties": [
                Property(name="problema", dataType=[DataType.TEXT]),
                Property(name="solucao", dataType=[DataType.TEXT])
            ]
        }
        weaviate_client.collections.create(**class_schema)
        print("Collection AggiocorpData was created.")

    batch_insert(weaviate_client, data)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model='gpt-3.5-turbo')
    vectorstore = Weaviate(weaviate_client, "AggiocorpData", "solucao", embeddings, by_text=False)
    my_prompt = r"""
    Você é o chatbot da aggiocorp uma empresa de operações de TI e deve sempre responder como OpsBuddy um especialista em tecnologia.
    Considere o contexto definido entre [].
    Responda a pergunta entre <>

    Contexto: [{context}]

    Pergunta: <{question}>

    Se você não encontrou contexto para esta pergunta ou não sabe respondê-la, responda "OpsBuddy não tem essa informação."
    """

    my_prompt = PromptTemplate.from_template(my_prompt)

    chain = ConversationalRetrievalChain.from_llm(
        llm=MyOpenAI,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        verbose=True,
        combine_docs_chain_kwargs={'prompt': my_prompt}
    )

    while True:
        print("Olá, sou o opsbuddy, como posso te ajudar?")
        query = input("")
        start = time.time()
        response = conversational_chat(query, chain)
        end = time.time()
        print("Tempo de resposta: ", end - start, "\tResposta:", response)


if __name__ == "__main__":
    main()
