import os
import openai
import pandas as pd
import weaviate
from langchain_openai import OpenAI  # Update the import
from weaviate.connect import ConnectionParams, ProtocolParams

# Initialize the OpenAI model
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


def insert_data_weaviate(data, client, class_name):
    for item in data:
        client.data_object.create(item, class_name)
    print("Data inserted successfully.")


def get_question():
    return input("Please enter your question: ")


def enrich_question(question):
    response = llm.generate(f"Enrich the following question for better understanding: {question}", temperature=0.5, max_tokens=60)
    return response.strip()


def search_weaviate(client, question, class_name):
    result = client.query.get(class_name, ["Questions", "Answers"]).with_near_text({"concepts": [question]}).do()
    return result


def generate_response(context, question, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"{context}\n\nQuestion: {question}\nAnswer:",
        temperature=0.7,
        max_tokens=150,
        top_p=1.0
    )
    return response


def main():
    # Configuration
    file_path = "sat_world_and_us_history.csv"
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_class_name = "MentalHealthFAQ"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    df = pd.read_csv(file_path)


    # Step 3: Get user question
    question = get_question()

    # Step 4: Enrich the question
    enriched_question = enrich_question(question)

    # Step 5: Query Weaviate for context
    weaviate_results = search_weaviate(client, enriched_question, weaviate_class_name)
    context = " ".join([item['properties']['Answers'] for item in weaviate_results['data']['Get'][weaviate_class_name]])

    # Step 6: Generate response using OpenAI
    response = generate_response(context, question, openai_api_key)
    print("Response:", response.choices[0].text.strip())


if __name__ == "__main__":
    main()
